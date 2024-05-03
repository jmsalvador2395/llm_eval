"""
loops over the all-sides/privacy-policy datasets and prompts LLMs based 
off certain fields
"""

# external imports
import ray
import torch
import gc
import os
import sys
import traceback
import json
import datasets
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from datasets import Dataset
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel
)
from torch.distributed import destroy_process_group

# local imports
from llm_eval.utils import (
    display,
    files,
)
from llm_eval.llm import *
from llm_eval.prompt_generator import TELeR, PromptGenerator

def run(args, cfg, keywords):

    ######### unpack vars from cfg ##########

    ds_cache = cfg.resp_coll['ds_cache']
    num_attempts = cfg.resp_coll['num_attempts']
    step_size = cfg.resp_coll['checkpoint_interval']
    if args.from_ckpt:
        # try to load checkpoint file
        save_dir = files.full_path(args.from_ckpt)
        ckpt_path = f'{save_dir}/checkpoint.json'
        try:
            with open(ckpt_path) as f:
                ckpt = json.loads(f.read())
        except Exception as e:
            display.error(
                f'Argument `--from_ckpt`: '
                f'checkpoint directory `{save_dir}` does not contain '
                f'any checkpointing info.'
            )
            print(e)
            os._exit(1)
    else:
        # initilaize save directory and progress file
        save_dir = (
            f'{cfg.resp_coll["save_dir"]}/'
            f'{keywords["timestamp"]}'
        )
        ckpt_path = f'{save_dir}/checkpoint.json'
        ckpt = {}
        files.create_path(save_dir)
        with open(ckpt_path, 'w') as f:
            f.write(json.dumps(ckpt))


    ######### unpack vars from args ######### 
    
    limit = args.limit

    #########################################

    # load in datasets
    ds_dict = {}
    try:
        d_sets = cfg.datasets
        for ds_name, d_set in d_sets.items():
            ds = datasets.load_dataset(
                d_set['type'],
                data_files=d_set['path'],
                cache_dir=ds_cache,
            )
            ds_dict[ds_name] = ds['train']
    except Exception as e:
        display.error(
            f'required config parameters not provided for resp_coll -> '
            f'datasets. refer to '
            f'\'{files.project_root()}/cfg/template.yaml\' for examples'
        )
        #traceback.print_exception(*sys.exc_info())
        print(e)
        os._exit(1)

    prmpt_gen = PromptGenerator(cfg)
    prompt_data = prmpt_gen.prepare_data(ds_dict)

    start_time = time.time()
    num_generated = 0
    for ds_name in ds_dict.keys():
        for model in cfg.resp_coll.get('models', ['gpt-3.5-turbo']):

            # load checkpointed data or initialize the list for data
            responses = ckpt.get(model, [])
            out_pth = (f'{save_dir}/{model}/results.json')

            # check if generation is already completed
            if len(responses) == len(prompt_data):
                display.info(
                    'checkpoint data contains allresponses. '
                    'checking if data exists ...'
                )
                if files.path_exists(out_pth):
                    display.ok(
                        f'data found. skipping generation for '
                        f'{model}'
                    )
                    continue
                else:
                    # save ds to output directory
                    display.info(f'data not found. saving to {out_pth}')
                    out_ds = prompt_data.add_column(
                        'response', 
                        responses
                    )
                    files.create_path(files.dirname(out_pth))
                    out_ds.to_json(out_pth)
                    continue

            session = select_chat_model(cfg, model)

            if files.path_exists(out_pth):
                pass

            # find starting index (based on whether checkpoint exists
            start_idx = len(responses)
            if start_idx != 0:
                display.info(
                    f'checkpoint found. starting from sample '
                    f'{start_idx}.'
                )
            milestones = sliding_window_view(
                np.arange(
                    start_idx, 
                    len(prompt_data), 
                    step_size),
                2
            )

            # collect responses and save in batches
            display.info(f'collecting responses with {model}')
            for start, stop in tqdm(milestones, total=len(milestones)):
                
                responses += session.get_response(
                    prompt_data[start:stop]['prompt'], 
                    prompt_data[start:stop]['system'],
                    prog_bar=False,
                )
                num_generated += step_size
                with open(ckpt_path, 'w') as f:
                    ckpt[model] = responses
                    f.write(json.dumps(ckpt))


            # saves ds to output directory
            out_ds = prompt_data.add_column('response', responses)
            files.create_path(files.dirname(out_pth))
            out_ds.to_json(out_pth)

            ckpt_time = time.time()
            elapsed = ckpt_time-start_time
            display.info(
                f'Elapsed Time: {elapsed:.02f} seconds '
                f'({elapsed/60:.02f} minutes or {elapsed/3600:.02f} '
                f'hours)'
            )

            # session cleanup
            del session
            ray.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                destroy_process_group()

    display.ok('Finished generating responses')
