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
#from vllm.model_executor.parallel_utils.parallel_state import ( destroy_model_parallel)
from vllm.distributed import destroy_model_parallel
from torch.distributed import destroy_process_group

# local imports
from llm_eval.utils import (
    display,
    files,
)
from llm_eval.llm import *
from llm_eval.prompt_generator import TELeR, PromptGenerator

def do_infilling(args, cfg, keywords):

    ######### unpack vars from cfg ##########

    ds_cache = cfg.infill['ds_cache']
    step_size = cfg.infill['checkpoint_interval']
    model = args.model

    if args.from_ckpt:
        # try to load checkpoint file
        save_dir = files.full_path(args.from_ckpt)
        prog_path = f'{save_dir}/progress.json'
#        ckpt_path = f'{save_dir}/checkpoint.json'
        try:
#            with open(ckpt_path) as f:
#                ckpt = json.loads(f.read())
            with open(prog_path) as f:
                prog = json.loads(f.read())
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
        prog_path = f'{save_dir}/progress.json'
#        ckpt_path = f'{save_dir}/checkpoint.json'
#        ckpt = {}
#        files.create_path(save_dir)
#        with open(ckpt_path, 'w') as f:
#            f.write(json.dumps(ckpt))
        prog = {}
        files.create_path(save_dir)
        with open(prog_path, 'w') as f:
            f.write(json.dumps(prog))

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

    # loop over models and generate data
    start_time = time.time()
    num_generated = 0

    # load checkpointed data or initialize the list for data
    #responses = ckpt.get(model, [])
    ckpt_path = f'{save_dir}/{model}/checkpoint.json'
    out_pth = f'{save_dir}/{model}/results.json'
    if model in prog:
        try:
            with open(ckpt_path) as f:
                responses = json.load(f)
        except Exception as e:
            display.warning(
                'progress data exists but checkpoint file does not '
                'exist. regenerating data ...'
            )
            prog[model] = 0
            responses = []
    else:
        responses = []

    # check if generation is already completed
    if len(responses) >= len(prompt_data):
        display.info(
            'checkpoint data contains all responses. '
            'checking if data exists ...'
        )
        if files.path_exists(out_pth):
            display.ok(
                f'data found. skipping generation for '
                f'{model}'
            )
            return
        else:
            # save ds to output directory
            display.info(f'data not found. saving to {out_pth}')
            out_ds = prompt_data.add_column(
                'response', 
                responses
            )
            files.create_path(files.dirname(out_pth))
            out_ds.to_json(out_pth)
            return

    session = select_chat_model(cfg, model)

    # find starting index (based on whether checkpoint exists)
    start_idx = len(responses)
    if start_idx != 0:
        display.info(
            f'checkpoint found. starting from sample '
            f'{start_idx}.'
        )

    milestones = np.arange(
        start_idx, 
        len(prompt_data), 
        step_size,
    )

    # collect responses and save in batches
    display.info(f'collecting responses with {model}')
    for start in tqdm(milestones, total=len(milestones)):
        stop = start+step_size
        responses += session.get_response(
            prompt_data[start:stop]['prompt'], 
            prompt_data[start:stop]['system'],
            prog_bar=False,
        )
        prog[model] = len(responses)
        num_generated += step_size
        files.create_path(ckpt_path, is_file=True)
        with open(ckpt_path, 'w') as f:
            #ckpt[model] = responses
            f.write(json.dumps(responses))
        with open(prog_path, 'w') as f:
            f.write(json.dumps(prog))


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

    display.ok('Finished generating responses')
