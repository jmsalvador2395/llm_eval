""" loops over the all-sides/privacy-policy datasets and prompts LLMs based off certain fields """

# external imports
import ray
import torch
import gc
import os
import sys
import traceback
import datasets
import time
from datasets import Dataset
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
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
    save_dir = cfg.resp_coll['save_dir']

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
            'required config parameters not provided for response_collection -> datasets. '
            + f'refer to \'{files.project_root()}/cfg/template.yaml\' for examples'
        )
        traceback.print_exception(*sys.exc_info())
        os._exit(0)

    prmpt_gen = PromptGenerator(cfg)
    prompt_data = prmpt_gen.prepare_data(ds_dict)

    # load teler
    start_time = time.time()
    for ds_name in ds_dict.keys():
        for model_name in cfg.response_collection.get('models', ['gpt-3.5-turbo']):
            session = select_chat_model(cfg, model_name)

            # collect responses
            responses = session.get_response(
                prompt_data['prompt'], 
                prompt_data['system'],
            )
            out_ds = prompt_data.add_column('response', responses)

            # saves ds to output directory
            out_pth = f'{save_dir}/{model_name}/results.json'
            files.create_path(files.dirname(out_pth))
            out_ds.to_json(out_pth)

            ckpt_time = time.time()
            elapsed = ckpt_time-start_time
            display.info(
                f'Elapsed Time: {elapsed:.02f} seconds ({elapsed/60:.02f} minutes or {elapsed/3600:.02f} hours)'
            )

            """
            for lv in teler.get_levels(ds_name):
                display.info(f'generating: dataset - {ds_name}, model - {model_name}, level - {lv} prompt')
                if limit is None:
                    limit = len(ds_dict[ds_name])

                out_ds = ds_dict[ds_name].select(range(limit))
                out_ds = teler.format_data(out_ds, ds_name, lv)

                responses = session.get_response(
                    out_ds['prompt_text'], 
                    out_ds['system_text'],
                )
                out_ds = out_ds.add_column('response', responses)

                # saves ds to output directory
                out_pth = f'{save_dir}/{keywords["timestamp"]}/{ds_name}/{model_name}/level{lv}.json'
                files.create_path(files.dirname(out_pth))
                out_ds.to_json(out_pth)
                
                ckpt_time = time.time()
                elapsed = ckpt_time-start_time
                display.info(
                    f'Elapsed Time: {elapsed:.02f} seconds ({elapsed/60:.02f} minutes or {elapsed/3600:.02f} hours)'
                )
            """

            # session cleanup
            del session
            ray.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
            destroy_model_parallel()
            if torch.distributed.is_initialized():
                destroy_process_group()

    display.ok('Finished generating responses')
