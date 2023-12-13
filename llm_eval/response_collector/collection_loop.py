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

# local imports
from llm_eval.utils import (
    display,
    files,
)
from llm_eval.llm import *
from llm_eval.teler import TELeR

def run(args, cfg, keywords):

    ######### unpack vars from cfg ##########

    ds_cache = cfg.response_collection['ds_cache']
    num_attempts = cfg.response_collection['num_attempts']
    prompt = cfg.response_collection['prompt']
    save_dir = cfg.response_collection['save_dir']

    ######### unpack vars from args ######### 
    
    limit = args.limit

    #########################################

    # load in datasets
    ds_dict = {}
    try:
        d_sets = cfg.response_collection['datasets']
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

    # load teler
    teler = TELeR(cfg)
    start_time = time.time()
    for ds_name in ds_dict.keys():
        display.info(f'generating data for dataset "{ds_name}"')
        if limit is None:
            limit = len(ds_dict[ds_name])

        for model_name in cfg.response_collection.get('models', ['gpt-3.5-turbo']):
            display.info(f'generating data for model "{model_name}"')
            session = select_chat_model(cfg, model_name)

            for lv in teler.get_levels(ds_name):
                display.info(f'generating data for level {lv}')

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
                    f'Elapsed Time: {elapsed:.02f} seconds ({elapsed/60:.02f} minutes or {elapsed/3600:.02f} hours'
                )

            del session
            ray.shutdown()
            gc.collect()
            torch.cuda.empty_cache()

    display.ok('Finished generating responses')
