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
import random
import sqlite3
from functools import partial
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from datasets import Dataset
#from vllm.model_executor.parallel_utils.parallel_state import ( destroy_model_parallel)
from vllm.distributed import destroy_model_parallel
from torch.distributed import destroy_process_group
from nltk import sent_tokenize
from multiprocessing import Pool, current_process

# local imports
from llm_eval.utils import (
    display,
    files,
)
from llm_eval.llm import *
from llm_eval.prompt_generator import TELeR, PromptGenerator

def create_problems(ds_names, row_ids, texts, **kwargs):
    db_file = kwargs['db_file']
    con = sqlite3.connect(db_file)
    cur = con.cursor()

    for ds_name, row_id, text in zip(ds_names, row_ids, texts):
        pass


def infilling(args, cfg, keywords):

    ######### unpack vars from cfg ##########

    ds_cache = cfg.infill['ds_cache']
    step_size = cfg.infill['checkpoint_interval']
    model = args.model

    limit = args.limit

    if args.from_ckpt:
        save_loc = args.from_ckpt
    else:
        # create database file
        save_loc = f'{cfg.infill["save_dir"]}/infill/{keywords["timestamp"]}'
        files.create_path(save_loc)
    db_file = f'{save_loc}/data.db'
    db_is_new = not files.path_exists(db_file)

    con = sqlite3.connect(db_file)
    cur = con.cursor()
    res = cur.execute(
        """
        CREATE TABLE IF NOT EXISTS source_data (
            dataset TEXT,
            row_id BIGINT,
            text TEXT,
            PRIMARY KEY (dataset, row_id)
        );
        """
    )

    # load in datasets
    ds_names = cfg.infill['target_data']
    for name in tqdm(ds_names, desc='populating source document table'):
        name_params = cfg.datasets[name]
        text_col = name_params['text_column']
        split = name_params['split']
        name_args = name_params.get('args', [])

        temp_ds = datasets.load_dataset(
            name,
            *name_args,
            cache_dir=cfg.infill['ds_cache']
        )
        temp_ds = temp_ds[split]
        temp_ds = temp_ds.rename_column(text_col, 'text')
        temp_ds = temp_ds.select_columns('text')

        N = len(temp_ds)
        cur.executemany(
            "INSERT OR IGNORE INTO source_data VALUES(?, ?, ?)",
            zip([name]*N, range(N), temp_ds['text'])
        )
        con.commit()

    text_data = list(cur.execute(
        'SELECT dataset, row_id, text FROM source_data'
    ))
    N = len(text_data)
    num_proc = cfg.infill['num_proc']
    chunk_size = cfg.infill['chunk_size']

    with Pool(num_proc) as p:
        p.starmap(
            func=partial(create_problems, **{'db_file': db_file}), 
            iterable=text_data, 
            chunksize=chunk_size,
        )

    breakpoint()

    res = cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fitb_problems (
            dataset TEXT,
            row_id BIGINT,
            problem TEXT,
            answer TEXT,
            unit INT,
            n INT,
            unit_idx INT,
            PRIMARY KEY (dataset, row_id, unit, n, unit_idx)
            FOREIGN KEY (dataset, row_id) REFERENCES source_data
        );
        """
    )
    con.close()


    # create dataset samples
    def map_fn(batch, **fn_kwargs):
        N = len(batch['source'])
        samples = [
            {'source': batch['source'][n], 
                'idx': batch['idx'][n],
                'ds': batch['ds'][n]}
            for n in range(N)
        ]
        out_batch = {
            'source': [],
            'idx': [],
            'ds': [],
            'problem': [],
            'answer': [],
            'unit': [],
            'n': [],
            'unit_idx': []
        }

        for sample in samples:

            #id_base = sample['id']

            # read in limits
            max_sents = fn_kwargs['max_sents']
            max_words = fn_kwargs['max_words']

            # split into words and sentences
            words = sample['source'].split()
            sents = sent_tokenize(sample['source'])

            # compute minimums then set to 0 if minimum is negative
            max_sents = max(min(max_sents, len(sents)-2), 0)
            max_words = max(min(max_words, len(words)-2), 0)
            blank = ' ______ '
            count = 0
            for n_words in range(1, max_words+1):
                for start_idx in range(1, len(words)-n_words):
                    count += 1
                    out_batch['answer'].append(' '.join(
                        words[start_idx:start_idx+n_words]
                    ))
                    out_batch['problem'].append(
                        ' '.join(words[:start_idx])
                        + blank
                        + ' '.join(words[start_idx+n_words:])
                    )
                    out_batch['unit'].append('word(s)')
                    out_batch['n'].append(n_words)
                    out_batch['unit_idx'].append(start_idx)
            
            for n_sents in range(1, max_sents+1):
                for start_idx in range(1, len(sents)-n_sents):
                    count += 1
                    out_batch['answer'].append(' '.join(
                        sents[start_idx:start_idx+n_sents]
                    ))
                    out_batch['problem'].append(
                        ' '.join(sents[:start_idx])
                        + blank
                        + ' '.join(sents[start_idx+n_words:])
                    )
                    out_batch['unit'].append('sentence(s)')
                    out_batch['n'].append(n_sents)
                    out_batch['unit_idx'].append(start_idx)
            out_batch['source'] += [sample['source']]*count
            out_batch['ds'] += [sample['ds']]*count
            out_batch['idx'] += [sample['idx']]*count

        return out_batch
    
    ds = ds.map(
        map_fn,
        batched=True,
        batch_size=32,
        fn_kwargs={
            'max_sents': cfg.infill['max_blank_sents'],
            'max_words': cfg.infill['max_blank_words']},
    )

    gen = PromptGenerator(cfg, args.procedure)
    ds = gen.prepare_infill_data(ds)
    breakpoint()
    
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
    if len(responses) >= len(ds):
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
            out_ds = ds.add_column(
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
        len(ds), 
        step_size,
    )

    # collect responses and save in batches
    display.info(f'collecting responses with {model}')
    for start in tqdm(milestones, total=len(milestones)):
        stop = start+step_size
        responses += session.get_response(
            ds[start:stop]['prompt_text'], 
            ds[start:stop]['sys_text'],
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
    out_ds = ds.add_column('response', responses)
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