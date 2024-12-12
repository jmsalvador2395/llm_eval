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
from typing import List
import re

# local imports
from llm_eval.utils import (
    display,
    files,
)
from llm_eval.llm import *
from llm_eval.prompt_generator import TELeR, PromptGenerator
from llm_eval.llm.generators.vllm import VLLM
from llm_eval.llm.session import Session

def create_problems(
    ds_name: str, 
    ref_id: int, 
    text: str, 
    max_sents: int,
    max_words: int,
    db_file: str,
    one_of_each: bool=False,
    **kwargs
):

    """populates the 'fitb_problems' table"""
    if one_of_each:
        (ds_names, row_ids, answers, 
         problems, units, ns, unit_indices) = get_one_of_each_problem(
            text, ref_id, ds_name, max_sents, max_words
        )
    else:
        (ds_names, row_ids, answers, 
         problems, units, ns, unit_indices) = get_all_problems(
            text, ref_id, ds_name, max_sents, max_words
        )

    N = len(ds_names)
    con = sqlite3.connect(db_file, check_same_thread=False)
    cur = con.cursor()
    keys = [
        'dataset', 'ref_id', 'problem', 'answer', 
        'unit', 'n', 'unit_idx',
    ]
    res = cur.executemany(
        f"""
            INSERT OR IGNORE INTO fitb_problems 
            ({','.join(keys)})
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        zip(ds_names, row_ids, problems, 
            answers, units, ns, unit_indices)
    )

    con.commit()
    cur.close()
    con.close()

def get_one_of_each_problem(text, row_id, ds_name, max_sents, max_words):
    answers = []
    problems = []
    units = []
    ns = []
    unit_indices = []

    # split into words and sentences
    words = text.split()
    sents = sent_tokenize(text)

    # clean the text
    clean_words = [re.sub(r"[^\w\s]+", '', x) for x in words]
    clean_words = list(filter(lambda x: x != '', clean_words))

    clean_sents = [re.sub(r"[^\w\s]+", '', x) for x in sents]
    clean_sents = list(filter(lambda x: x != '', clean_sents))

    blank = ' ______ '
    count = 0

    # compute minimums then set to 0 if minimum is negative
    max_sents = max(min(max_sents, len(clean_sents)-2), 0)
    max_words = max(min(max_words, len(clean_words)-2), 0)

    for n_words in range(1, max_words+1):
        start_idx = random.randint(1, len(clean_words)-n_words-1)
        count += 1
        answers.append(' '.join(
            clean_words[start_idx:start_idx+n_words]
        ))
        problems.append(
            ' '.join(clean_words[:start_idx])
            + blank
            + ' '.join(clean_words[start_idx+n_words:])
        )
        units.append('word(s)')
        ns.append(n_words)
        unit_indices.append(start_idx)
    
    for n_sents in range(1, max_sents+1):
        start_idx = random.randint(1, len(clean_sents)-n_sents-1)
        count += 1
        answers.append(' '.join(
            clean_sents[start_idx:start_idx+n_sents]
        ))
        problems.append(
            ' '.join(clean_sents[:start_idx])
            + blank
            + ' '.join(clean_sents[start_idx+n_sents:])
        )
        units.append('sentence(s)')
        ns.append(n_sents)
        unit_indices.append(start_idx)
    N = len(answers)
    ds_names = [ds_name]*N
    row_ids = [row_id]*N

    return (
        ds_names, row_ids, answers, 
        problems, units, ns, unit_indices
    )


def get_all_problems(text, row_id, ds_name, max_sents, max_words):
    answers = []
    problems = []
    units = []
    ns = []
    unit_indices = []

    # split into words and sentences
    words = text.split()
    sents = sent_tokenize(text)

    blank = ' ______ '
    count = 0

    # clean the text
    clean_words = [re.sub(r"[^\w\s]+", '', x) for x in words]
    clean_words = list(filter(lambda x: x != '', clean_words))

    clean_sents = [re.sub(r"[^\w\s]+", '', x) for x in sents]
    clean_sents = list(filter(lambda x: x != '', clean_sents))

    # compute minimums then set to 0 if minimum is negative
    max_sents = max(min(max_sents, len(clean_sents)-2), 0)
    max_words = max(min(max_words, len(clean_words)-2), 0)

    for n_words in range(1, max_words+1):
        for start_idx in range(1, len(clean_words)-n_words):
            count += 1
            answers.append(' '.join(
                clean_words[start_idx:start_idx+n_words]
            ))
            problems.append(
                ' '.join(clean_words[:start_idx])
                + blank
                + ' '.join(clean_words[start_idx+n_words:])
            )
            units.append('word(s)')
            ns.append(n_words)
            unit_indices.append(start_idx)
    
    for n_sents in range(1, max_sents+1):
        for start_idx in range(1, len(clean_sents)-n_sents):
            count += 1
            answers.append(' '.join(
                clean_sents[start_idx:start_idx+n_sents]
            ))
            problems.append(
                ' '.join(clean_sents[:start_idx])
                + blank
                + ' '.join(clean_sents[start_idx+n_sents:])
            )
            units.append('sentence(s)')
            ns.append(n_sents)
            unit_indices.append(start_idx)
    N = len(answers)
    ds_names = [ds_name]*N
    row_ids = [row_id]*N

    return (
        ds_names, row_ids, answers, 
        problems, units, ns, unit_indices
    )

def infill_setup(args, cfg, keywords):

    ######### unpack vars from cfg ##########

    ds_cache = cfg.infill['ds_cache']

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

    ds_names = cfg.infill['target_data']

    # Determine if evaluation data configuration has changed
    if db_is_new:
        skip = True
        res = cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cfg (
                data_cfg TEXT,
                max_blank_sents INT,
                max_blank_words INT
            );
            """
        )
        res = cur.execute(
            f"""
            INSERT INTO cfg VALUES (
                '{json.dumps(cfg.infill['target_data'])}',
                {cfg.infill['max_blank_sents']},
                {cfg.infill['max_blank_words']}
            );
            """
        )
    # if dataset is not new, check for changes to configuration
    else:
        res = cur.execute('SELECT * FROM cfg').fetchone()
        # check if target datasets are different
        if (
            res[0] != json.dumps(cfg.infill['target_data'])
            or res[1] != cfg.infill['max_blank_sents']
            or res[2] != cfg.infill['max_blank_words']
        ):
            prev_datasets, n_blank_sents, n_blank_words = res
            res = cur.execute(
                f"""
                UPDATE cfg 
                SET 
                    data_cfg='{json.dumps(cfg.infill['target_data'])}',
                    max_blank_sents={cfg.infill['max_blank_sents']},
                    max_blank_words={cfg.infill['max_blank_words']}
                WHERE data_cfg='{res[0]}'
                """
            )
            prev_dsets = json.loads(prev_datasets)
            new_dsets = ds_names
            # goes here if target datasets change but n_blanks stay same
            if (
                n_blank_sents == cfg.infill['max_blank_sents']
                and n_blank_words == cfg.infill['max_blank_words']
            ):
                ds_names = list(
                    set(new_dsets) - set(json.loads(prev_dsets))
                )
        # skip building source_data table if there's no difference
        else:
            display.done('Data already exists')
            return 

    cur = con.cursor()
    res = cur.execute(
        """
        CREATE TABLE IF NOT EXISTS source_data (
            dataset TEXT,
            ref_id BIGINT,
            text TEXT,
            PRIMARY KEY (ref_id)
        );
        """
    )
    cur.close()

    # load in datasets
    #ds_names = cfg.infill['target_data']
    for name in tqdm(ds_names, desc='populating source document table'):
        cur = con.cursor()
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
        cur.close()

    con.commit()
    cur = con.cursor()

    text_data = cur.execute(
        'SELECT dataset, ref_id, text FROM source_data'
    ).fetchall()

    # create fitb_problems table
    res = cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fitb_problems (
            dataset TEXT,
            ref_id BIGINT,
            problem TEXT,
            answer TEXT,
            unit TEXT,
            n INT,
            unit_idx INT,
            PRIMARY KEY (ref_id, unit, n, unit_idx)
            FOREIGN KEY (ref_id) REFERENCES source_data
        );
        """
    )
    
    cur.close()
    con.close()

    # populate fitb_problems table
    N = len(text_data)
    fn_kwargs = {
        'db_file': db_file,
        'max_words': cfg.infill['max_blank_words'],
        'max_sents': cfg.infill['max_blank_sents'],
        'one_of_each': cfg.infill['one_of_each'],
    }

    for el in tqdm(text_data, desc='creating fitb problems'):
        create_problems(*el, **fn_kwargs)

    keys = ['dataset', 'ref_id', 'problem', 'answer',
            'unit', 'n', 'unit_idx']
    gen = PromptGenerator(cfg, 'infill')
    gen.prepare_infill_data(keys, db_path=db_file)

def infill_solve(args, cfg, keywords):
    """solve problems created by infill_setup()

    takes the problems created by infill_setup() and runs inference on
    a given LLM to try and solve the task.
    """
    batch_size = cfg.infill['batch_size']
    limit = args.limit

    db_path = f"{args.from_ckpt}/data.db"
    con = sqlite3.connect(db_path)
    cur = con.cursor()


    tables = list(list(zip(*cur.execute(
        'select name from sqlite_master where type="table";'
    ).fetchall()))[0])

    cols = dict()
    for name in tables:
        tbl_info = cur.execute(f"PRAGMA table_info({name});").fetchall()
        _, cnames, _, _, _, _ = zip(*tbl_info)
        cols[name] = list(cnames)

    model = VLLM(args.model)

    write_cur = con.cursor()
    write_cur.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            prompt_id int,
            response text,
            model text,
            PRIMARY KEY (prompt_id, model)
            FOREIGN KEY (prompt_id) REFERENCES prompts(rowid)
        )
        """
    )

    keys = ['rowid'] + cols['prompts']
    N, = cur.execute('select count(*) from prompts').fetchone()
    N_batch = N // batch_size
    if N % batch_size:
        N_batch += 1
    prompt_cur = cur.execute(f"select {', '.join(keys)} from prompts")
    for batch in tqdm(
        fetch_batches(prompt_cur, batch_size, keys=keys), 
        total=N_batch, desc='generating responses'
    ):
        sessions = [
            Session(system_role=sys_role) 
            for sys_role in batch['sys_text']
        ]
        out_sess, resp = model.generate(sessions, batch['prompt_text'])
        write_cur.executemany(
            f"""
            INSERT INTO responses (prompt_id, response, model)
            VALUES (?, ?, ?)
            """,
            zip(batch['rowid'], resp, [args.model]*batch_size)
        )
        con.commit()

def infill_evaluate(args, cfg, keywords):

    metric = args.metric
    db_path = args.from_ckpt

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            resp_id int,
            metric text,
            score double,
            FOREIGN KEY (resp_id) REFERENCES responses(rowid)
        )
        """
    )

    breakpoint()

def fetch_batches(cursor, batch_size, keys=None):
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:  # No more rows to fetch
            break
        if keys:
            yield dict(zip(keys, zip(*batch)))
        else:
            yield batch