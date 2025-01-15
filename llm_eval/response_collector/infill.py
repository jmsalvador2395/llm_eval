"""
loops over the all-sides/privacy-policy datasets and prompts LLMs based 
off certain fields
"""

# external imports
import evaluate
import json
import datasets
import numpy as np
import random
import sqlite3
from itertools import product
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from nltk import sent_tokenize
from typing import List
import re
import time

# local imports
from llm_eval.utils import (
    display,
    files,
)
from llm_eval.llm import *
from llm_eval.prompt_generator import TELeR, PromptGenerator
from llm_eval.llm.generators.vllm import VLLM
from llm_eval.llm.session import Session
from .datamodule import load_data

def create_problems(
    ds_name: str, 
    ref_id: int, 
    text: str, 
    max_sents: int,
    max_words: int,
    db_file: str,
    one_of_each: bool=False,
    cfg=None,
    **kwargs
):

    split_tokens = cfg.infill.get('split_tokens', dict())
    split_token = split_tokens.get(ds_name)

    """populates the 'fitb_problems' table"""
    if one_of_each:
        (ds_names, row_ids, answers, 
         problems, units, ns, unit_indices) = get_one_of_each_problem(
            text, ref_id, ds_name, max_sents, max_words, split_token
        )
    else:
        (ds_names, row_ids, answers, 
         problems, units, ns, unit_indices) = get_all_problems(
            text, ref_id, ds_name, max_sents, max_words, split_token
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

def get_one_of_each_problem(
        text, row_id, ds_name, max_sents, max_words, split_token=None
    ):
    answers = []
    problems = []
    units = []
    ns = []
    unit_indices = []

    # split into words and sentences
    if split_token:
        words = text.replace(split_token, '').split()
        sents = text.split(split_token)
        sents = [sent.strip() for sent in sents]
    else:
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
    max_sents = max(min(max_sents, len(sents)-2), 0)
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
        start_idx = random.randint(1, len(sents)-n_sents-1)
        count += 1
        answers.append(' '.join(
            sents[start_idx:start_idx+n_sents]
        ))
        problems.append(
            ' '.join(sents[:start_idx])
            + blank
            + ' '.join(sents[start_idx+n_sents:])
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


def get_all_problems(
        text, row_id, ds_name, max_sents, max_words, split_token=None
    ):
    answers = []
    problems = []
    units = []
    ns = []
    unit_indices = []

    # split into words and sentences
    if split_token:
        words = text.replace(split_token, '').split()
        sents = text.split(split_token)
        sents = [sent.strip() for sent in sents]
    else:
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

    if args.path:
        save_loc = args.path
    else:
        # create database file
        save_loc = f'{cfg.infill["save_dir"]}/infill/{keywords["timestamp"]}'
        files.create_path(save_loc)
    db_file = f'{save_loc}/data.db'
    db_is_new = not files.path_exists(db_file)

    
    display.info('connecting to database')
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    display.ok('')

    ds_names = cfg.infill['target_data']

    # Determine if evaluation data configuration has changed
    if db_is_new:
        display.info('saving contents of cfg to database')
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
        display.ok('')
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

    display.info('creating table `source_data`')
    cur = con.cursor()
    res = cur.execute(
        """
        CREATE TABLE IF NOT EXISTS source_data (
            dataset TEXT,
            ref_id BIGINT,
            text TEXT,
            PRIMARY KEY (dataset, ref_id)
        );
        """
    )
    display.ok('')

    # load in datasets
    #ds_names = cfg.infill['target_data']
    ds_info = cfg.datasets
    for name in tqdm(ds_names, desc='populating source document table'):
        text = load_data(name, cfg, ds_info[name], args.limit)
        N = len(text)
        cur.executemany(
            "INSERT OR IGNORE INTO source_data VALUES(?, ?, ?)",
            zip([name]*N, range(N), text)
        )
        con.commit()

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
        'cfg': cfg,
    }

    for el in tqdm(text_data, desc='creating fitb problems'):
        create_problems(*el, **fn_kwargs)

    keys = ['dataset', 'ref_id', 'problem', 'answer',
            'unit', 'n', 'unit_idx']
    gen = PromptGenerator(cfg, 'infill')
    gen.prepare_infill_data(keys, db_path=db_file)
    display.done(f'database file saved to: {db_file}')

def count_picked_templates(cur, picked_templates, batch_size):
    N_batch = 0
    for template, picked_ids in tqdm(
        picked_templates.items(), total=len(picked_templates)
    ):
        N_sub = 0
        total = 0
        for sid, tid in zip(picked_ids['sids'], picked_ids['tids']):
            count, = cur.execute(
                f"""
                SELECT count(*) FROM prompts 
                WHERE 
                    sys_id={sid} 
                    AND template_id={tid}
                    AND template_name="{template}"
                """
            ).fetchone()
            total += count
            N_sub += count
            N_batch += count // batch_size
            if count % batch_size:
                N_batch += 1

        display.info(
            f'number of prompts picked for `{template}`: {N_sub}'
        )
    display.info(f'prompts to process: {total}')
    return N_batch

def fetch_picked_templates(
        cur, picked_templates, batch_size, keys=None
    ):
    for template, picked_ids in picked_templates.items():
        N_sub = 0
        for sid, tid in zip(picked_ids['sids'], picked_ids['tids']):
            res = cur.execute(
                f"""
                SELECT {', '.join(keys)} FROM prompts 
                WHERE 
                    sys_id={sid} 
                    AND template_id={tid}
                    AND template_name="{template}"
                """
            )
            while True:
                batch = res.fetchmany(batch_size)
                if not batch:
                    break
                if keys:
                    yield dict(zip(keys, zip(*batch)))
                else:
                    yield batch


def count_subsamples(cur, batch_size, limit):
    start = time.time()
    res = cur.execute(
        'SELECT DISTINCT template_name FROM prompts'
    ).fetchall()
    template_names, = list(zip(*res))
    stop = time.time()
    display.info(f'unique template_name query took {stop-start:.02f} seconds')
    N = 0
    for tmplt in template_names:
        start = time.time()
        res = cur.execute(f'SELECT DISTINCT template_id from prompts P where P.template_name="{tmplt}"')
        tids, = list(zip(*res.fetchall()))
        stop = time.time()
        display.info(f'unique template_id query took {stop-start:.02f} seconds')

        start = time.time()
        res = cur.execute(f'SELECT DISTINCT sys_id from prompts P where P.template_name="{tmplt}"')
        sids, = list(zip(*res.fetchall()))
        stop = time.time()
        display.info(f'unique sys_id query took {stop-start:.02f} seconds')
        N += len(tids)*len(sids)*limit
        print(f'{tmplt}, tids: {len(tids)}, sids: {len(sids)}')

    display.info(f'prompts to process: {N}')
    N_batch = N // batch_size
    if N % batch_size:
        N_batch += 1
    return N_batch


def fetch_subsamples(cur, batch_size, limit, pids, keys=None):
    res = cur.execute('SELECT DISTINCT template_name from prompts')
    template_names, = list(zip(*res.fetchall()))

    for tmplt in template_names:
        res = cur.execute(f'SELECT DISTINCT template_id from prompts P where P.template_name="{tmplt}"')
        tids, = list(zip(*res.fetchall()))
        res = cur.execute(f'SELECT DISTINCT sys_id from prompts P where P.template_name="{tmplt}"')
        sids, = list(zip(*res.fetchall()))
        for tid, sid in product(tids, sids):
            res = cur.execute(
                f"""
                SELECT {', '.join(keys)} FROM prompts P
                WHERE P.template_name="{tmplt}" 
                    AND P.sys_id={sid} 
                    AND P.template_id={tid}
                    AND P.problem_id in {tuple(pids)}
                """
            )
            while True:
                batch = res.fetchmany(batch_size)
                if not batch:
                    break
                if keys:
                    yield dict(zip(keys, zip(*batch)))
                else:
                    yield batch

def fetch_batches(cur, batch_size, keys=None):
    res = cur.execute(f"select {', '.join(keys)} from prompts")
    while True:
        batch = res.fetchmany(batch_size)
        if not batch:  # No more rows to fetch
            break
        if keys:
            yield dict(zip(keys, zip(*batch)))
        else:
            yield batch

def infill_solve(args, cfg, keywords):
    """solve problems created by infill_setup()

    takes the problems created by infill_setup() and runs inference on
    a given LLM to try and solve the task.
    """
    display.info(f'running infill_solve using {args.model}')
    batch_size = cfg.infill['batch_size']
    limit = args.limit

    db_path = args.path
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

    keys = ['rowid'] + cols['prompts']
    if args.limit:
        display.info('fetching subsample IDs')
        # check if table of selected problem ids exists
        res = cur.execute(
            f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='sample_pids';
            """
        ).fetchall()
        if res == []:
            # create table of selected problem ids
            all_pids, = list(zip(*cur.execute(
                'SELECT DISTINCT(problem_id) from prompts'
            )))
            pids = np.random.permutation(all_pids)[:limit]
            pids = tuple(pids.tolist())
            res = cur.execute(
                """
                CREATE TABLE sample_pids (
                    pid int
                )
                """
            )
            res = cur.executemany(
                """
                INSERT INTO sample_pids (pid)
                VALUES (?)
                """,
                zip(pids)
            )
            con.commit()
        else:
            pids, = zip(*cur.execute(f'SELECT pid FROM sample_pids'))

        print(f'counting prompt variations')
        N_batch = count_subsamples(cur, batch_size, args.limit)
        print(f'creating generator function')
        gen_fn = fetch_subsamples(
            cur, batch_size, args.limit, pids, keys=keys
        )
    elif args.picked_templates:
        picked_templates = (
            cfg.infill['picked_templates'].get(args.model)
        )
        if picked_templates is None:
            display.error(
                f'config does not contain picked templates for '
                f'model `{args.model}`'
            )
            raise ValueError(f'')
        N_batch = count_picked_templates(
            cur, picked_templates, batch_size
        )
        gen_fn = fetch_picked_templates(
            cur, picked_templates, batch_size, keys=keys
        )
    else:
        N, = cur.execute('select count(*) from prompts').fetchone()
        N_batch = N // batch_size
        if N % batch_size:
            N_batch += 1
        gen_fn = fetch_batches(cur, batch_size, keys=keys)
        display.info(f'prompts to process: {N}')

    display.info(f'loading model: {args.model}')
    model_cache = cfg.model_params.get('model_cache')
    model_args = cfg.model_params.get(args.model, dict())
    model = VLLM(args.model, model_cache=model_cache, **model_args)

    write_cur = con.cursor()
    write_cur.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            prompt_id int,
            response text,
            model text,
            temperature double,
            PRIMARY KEY (prompt_id, model, temperature)
            FOREIGN KEY (prompt_id) REFERENCES prompts(rowid)
        )
        """
    )

    display.info('begin collecting responses')
    temps = cfg.infill.get('temps', [1.0])
    start_point, = cur.execute(
        f'SELECT count(*) FROM responses WHERE model="{args.model}"'
    ).fetchone()
    start_point = start_point // batch_size

    for step, (batch, temp) in tqdm(
        enumerate(product(gen_fn, temps)), 
        total=N_batch*len(temps), 
        desc='generating responses'
    ):
        # skip until you get to start point
        if step < start_point:
            continue

        sessions = [
            Session(system_role=sys_role) 
            for sys_role in batch['sys_text']
        ]
        out_sess, resp = model.generate(
            sessions, 
            batch['prompt_text'],
            temp=temp,
        )
        write_cur.executemany(
            f"""
            INSERT INTO responses (prompt_id, response, model, temperature)
            VALUES (?, ?, ?, ?)
            """,
            zip(batch['rowid'], resp, [args.model]*batch_size, [temp]*batch_size,)
        )
        con.commit()

def infill_evaluate(args, cfg, keywords):

    metric = args.metric
    db_path = args.path
    batch_size = cfg.infill['batch_size']

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    kwargs = cfg.infill['kwargs'].get(metric, dict())
    mets = ['bertscore', 'rouge']
    if metric == 'bertscore':
        critic = evaluate.load('evaluate-metric/bertscore')
        met_cols = ['precision', 'recall', 'f1']
        table_names = [f'bertscore_{met}' for met in met_cols]
    elif metric == 'rouge':
        critic = evaluate.load('rouge')
        met_cols = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        table_names = met_cols
    else:
        raise ValueError(f'choose from either [{", ".join(mets)}]')
    
    # create table to keep track of which tables actually contain scores
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metric_names (name text PRIMARY KEY)
        """
    )
    res = cur.executemany(
        "INSERT OR IGNORE INTO metric_names VALUES (?)", 
        zip(table_names)
    )
    con.commit()

    for table_name in table_names:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                resp_id int PRIMARY KEY,
                score double,
                FOREIGN KEY (resp_id) REFERENCES responses(rowid)
            )
            """
    )

    read_cur = con.cursor()
    data_cur = read_cur.execute(
        f"""
        SELECT R.rowid, F.problem, F.answer, R.response
        FROM responses R, prompts P, fitb_problems F
        WHERE R.prompt_id=P.rowid AND P.problem_id=F.rowid
        """
    )

    keys = ['resp_id', 'problem', 'answer', 'response']
    
    N, = cur.execute('select count(*) from responses').fetchone()
    N_batch = N // batch_size
    if N % batch_size:
        N_batch += 1
    for batch in tqdm(
        batch_generator(data_cur, batch_size, keys=keys), 
        total=N_batch, desc=f'evaluating {metric}'
    ):
        # TODO check if samples exist before calculating scores

        scores = critic.compute(
            predictions=batch['response'],
            references=batch['answer'],
            **kwargs
        )

        if metric == 'bertscore':
            for table_name, met in zip(table_names, met_cols):
                cur.executemany(
                    f"""
                    INSERT INTO {table_name} (resp_id, score)
                    VALUES (?, ?)
                    """,
                    zip(
                        batch['resp_id'], 
                        scores[met]
                    )
                )
                con.commit()
        elif metric == 'rouge':
            for table_name in table_names:
                cur.executemany(
                    f"""
                    INSERT INTO {table_name} (resp_id, score)
                    VALUES (?, ?)
                    """,
                    zip(
                        batch['resp_id'],
                        scores[table_name]
                    )
                )
                con.commit()
def batch_generator(cur, batch_size, keys=None):
    while True:
        batch = cur.fetchmany(batch_size)
        if not batch:  # No more rows to fetch
            break
        if keys:
            yield dict(zip(keys, zip(*batch)))
        else:
            yield batch
