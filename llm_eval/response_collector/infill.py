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
from itertools import product, chain
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from nltk import sent_tokenize
from typing import List
from difflib import ndiff
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
from . import db_funcs

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
    name = args.name or keywords['timestamp']

    # create database file
    save_loc = f'{cfg.infill["save_dir"]}/infill/{name}'
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

    # create source_data table if it doesn't exist
    display.info('creating table `source_data`')
    cur = con.cursor()
    src_data_exists, = cur.execute(
        f"""
        SELECT EXISTS(
            SELECT 1 FROM sqlite_master 
            WHERE type="table" AND name="source_data"
        );
        """
    ).fetchone()
    if not src_data_exists:
        display.in_progress('building source_data table')
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
        tok_names = cfg.infill.get(
            'limit_tokenizers', 
            ['meta-llama/Llama-3.2-1B-Instruct']
        )
        for name in tqdm(ds_names, desc='populating source document table'):
            text = load_data(
                name, cfg, ds_info[name], args.limit, tok_names
            )
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
    problems_exists, = cur.execute(
        f"""
        SELECT EXISTS(
            SELECT 1 FROM sqlite_master 
            WHERE type="table" AND name="fitb_problems"
        );
        """
    ).fetchone()
    if not problems_exists:
        display.in_progress('building problems table')
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

    # create prompts table
    prompts_exists, = cur.execute(
        f"""
        SELECT EXISTS(
            SELECT 1 FROM sqlite_master 
            WHERE type="table" AND name="prompts"
        );
        """
    ).fetchone()
    if not prompts_exists:
        display.in_progress('building prompts table')
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


def count_subsamples(cur, batch_size, limit, pids):
    start = time.time()
    res = cur.execute(
        'SELECT DISTINCT template_name FROM prompts'
    ).fetchall()
    template_names, = list(zip(*res))
    stop = time.time()
    display.info(f'unique template_name query took {stop-start:.02f} seconds')
    N = 0
    N_batch = 0
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
        #N += len(tids)*len(sids)*limit
        print(f'{tmplt}, tids: {len(tids)}, sids: {len(sids)}')
        display.info(f'counting samples from each query')
        start = time.time()
        for tid, sid in product(tids, sids):
            count, = cur.execute(
                f"""
                SELECT count(*) FROM prompts P
                WHERE P.template_name="{tmplt}" 
                    AND P.sys_id={sid} 
                    AND P.template_id={tid}
                    AND P.problem_id in {tuple(pids)}
                """
            ).fetchone()
            N += count
            N_batch += count // batch_size 
            if count % batch_size:
                N_batch += 1
        stop = time.time()
        display.info(f'counting samples took {stop-start:.02f} seconds')

    display.info(f'batch_size: {batch_size}, limit: {limit}, N: {N}, N_batch: {N_batch}')
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
    batch_size = args.batch_size
    limit = args.limit
    supports_sys = not args.no_sys

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
        N_batch = count_subsamples(cur, batch_size, args.limit, pids)
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
    model_args = cfg.model_params.get(args.model)
    if model_args is None:
        model_args = cfg.model_params['default']
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
            Session(system_role=sys_role, supports_sys=supports_sys) 
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

def extract_answers(
        sample: dict, 
        patterns=[], 
        key_tokens=[],
        rm_up_to=[],
        blank_str='______',
        debug=False,
    ):
    response = sample['response']
    extr = dict()

    # convert to list if patterns is just a string
    if isinstance(patterns, str):
        patterns = [patterns]
    tags, texts = [], []

    # remove_up_to
    def remove_up_to(s, substring):
        before, sep, after = s.partition(substring)
        return after if sep else ""

    matched = False
    for m_key in rm_up_to:
        if sample['model'].startswith(m_key):
            pat = rm_up_to[m_key]
            response = remove_up_to(response, pat)
            texts.append(response)
            tags.append(f'rm_up_to_{pat}')
            matched = True
            break
    if not matched:
        pat = rm_up_to['default']
        response = remove_up_to(response, pat)
        texts.append(response)
        tags.append(f'rm_up_to_{pat}')

    tags, texts = ['full'], [response]

    # extract using search patterns
    match_keys = []
    match_vals = []
    for i, pattern in enumerate(patterns):
        pattern = pattern.strip()
        matches = re.findall(pattern, response)
        match_keys += [
            f'pattern:{i},pos:{k}' for k in range(len(matches))
        ]
        match_vals += matches

    if debug:
        print(f"match keys: {match_keys}")
        print(f"match vals: {match_vals}")

    tags += match_keys
    texts += match_vals

    diffs = list(ndiff(
        sample['problem'].split(), 
        response.split()
    ))

    # A = problem text, B = response text
    A_only = [word[2:] for word in diffs if word.startswith('- ')]
    B = [
        word[2:] for word in diffs 
        if re.match('^\+\s|^\s\s', word)
    ]
    B_only = [word[2:] for word in diffs if word.startswith('+ ')]

    tags.append('diff_seqs:all')
    texts.append(' '.join(B_only))

    # find uninterupted exclusive sub-sequences in text
    seqs = []
    candidate = ''
    for word in response.split():
        if word in B_only:
            candidate += f' {word}'
        elif candidate != '':
            seqs.append(candidate.strip())
            candidate = ''
    if candidate != '':
        seqs.append(candidate.strip())
    seqs = list(filter(lambda x: len(x.split()) > 3, seqs))

    tags += [f'diff_seqs:{i}' for i in range(len(seqs))]
    texts += seqs
    lines = response.split('\n')
    lines = [line for line in lines if line != '']
    if len(lines) > 1:
        tags += [f'lines:{i}' for i in range(len(lines))]
        texts += lines

    # Matches numbered lists (e.g., "1. Item", "2) Item", "(3) Item")
    numbered_list_pattern = r'(?m)^\s*(?:\d+[\).]|\(\d+\))\s+(.+)$'
    
    # Matches bullet points (e.g., "- Item", "* Item", "• Item")
    bullet_list_pattern = r'(?m)^\s*[-•*]\s+(.+)$'
    
    matches = (
        re.findall(numbered_list_pattern, response) 
        + re.findall(bullet_list_pattern, response)
    )
    list_elements = [item.strip() for item in matches]
    tags += [f'list_el:{i}' for i in range(len(list_elements))]
    texts += list_elements

    matches = re.findall(r':\s*(.+)', response)
    tags += [f'post_colon:{i}' for i in range(len(matches))]
    texts += [match.strip() for match in matches]

    # Use left context of problem and right context of problem to match
    # the blank
    left_half, right_half = sample['problem'].split('______')
    matches = re.findall(
        f'{re.escape(left_half)}(.*?){re.escape(right_half)}', 
        response
    )
    tags += [f'context_match:{i}' for i in range(len(matches))]
    texts += [match.strip() for match in matches]
    
    # remove tokens
    def remove_key_tokens(s, tokens):
        for token in tokens:
            s = s.replace(token, '')
        return s.strip()
    texts.append(remove_key_tokens(response, key_tokens))
    tags.append(f'rm_key_tokens')

    ##############################
    # filter and reduction step #
    ##############################

    # instantiate text_set with strings we already know to filter out
    invalid_set = {
        '', blank_str,
        left_half.strip() + ' ' + right_half.strip(),
        right_half.strip(), left_half.strip(),
    }
    if debug:
        print(f"left half: {left_half.strip()}")
        print(f"right half: {right_half.strip()}")
        print(f"invalid set: {invalid_set}")

    # combine tags that have matching text
    text_tag_map = {
        text: [] for text in texts if text not in invalid_set
    }
    if debug:
        print(f"text_tag_map: {text_tag_map}")
    for tag, text in zip(tags, texts):
        if text not in invalid_set:
            text_tag_map[text].append(tag)
    extr = {
        ','.join(tag_set): text 
        for text, tag_set 
        in text_tag_map.items()
    }

    if debug:
        return extr, diffs, texts, tags
    else:
        return extr

def infill_extract(args, cfg, keywords):
    db_path = args.path
    batch_size = args.batch_size
    num_proc = args.num_proc

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    stream_cur = con.cursor()

    match_patterns = cfg.infill.get('ans_extract_patterns', [])

    table_info = db_funcs.get_tables(con)
    if 'responses' not in table_info:
        raise Exception(
            'table `responses` does not exist in the provided database'
        )

    N, = cur.execute(f'select count(*) from responses').fetchone()
    keys = table_info['responses']

    N_batch = N // batch_size
    if N % batch_size:
        N_batch += 1

    if 'extractions' in table_info:
        checkpoint, = cur.execute(
            f'SELECT MAX(resp_id) FROM extractions'
        ).fetchone()
        if checkpoint is None:
            checkpoint = 0
        else:
            checkpoint = N // batch_size
            if N % batch_size:
                checkpoint += 1
    else:
        # create table for extractions 
        checkpoint = 0
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS extractions (
                resp_id int,
                tag text,
                text text,
                PRIMARY KEY (resp_id, tag)
                FOREIGN KEY (resp_id) REFERENCES responses(rowid)
            )
            """
        )
    res = stream_cur.execute(
        f"""
        SELECT R.rowid, R.response, F.problem, R.model
        FROM responses R, prompts P, fitb_problems F
        WHERE
            R.prompt_id=P.rowid
            AND P.problem_id=F.rowid
        """
    )
    data_stream = db_funcs.cur_gen(
        res, 
        batch_size=batch_size, 
        keys=['rowid', 'response', 'problem', 'model'],
        list_of_dicts=True,
    )
    extract_answers_partial = partial(
        extract_answers, 
        patterns=match_patterns,
        key_tokens = cfg.infill['key_tokens'],
        rm_up_to = cfg.infill['rm_up_to'],
        blank_str='______',
        debug=False,
    )
    for n, batch in tqdm(
        enumerate(data_stream), total=N_batch,
        desc='extracting candidates from responses'
    ):
        if n < checkpoint:
            continue

        if num_proc == 1:
            extr = [
                extract_answers(
                    sample, 
                    patterns=match_patterns, 
                    key_tokens = cfg.infill['key_tokens'],
                    rm_up_to = cfg.infill['rm_up_to'],
                    blank_str='______',
                    debug=False,
                )
                for sample in batch
            ]
        else:
            with ProcessPoolExecutor(max_workers=num_proc) as executor:
                extr = list(executor.map(extract_answers_partial, batch))

        tags = [key for d in extr for key in d.keys()]
        texts = [value for d in extr for value in d.values()]
        n_vals = [len(sample) for sample in extr]
        rowids = list(chain(*[
            [sample['rowid']]*nval 
            for sample, nval in zip(batch, n_vals)
        ]))
        #tags, texts, = zip(*extr.items())

        # sometimes no answers are extracted
        if len(tags) != 0:
            cur.executemany(
                """
                INSERT INTO extractions (resp_id, tag, text)
                VALUES (?, ?, ?)
                """,
                zip(rowids, tags, texts)
            )
            con.commit()

def infill_evaluate(args, cfg, keywords):

    metric = args.metric
    db_path = args.path
    batch_size = args.batch_size

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    kwargs = cfg.infill['kwargs'].get(metric)
    if kwargs is None: 
        kwargs = cfg.infill['kwargs'].get('default')

    mets = ['bertscore', 'rouge', 'perplexity']
    if metric == 'bertscore':
        critic = evaluate.load('evaluate-metric/bertscore')
        met_cols = ['precision', 'recall', 'f1']
        table_names = [f'bertscore_{met}' for met in met_cols]
    elif metric == 'rouge':
        critic = evaluate.load('rouge')
        met_cols = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        table_names = met_cols
    elif metric == 'perplexity':
        critic = evaluate.load('perplexity', module_type='metric')
        met_model = kwargs['model_id']
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', met_model)
        table_names = [
            f'ppl_{clean_name}', f'ppl_diff_{clean_name}'
        ]
        display.info(f'evaluating perplexity with: {met_model}')
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
                extraction_id int PRIMARY KEY,
                score double,
                FOREIGN KEY (extraction_id) 
                    REFERENCES extractions(rowid)
            )
            """
    )

    read_cur = con.cursor()
    data_cur = read_cur.execute(
        f"""
        SELECT E.rowid, F.problem, F.answer, R.response, E.text
        FROM extractions E, responses R, prompts P, fitb_problems F
        WHERE 
            R.prompt_id=P.rowid 
            AND P.problem_id=F.rowid
            AND E.resp_id=R.rowid
        """
    )

    keys = ['extraction_id', 'problem', 'answer', 'response', 'text']
    
    N, = cur.execute(f'select count(*) from extractions').fetchone()
    N_batch = N // batch_size
    if N % batch_size:
        N_batch += 1
    n_processed, = cur.execute(
        f'SELECT COUNT(*) from {table_name}'
    ).fetchone()
    checkpoint = n_processed // batch_size
    for n, batch in tqdm(
        enumerate(batch_generator(data_cur, batch_size, keys=keys)), 
        total=N_batch, desc=f'evaluating {metric}'
    ):
        if n < checkpoint:
            continue

        if metric == 'perplexity':
            candidates = [
                problem.replace('______', text)
                for problem, text
                in zip(batch['problem'], batch['text'])
            ]
            answers = [
                problem.replace('______', answer )
                for problem, answer 
                in zip(batch['problem'], batch['answer'])
            ]
            score_cand = critic.compute(
                predictions=candidates, **kwargs
            )
            score_cand = np.array(score_cand['perplexities'])
            score_ans = critic.compute(
                predictions=answers, **kwargs
            )
            score_ans = np.array(score_ans['perplexities'])
            diffs = (score_cand - score_ans).tolist()
            score_cand = score_cand.tolist()
            # table_names = [
            #     f'ppl_{clean_name}', f'ppl_diff_{clean_name}'
            # ]
            cur.executemany(
                f"""
                INSERT INTO ppl_{clean_name} (extraction_id, score)
                VALUES (?, ?)
                """,
                zip(
                    batch['extraction_id'],
                    score_cand
                )
            )
            cur.executemany(
                f"""
                INSERT INTO ppl_diff_{clean_name} (extraction_id, score)
                VALUES (?, ?)
                """,
                zip(
                    batch['extraction_id'],
                    diffs
                )
            )
        else:
            predictions = batch['text']
            references = batch['answer']

            scores = critic.compute(
                predictions=predictions,
                references=references,
                **kwargs
            )

            if metric == 'bertscore':
                for table_name, met in zip(table_names, met_cols):
                    cur.executemany(
                        f"""
                        INSERT INTO {table_name} (extraction_id, score)
                        VALUES (?, ?)
                        """,
                        zip(
                            batch['extraction_id'], 
                            scores[met]
                        )
                    )
                    con.commit()
            elif metric == 'rouge':
                for table_name in table_names:
                    cur.executemany(
                        f"""
                        INSERT INTO {table_name} (extraction_id, score)
                        VALUES (?, ?)
                        """,
                        zip(
                            batch['extraction_id'],
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
