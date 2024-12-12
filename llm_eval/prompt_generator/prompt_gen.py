import os
import traceback
import copy
import random
import itertools
from datasets import Dataset
from itertools import product
import sqlite3
from tqdm import tqdm

from llm_eval.utils import (
    files,
    display,
    strings,
)

class PromptGenerator:
    def __init__(self, cfg, procedure='response_collection'):

        self.cfg = cfg
        if procedure == 'response_collection':
            prompt_groups = cfg.resp_coll.get('prompt_templates')
        elif procedure == 'infill':
            prompt_groups = cfg.infill.get('prompt_templates')
        else:
            raise Exception('Invalid procedure running: `{procedure}`')

        if prompt_groups is None:
            display.error(
                'prompt templates and paths not specified '
                'in config file'
            )
            raise ValueError()

        try:
            templates = {
                grp: files.load_yaml(path) 
                for grp, path in prompt_groups.items()
            }
        except Exception as e:
            display.error(f'error reading from path')
            display.error(str(e))
            raise ValueError()
        if 'sys' in templates:
            self.sys_text = templates.pop('sys')
        else:
            display.error('no "sys" config provided')
            raise KeyError('no "sys" config provided')
            
        self.templates = templates
    
    def prepare_infill_data(self, keys, db_path):

        con = sqlite3.connect(db_path)
        cur = con.cursor()

        keys = ['rowid'] + keys

        # create prompt table
        res = cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                template_name text,
                template_id int,
                sys_id int,
                sys_text text,
                prompt_text text,
                problem_id int,
                PRIMARY KEY (problem_id, template_name, sys_id, template_id)
                FOREIGN KEY (problem_id) references fitb_problems(rowid)
            );
            """
        )

        # create combinations
        tmplt_names = []
        tmplt_idx = []
        prompts = []
        for key, val in self.templates.items():
            prompts += val
            tmplt_names += [key]*len(val)
            tmplt_idx += list(range(len(val)))
        global_tmplt_idx = list(range(len(prompts)))
        sys_idx = list(range(len(self.sys_text)))
        combos = list(itertools.product(sys_idx, global_tmplt_idx))

        prompt_data = [
            {
                'template_name': tmplt_names[tmplt_id],
                'template_id': tmplt_idx[tmplt_id],
                'sys_id': sys_id,
                'sys_text': self.sys_text[sys_id],
                'prompt_text': prompts[tmplt_id],
            }
            for sys_id, tmplt_id in combos
        ]

        N = cur.execute(
            'select count(*) from fitb_problems'
        ).fetchone()[0]
        data_cur = cur.execute(
            f"select {','.join(keys)} from fitb_problems"
        )

        writer_cur = con.cursor()
        out_keys = [
            'template_name', 'template_id', 'sys_id', 
            'sys_text', 'prompt_text', 'problem_id',
        ]
        data_queue = {key: [] for key in out_keys}
        for count, sample in enumerate(
            tqdm(data_cur, total=N, desc='creating LLM prompts')
        ):
            sample_dict = dict(zip(keys, sample))
            for prompt in prompt_data:
                union_dict = sample_dict | prompt
                prompt_text = strings.replace_slots(
                    union_dict['prompt_text'], union_dict
                )
                sys_text = strings.replace_slots(
                    union_dict['sys_text'], union_dict
                )
                data_queue['prompt_text'].append(prompt_text)
                data_queue['sys_text'].append(sys_text)
                data_queue['problem_id'].append(sample_dict['rowid'])
                for key in ['template_name', 'template_id', 'sys_id']:
                    data_queue[key].append(union_dict[key])

            # dump samples into the database
            if count % 1000 == 0:
                writer_cur.executemany(
                    f"""
                    INSERT INTO prompts ({', '.join(data_queue.keys())}) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, 
                    zip(*data_queue.values())
                )
                con.commit()
                data_queue = {key: [] for key in out_keys}

        cur.executemany(
            f"""
            INSERT INTO prompts ({', '.join(data_queue.keys())}) 
            VALUES (?, ?, ?, ?, ?, ?)
            """, 
            zip(*data_queue.values())
        )
        con.commit()

    def prepare_data(self, data):
        cfg = self.cfg
        dsvar_dict = cfg.datasets
        out_data = {
            'id': [],
            'template': [],
            'dataset': [],
            'system': [],
            'prompt': [],
            'sys_text_id': [],
            'prompt_text_id': [],
        }

        for ds_name, ds in data.items():
            data_vars = dsvar_dict[ds_name]
            aliases = data_vars['aliases']
            aliases_inv = {v: k for k, v in aliases.items()}
            for sid, sample in enumerate(ds):
                # apply aliases to sample
                sample_tmp = copy.deepcopy(sample)
                for K, V in cfg.datasets[ds_name]['aliases'].items():
                    sample_tmp[V] = sample_tmp.pop(K)

                for tmplt_name, tmplt in self.templates.items():
                    templates = (
                        tmplt.get('common', []) 
                        + tmplt.get(ds_name, [])
                    )
                    sys_text = (
                        self.sys_text.get('common', [])
                        + self.sys_text.get(ds_name, [])
                    )

                    # do procedure icl
                    if tmplt_name == 'icl':
                        # get a random example
                        exid = random.randint(0, len(ds)-1)
                        while exid == sid:
                            exid = random.randint(0, len(ds)-1)
                        example = ds[exid]
                        trgt_ref = data_vars['references'][0]
                        sample_tmp.update({
                            'example_doc1': example[aliases_inv['doc1']],
                            'example_doc2': example[aliases_inv['doc2']],
                            'example_ref': example[trgt_ref],
                        })

                    prompts = [
                        strings.replace_slots(text, sample_tmp)
                        for text in templates
                    ]

                    sys_text_pos, prompt_pos = zip(*list(product(
                        range(len(sys_text)), range(len(prompts))
                    )))

                    sys_text, prompts = zip(
                        *list(product(sys_text, prompts))
                    )

                    out_data['id'] += [sid]*len(prompts)
                    out_data['template'] += [tmplt_name]*len(prompts)
                    out_data['dataset'] += [ds_name]*len(prompts)
                    out_data['system'] += list(sys_text)
                    out_data['prompt'] += list(prompts)
                    out_data['sys_text_id'] += list(sys_text_pos)
                    out_data['prompt_text_id'] += list(prompt_pos)

        return Dataset.from_dict(out_data)