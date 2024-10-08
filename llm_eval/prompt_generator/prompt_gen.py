import os
import traceback
import copy
import random
import itertools
from datasets import Dataset
from itertools import product
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
            raise Exception('Invalid procedure running?')

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
    
    def prepare_infill_data(self, ds):

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
                'tmplt_name': tmplt_names[tmplt_id],
                'tmplt_id': tmplt_idx[tmplt_id],
                'sys_id': sys_id,
                'sys_text': self.sys_text[sys_id],
                'prompt_text': prompts[tmplt_id],
            }
            for sys_id, tmplt_id in combos
        ]

        def map_fn(batch, **fn_kwargs):
            prompts = fn_kwargs['prompts']
            sample_keys = list(dict(batch).keys())
            prompt_keys = list(prompts[0].keys())
            all_keys = sample_keys + prompt_keys

            out_batch = {key: [] for key in sample_keys+prompt_keys}
            N = len(batch[sample_keys[0]])
            for prompt in prompts:
                for n in range(N):
                    sample = (
                        prompt 
                        | {key:batch[key][n] for key in sample_keys}
                    )
                    sample['prompt_text'] = strings.replace_slots(
                        sample['prompt_text'],
                        sample
                    )
                    sample['sys_text'] = strings.replace_slots(
                        sample['sys_text'],
                        sample
                    )
                    for key in sample.keys():
                        out_batch[key].append(sample[key])

            return out_batch
        
        ds = ds.map(
            map_fn,
            fn_kwargs={
                'prompts': prompt_data},
            batched=True,
            batch_size=16,
            num_proc=8,
        )

        return ds


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