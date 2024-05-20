import os
import traceback
import copy
import random
from datasets import Dataset
from itertools import product
from llm_eval.utils import (
    files,
    display,
    strings,
)

class PromptGenerator:
    def __init__(self, cfg):

        self.cfg = cfg
        prompt_groups = cfg.resp_coll.get('prompt_templates')
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
        self.sys_text = templates.pop('sys')
        self.templates = templates

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