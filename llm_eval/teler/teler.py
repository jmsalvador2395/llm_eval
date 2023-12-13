import os
import traceback
from llm_eval.utils import (
    files,
    display,
    strings,
)

class TELeR:
    def __init__(self, cfg):

        self.cfg = cfg
        prompt_file = cfg.response_collection.get('prompt_template')
        if prompt_file is None:
            display.error('"prompt_template" location not specified in config file')
            raise ValueError()
        self.template = files.load_yaml(prompt_file)

    def get_num_levels(self, ds_name):
        try:
            num_levels = len(self.template[ds_name])
        except Exception as e:
            display.error(f'prompt template for dataset "{ds_name}" does not exist')
        return num_levels
    def get_levels(self, ds_name):
        try:
            levels = list(self.template[ds_name].keys())
            return levels
        except Exception as e:
            display.error(f'prompt template for dataset "{ds_name}" does not exist')

    def format_data(self, 
                    ds,
                    ds_name,
                    lv,
                    sys_title='system_text',
                    prompt_title='prompt_text'):

        trgt_template = self.template.get(ds_name)
        if trgt_template is None:
            display.error('Target dataset does not exist in the prompt template file')
            traceback.print_exception(*sys.exc_info())
            os._exit(0)

        trgt_template = trgt_template.get(lv)
        if trgt_template is None:
            display.error('Target level does not exist in the prompt template file')
            traceback.print_exception(*sys.exc_info())
            os._exit(0)

        try:
            msg_text = trgt_template.get('message')
            sys_text = trgt_template.get('system')
        except:
            breakpoint()

        if sys_text is None:
            display.warning('No system message specified in config')
        if msg_text is None:
            display.error('No prompt specified in config')

        prompts = []
        sys_roles = []
        for sample in ds:
            prompts.append(strings.replace_slots(msg_text, sample))
            if sys_text is not None:
                sys_roles.append(strings.replace_slots(sys_text, sample))
        if sys_roles == []:
            sys_roles = [None]*len(prompts)

        out_ds = ds.add_column(prompt_title, prompts)
        out_ds = out_ds.add_column(sys_title, sys_roles)

        return out_ds
