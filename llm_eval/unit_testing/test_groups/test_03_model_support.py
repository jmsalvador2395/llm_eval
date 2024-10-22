import re
import sys
import unittest
import  yaml
from transformers import AutoTokenizer

from llm_eval.llm.session import Session, apply_custom_format
from llm_eval.utils import (files, strings)

class TestLLMs(unittest.TestCase):
    
    def setUp(self):
        """create list of models to check for tokenization support"""

        # set verbosity
        self.verbosity = 1
        if '-v' in sys.argv:
            self.verbosity = int(sys.argv[sys.argv.index('-v') + 1])

        # read model list
        model_list_path = (
            f'{files.project_root()}'
            f'/cfg/supported_models.yaml'
        )
        with open(model_list_path) as f:
            try:
                self.model_list = yaml.safe_load(f.read())
            except Exception as e:
                print(
                    f'model list not found at {model_list_path}. '
                    f'this file needs to be moved back or restored.'
                )

        # read custom formats
        custom_formats_path = (
            f'{files.project_root()}'
            f'/cfg/custom_formats.yaml'
        )
        with open(custom_formats_path) as f:
            try:
                self.custom_formats = yaml.safe_load(f.read())
            except Exception as e:
                print(
                    f'custom formats not found at {custom_formats_path}. '
                    f'this file needs to be moved back or restored.'
                )

        self.prompt_no_sys = [
            {'role': 'user', 'content': 'who are the adeptus astartes'},
        ]

        self.prompt_with_sys = [
            {'role': 'system', 
             'content': 'you are the empereor of the imperium'},
            {'role': 'user', 
             'content': 'who are the adeptus astartes'},
        ]

        self.session = Session(sess_type='chat_custom', hist=[
            {'role': 'system', 'content': 'you are a senior developer'},
            {'role': 'user', 'content': 'how do i print hello world'},
            {'role': 'assistant', 'content': 'print(\'hello world\')'},
        ])
    def _check_tok(self, models, chat_hist, with_sys):
        """check if model fails to initialize or fails to tokenize"""
        failed_to_init = []
        init_fail_reasons = []
        failed_to_tok = []
        tok_fail_reasons = []

        for model in models:
            is_init = False
            try:
                tok = AutoTokenizer.from_pretrained(model)
                is_init = True
            except Exception as e:
                if self.verbosity == 2:
                    model += f', reason: {str(e)}'
                failed_to_init.append(model)

            if is_init:
                try:
                    tok.apply_chat_template(
                        self.prompt_no_sys,
                        tokenize=False,
                    )
                except Exception as e:
                    if self.verbosity == 2:
                        model += f', reason: {str(e)}'
                    failed_to_tok.append(model)

        fail_str = ''
        if len(failed_to_init) > 0:
            fail_str += (
                f'\n\n1 or more tokenizers failed to initialize:\n\t'
                + '\n\t'.join(failed_to_init)
            )
        if len(failed_to_tok) > 0:
            fail_str += (
                f'\n\n1 or more tokenizers failed to apply '
                f'tokenization {"with" if with_sys else "without"} '
                f'system role:\n\t'
                + '\n\t'.join(failed_to_tok)
            )

        if fail_str != '':
            self.fail(fail_str)
            
    def test_01_chat_template_no_sys(self):
        """check if tokenizers work without role"""
        self._check_tok(
            self.model_list['chat'], 
            self.prompt_no_sys,
            with_sys=False,
        )

    def test_02_chat_template_with_sys(self):
        """check if tokenizers work with role"""
        self._check_tok(
            self.model_list['chat'], 
            self.prompt_with_sys,
            with_sys=True,
        )

    def test_03_load_standard_tokenizers(self):
        """check if you can load the tokenizers of candidate models"""
        models = self.model_list['standard']
        failed_to_init = []

        for model in models:
            try:
                tok = AutoTokenizer.from_pretrained(model)
            except Exception as e:
                if self.verbosity == 2:
                    model += f', reason:\n{str(e)}'
                failed_to_init.append(model)
        
        if len(failed_to_init) > 0:
            self.fail(
                'failed to initialize some models:\n\t'
                + '\n\t'.join(failed_to_init)
            )

    def test_04_apply_custom_formats(self):
        """checks if custom formats are properly applied"""
        models = self.model_list['custom_formats']
        keys = list(self.custom_formats.keys())
        failed = False

        multi_matched = []
        none_matched = []
        for model in models:
            matched_keys = list(filter(
                lambda x: re.match(x, model),
                keys
            ))
            if len(matched_keys) == 1:
                key = matched_keys[0]
                try:
                    text = apply_custom_format(
                        self.custom_formats[key], 
                        self.session,
                    )
                except Exception as e:
                    self.fail(
                        f'failed while calling apply_custom_format():\n'
                        f'{e}'
                    )
            # check failure conditions
            elif len(matched_keys) == 0:
                none_matched.append(model)
                failed = True
                continue
            elif len(matched_keys) > 1:
                multi_matched.append((model, matched_keys))
                failed = True
                continue
        
        fail_str = ''
        if len(multi_matched) > 0:
            fail_str += (
                f'\n{len(multi_matched)} model(s) matched more than '
                f'1 custom format:'
            )
            for model, matched_keys in multi_matched:
                fail_str += f'\n\t{model} matched with: {matched_keys}'
        if len(none_matched) > 0:
            fail_str += (
                f'\n\n{len(none_matched)} model(s) did not match any '
                f'custom format key:\n\t'
                + '\n\t'.join(none_matched)
            )

        if failed:
            self.fail(fail_str)
