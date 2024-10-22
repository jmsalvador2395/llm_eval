import unittest

from llm_eval.llm.session import Session
from llm_eval.llm.generators import VLLM

llm_initialized = False

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.empty_chats = [Session(), Session()]
        self.empty_gens = [Session('standard'), Session('standard')]

        self.chat_prompts = ["how do i print 'hello world' in python?",
                             "who is the primarch of the ultramarines?"]
        self.gen = VLLM('allenai/OLMoE-1B-7B-0924-Instruct')

    def test_01_inference(self):
        if not self.gen:
            self.fail('VLLM initialization failed in setUp()')

        try:
            sess, resp = self.gen.generate(
                self.empty_chats, 
                self.chat_prompts
            )
        except:
            self.fail('failed to run inference')