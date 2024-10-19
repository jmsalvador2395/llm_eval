import unittest

from llm_eval.llm.session import Session
from llm_eval.llm.generators import ChatVLLM

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.empty_chats = [Session(), Session()]
        self.empty_gens = [Session('gen'), Session('gen')]

        self.chat_prompts = ["how do i print 'hello world' in python?",
                             "who is the primarch of the ultramarines?"]
        self.gen = ChatVLLM('allenai/OLMoE-1B-7B-0924-Instruct')

    def test_temp(self):
        sess, resp = self.gen.generate(
            self.empty_chats, 
            self.chat_prompts
        )
        print(sess)
        breakpoint()