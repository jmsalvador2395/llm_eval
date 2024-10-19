
import unittest

from llm_eval.llm.session import *

class TestSessions(unittest.TestCase):

    def test_init_default_session(self):
        """test if you can initialize a session with no args"""
        session = Session()

    def test_init_chat_session(self):
        """test if you can initialize a session with sess_type='chat'"""
        session = Session(sess_type='chat')
    
    def test_init_chat_gen(self):
        """test if you can initialize a session with sess_type='gen'"""
        session = Session(sess_type='gen')
    
    def test_invalid_init(self):
        """checks if an invalid sess_type raises an error"""
        self.assertRaises(
            ValueError, 
            Session,
            **{'sess_type': 'asdfasdf'}
        )
    
    def test_init_with_hist(self):
        session = Session(hist=[])