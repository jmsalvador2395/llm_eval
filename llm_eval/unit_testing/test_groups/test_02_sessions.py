
import unittest

from llm_eval.llm.session import *

class TestSessions(unittest.TestCase):

    def test_01_init_default_session(self):
        """test if you can initialize a session with no args"""
        session = Session()

    def test_02_init_chat_session(self):
        """test if you can initialize a session with sess_type='chat'"""
        session = Session(sess_type='chat')
    
    def test_03_init_chat_standard(self):
        """test if you can initialize a session with sess_type='gen'"""
        session = Session(sess_type='standard')
    
    def test_04_invalid_init(self):
        """checks if an invalid sess_type raises an error"""
        self.assertRaises(
            ValueError, 
            Session,
            **{'sess_type': 'asdfasdf'}
        )
    
    def test_05_init_without_hist(self):
        session = Session(hist=[
            {'role': 'user', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
        ])
 
    
    def test_06_init_with_hist(self):
        session = Session(hist=[
            {'role': 'system', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
        ])
    
    @unittest.expectedFailure
    def test_07_init_with_multiple_system_role_hist(self):
        sessions = Session(hist=[
            {'role': 'system', 'content': 'content'},
            {'role': 'system', 'content': 'content'},
        ])
    
    @unittest.expectedFailure
    def test_08_init_with_invalid_order_01(self):
        sessions = Session(hist=[
            {'role': 'user', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
        ])

    @unittest.expectedFailure
    def test_09_init_with_invalid_order_02(self):
        sessions = Session(hist=[
            {'role': 'assistant', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
        ])

    @unittest.expectedFailure
    def test_10_init_with_invalid_order_03(self):
        sessions = Session(hist=[
            {'role': 'user', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
        ])

    @unittest.expectedFailure
    def test_11_init_with_invalid_order_04(self):
        sessions = Session(hist=[
            {'role': 'user', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
        ])

    @unittest.expectedFailure
    def test_12_init_with_invalid_order_05(self):
        sessions = Session(hist=[
            {'role': 'system', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
        ])

    @unittest.expectedFailure
    def test_13_init_with_invalid_order_06(self):
        sessions = Session(hist=[
            {'role': 'system', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
            {'role': 'assistant', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
            {'role': 'user', 'content': 'content'},
        ])