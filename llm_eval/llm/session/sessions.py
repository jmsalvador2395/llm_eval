from typing import List, Tuple, Dict, Self, Optional

class Session:
    """A wrapper for chat/inference dialogues
    
    This provides an interface for managing chat sessions with LLMs.
    This is meant to be used in conjunction with the SessionManager 
    class. 
    """
    def __init__(self, 
        sess_type='chat', 
        system_role: Optional[str]=None,
        hist: List[Dict[str, str]] | List[str]=[],
        override_sys: bool=False,
    ):
        """
        Args:
            sess_type: a string that is either 'chat' or 'standard'. 
              (Default: 'chat')
        Raises:
            ValueError: argument 'sess_type' must be either 'chat' or 
              'standard'
            ValueError: attempted to add a system role with history that
              already has it. To overwrite the existing role, set the
              'override_sys' flag to True
        """
        self.hist = hist

        if sess_type not in ['chat', 'standard']:
            raise ValueError(
                'argument \'sess_type\' must be either \'chat\' or '
                '\'standard\''
            )
        self.sess_type=sess_type

        el = {'role': 'system', 'content': system_role}

        if system_role and len(self.hist) == 0:
            self.hist.append(el)
        elif system_role and len(self.hist) > 0:
            if self.hist[0]['role'] != 'system':
                self.hist = [el] + self.hist
            elif override_sys:
                self.hist = [el] + self.hist[1:]
            else:
                raise ValueError(
                    "attempted to add a system role with history that "
                    "already has it. To overwrite the existing role, "
                    "set the 'override_sys' flag to True"
                )
    
    def __str__(self):
        return f'Session({self.hist})'

    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.hist)

    def get_hist(self) -> List[Dict[str, str]] | List[str]:
        """returns the chat history"""
        return self.hist
    
    def get_last(self) -> str | None:
        """Returns the last message in the session history

        Returns the last message in the session history. If there is
        nothing in the history, None is returned

        Raises:
            Exception: Unexpected case reached due to invalid Session.
        """
        if len(self.hist) == 0:
            return None
        elif self.sess_type == 'chat':
            return self.hist[-1]['content']
        elif self.sess_type == 'standard':
            return self.hist[-1]
        else:
            raise Exception(
                'Unexpected case reached due to invalid Session'
            )
    def add_prompt(self, 
        prompt: str, 
        role: str='user',
        inplace: bool=False
    ) -> Optional[Self]:
        """adds a prompt to the session

        takes a prompt and appends it to the end of the session history.
        'role' is only used if self.sess_type=='chat'.

        Example usage:
          >>> sess = Session('chat')
          >>> sess.add_prompt('hello')
          >>> print(sess.hist)
          [{'role': 'user', 'content': 'hello'}]
        """
        el = {'role': role, 'content': prompt}
        if inplace:
            self.hist.append(el)
            return self
        else:
            return Session(
                sess_type=self.sess_type, 
                hist=self.hist + [el]
            )
    
    def __add__(self, prompt: str):
        return self.add_prompt(prompt)


class SessionManager:
    """used to manage multiple chat/inference sessions

    This is the main interface for running inference on LLMs. This will
    automatically manage the inference model and the multiple chat 
    sessions of the user so the user does not have to worry so much 
    about managing managing multiple chat dialogues.

    """
    def __init__(self, 
        model_name: str,
    ):
        pass

    def prepare_to_log(self, sessions: List[Session]):
        """converts chat sessions to be used with SQL updates"""
        raise NotImplementedError()
    
    def create_sessions(self, 
        N: int, 
        session_type: str
    ) -> List[Session]:
        """initializes new sessions

        returns a list of 'N' sessions depending on the 'session_type'

        Args:
            N: the number of sessions to be initialized
            session_type: the type of sessions to be initialized. Must 
              be one of ['chat', 'standard']
        Returns:
            A list of 'Session' objects.
            for example:

            [Session([]), Session([])]
        Raises:
            ValueError: 'session_type' must be one of ['chat', 'standard']'
        """

        if session_type not in ['chat', 'standard']:
            raise ValueError(
                '\'session_type\' must be one of [\'chat\', \'standard\']'
            )
        raise NotImplementedError()

    def gen_response(
        self, 
        gen_model: str,
        sessions: List[Session], 
        prompts: List[str],
    ) -> Tuple[List[Session], List[str]]:
        """Runs inference on a generator given a set of prompts

        Runs inference on batch of prompts given the generator and
        corresponding sessions.

        Args:
            gen_model: the model name of the generator
            generator: a generator class from 
              llm_eval.llm.chat_session.generators
            sessions: a list of Sessions containing chat history
            prompts: a list of prompts that correspond to the sessions
              argument
        
        Returns:
            a tuple of sessions updated with the LLM responses and also
            a list of the same responses responses.
            for example:

            ([Session([{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}])],
             ["hello"])

        """

        assert len(sessions) == len(prompts)