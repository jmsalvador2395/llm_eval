import os
import google.generativeai as palm
from tqdm import tqdm

# local imports
from .chat_session import ChatSession
from llm_eval.utils import (
    display,
)

class PalmSession(ChatSession):
    """
    A subclass of ChatSession specifically for OpenAI's language models, using YAML configuration.
    """

    def __init__(self, config, model_name):
        """
        Initializes the OpenAI chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used (e.g., '3.5' or '4').
        """
        # # Load configuration from YAML file
        # with open(config_path, 'r') as file:
        #     full_config = yaml.safe_load(file)
        
        # config = full_config['data_path']
        
        super().__init__(config, model_name)  # Initialize the base class with the loaded configuration
        palm.configure(api_key=config.api_keys['palm'])

        # Initialize usage statistics
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }


    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list=None):
        """
        Retrieves a response from a PALM language model.

        :param user_message: The user message to be sent to the model.
        :param system_message: An optional system message.
        :return: The response from the model.
        """

        #session = ChatSession(system_msg=cfg.response_collection['sys_role'])
        usr_msg, sys_msg, return_str = self._prepare_batch(user_message, system_message)

        responses = []
        for prompt, context in tqdm(zip(usr_msg, sys_msg), total=len(usr_msg)):
            session = palm.chat(
                context=context,
                messages=prompt,
                temperature=self.temperature,
            )
            #reply = session.get_response(msg)
            responses.append(session.last)

        if return_str:
            return responses[0]
        else:
            return responses
