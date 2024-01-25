import os
import google.generativeai as genai
from tqdm import tqdm
import time

# local imports
from .chat_session import ChatSession
from llm_eval.utils import (
    display,
)

class GeminiSession(ChatSession):
    """
    A subclass of ChatSession specifically for Google's language models, using YAML configuration.
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
        genai.configure(api_key=config.api_keys['palm'])
        self.model = genai.GenerativeModel(model_name)

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

        chat = self.model.start_chat(history=[])
        responses = []
        start = time.time()
        num_queries = 0
        for prompt, context in tqdm(zip(usr_msg, sys_msg), total=len(usr_msg)):
            response = chat.send_message(content=prompt)
            #reply = session.get_response(msg)
            responses.append(response.text)
            num_queries += 1
            end = time.time()

            # set cooldown time due query limits (https://ai.google.dev/pricing)
            if num_queries >= 60 and end-start > 55:
                time.sleep(int(65-(end-start)))
                start = time.time()
                num_queries = 0

        if return_str:
            return responses[0]
        else:
            return responses
