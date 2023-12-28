#external imports
import os
import yaml
import torch
import transformers
import time
from vllm import LLM, SamplingParams


# local imports
from .chat_session import ChatSession

class VLlmSession(ChatSession):
    """
    A subclass of ChatSession specifically for the LLAMA2 language model, using YAML configuration.
    """

    def __init__(self, config, model_name):
        """
        Initializes the LLAMA2 chat session.

        :param config_path: Path to the YAML configuration file.
        :param model_name: The name of the model to be used.
        """
        
        super().__init__(config, model_name)  # Initialize the base class with the loaded configuration

        # Set up LLAMA2 API credentials and model
        # Assuming LLAMA2 uses an API key and has a similar setup to OpenAI

        model_cache = config.model_params.get('model_cache', None)

        two_gpu_models = [
            'WizardLM/WizardCoder-3B-V1.0',
            'WizardLM/WizardCoder-15B-V1.0',
        ]
        single_gpu_models = [
            'tiiuae/falcon-7b-instruct',
            'tiiuae/falcon-40b-instruct',
        ]
        tensor_parallel_size=torch.cuda.device_count()
        if self.model_name in single_gpu_models and tensor_parallel_size > 1:
            tensor_parallel_size=1
        elif self.model_name in two_gpu_models and tensor_parallel_size > 2:
            tensor_parallel_size=2

        # Initialize usage statistics
        self.usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0,
        }

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.num_output_tokens,
        )

        self.model = LLM(
            model_name,
            trust_remote_code=True,
            download_dir=model_cache,
            #gpu_memory_utilization=1,
            dtype='auto',
            #dtype=torch.float16,
            tensor_parallel_size=tensor_parallel_size,
            seed=int(time.time())
        )

    def get_response(self,
                     user_message:   str | list,
                     system_message: str | list=None):
        """
        Retrieves a response from the vLLM language model.
        """

        msg, return_str = self._prepare_batch(user_message, system_message)
        # Implement the logic to interact with the LLAMA2 model's API
        # This is a placeholder implementation

        # generate response
        seqs = self.model.generate(
            msg,
            sampling_params=self.sampling_params,
        )


        # Update history and usage statistics
        # [Rest of the method should handle response parsing and updating the session similar to OpenAISession]
        if return_str:
            response = seqs[0].outputs[0].text
        else:
            response = [seq.outputs[0].text for seq in seqs]

        return response
