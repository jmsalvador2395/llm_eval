from typing import List, Tuple
from functools import wraps
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

from .base import Generator
from llm_eval.llm.session import Session

class ChatVLLM(Generator):
    
    def __init__(self,
        *args,
        **kwargs,
    ):
        super(ChatVLLM, self).__init__(*args, **kwargs)

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = LLM(
            model=self.model_name,
            seed=int(time.time()),
            #download_dir=model_cache,
            max_model_len=self.max_length,
            enforce_eager=True,
            worker_use_ray=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            max_tokens=self.max_length,
        )



    @wraps(Generator.generate)
    def generate(self, 
        sessions: List[Session],
        prompts: List[str]
    ) -> Tuple[List[Session], List[str]]:

        # prepare model input
        out_sessions = [
            sess.add_prompt(prompt)
            for sess, prompt in zip(sessions, prompts)
        ]
        formatted_prompts = [
            self.tok.apply_chat_template(
                sess.get_hist(), 
                tokenize=False) 
            for sess in out_sessions
        ]

        # make inference call
        responses = self.model.generate(
            formatted_prompts, self.sampling_params
        )
        
        # convert responses to text
        responses = [resp.outputs[0].text for resp in responses]

        # add model responses to history
        out_sessions = [
            sess.add_prompt(resp, role='assistant')
            for sess, resp in zip(out_sessions, responses)
        ]

        return out_sessions, responses