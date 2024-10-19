from typing import List, Optional
import time

class Generator:
    """Defines the standard API inference interface 

    Defines the standard API inference interface. 

    """
    def __init__(self, 
        model_name: str, 
        top_p: float=1.0,
        top_k: int=-1,
        num_beams: int=1,
        temperature: float=1.0,
        sampling: bool=True,
        max_length: int=4096,
        seed: Optional[int]=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.temperature = temperature
        self.sampling = sampling
        self.max_length = max_length
        self.seed = seed


    def generate(self,
        prompts: List[str],
    ) -> List[str]:
        """passes prompts to the generator and returns the responses

        Args:
            prompts: the list of prompts to pass to the generator
        Returns:
            A list of responses from the generator.
            Example:

            ['this is a response to the first prompt',
             'this is a response to the second prompt']
        """
        raise NotImplementedError(
            'Subclasses must implement this function'
        )
        