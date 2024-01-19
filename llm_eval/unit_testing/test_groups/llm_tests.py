import sys
import gc
import torch
import time
import ray
from datasets import Dataset
import traceback
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from torch.distributed import destroy_process_group

from llm_eval.llm import *
from llm_eval.utils import (
    files,
    strings,
    display
)

def test_llms(args, cfg, keywords):

    batch_size = cfg.model_params.get('batch_size', 10)

    llms = [
        'tiiuae/falcon-7b-instruct'
        'mistralai/Mistral-7B-Instruct-v0.1',
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-7b-v1.5-16k',
        'lmsys/vicuna-13b-v1.5',
        'lmsys/vicuna-13b-v1.5-16k',
    ]

    llms = [
        #'chat-bison-001',
        'gpt-3.5-turbo',
        #'meta-llama/Llama-2-7b-chat-hf',
        #'mosaicml/mpt-7b-instruct',
        #'mosaicml/mpt-7b-chat',
        #'mosaicml/mpt-30b-chat',
        #'mosaicml/mpt-30b-instruct',
    ]

    usr_messages = [
        'what does the symbol "eta" usually represent in math',
        'what is the difference between supervised and unsupervised learning',
    ]
    sys_messages = ['respond to the following instructions to the best of your ability']*len(usr_messages)
    #usr_messages = ds[1:301]['directive']
    #sys_messages = ds[1:301]['system_role']

    out_dir = f'{files.project_root()}/data/unit_test/llm_responses'

    for llm in llms:
        display.in_progress(f'Testing {llm}')
        success=False
        try:

            # create session and generate response
            session = select_chat_model(cfg, llm)
            display.info(f'session type for {llm} is: {type(session)}')

            start2 = time.time()
            response = session.get_response(usr_messages, sys_messages)
            #response = session.get_response(usr_messages)
            end2 = time.time()

            display.info(f'generation with sys_msg took {end2-start2:.02f} seconds')

            # save response to text file for inspection
            trgt_path = f'{out_dir}/{llm}.txt'
            files.create_path(files.dirname(trgt_path))
            with open(trgt_path, 'w', encoding='utf-8') as f:
                if type(response) == list:
                    out_str = '\n\n==================\n\n'.join(response)
                else:
                    out_str = response
                out_str = out_str.replace('\r\n', '\n')
                f.write(out_str)

            display.ok('PASS')
            success=True

        except KeyboardInterrupt as e:
            print(e)
            sys.exit(0)
        except Exception as e:
            display.fail(f'Error in running inference on {llm}: {str(e)}')
            if args.verbose:
                traceback.print_exception(*sys.exc_info())

        # free memory (not doing so causes CUDA_OUT_OF_MEMORY
        if success:
            del session
            ray.shutdown()
            gc.collect()
            torch.cuda.empty_cache()
            destroy_model_parallel()
            destroy_process_group()


