import sys
import gc
import torch
import time
import ray
from datasets import Dataset
import traceback
#from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm import distributed
from vllm.distributed import destroy_model_parallel
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
        #"mistralai/Mistral-7B-Instruct-v0.1",
        #"mistralai/Mistral-7B-Instruct-v0.2",
        'microsoft/Phi-3-mini-4k-instruct',
        'microsoft/Phi-3-mini-128k-instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
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
            torch.cuda.empty_cache()
            display.info('cuda empty cache')
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
            try:
                del session.model
                display.info('deleted session.model')
            except:
                pass
            gc.collect()
            display.info('gc collect')
            torch.cuda.empty_cache()
            display.info('cuda empty cache')

            del session
            try:
                ray.shutdown()
                display.info('ray shutdown')
            except:
                pass

            try:
                destroy_model_parallel()
                display.info('destroy model parallel')
            except:
                pass

            try:
                destroy_process_group()
                display.info('destroy process group')
            except:
                pass

            try:
                distributed.device_communicators.pynccl_utils.destroy_process_group()
                display.info('vllm destroy process group')
            except:
                pass



