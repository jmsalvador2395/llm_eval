# external imports
from datasets import Dataset
import traceback
import sys

# local imports
from llm_eval.evaluation import collect_scores
from llm_eval.utils import (
    files,
    display,
)

def test_eval(args, cfg, keywords):
    display.in_progress('Testing Evaluation Pipeline')

    try:
        collect_scores(args, cfg, keywords)

        display.ok('PASS')
        breakpoint()
    except Exception as e:
        display.fail('Failed to run TELeR class without errors')
        if args.verbose:
            traceback.print_exception(*sys.exc_info())

    display.done()

