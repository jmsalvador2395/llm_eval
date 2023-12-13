# external imports
from datasets import Dataset
import traceback
import sys

# local imports
from llm_eval.teler import TELeR
from llm_eval.utils import (
    files,
    display,
)

def test_teler(args, cfg, keywords):
    display.in_progress('Testing prompt creation using TELeR class')

    try:
        template_dir = f'{files.project_root()}/cfg/prompts.yaml'
        teler = TELeR(cfg, template_dir)

        ds = Dataset.from_json(cfg.response_collection['datasets']['all_sides'])
        test_ds = teler.format_data(ds, 'all_sides', 1)

        display.ok('PASS')
        breakpoint()
    except Exception as e:
        display.fail('Failed to rune TELeR class without errors')
        if args.verbose:
            traceback.print_exception(*sys.exc_info())

    display.done()

