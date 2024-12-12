# external imports
import os
import unittest

# local imports
from llm_eval import (
    arguments, 
    cfg_reader,
    unit_testing,
    evaluation,
    response_collector,
    utils,
)


def main():
    """
    this is the entry point for the program
    look at the local functions imported from the top to see where the next steps are executed.
    to add more options, import the function, apply the same decorators that you see in the function
    definitions and add an 'add_argument()' entry below
    """

    # read args
    args = arguments.parse()
    cfg, keywords = cfg_reader.load(args.cfg)

    # choose execute the desired procedure
    match args.procedure:
        case 'exec_all':
            response_collector.run(args, cfg, keywords)
            evaluation.collect_scores(args, cfg, keywords)
        case 'response_collection':
            response_collector.run(args, cfg, keywords)
        case 'evaluate':
            evaluation.collect_scores(args, cfg, keywords)
        case 'unittest':
            unit_testing.run_tests(args, cfg, keywords)
        case 'infill_setup':
            response_collector.infill_setup(args, cfg, keywords)
        case 'infill_solve':
            response_collector.infill_solve(args, cfg, keywords)
        case 'infill_evaluate':
            response_collector.infill_evaluate(args, cfg, keywords)
        case _:
            raise NotImplementedError(utils.strings.clean_multiline(
                """
                Procedure added to args but case not added to main function in <project root>/lm_toolkit/__init__.py.
                """
            ))
            
