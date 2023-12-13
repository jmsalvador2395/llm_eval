# external imports
import os

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

    if args.procedure == 'exec_all':
        response_collector.run(args, cfg, keywords)
        evaluator.evaluate(args, cfg, keywords)

    # choose execute the desired procedure
    match args.procedure:
        case 'response_collection':
            response_collector.run(args, cfg, keywords)
        case 'evaluate':
            evaluation.collect_scores(args, cfg, keywords)
        case 'unit_test':
            unit_testing.unit_test(args, cfg, keywords)
        case _:
            raise NotImplementedError(utils.strings.clean_multiline(
                """
                Procedure added to args but case not added to main function in <project root>/lm_toolkit/__init__.py.
                """
            ))
            
