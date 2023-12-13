from .test_groups import (
    test_llms,
    test_teler,
    test_eval,
)

from llm_eval.utils import (
    strings,
    display,
    validate,
)

def unit_test(args, cfg, keywords):

    test_cases = cfg.unit_test.get('cases', [])

    display.info(f'verbose: {args.verbose}')

    # start routine
    display.title('BEGIN UNIT TESTING')

    """ unit testing starts """

    if 'llm-responses' in test_cases:
        display.title('Testing LLM Interface')
        test_llms(args, cfg, keywords)
        display.title('Done')

    if 'teler' in test_cases:
        display.title('Testing TELeR Interface')
        test_teler(args, cfg, keywords)
        display.title('Done')

    if 'eval' in test_cases:
        display.title('Testing Evaluation Interface')
        test_eval(args, cfg, keywords)

    """ unit testing ends """

    # exit routine
    display.title('END UNIT TESTING')
