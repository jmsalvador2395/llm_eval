"""
this is for reading in yaml files and replacing keywords with their 
intended values
"""
# external imports
import argparse

# internal imports
from llm_eval.utils import (
    files,
    strings,
)

def parse():
    """
    builds and returns the program argument parser
    """
    keywords = {
        'project_root' : files.project_root(),
        'home' : files.home(),
        'timestamp' : strings.now()
    }

    # base parser
    parser = argparse.ArgumentParser()

    # create subparser for procedures
    subparser = parser.add_subparsers(
        required=True, dest='procedure',
        description='decides on which procedure to run',
    )

    # add subparser for full execution of script
    parser_ea = subparser.add_parser('exec_all')
    parser_ea.add_argument(
        '-c', '--cfg', help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )

    parser_ea.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help=(
            'sets the program to debug mode. moves outputs to special '
            'locations'),
    )
    parser_ea.add_argument(
        '-l',
        '--limit',
        help='sets the number of samples to work on',
        default=None,
        type=int,
    )

    # add subparser for response collection procedure
    parser_rc = subparser.add_parser('response_collection')
    parser_rc.add_argument(
        '-c', '--cfg', help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )
    parser_rc.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help=(
            'sets the program to debug mode. moves outputs to '
            'special locations'),
    )
    parser_rc.add_argument(
        '-l', '--limit', type=int,
        help='sets the number of samples to work on', default=None,
    )
    parser_rc.add_argument(
        '--from_ckpt', default=None, type=str,
        help='use this to provide a di',
    )
 
    # add subparser for infilling procedure
    parser_rc = subparser.add_parser('infilling')
    parser_rc.add_argument(
        '-c', '--cfg', help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )
    parser_rc.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help=(
            'sets the program to debug mode. moves outputs to '
            'special locations'),
    )
    parser_rc.add_argument(
        '-l', '--limit', type=int,
        help='sets the number of samples to work on', default=None,
    )
    parser_rc.add_argument(
        '--from_ckpt', default=None, type=str,
        help='use this to provide a di',
    )
    parser_rc.add_argument(
        '--model', default=None, type=str, required=True,
        help='the model to collect data on',
    )

    # parser for the evaluation procedure
    parser_eval = subparser.add_parser('evaluate')
    parser_eval.add_argument(
        '-c',
        '--cfg',
        help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )
    parser_eval.add_argument(
        '-t',
        '--timestamp',
        help='the timestamp of the experiment you want to collect scores for',
        required=True,
    )

    # subparser for unit testing
    parser_gen = subparser.add_parser('unit_test')
    parser_gen.add_argument(
        '-c',
        '--cfg',
        help='config path',
        type=str,
        default=f'{files.project_root()}/cfg/config.yaml',
    )

    parser_gen.add_argument(
        '-v',
        '--verbose',
        help='set to true to display more verbose failure messages',
        action='store_true',
        default=False,
    )
    
    # call to parse arguments
    parser = parser.parse_args()

    # replace with keywords
    parser.cfg= strings.replace_slots(
        parser.cfg,
        keywords
    )

    return parser
