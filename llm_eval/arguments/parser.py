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
 
    # add subparser for infilling setup
    parser_is = subparser.add_parser('infill_setup')
    parser_is.add_argument(
        '-c', '--cfg', help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )
    parser_is.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help=(
            'sets the program to debug mode. moves outputs to '
            'special locations'),
    )
    parser_is.add_argument(
        '-p', '--path', default=None, type=str,
        help=('if continuing from a checkpoint, provide the directory '
              'with this flag'),
    )

    # subparser for solving the infilling problems
    parser_is = subparser.add_parser('infill_solve')
    parser_is.add_argument(
        '-c', '--cfg', help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )
    parser_is.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help=(
            'sets the program to debug mode. moves outputs to '
            'special locations'),
    )
    parser_is.add_argument(
        '-m', '--model', default=None, type=str, required=True,
        help='the model to collect data on',
    )
    parser_is.add_argument(
        '-l', '--limit', type=int,
        help='sets the number of samples to work on', default=None,
    )
    parser_is.add_argument(
        '-p', '--path', default=None, type=str,
        help=('if continuing from a checkpoint, provide the directory '
              'with this flag'),
    )

    # subparser for evaluating the infilling responses
    parser_is = subparser.add_parser('infill_evaluate')
    parser_is.add_argument(
        '-c', '--cfg', help='config path',
        default=f'{files.project_root()}/cfg/config.yaml',
    )
    parser_is.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help=(
            'sets the program to debug mode. moves outputs to '
            'special locations'),
    )
    parser_is.add_argument(
        '-m', '--metric', type=str, required=True,
        help='the metric to evaluate with',
    )
    parser_is.add_argument(
        '-p', '--path', default=None, type=str,
        help=('if continuing from a checkpoint, provide the directory '
              'with this flag'),
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
    parser_gen = subparser.add_parser('unittest')
    parser_gen.add_argument(
        '-c',
        '--cfg',
        help='config path',
        type=str,
        default=f'{files.project_root()}/cfg/config.yaml',
    )

    parser_gen.add_argument(
        '-v',
        '--verbosity',
        help=('set to true to display more verbose failure messages. '
              '(0, 1, 2)'),
        type=int,
        default=1,
    )
    
    # call to parse arguments
    parser = parser.parse_args()

    # replace with keywords
    parser.cfg= strings.replace_slots(
        parser.cfg,
        keywords
    )

    return parser
