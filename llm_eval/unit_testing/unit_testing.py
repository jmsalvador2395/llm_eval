from llm_eval.utils import files

import unittest

def run_tests(args, cfg, keywords):

    suite = unittest.defaultTestLoader.discover(
        f'{files.project_root()}/llm_eval/unit_testing/test_groups',
        top_level_dir=files.project_root(),
    )

    runner = unittest.TextTestRunner(verbosity=args.verbosity)
    runner.run(suite)