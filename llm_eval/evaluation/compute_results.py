import os

# external dependencies
from typing import List, Dict
import datasets
import torch
import evaluate
import os
import re
import json
import gc
import numpy as np
import sys
#import tensorflow as tf
from datasets import Dataset, DatasetDict, concatenate_datasets
from nltk import sent_tokenize
from itertools import product, chain
from pprint import pprint
from tqdm import tqdm
import time

# local dependencies
from llm_eval.llm import *
from llm_eval.utils import (
    files,
    strings,
    display,
)
from .similarity_metrics import use_similarity, sbert_similarity
#from .compute_fans import compute_fans

def collect_scores(args, cfg, keywords):

    if args.procedure == 'unit_test':
        base_path = f'{files.project_root()}/data/unit_test/eval'
    elif args.procedure == 'exec_all':
        base_path = f'{cfg.resp_coll["save_dir"]}/{keywords["timestamp"]}'
    elif args.procedure == 'evaluate':
        base_path = f'{cfg.resp_coll["save_dir"]}/{args.timestamp}'

    # try to read in evaluation progress
    prog_path = f'{base_path}/eval_progress.json'
    try:
        with open(prog_path) as f:
            progress = json.load(f)
        display.info('checkpoint info loaded')
    except Exception as e:
        progress = {}
        with open(prog_path, 'w') as f:
            f.write(json.dumps(progress))
        display.info('checkpoint info not found. creating data')

    # get file names in target path
    targets = []
    for path, subdirs, fnames in os.walk(base_path):
        for name in fnames:
            if name.endswith('results.json'):
                targets.append(os.path.join(path, name))
    targets = sorted(targets)

    # extract model names from each target
    model_names = [
        re.match(f'{base_path}/(.*)/results.json', trgt).groups()[0]
        for trgt in targets
    ]

    metrics = cfg.eval.get('metrics', ['rouge'])
    if type(metrics) != list:
        display.error('Error in config: "eval -> metrics" should be a list')
        raise ValueError('config -> eval -> metrics should be a list')
    elif metrics == []:
        display.error('Error in config: "eval -> metrics" should not be empty')
        raise ValueError('config -> eval -> metrics should not be empty')
            
    # get set of dataset names
    ds_names = sorted(set(cfg.datasets.keys()))

    # read in reference datasets
    ds_cache = cfg.resp_coll['ds_cache']
    ref_data = {}
    try:
        d_sets = cfg.datasets
        for ds_name, d_set in d_sets.items():
            ds = datasets.load_dataset(
                d_set['type'],
                data_files=d_set['path'],
                cache_dir=ds_cache,
            )
            ref_data[ds_name] = ds['train']
    except Exception as e:
        display.error(
            f'required config parameters not provided for resp_coll -> '
            f'datasets. refer to '
            f'\'{files.project_root()}/cfg/template.yaml\' for examples'
        )
        #traceback.print_exception(*sys.exc_info())
        print(e)
        os._exit(1)

    all_scores = []
    start = time.time()
    display.info('evaluating: ' + "\n\t".join(targets))

    for trgt, model_name in tqdm(
            zip(targets, model_names), 
            total=len(targets)):
        
        if model_name in progress:
            display.info(
                f'`{model_name}` found in progress file'
            )
            if set(metrics) - set(progress[model_name]) == set():
                display.info('all metrics accounted for. skipping ...')
                continue
        else:
            progress[model_name] = {}

        sample_start = time.time()
        display.info(f'computing metrics for {model_name}')

        # read in target
        ds = Dataset.from_json(
            trgt,
            cache_dir=cfg.resp_coll['ds_cache'],
        )

        # add indiices
        ds = ds.map(
            lambda x, idx: x | {'index': idx},
            with_indices=True,
        )

        # remove empty samples
        before_prune = len(ds)
        before_prune = len(ds)
        ds = ds.filter(lambda x: x['response'] is not None)
        ds = ds.filter(lambda x: x['response'].strip() != '')
        after_prune = len(ds)

        display.info(
            f'before prune: {before_prune}, after_prune: {after_prune} '
            f'difference: {before_prune - after_prune}'
        )

        # split by dataset names
        splits = DatasetDict({
            ds_name: ds.filter(lambda x: x['dataset'] == ds_name)
            for ds_name in ds_names
        })

        for ds_name, trgt_ds in splits.items():
            # compute metrics
            out_ds = trgt_ds
            ref_cols = cfg.datasets[ds_name]['references']
            ref_indices = trgt_ds['id']

            unique_refs = {}
            """
            unique_refs = list(zip(*(
                ref_data[ds_name][ref_col]
                for ref_col in ref_cols
            )))
            references = [list(unique_refs[ref_id]) for ref_id in ref_indices]
            """
            unique_references = ref_data[ds_name].select_columns(
                ref_cols
            )
            references = unique_references.select(ref_indices)
            out_ds = concatenate_datasets([out_ds, references], axis=1)

            key = f'{ds_name}-rouge'
            if 'rouge' in metrics:
                if key not in progress[model_name]:
                    display.info(f'computing rouge for {model_name}-{ds_name}')
                    rouge_scores = compute_rouge(out_ds, ref_cols)
                    progress[model_name][key] = rouge_scores
                    with open(prog_path, 'w') as f:
                        f.write(json.dumps(progress))
                else:
                    display.info(
                        f'rouge checkpoint for {model_name}-{ds_name} found'
                    )
                    rouge_scores = progress[model_name][key]

                out_ds = concatenate_datasets(
                    [out_ds, Dataset.from_dict(rouge_scores)],
                    axis=1,
                )

            key = f'{ds_name}-semf1'
            if 'semf1' in metrics:
                if key not in progress[model_name]:
                    display.info(f'computing semf1 for {model_name}-{ds_name}')
                    semf1_scores = compute_semf1(out_ds, ref_cols)
                    progress[model_name][key] = semf1_scores
                    with open(prog_path, 'w') as f:
                        f.write(json.dumps(progress))
                else:
                    display.info(
                        f'sem-f1 checkpoint for {model_name}-{ds_name} found'
                    )
                    semf1_scores = progress[model_name][key]
                out_ds = concatenate_datasets(
                    [out_ds, Dataset.from_dict(semf1_scores)],
                    axis=1,
                )
            
            key = f'{ds_name}-bertscore'
            if 'bertscore' in metrics:
                if key not in progress[model_name]:
                    display.info(
                        f'computing bertscore for {model_name}-{ds_name}'
                    )
                    bertscores = compute_bertscore(out_ds, ref_cols)
                    progress[model_name][key] = bertscores
                    with open(prog_path, 'w') as f:
                        f.write(json.dumps(progress))
                else:
                    display.info(
                        f'bertscore checkpoint for {model_name}-{ds_name} found'
                    )
                    bertscores = progress[model_name][key]
                out_ds = concatenate_datasets(
                    [out_ds, Dataset.from_dict(bertscores)],
                    axis=1,
                )
                """
                sample_data.update({
                    'bert_score': bert_score,
                })
                """

            # add data for output dataset
            #all_scores.append(sample_data)
            out_ds = out_ds.remove_columns(ref_cols)
            out_ds.to_json(f'{base_path}/{model_name}/{ds_name}-scores.json')

            # cleanup
            trgt_ds.cleanup_cache_files()
            out_ds.cleanup_cache_files()
            del out_ds
            gc.collect()

        # write out data
        #Dataset.from_list(all_scores).to_json(f'{base_path}/scores.json')

        end = time.time()
        print('=================')
        print(f'Time to Finish Sample: {end-sample_start:.02f} seconds\n'
            + f'Total Elapsed Time: {(end-start)/60:.02f} minutes')
        print('=================')

def is_float(val):

    # try to convert to float. fails if not float
    try:
        float(val)
    except:
        return False

    # try to convert to int. this will fail if val is a float
    if str(val).isnumeric():
        return False
    else:
        return True

def replace_nones(responses: List[str]):
    out = [el if el is not None else '' for el in responses]
    return out

def compute_rouge(
            ds: Dataset, 
            ref_cols: List[str]):
    """ 
    computes individual rouge scores for the given dataset and 
    references

    :param ds: the dataset that contains the predictions
    :type ds: Dataset

    :param references: the list o
    """
    rouge_preds = ds['response']

    #rouge_refs = list(zip(*[ds[rc] for rc in ref_cols]))
    references = list(zip(*[ds[rc] for rc in ref_cols]))
    rouge_scorer = evaluate.load('rouge')

    #rouge_preds = [doc if doc is not None else '' for doc in rouge_preds]

    rouge_metrics = rouge_scorer.compute(
        predictions=rouge_preds,
        references=references,
        use_aggregator=False,
    )
    return rouge_metrics

def compute_bertscore(ds: Dataset,
                      ref_cols: List[str], 
                      rescale_with_baseline: bool=True):

    # set preds and refs
    bertscore_preds = ds['response']
    bertscore_preds = replace_nones(bertscore_preds)
    bertscore_refs = {rc: ds[rc] for rc in ref_cols}

    # load metric
    bertscore = evaluate.load("bertscore")

    # initialize metrics dictionary
    bertscore_metrics = {}

    # evaluation loop
    scores = []
    for rc in ref_cols:
        try:
            gc.collect()
            torch.cuda.empty_cache()
            scores.append(
                bertscore.compute(
                    predictions=bertscore_preds, 
                    references=bertscore_refs[rc], 
                    lang="en",
                    rescale_with_baseline=rescale_with_baseline,
                    batch_size=60,
                    verbose=True,
                )
            )

        except Exception as e:
            display.error('failed to compute bertscore at ')
            print(e)
            sys.exit(1)
    #metrics_stacked = np.vstack([metric1['f1'], metric2['f1'], metric3['f1'], metric4['f1']])
    metrics_stacked = np.vstack([score['f1'] for score in scores])
    metrics_maxed = np.max(metrics_stacked, axis=0)
    #metrics_ag = np.mean(metrics_maxed)
    
    #bertscore_metrics = metrics_ag
    bertscore_metrics = {
        'bertscore': metrics_maxed.tolist(),
        'hashcode': [scores[-1]['hashcode']]*len(ds),
    }
        
    return bertscore_metrics


def compute_semf1(ds: Dataset,
                  ref_cols: List[str]):

    semf1_preds = ds['response']

    # make dictionary of references. each key represents 1 column of references
    semf1_refs = {rc: ds[rc] for rc in ref_cols} # rc = ref_col

    # aggregate the references so that iterating provides one set of references for one sample
    semf1_refs_ag = list(zip(*[ds[rc] for rc in ref_cols]))

    semf1_refs_ag = ['\n'.join(grp) for grp in semf1_refs_ag]
    semf1_recall_preds = []

    for grp in zip(*[ds[rc] for rc in ref_cols]):
        semf1_recall_preds.extend(list(grp))

    semf1_recall_refs  = []
    for ref in ds['response']:
        semf1_recall_refs.extend([ref]*len(ref_cols))

    semf1_preds = [sent_tokenize(doc) if doc is not None else [''] for doc in semf1_preds]
    for key in semf1_refs.keys():
        semf1_refs[key] = [sent_tokenize(doc) if doc is not None else [''] for doc in semf1_refs[key]]
    semf1_recall_refs = [sent_tokenize(doc) if doc != '' else [''] for doc in semf1_recall_refs]
        
    semf1_refs_ag = [sent_tokenize(doc) for doc in semf1_refs_ag]
    semf1_recall_preds = [sent_tokenize(doc) for doc in semf1_recall_preds]

    semf1_metrics = {}

    # compute metrics
    # precision computed as `score(pred, ' '.join([ref1, ref2, ref3, ref4])`
    p_rob = sbert_similarity(semf1_preds, semf1_refs_ag, 'stsb-roberta-large')
    r_rob = sbert_similarity(semf1_recall_preds, semf1_recall_refs, 'stsb-roberta-large')
    p_use = use_similarity(semf1_preds, semf1_refs_ag)
    r_use = use_similarity(semf1_recall_preds, semf1_recall_refs)
    p_distil = sbert_similarity(semf1_preds, semf1_refs_ag, 'paraphrase-distilroberta-base-v1')
    r_distil = sbert_similarity(semf1_recall_preds, semf1_recall_refs, 'paraphrase-distilroberta-base-v1')

    # compute averages for each sample
    p_use = np.array([np.mean(sample) for sample in p_use])
    r_use = np.array([np.mean(sample) for sample in r_use])
    p_distil = np.array([np.mean(sample) for sample in p_distil])
    r_distil = np.array([np.mean(sample) for sample in r_distil])
    p_rob = np.array([np.mean(sample) for sample in p_rob])
    r_rob = np.array([np.mean(sample) for sample in r_rob])

    # aggregate the recall scores as `mean([score(ref1, pred), score(ref2, pred), ...])`
    r_use = np.mean(r_use.reshape((len(ds), -1)), axis=-1)
    r_distil = np.mean(r_distil.reshape((len(ds), -1)), axis=-1)
    r_rob = np.mean(r_rob.reshape((len(ds), -1)), axis=-1)

    # compute individual f1 scores
    semf1_metrics = {
        'semf1-use': (2*(p_use*r_use)/(p_use+r_use)).tolist(),
        'semf1-distil': (2*(p_distil*r_distil)/(p_distil+r_distil)).tolist(),
        'semf1-rob': (2*(p_rob*r_rob)/(p_rob+r_rob)).tolist(),
    }

    gc.collect()
    torch.cuda.empty_cache()


    return semf1_metrics
