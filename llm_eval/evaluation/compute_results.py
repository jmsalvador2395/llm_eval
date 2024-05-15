import os

# external dependencies
from typing import List, Dict
import datasets
import torch
import evaluate
import os
import re
import json
import numpy as np
#import tensorflow as tf
from datasets import Dataset
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
from .similarity_metrics import (
    use_similarity,
    sbert_similarity,
)
#from .compute_fans import compute_fans

def collect_scores(args, cfg, keywords):

    if args.procedure == 'unit_test':
        base_path = f'{files.project_root()}/data/unit_test/eval'
    elif args.procedure == 'exec_all':
        base_path = f'{cfg.resp_coll["save_dir"]}/{keywords["timestamp"]}'
    elif args.procedure == 'evaluate':
        base_path = f'{cfg.resp_coll["save_dir"]}/{args.timestamp}'

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
    ds_names = set(cfg.datasets.keys())
    breakpoint()

    all_scores = []
    start = time.time()

    for trgt, model_name in tqdm(zip(targets, model_names), total=len(targets)):
        sample_start = time.time()
        
        display.info(f'computing metrics for {model_name}')
        ds = Dataset.from_json(
            trgt,
            cache_dir=cfg.resp_coll['ds_cache'],
        )

        # get dataset name for given target
        trgt_split = set(trgt.replace(base_path, '').split('/'))
        trgt_ds = trgt_split.intersection(ds_names).pop()

        # remove empty samples
        before_prune = len(ds)
        ds = ds.filter(lambda x: x['response'] is not None)
        ds = ds.filter(lambda x: x['response'].strip() != '')
        after_prune = len(ds)
        #print(f'size before filter: {before_prune}\nsize after filter: {after_prune}')

        # compute metrics
        ref_cols = cfg.datasets[trgt_ds]['references']
        
        # extract meta-data from file name
        data_info = trgt[len(base_path)+1:].split('/')
        data_info.pop(0)
        trgt_level = data_info.pop(-1).replace('level', '').replace('.json', '')

        # create sample for output dataset
        trgt_data = {
            'name': '/'.join(data_info),
            'dataset': trgt_ds,
            'level': int(trgt_level),
            'num_samples': before_prune,
            'num_evaluated': after_prune,
        }
        if 'rouge' in metrics:
            rouge_scores = compute_rouge(ds, ref_cols)
            trgt_data.update(rouge_scores)
        if 'sem-f1' in metrics:
            semf1 = compute_semf1(ds, ref_cols)
            trgt_data.update({
                'sem-f1-use': semf1['f_use'],
                'sem-f1-distilroberta': semf1['f_distil'],
                'sem-f1-roberta': semf1['f_rob'],
            })
        if 'bert_score' in metrics:
            bert_score = compute_bertscore(ds, ref_cols)
            trgt_data.update({
                'bert_score': bert_score,
            })
        """
        if 'fans' in metrics:
            fans_score = compute_fans(ds, ref_cols, cfg)
            trgt_data.update({
                'fans': fans_score['fans_f1'],
                'fans_num_evaluated': fans_score['num_evaluated'],
            })
        """

        # add data for output dataset
        all_scores.append(trgt_data)

        # write out data
        Dataset.from_list(all_scores).to_json(f'{base_path}/scores.json')

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

def compute_rouge(ds: Dataset, 
                  ref_cols: List[str]):

    rouge_preds = ds['response']

    rouge_refs = list(zip(*[ds[rc] for rc in ref_cols]))
    rouge_scorer = evaluate.load('rouge')

    rouge_preds = [doc if doc is not None else '' for doc in rouge_preds]

    rouge_metrics = rouge_scorer.compute(
        predictions=rouge_preds,
        references=rouge_refs,
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
    torch.cuda.empty_cache()

    # initialize metrics dictionary
    bertscore_metrics = {}

    # evaluation loop
    scores = []
    for rc in ref_cols:
        try:
            scores.append(
                bertscore.compute(
                    predictions=bertscore_preds, 
                    references=bertscore_refs[rc], 
                    lang="en",
                    rescale_with_baseline=rescale_with_baseline,
                    device='cpu',
                )
            )
            torch.cuda.empty_cache()

        except Exception as e:
            display.error('failed to compute bertscore at ')
            print(e)
    #metrics_stacked = np.vstack([metric1['f1'], metric2['f1'], metric3['f1'], metric4['f1']])
    metrics_stacked = np.vstack([score['f1'] for score in scores])
    metrics_maxed = np.max(metrics_stacked, axis=0)
    metrics_ag = np.mean(metrics_maxed)
    
    bertscore_metrics = metrics_ag
        
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

    # aggregate scores
    p_use_ag = np.mean(p_use)
    r_use_ag = np.mean(r_use)
    p_distil_ag = np.mean(p_distil)
    r_distil_ag = np.mean(r_distil)
    p_rob_ag = np.mean(p_rob)
    r_rob_ag = np.mean(r_rob)
    
    semf1_metrics = {
        'p_use': p_use_ag,
        'r_use': r_use_ag,
        'f_use': 2*(p_use_ag*r_use_ag)/(p_use_ag+r_use_ag),
        'p_distil': p_distil_ag,
        'r_distil': r_distil_ag,
        'f_distil': 2*(p_distil_ag*r_distil_ag)/(p_distil_ag+r_distil_ag),
        'p_rob': p_rob_ag,
        'r_rob': r_rob_ag,
        'f_rob': 2*(p_rob_ag*r_rob_ag)/(p_rob_ag+r_rob_ag),
    }
    torch.cuda.empty_cache()

    return semf1_metrics
