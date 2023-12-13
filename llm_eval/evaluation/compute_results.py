import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# external dependencies
from typing import List, Dict
import datasets
import torch
import evaluate
import os
import numpy as np
import tensorflow as tf
from datasets import Dataset
from nltk import sent_tokenize
from itertools import product
from pprint import pprint
from tqdm import tqdm
import time

# local dependencies
from .similarity_metrics import (
    use_similarity,
    sbert_similarity,
)

from llm_eval.utils import (
    files,
    strings,
)

def replace_nones(responses: List[str]):
    out = [el if el is not None else '' for el in responses]
    return out

def compute_rouge(ds):
    rouge_preds = ds['response']

    rouge_refs = list(zip(
        ds['Ahmed_Intersection'],
        ds['Naman_Intersection'],
        ds['Helen_Intersection'],
        ds['AllSides_Intersection'],
    ))
    rouge_scorer = evaluate.load('rouge')

    rouge_preds = [doc if doc is not None else '' for doc in rouge_preds]

    rouge_metrics = rouge_scorer.compute(
        predictions=rouge_preds,
        references=rouge_refs,
    )
    return rouge_metrics

def compute_bertscore(ds: Dataset, rescale_with_baseline: bool=True):
    # set preds and refs
    bertscore_preds = ds['response']
    bertscore_preds = replace_nones(bertscore_preds)
    bertscore_refs = {
        'Ahmed': ds['Ahmed_Intersection'],
        'Naman': ds['Naman_Intersection'],
        'Helen': ds['Helen_Intersection'],
        'AllSides': ds['AllSides_Intersection'],
    }

    # load metric
    bertscore = evaluate.load("bertscore")

    # initialize metrics dictionary
    bertscore_metrics = {}

    # evaluation loop
    metric1 = bertscore.compute(
        predictions=bertscore_preds, 
        references=bertscore_refs['Ahmed'], 
        lang="en",
        rescale_with_baseline=rescale_with_baseline,
    )
    metric2 = bertscore.compute(
        predictions=bertscore_preds, 
        references=bertscore_refs['Naman'], 
        lang="en"
    )
    metric3 = bertscore.compute(
        predictions=bertscore_preds, 
        references=bertscore_refs['Helen'], 
        lang="en"
    )
    metric4 = bertscore.compute(
        predictions=bertscore_preds, 
        references=bertscore_refs['AllSides'], 
        lang="en"
    )
    metrics_stacked = np.vstack([metric1['f1'], metric2['f1'], metric3['f1'], metric4['f1']])
    metrics_maxed = np.max(metrics_stacked, axis=0)
    metrics_ag = np.mean(metrics_maxed)
    
    bertscore_metrics = metrics_ag
        
    return bertscore_metrics


def compute_semf1(ds):

    semf1_preds = ds['response']

    semf1_refs = {
        'Ahmed_Intersection': ds['Ahmed_Intersection'],
        'Naman_Intersection': ds['Naman_Intersection'],
        'Helen_Intersection': ds['Helen_Intersection'],
        'AllSides_Intersection': ds['AllSides_Intersection'],
    }
    semf1_refs_ag = list(zip(
        ds['Ahmed_Intersection'],
        ds['Naman_Intersection'],
        ds['Helen_Intersection'],
        ds['AllSides_Intersection'],
    ))

    semf1_refs_ag = ['\n'.join(grp) for grp in semf1_refs_ag]
    semf1_recall_preds = []
    for grp in zip(ds['Ahmed_Intersection'],
                   ds['Naman_Intersection'],
                   ds['Helen_Intersection'],
                   ds['AllSides_Intersection']):
        semf1_recall_preds.extend(list(grp))

    semf1_recall_refs  = []
    for ref in ds['response']:
        semf1_recall_refs.extend([ref]*4)

    semf1_preds = [sent_tokenize(doc) if doc is not None else [''] for doc in semf1_preds]
    for key in semf1_refs.keys():
        semf1_refs[key] = [sent_tokenize(doc) if doc is not None else [''] for doc in semf1_refs[key]]
    semf1_recall_refs = [sent_tokenize(doc) if doc != '' else [''] for doc in semf1_recall_refs]
        
    semf1_refs_ag = [sent_tokenize(doc) for doc in semf1_refs_ag]
    semf1_recall_preds = [sent_tokenize(doc) for doc in semf1_recall_preds]

    semf1_metrics = {}

    # compute metrics
    p_use = use_similarity(semf1_preds, semf1_refs_ag)
    r_use = use_similarity(semf1_recall_preds, semf1_recall_refs)
    p_distil = sbert_similarity(semf1_preds, semf1_refs_ag, 'paraphrase-distilroberta-base-v1')
    r_distil = sbert_similarity(semf1_recall_preds, semf1_recall_refs, 'paraphrase-distilroberta-base-v1')
    p_rob = sbert_similarity(semf1_preds, semf1_refs_ag, 'stsb-roberta-large')
    r_rob = sbert_similarity(semf1_recall_preds, semf1_recall_refs, 'stsb-roberta-large')

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

    return semf1_metrics

def collect_scores(args, cfg, keywords):

    breakpoint()
    if args.procedure == 'unit_test':
        base_path = f'{files.project_root()}/data/unit_test/eval/'
    elif args.procedure == 'exec_all':
        base_path = f'{cfg.response_collection["save_dir"]}/{keywords["timestamp"]}'
    elif args.procedure == 'evaluate':
        base_path = f'{cfg.response_collection["save_dir"]}/{args.timestamp}'

    targets = []
    for path, subdirs, files in os.walk(base_path):
        for name in files:
            targets.append(os.path.join(path, name))

    out_table = '| Model | Sem-F1 (USE) | Sem-F1 (Distil) | Sem-F1 (RoBERTa) | Rouge-1 | ' \
                + 'Rouge-2 | Rouge-L | Rouge-L Sum | BERTscore |\n' \
                + '| - | - | - | - | - | - | - | - | - |\n'
            
    scores = {}
    start = time.time()
    for trgt in tqdm(targets, total=len(targets)):
        sample_start = time.time()
        
        print(f'***** computing metrics for {trgt[len(base_path):]} *****')
        ds = Dataset.from_json(trgt)

        # remove empty samples
        before_prune = len(ds)
        ds = ds.filter(lambda x: x['response'] != '')
        ds = ds.filter(lambda x: x['response'] is not None)
        after_prune = len(ds)

        print(f'size before filter: {before_prune}\nsize after filter: {after_prune}')

        rouge_scores = compute_rouge(ds)
        bert_score = compute_bertscore(ds)
        semf1 = compute_semf1(ds)

        scores[trgt[:len(base_path)]] = {
            'rouge': rouge_scores,
            'bert_score': bert_score,
            'semf1': semf1,
        }

        out_table += f'| {trgt[len(base_path):]} | {semf1["f_use"]:.02f} | {semf1["f_distil"]:.02f} ' \
                   + f'| {semf1["f_rob"]:.02f} | {rouge_scores["rouge1"]:.02f} | {rouge_scores["rouge2"]:.02f} ' \
                   + f'| {rouge_scores["rougeL"]:.02f} | {rouge_scores["rougeLsum"]:.02f} | {bert_score:.02f} |\n'

        with open('./results_table.md', 'w') as f:
            f.write(out_table)

        end = time.time()
        print('=================')
        print(f'Time to Finish Sample: {end-sample_start:.02f}'
            + f'Total Elapsed Time: {end-start:.02f}')
        print('=================')

if __name__ == '__main__':
    main()
