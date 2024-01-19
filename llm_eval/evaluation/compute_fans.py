import os

# external dependencies
from typing import List, Dict
import evaluate
import os
import re
import json
import numpy as np
#import tensorflow as tf
from datasets import Dataset
from itertools import product, chain
from pprint import pprint
from tqdm import tqdm

# local dependencies
from llm_eval.llm import *
from llm_eval.utils import (
    files,
    strings,
    display,
)
from .fans import compute_all_f1

def compute_fans(ds: Dataset,
                 ref_cols: List[str],
                 cfg: Dict):
    session = select_chat_model(
        cfg,
        'gpt-3.5-turbo'
        #'mistralai/Mistral-7B-Instruct-v0.2'
    )

    template = """
        I want you to act as a native English linguistic expert. Your task
        is to help structure a long narrative based on 5W1H, given inside
        curly brackets {like this}. Give the output in json format like
        this {who: "who", when: "when", where: "where", what: "what", why: "why", how: "how"}

        Perform the following actions to analyze the narrative.

        Step 1 - Find out “Who” in the narrative. You can ask the questions
        delimited by triple backticks to analyze ```Who is speaking in the
        narrative? Who is involved in the situation? Who is the subject of concern? ```

        Step 2 - Find out “When” in the narrative. You can ask the questions
        delimited by triple backticks to analyze ``` When did the events take
        place? When did these concerns arise?  When did this discussion happen?```

        Step 3 - Find out “Where” in the narrative. You can ask the questions
        delimited by triple backticks to analyze ``` Where did this happen? Where
        did the events take place? Where did the discussion take place? ```

        Step 4 - Find out “What” in the narrative. You can ask the questions
        delimited by triple backticks to analyze ```What happened? What is the
        reason of the concern? What is the significance of this action? What is
        the focus of the discussion? ``` Use at most 20 words for "what" response.

        Step 5 - Find out “Why” in the narrative. You can ask the questions
        delimited by triple backticks to analyze ``` Why did this take palce?
        Why did the actors of the event raise these concerns? Why did these events
        happen? Why is the discussion taking place?```.Use at most 20 words for "Why" response.

        Step 6 - Find out “How” in the narrative. You can ask the questions
        delimited by triple backticks to analyze ```How did the event happen?
        How did these events transpire? How is the actors of the involved in
        the  situation?```. Use at most 20 words for "How" response.

        Narrative is:
        """
    template = template.replace('\n' + 8*' ', '\n')

    # ppr = prompts per row
    ppr = 1 + len(ref_cols) 

    prompts = []
    cols = ['response'] + ref_cols
    for col in cols:
        for nar in ds[col]:
            prompts.append(template + nar)

    nar_extracts = session.get_response(prompts)

    facets_data = []
    for idx, nar in enumerate(nar_extracts):
        re_match = re.search(r'\{.*\}', nar, flags=re.S)
        if re_match:
            try:
                facets = json.loads(re_match.group(0))
                facets.update({'id': idx})
                facets_data.append(facets)
            except:
                pass


    # cleanup the json and convert to a huggingface dataset
    ds_keys = {"id", "who", "what", "when", "where", "why", "how"}
    for sample in facets_data:
        # set sample keys to lower case
        sample_keys = set(sample.keys())
        for sk in sample_keys:
            sample[sk.lower()] = sample.pop(sk)


        # re-read sample-keys and remove any extraneous keys
        sample_keys = set(sample.keys())
        extra_keys = sample_keys - ds_keys
        for ek in extra_keys:
            print(f'popped {ek} from sample {sample["id"]}')
            sample.pop(ek)

        for dsk in ds_keys - {'id'}:
            sample[dsk] = str(sample.get(dsk, ''))

    fds = Dataset.from_list(facets_data)
    # make evaluation pairs
            
    ids = list(filter(lambda x: x < len(ds), fds['id']))
    step = len(ds)

    pair_ds = []
    for idx, sid in enumerate(ids):
        # get the facets of the prediction
        pf = fds[idx]

        # pair predection facets with reference facets and add to pair_ds
        ref_ids = list(range(sid + step, len(ds)*ppr, step))
        ref_facets = fds.filter(lambda x: x['id'] in ref_ids, keep_in_memory=True)
        ref_name_map = {rid: col for rid, col in zip(ref_ids, ref_cols)}
        for rf in ref_facets:
            ref_name = ref_name_map[rf['id']]

            # pf for pred facets and rf for reference facets
            pair_sample = {
                'sample_id': sid,
                'ref_name': ref_name,
                'who_pred': pf['who'],
                'what_pred': pf['what'],
                'when_pred': pf['when'],
                'where_pred': pf['where'],
                'why_pred': pf['why'],
                'how_pred': pf['how'],
                'who_ref': rf['who'],
                'what_ref': rf['what'],
                'when_ref': rf['when'],
                'where_ref': rf['where'],
                'why_ref': rf['why'],
                'how_ref': rf['how'],
            }
            
            pair_ds.append(pair_sample)

    pair_ds = Dataset.from_list(pair_ds)
    
    # aggregate facets into single score with alpha=0.2
    f1_scores = compute_all_f1(pair_ds)

    alpha = 0.2
    who =   alpha*np.array(f1_scores['who_f1'])[:, -1]
    when =  alpha*np.array(f1_scores['when_f1'])[:, -1]
    where = alpha*np.array(f1_scores['where_f1'])[:, -1]
    what =  (1-alpha)*np.array(f1_scores['what_f1'])[:, -1]
    why =   (1-alpha)*np.array(f1_scores['why_f1'])[:, -1]
    how =   (1-alpha)*np.array(f1_scores['how_f1'])[:, -1]

    aggregate_f1 = np.mean(
        np.stack((who, what, when, where, why, how)),
        axis=0
    )
    f1_scores = f1_scores.add_column('aggregate_f1', aggregate_f1)

    # take the max f1 score per sample id
    max_f1_per_sample_id = []
    ids = set(f1_scores['sample_id'])
    for sample_id in ids:
        scores_per_sample_id = f1_scores.filter(
            lambda x: sample_id == x['sample_id'],
            keep_in_memory=True
        )
        score = np.max(scores_per_sample_id['aggregate_f1'])
        max_f1_per_sample_id.append(score)

    final_score = np.mean(max_f1_per_sample_id)
    
    return {
        'fans_f1': final_score,
        'num_evaluated': len(max_f1_per_sample_id)
    }

