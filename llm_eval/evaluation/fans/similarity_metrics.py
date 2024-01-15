import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from transformers import logging
logging.set_verbosity_error()

import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import nltk
nltk.download('punkt')

def sent_tokenize(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def compute_cosine_similarity(pred_embeds, ref_embeds):
    cosine_scores = cosine_similarity(pred_embeds, ref_embeds)
    return np.max(cosine_scores, axis=-1).tolist() #, np.argmax(cosine_scores, axis=-1).tolist()

def sbert_similarity(predictions: List[List[str]], references: List[List[str]], model_name):
    '''
    Paraphrase similarity and Semantic Textual similarity
        :param predictions: List of predicted summaries. Each sample should also be a list, a list of sentences
        :param references: List of reference summaries. Each sample should also be a list, a list of sentences
        :return:
    '''

    assert model_name in ["paraphrase-distilroberta-base-v1", 'stsb-roberta-large']
    assert len(predictions) == len(references), 'Mismatch in number of predictions to the number of references'

    # print("In sbert sim. Model: {model_name}")

    model = SentenceTransformer(model_name)
    out_scores = [0] * len(predictions)
    for idx, (preds, refs) in enumerate(zip(predictions, references)):
        pred_embeddings = model.encode(preds)
        ref_embeddings = model.encode(refs)
        out_scores[idx] = compute_cosine_similarity(pred_embeddings, ref_embeddings)

    return out_scores

def sem_f1(in1, in2):
    if in1 and in2:
        in1 = [sent_tokenize(in1)]
        in2 = [sent_tokenize(in2)]
        pr = sbert_similarity(in1, in2, model_name="stsb-roberta-large")
        re = sbert_similarity(in2, in1, model_name="stsb-roberta-large")
        pr = np.mean(pr)
        re = np.mean(re)
        f1_score = lambda p, r: 2 * ((p * r) / (p + r)) if p + r > 0 else 0
        return (pr, re, f1_score(pr, re))
    else:
        return (0.0, 0.0, 0.0)