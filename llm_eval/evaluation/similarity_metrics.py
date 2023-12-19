# TODO: Not working on gpu rn.
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List
import gc


def compute_cosine_similarity(pred_embeds, ref_embeds):
    cosine_scores = cosine_similarity(pred_embeds, ref_embeds)
    return np.max(cosine_scores, axis=-1).tolist() #, np.argmax(cosine_scores, axis=-1).tolist()


def use_similarity(predictions: List[List[str]], references: List[List[str]]):
    '''
    Universal Sentence Encoder
    :param predictions: List of predicted summaries. Each sample should also be a list, a list of sentences
    :param references: List of reference summaries. Each sample should also be a list, a list of sentences
    :return:
    '''

    assert len(predictions) == len(references), 'Mismatch in number of predictions to the number of references'
    print("In Universal Sentence Encoder")

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    out_scores = [0]*len(predictions)
    for idx, (preds, refs) in enumerate(zip(predictions, references)):
        pred_embeddings = model(preds)
        ref_embeddings = model(refs)
        out_scores[idx] = compute_cosine_similarity(pred_embeddings, ref_embeddings)

    del model
    gc.collect()

    return out_scores


def sbert_similarity(predictions: List[List[str]], references: List[List[str]], model_name):
    '''
    Paraphrase similarity and Semantic Textual similarity
        :param predictions: List of predicted summaries. Each sample should also be a list, a list of sentences
        :param references: List of reference summaries. Each sample should also be a list, a list of sentences
        :return:
    '''

    assert model_name in ["paraphrase-distilroberta-base-v1", 'stsb-roberta-large']
    assert len(predictions) == len(references), 'Mismatch in number of predictions to the number of references'

    print(f"In sbert sim. Model: {model_name}")

    model = SentenceTransformer(model_name)
    out_scores = [0] * len(predictions)
    for idx, (preds, refs) in enumerate(zip(predictions, references)):
        pred_embeddings = model.encode(preds)
        ref_embeddings = model.encode(refs)
        out_scores[idx] = compute_cosine_similarity(pred_embeddings, ref_embeddings)

    del model
    gc.collect()

    return out_scores
