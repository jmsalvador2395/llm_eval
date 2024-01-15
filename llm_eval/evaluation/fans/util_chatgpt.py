import openai
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import spacy
import math

import numpy as np
from transformers import logging
logging.set_verbosity_error()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_md")

MAX_TOKENS = 4096

def reduce_message(message):
    message=message.strip()
    tokens = message.split()[:MAX_TOKENS]
    return ' '.join(tokens)

def cosine_similarity(embed1, embed2):
    a = torch.tensor(embed1, dtype=torch.float32, device=device)
    b = torch.tensor(embed2, dtype=torch.float32, device=device)
    similarity = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return similarity.item()

def get_entity_embedding(word):
    return nlp(word).vector

def read_excel_file(file='all_news.xlsx'):
    df = pd.read_excel('../Data/'+file)
    return df['theme-description'], df['center-context'], df['left-context'], df['right-context'] 


def create_response(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    api_key ="",
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

def main(args):
    theme, center, left, right = read_excel_file()
    # level1
    # prompt = f"""I want you to act as a native English linguistic expert.
    #         For a given narrative inside curly brackets {{like this}}. give
    #         the response like who: when: where: what: why: how: like this. 
    #         Give the output in json format.
    #         """
    
    #level2
#     prompt = f"""
#         I want you to act as a native English linguistic expert. Your task is to
#         help structure a long narrative based on 5W1H, given inside curly
#         brackets {{like this}}. You can ask the questions delimited by triple backticks
#         to analyze the narrative.
            
# ​​        ```Who is involved in the narrative? When did this happen? Where did this happen?
#          What happened and why? How did this happen?```

#         Give the output in json format like this {{who: when: where: what: why: how:}}
        
#         Narrative is 
#         """
    
    #level3
    prompt = f"""
        I want you to act as a native English linguistic expert. Your task
        is to help structure a long narrative based on 5W1H, given inside
        curly brackets {{like this}}. Give the output in json format like
        this {{who: when: where: what: why: how:}}

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

        Narrative is 
        """
    
    if args.type=="left":
        sample = left
    if args.type =="right":
        sample = right
    if args.type=="center":
        sample = center
    if args.type=="theme":
        sample = theme
    file1 = open("../Output/out-all-level3-chatgpt/output_"+args.type+".txt", "a")
    if isinstance(sample[args.no], float):
        print(f"For Sample {args.no} NAN Value found")
    else:
        prompt = prompt+'{'+sample[args.no]+'}'
        try:
            response = create_response(prompt)
        except openai.error.InvalidRequestError as e:
            print("4096 Max Token Limit Exception Occured!")
            prompt = reduce_message(prompt)  #chatgpt max limit 4096 tokens
            response = create_response(prompt)
        print(f'Prompt: {prompt}')
        file1.write(f'Sample: {args.no} \n')
        file1.write(f'Type: {args.type} \n')
        file1.write(response+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='left or right')
    parser.add_argument('--no', type=int, help='sample no')
    args = parser.parse_args()
    main(args)
   
