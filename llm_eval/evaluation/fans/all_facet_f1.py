##################################################
# About: Calculates F1 score for all facets using level 1, level 2, and level 3 prompts.
##################################################
import jsonlines
import pandas as pd
import argparse
from .who_matcher import who_matcher_string_fuzzy_F1
from .when_matcher import when_matcher_F1
from .where_matcher import where_matcher_f1
from .similarity_metrics import sem_f1

def debug(level, score):
    rtDir='../Output/'
    file = rtDir+'pair_final/'+level+'-chatgpt.jsonl'
    df=read_jsonl_to_df(file)
    df = fil_df(df, score)

    file_path = '../Result/'+level+'_chatgpt_Score'+str(score)+'.jsonl'

    dfT = pd.read_json(file_path, lines=True)

    print(f"Original Level {level} Score{score} Samples {len(df)}")
    print(f"Generated Level {level} Score{score} Samples {len(dfT)}")

def read_jsonl_to_df(file):
    data = []
    with jsonlines.open(file) as reader:
        for line in reader:
            data.append(line)
    df = pd.DataFrame(data)
    return df.sort_values(by='Similarity', ascending=True)

def fil_df(df, score):
    return df[df['Similarity'] == score]

def read_facets(df, Nar, row):
    return df.iloc[row][Nar]['Sample'], df.iloc[row][Nar]['Type'], df.iloc[row][Nar]['Facets']

def all_f1(level, type, score):
    # Define the file path for the JSONL file
    rtDir='../Output/'
    file = rtDir+'pair_final/'+level+'-'+type+'.jsonl'
    df=read_jsonl_to_df(file)
    df = fil_df(df, score)
    if score==1 or score==0:
        n=500
    else:
        n=len(df)

    # Write the data to the JSONL file
    file_path = '../Result/'+type+'/'+level+'_'+type+'_Score'+str(score)+'.jsonl'
    with jsonlines.open(file_path, mode='a') as writer:
        for i in range(n):
            sm1, tp1, fac1 = read_facets(df,  Nar='Nar1', row = i)
            sm2, tp2, fac2 = read_facets(df, Nar='Nar2', row = i)
            who_f1 = who_matcher_string_fuzzy_F1(fac1['who'], fac2['who'])
            when_f1 = when_matcher_F1(fac1['when'], fac2['when'])
            where_f1 = where_matcher_f1(fac1['where'], fac2['where'])
            what_f1 = sem_f1(fac1['what'], fac2['what'])
            how_f1 = sem_f1(fac1['how'], fac2['how'])
            why_f1 = sem_f1(fac1['why'], fac2['why'])

            data = {
            'Row': i,
            'Nar1': sm1,
            'Nar2': sm2,
            'Nar1_Type': tp1,
            'Nar2_Type': tp2,
            'fac1': fac1,
            'fac2': fac2,
            'who_f1': who_f1,
            'when_f1': when_f1,
            'where_f1': where_f1,
            'what_f1': what_f1,
            'how_f1': how_f1,
            'why_f1': why_f1}
            writer.write(data)
            print(f"Sample: {i}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate F1 score based on level and score.')
    parser.add_argument('--level', type=str, help='Level argument (default: level3)')
    parser.add_argument('--score', type=int, help='Score argument (default: 2)')
    parser.add_argument('--type', type=str, help='Chatgpt or bard?')

    args = parser.parse_args()

    all_f1(args.level, args.type, args.score)
    


