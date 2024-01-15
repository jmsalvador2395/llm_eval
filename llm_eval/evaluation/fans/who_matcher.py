from fuzzywuzzy import fuzz
from .util_chatgpt import get_entity_embedding, cosine_similarity

sim_threshold = 0.75

def preprocess(in1):
    in1 = in1.replace(" and", ", ") #chatgpt sometimes separate by and
    in1 =  [string.strip() for string in in1.split(',')]
    in1 = list(filter(lambda s: s != '', list(set(in1))))
    return in1

def fuzzy_overlap_count(list1, list2):
    matched_list = []
    overlap_count = 0
    for str1 in list1:
        for str2 in list2:
            if fuzz.partial_ratio(str1, str2) >= 80: # adjust this value to your preference
                overlap_count += 1
                matched_list.append(str1)
            else:
                embed1 = get_entity_embedding(str1)
                embed2 = get_entity_embedding(str2)
                cos_sim = cosine_similarity(embed1, embed2)
                if cos_sim>=sim_threshold:
                    overlap_count += 1
                    matched_list.append(str1)
    return overlap_count, matched_list

def who_matcher_string_fuzzy_F1(in1, in2):
    """
    :type in1: str
    :type in2: str
    :rtype: float
    """
    f1_score = lambda p, r: 2 * ((p * r) / (p + r)) if p + r > 0 else 0
    in1 = preprocess(in1)
    in2 = preprocess(in2)

    overlap_count, matched_list = fuzzy_overlap_count(in1, in2)
    pr = overlap_count/(1.0*(len(in1))) if len(in1)>0 else 0
    re = overlap_count/(1.0*(len(in2))) if len(in2)>0 else 0
    # print(in1)
    # print(in2)
    # print(matched_list)
    return pr, re, f1_score(pr, re)

if __name__=="__main__":
    input1 = "Special counsel Robert Mueller's team, Trump campaign officials, Jared Kushner"
    input2 = "Special Counsel Robert Mueller, Trump campaign officials, former campaign manager Paul Manafort, former campaign official George Papadopoulos, President Trump, Attorney General Jeff Sessions, federal officials."
    print(who_matcher_string_fuzzy_F1(input1, input2))





