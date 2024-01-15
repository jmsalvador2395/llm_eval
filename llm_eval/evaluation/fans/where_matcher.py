import spacy
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import requests
from .util_chatgpt import get_entity_embedding, cosine_similarity
sim_threshold = 0.75
concept_top = 5

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def get_associated_words(concept, top):  
    url = f"http://api.conceptnet.io/c/en/{concept}?rel=/r/en/RelatedTo&limit=1000"
    response = requests.get(url).json()
    words = [edge["end"]["label"] for edge in response["edges"] if edge["start"]["label"] == concept and "en"==edge["end"].get("language", "")]
    word_counts = {word: words.count(word) for word in words}
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_words[:top]]

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
                else:
                    concept_list1 = get_associated_words(str1, top=concept_top)
                    concept_list2 = get_associated_words(str2, top=concept_top)
                    match_count = sum(1 for item in concept_list1 if item in concept_list2)
                    if match_count>=1:
                        overlap_count += 1
                        matched_list.append(str1)
    return overlap_count, matched_list

def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def replace_abbreviated_form(text):
    state_dict = {'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'}

    for abbr, name in state_dict.items():
        text = text.replace(abbr, name)

    abbreviations = {
    "US": "United States",
    "USA": "United States",
    "st": "saint",
    "U.S.A.": "United States",
    "U.S.": "United States"
    }
    for abbreviation, full_form in abbreviations.items():
        text = text.replace(abbreviation, full_form)
    return text


def location_overlap_finder(s1, s2):
    """
    :type s1: str
    :type s2: str
    :rtype loc1: locations in s1
    :rtype loc2: locations in s2
    :rtype common_locs: overlap locations
    """
    # s1 = remove_stop_words(s1)
    # s2 = remove_stop_words(s2)

    s1 = replace_abbreviated_form(s1)
    s2 = replace_abbreviated_form(s2)
  
    # Process strings with spaCy
    doc1 = nlp(s1)
    doc2 = nlp(s2)


    # Get the set of all locations in both strings
    locs1 = set([ent.text.lower() for ent in doc1.ents if ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ =="FAC" or ent.label_ =="ORG"])
    locs2 = set([ent.text.lower() for ent in doc2.ents if ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ =="FAC" or ent.label_ =="ORG"])

    _, common_locs = fuzzy_overlap_count(locs1, locs2)
    return list(locs1), list(locs2), list(common_locs)

def where_matcher_f1(s1, s2):
    """
    :type s1: str
    :type s2: str
    :rtype: float
    """
    loc1, loc2, overlap = location_overlap_finder(s1, s2)
    f1_score = lambda p, r: 2 * ((p * r) / (p + r)) if p + r > 0 else 0
    pr = len(overlap)/(1.0*len(loc1)) if len(loc1)>0 else 0
    re = len(overlap)/(1.0*len(loc2)) if len(loc2)>0 else 0
    return pr, re, f1_score(pr, re)
    

if __name__=="__main__":
    # Sample strings
    s1 = "The Eiffel Tower is in Paris, France, Illinois, USA, New York, CA"
    s2 = "Brown University campus, Paris is a beautiful city that has the Eiffel Tower, united states, NY, California"
    # print(location_overlap_finder(s1, s2))
    print(where_matcher_f1(s1, s2))

