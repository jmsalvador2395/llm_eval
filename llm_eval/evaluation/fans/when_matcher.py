from dateutil.parser import parse
import re
import sys

def preprocess_for_when(text):
    # Date pattern
    date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{4}')

    # Day pattern
    day_pattern = re.compile(r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday')

    # Month pattern
    month_pattern = re.compile(r'January|February|March|April|May|June|July|August|September|October|November|December')

    # Year pattern
    year_pattern = re.compile(r'\d{4}')

    # Time pattern
    time_pattern = re.compile(r'\d{1,2}:\d{2}([ap]m)?')

    daytime_pattern = re.compile(r'Morning|Afternoon|Evening|Night')

    # Find all matches in the text
    dates = re.findall(date_pattern, text)
    days = re.findall(day_pattern, text)
    months = re.findall(month_pattern, text)
    years = re.findall(year_pattern, text)
    times = re.findall(time_pattern, text)
    daytime = re.findall(daytime_pattern, text)
    return dates+days+months+years+times+daytime

# Function to check if two datetime objects are the same
def match_datetime(dt1, dt2):
    return dt1.date() == dt2.date() and dt1.time() == dt2.time()

def when_matcher_overlap_count(list1, list2):
    count = 0
    for str1 in list1:
        for str2 in list2:
            try:
                dt1 = parse(str1)
                dt2 = parse(str2)
                if match_datetime(dt1, dt2):
                    count += 1
            except:
                pass
    return count

def when_matcher_F1(text1, text2):
    """
    :type text1: str
    :type text2: str
    """
    text1 = [preprocess_for_when(t1.capitalize()) for t1 in text1.split(' ')]
    text2 = [preprocess_for_when(t2.capitalize()) for t2 in text2.split(' ')]

    #remove duplicates and empty string
    text1 = [elem[0] for elem in text1 if elem]
    text2 = [elem[0] for elem in text2 if elem]

    text1 = list(set(text1))
    text2 = list(set(text2))
    

    f1_score = lambda p, r: 2 * ((p * r) / (p + r)) if p + r > 0 else 0
    overlap_count = when_matcher_overlap_count(text1, text2)
    pr = overlap_count/(1.0*len(text1)) if len(text1)>0 else 0
    re = overlap_count/(1.0*len(text2)) if len(text2)>0 else 0
    # print(text1)
    # print(text2)
    # print(overlap_count)
    return pr, re, f1_score(pr, re)

if __name__=="__main__":
    # text1 = "monday january, evening happened i don't know "
    # text2 = "Thursday afternoon april monday"

    text1 = "Approximately 2006 through at least 2016."
    text2 = "2006 to 2015"
    print(when_matcher_F1(text1, text2))