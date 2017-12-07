'''
Created on Dec 6, 2017

@author: fernando


Unique words processing
'''

from collections import Counter
import pandas
import re

def make_matrix(headlines, vocab):
    matrix = []
    for h in headlines:
        h = h.split(" ")
        counter = Counter(h)
        row = [counter.get(w,0) for w in vocab]
        matrix.append(row)
    df = pandas.DataFrame(matrix)
    df.columns = unique_words
    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.max_rows', None)
    return df

headlines = ["New dinosaur is part penguin, part duck, part swan: ‘kind of a bizarre one’",
            "Kenow wildfire report points to lack of communication during event, Alberta minister disagrees",
            "Donald Trump Jr. refuses to discuss meeting with dad during congressional panel appearance",
            "Edmonton property tax hike expected to come in at 3.3% or under: Iveson",
            "Alberta government denies increase in support funding despite more addicts dying"
    ]

filename = '../data/stop_words.txt'
with open(filename,'r') as f:
    stopwords = f.read().split("\n")
    
stopwords = [re.sub(r'[^\w\s\d]','',s.lower()) for s in stopwords]
new_headlines = [re.sub(r'[^\w\s\d]','',h.lower()) for h in headlines]

unique_words = list(set(" ".join(new_headlines).split(" ")))

unique_words = [w for w in unique_words if w not in stopwords]

print(make_matrix(headlines, unique_words))
