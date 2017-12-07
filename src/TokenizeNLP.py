'''
Created on Dec 6, 2017

@author: fernando

Starting with NLP using NTLK
'''

import nltk
import pprint

tokenizer = None
tagger = None
nltk.download('brown')

def init_nltk():
    global tokenizer
    global tagger
    #print("init")
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|\S\w') #remove spaces
    tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())
    
def tag(text):
    global tokenizer
    global tagger
    
    #print("tag")
    if not tokenizer:
        init_nltk()
    tokenized = tokenizer.tokenize(text)
    tagged = tagger.tag(tokenized)
    #tagged.sort()
    return tagged

def main():
    text = """
            This Swedish electric car comes with 5 years of free electricity Uniti is on a mission 
            to create an intelligent, small electric car – and they just partnered with energy company 
            E.ON to provide customers with five years of free solar energy. Uniti is just a couple days 
            away from the worldwide debut of the vehicle.
            """
    
    tagged = tag(text)
    l = list(set(tagged))
    l.sort(key=lambda l: (l[1] is None, (l[1],l[0])))
    pprint.pprint(l)
    
if __name__ == '__main__':
    main()

