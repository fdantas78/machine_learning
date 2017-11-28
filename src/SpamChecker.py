'''
Created on Nov 27, 2017

@author: fernando
'''
import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
#from dask.tests.test_dot import test_label

def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:
                    words = line.split()
                    all_words += words
    
    dictionary =  Counter(all_words)
    
    #remove non-words from dataset
    list_to_remove = dictionary.keys()
    keys_to_remove = []
    
    for k in list_to_remove:
        if k.isalpha() == False:
            keys_to_remove.append(k)
        elif len(k) == 1:
            keys_to_remove.append(k)
    
    for item in keys_to_remove:
        del dictionary[item]
    
    dictionary = dictionary.most_common(3000)
    
    #return just the 3000 words most used in our data
    return dictionary

#extract features of training data        
def extract_features(mail_dir):
    files = [os.path.join(mail_dir, f) for f in os.listdir(mail_dir)]  
    #create a matrix with zeros
    features_matrix = np.zeros((len(files), 3000))  
    doc_id = 0
    
    for file_to_read in files:
        with open(file_to_read) as f:
            for i, line in enumerate(f):
                if i == 2:
                    words = line.split()
                    for word in words:
                        word_id = 0
                        for i, id in enumerate(dictionary):   
                            if id[0] == word:
                                word_id = i
                                features_matrix[doc_id, word_id] = words.count(word)  
            doc_id += 1
    
    return features_matrix  
        
print("New Spam checker")
print("Using 2 types of data:")
print("Training data and test data")

train_dir = '../data/train-mail'
print("Process trainning!")
dictionary = make_Dictionary(train_dir)

print("\nExtracting Features!")
train_labels = np.zeros(702)
train_labels[351:701] = 1

#get features from data
train_matrix = extract_features(train_dir)

#train SVM and Naive bays classifier
print("\n Train out system with SVM and Naive Bayes!")
multinomial_model = MultinomialNB()
svc_model = LinearSVC()
multinomial_model.fit(train_matrix, train_labels)
svc_model.fit(train_matrix, train_labels)

#test unseen mail for spam
test_dir = '../data/test-mail'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = multinomial_model.predict(test_matrix)
result2 = svc_model.predict(test_matrix)

print("Our results are as follows:")
print("0 represents no spam and 1 represents spam")
print(result1)
print(result2)


