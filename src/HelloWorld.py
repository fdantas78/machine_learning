'''
Created on Nov 22, 2017

@author: fernando
'''

import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width','class']
dataset = pandas.read_csv(url, names=names)
print(dataset.shape)
print(dataset.describe())
print(dataset.groupby('class').size()) 

#testing installation of theano tensorflow and keras

import theano
print('theano: %s' % theano.__version__)

import tensorflow
print('tensorflow: %s' % tensorflow.__version__)

import keras
print('keras: %s' % keras.__version__)