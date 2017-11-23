'''
Created on Nov 22, 2017

@author: fernando
'''

import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width','class']
dataset = pandas.read_csv(url, names=names)
