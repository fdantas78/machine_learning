'''
Created on Dec 2, 2017

@author: fernando

Linear Regression sqrt feet x house price
'''
from random import randrange
from csv import reader
from math import sqrt
from astropy.wcs.docstrings import row

def load_csv(file):
    data = list()
    with open(file, 'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data

def str_column_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip())
        
def train_test_split(data, split, random):
    train = list()
    train_size = split * len(data)
    data_copy = list(data)
    if random == 1: # random
        while len(train) < train_size:
            index = randrange(len(data_copy))
            train.append(data_copy.pop(index))
    else: #sequential
        for i in range(int(train_size)):
            train.append(data_copy[i])
        
    return train, data_copy

#calculate root mean square error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

#evaluate regression algorithm on training dataset
def evaluate_algorithm(data, algorithm, split, random, *args):
    train, test = train_test_split(data, split, random)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    #print(predicted)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual,predicted)
    return rmse

#calculare the mean value
def mean(values):
    return sum(values) / float(len(values))

#calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

#calculate the variance of a list
def variance(values, mean):
    return sum([(x-mean) ** 2 for x in values])

#calculate coefficients
def coefficients(data):
    x = [row[0] for row in data]
    y = [row[1] for row in data]
    mean_x, mean_y = mean(x), mean(y)
    b1 = covariance(x, mean_x, y, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    #print([b0,b1])
    return [b0, b1]

#simple linear regression algorithm
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions
            

filename = '../data/insurance.csv'
data = load_csv(filename)
for i in range(len(data[0])):
    str_column_to_float(data, i)
split = 10
while split < 100:
    #print(split)
    rsme = evaluate_algorithm(data, simple_linear_regression, split/100, 0)
    rsme2 = evaluate_algorithm(data, simple_linear_regression, split/100, 1)
    
    #comparison between calculated vs in flie values
    print("Train data: %s%% RMSE seq: %.3f RMSE random: %.3f" % (split,rsme, rsme2))
    
    split += 10



