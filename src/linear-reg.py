'''
Created on Dec 2, 2017

@author: fernando

Linear Regression Example

y = b0 + b1 * x
b1 = sum(x(i) - mean(x)) * (y(i) - mean(y)) / sum(x(i) - mean(y)ˆ2)
'''
from math import sqrt
from astropy.wcs.docstrings import row

#calculate root mean square error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

#simple linear regression algorithm
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions

#evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = simple_linear_regression(dataset, test_set)
    print(predicted)
    actual = [row[-1] for row in dataset]
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
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    mean_x, mean_y = mean(x), mean(y)
    b1 = covariance(x, mean_x, y, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean_x
    print([b0,b1])
    print(b0 + b1 * 6)
    return [b0, b1]

data = [[1,2],[2,4],[3,2],[4,4],[5,2], [6,4], [7,2], [8,4], [9,2], [10,4],
        [11,2],[12,4],[13,2],[14,4],[15,2], [16,4], [17,2], [18,4], [19,2]]
#data = [[1,1],[2,2],[3,3],[4,4],[5,5], [6,6], [7,7]]
rmse = evaluate_algorithm(data)
print('RMSE: %.3f' % (rmse))
