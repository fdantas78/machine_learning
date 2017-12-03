'''
Created on Nov 29, 2017

@author: fernando
'''
import pandas


print("Project to analyse Iris data!")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width', 'class']
print("\nDownloading data from \n%s" % url)
dataset = pandas.read_csv(url, names = names)
pandas.set_option('display.width',100)
pandas.set_option('precision',3)

print("\nTaking a look of the first 30 rows in the dataset:")
print(dataset.head(30))

print("\nDimensions of Dataset")
print(dataset.shape)

print("\nData types for each column")
print(dataset.dtypes)

print("\nDescriptive Statistics")
print(dataset.describe())

print("\nCorrelation Between Attributes")
print(dataset.corr(method='pearson'))

print("\nSkew for data")
print(dataset.skew())
