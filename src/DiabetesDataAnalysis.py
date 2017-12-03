'''
Created on Dec 02, 2017

@author: fernando
'''
import pandas


print("Project to analyse Pima Indians Diabetes data!")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

'''
names = ['Number of times pregnant',
   'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
   'Diastolic blood pressure (mm Hg)',
   'Triceps skin fold thickness (mm)',
   '2-Hour serum insulin (mu U/ml)',
   'Body mass index (weight in kg/(height in m)^2)',
   'Diabetes pedigree function',
   'Age (years)',
   'Class variable (0 or 1)']
'''
names = ['pregnant',
   'Plasma glucose',
   'blood pressure',
   'Triceps skin',
   'insulin',
   'BMI',
   'Diabetes pedigree',
   'Age',
   'Class']

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

#Data visualization
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

dataset.plot(kind='box',subplots=True, layout=(3,3),sharex=False,sharey=False)
#plt.show()

dataset.hist()
#plt.show()

scatter_matrix(dataset)
plt.show()

