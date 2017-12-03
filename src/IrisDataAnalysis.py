'''
Created on Nov 27, 2017

@author: fernando
'''
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model.tests.test_passive_aggressive import random_state

print("Project to understand several packges usisng Iris data!")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-lenght', 'sepal-width', 'petal-lenght', 'petal-width','class']
print("\nDownloading data from \n%s" % url)
dataset = pandas.read_csv(url, names=names)

print("\nNumber of rows and columns in the dataset:")
print(dataset.shape)

print("\nTaking a look at the dataset: \n%s" % dataset.head(10))

print("\n Mean. medium and max values \n%s" % dataset.describe())

print ("Number of classes in the dataset: \n%s" % dataset.groupby('class').size())

#getting data from the dataset

array = dataset.values

#all values minus classes
X = array[:,0:4]
#all classes
Y = array[:,4]

#size of the dataset that will be used for learning validation
validation_size = 0.30

#number of the iterations in trained data
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size=validation_size, random_state=seed)

#Start the training with 70% of the dataset
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print("\nNumber of records or precision on the bases of classes")
print(classification_report(Y_validation,predictions))
      
