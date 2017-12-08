'''
Created on Dec 7, 2017

@author: fernando
dendogram with iris data

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
'''

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

from sklearn import datasets



#random generates two clusters
np.set_printoptions(precision=5, suppress=True) 
np.random.seed(4711)
a = np.random.multivariate_normal([10,0], [[3,1], [1,4]], size=[100]) 
b = np.random.multivariate_normal([0,20], [[3,1], [1,4]], size=[50,]) 
X = np.concatenate((a,b),)

#using iris data
iris = datasets.load_iris()
X = iris.data 
print(X.shape)
print(X) 

print(X[1],X[45])

#matrix
plt.scatter(X[:,0], X[:,1])
plt.show()
#create dendrogram
Z = linkage(X, 'ward') 
plt.figure(figsize=(10,5))
plt.title('Hierachical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=8.,
)
plt.show()