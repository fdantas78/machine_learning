'''
Created on Dec 7, 2017

@author: fernando

clustering with iris data

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)
centers = [ [1,1], [-1,-1], [1,-1] ]
iris = datasets.load_iris()
X = iris.data 
Y = iris.target 

estimators = {'k_means_iris_3':KMeans(n_clusters=3),
            'k_means_iris_8':KMeans(n_clusters=8),
            'k_means_iris_bad_init':KMeans(n_clusters=3, n_init=1, init='random')}

fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4,3)) 
    plt.clf() 
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) 
    plt.cla() 
    est.fit(X) 
    labels = est.labels_

    ax.scatter(X[:,3], X[:,0], X[:,2], c=labels.astype(np.float)) 
    ax.w_xaxis.set_ticklabels([]) 
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width') 
    ax.set_ylabel('Sepal length') 
    ax.set_zlabel('Petal length') 
    fignum += 1


fig = plt.figure(fignum, figsize=(4,2))
plt.clf() 
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) 
plt.cla() 


for name, label in [
    ('Setosa', 0),
    ('Versicolour', 1),
    ('Virginica', 2)]:
    ax.text3D(X[Y == label,3].mean(),
        X[Y == label, 0].mean() + 1.5,
        X[Y == label, 2].mean(),name,
        horizontalalignment='center',
        bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

y = np.choose(Y,[1,2,0]).astype(np.float) 
ax.scatter(X[:,3], X[:,0], X[:,2], c=Y)

print(ax.scatter(X[:,3], X[:,0], X[:,2], c=Y))


ax.w_xaxis.set_ticklabels([]) 
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width') 
ax.set_ylabel('Sepal length') 
ax.set_zlabel('Petal length') 
plt.show()