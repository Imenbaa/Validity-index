# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:07:59 2018

@author: Imen
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

#Create a database of random values of 4 features and a fixed number of clusters 
n_clusters=6
dataset,y=make_blobs(n_samples=200,n_features=4,centers=n_clusters)
#plt.scatter(dataset[:,2],dataset[:,3])

#Firstly,i will calculate Vsc for this number of clusters
#Create the k-means 
kmeans=KMeans(init="k-means++",n_clusters=n_clusters,random_state=0)
kmeans.fit(dataset)
mu_i=kmeans.cluster_centers_
k_means_labels=kmeans.labels_
mu=dataset.mean(axis=0)
SB=np.zeros((4,4))
for line in mu_i:
    diff1=line.reshape(1,4)-mu.reshape(1,4)
    diff2=np.transpose(line.reshape(1,4)-mu.reshape(1,4))
    SB+=diff1*diff2
Sw=np.zeros((4,4))
sum_in_cluster=np.zeros((4,4))
comp_c=0
for k in range(n_clusters):
    mes_points=(k_means_labels==k)
    cluster_center=mu_i[k]
    for i in dataset[mes_points]:
        diff11=i.reshape(1,4)-cluster_center.reshape(1,4)
        diff22=np.transpose(i.reshape(1,4)-cluster_center.reshape(1,4))
        sum_in_cluster+=diff11*diff22
    Sw+=sum_in_cluster
    comp_c+=np.trace(Sw)

sep_c=np.trace(SB)
Vsc=sep_c/comp_c
print("For n_clusters=",n_clusters," => Vsc=",Vsc)

#Secondly,i will determine Vsc for each number of cluster from 2 to 10
#Define a function validity_index
def validity_index(c):
    kmeans=KMeans(init="k-means++",n_clusters=c,random_state=0)
    kmeans.fit(dataset)
    #mu_i is the centers of clusters
    mu_i=kmeans.cluster_centers_
    k_means_labels=kmeans.labels_
    #mu is the center of the whole dataset
    mu=dataset.mean(axis=0)
    #initialize the between clusters matrix
    SB=np.zeros((4,4))
    for line in mu_i:
        diff1=line.reshape(1,4)-mu.reshape(1,4)
        diff2=np.transpose(line.reshape(1,4)-mu.reshape(1,4))
        SB+=diff1*diff2
    comp_c=0
    #initialize the within matrix
    Sw=np.zeros((4,4))
    sum_in_cluster=np.zeros((4,4))
    for k in range(c):
        mes_points=(k_means_labels==k)
        cluster_center=mu_i[k]
        for i in dataset[mes_points]:
            diff11=i.reshape(1,4)-cluster_center.reshape(1,4)
            diff22=np.transpose(i.reshape(1,4)-cluster_center.reshape(1,4))
            sum_in_cluster+=diff11*diff22
        Sw+=sum_in_cluster
        #calculate the compactness in each cluster
        comp_c+=np.trace(Sw)
    #define the separation between clusters
    sep_c=np.trace(SB)
    #determin the Vsc
    Vsc=sep_c/comp_c
    return Vsc
#We have to find that the max Vsc is for the n_cluster defined initially
Vsc_vector=[]
cc=[2,3,4,5,6,7,8,9,10]
for i in cc:
    Vsc_vector.append(validity_index(i))
print("Number of clusters which has max of Vsc:",Vsc_vector.index(max(Vsc_vector))+2 ,"=> Vsc=",max(Vsc_vector))
