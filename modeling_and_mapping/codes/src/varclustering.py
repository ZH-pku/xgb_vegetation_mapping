# -*- coding: utf-8 -*-
"""
Created on Sat May 26 23:16:47 2018

@author: ZH
"""

import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans

import src.xgb_functions as xgbf

def KMeansLabel(DataSet,varnames,n_varlabels=5):
    [Y,X]=xgbf.trainingDataSet(DataSet,[],varnames)
    X_scaled = preprocessing.scale(X)
    kmeans = KMeans(n_clusters=n_varlabels).fit(X_scaled.T)
    varlabels=kmeans.labels_
    varlabel_counts=np.bincount(varlabels)
    #Print KMeans Clustering Result
    print("\n")
    print("Variables have been divided into %d groups: "%len(varlabel_counts))
    for i in range(len(varlabel_counts)):
        print("Label: %d, num = %d"%(i,varlabel_counts[i]))
    print("\n")
    return varlabels