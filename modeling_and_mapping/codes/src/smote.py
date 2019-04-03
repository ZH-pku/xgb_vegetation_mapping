# -*- coding: utf-8 -*-
"""
Created on Sun May  6 00:30:29 2018

@author: ZH
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE 

import src.xgb_functions as xgbf
import src.initialize as init

#SMOTE
def _calc_smoteratio(Y,class_counts,tar_ratio=16):
    class_num=len(class_counts)   
    class_counts_smote=np.zeros(class_num,dtype=np.int32)
    max_class_count=np.max(class_counts)
    min_class_count=np.min(class_counts)
    tar_max_count=max_class_count
    tar_min_count=np.round(tar_max_count/tar_ratio)
    for i in range(class_num):
        class_counts_smote[i]=np.ceil((class_counts[i]-min_class_count)/(max_class_count-min_class_count)\
                                *(tar_max_count-tar_min_count)+tar_min_count)
    smoteratiodict={}
    for i in range(class_num):
        smoteratiodict[i]=class_counts_smote[i]
    return smoteratiodict
    

def createSMOTEDataSet(DataSet,VegeTypes,varnames,method='regular',tar_ratio=-1,nthread=1,bool_pandas=True,bool_strclass=False,labelHeaderName=""):
    if bool_pandas:
        [Y,X]=xgbf.trainingDataSet(DataSet,VegeTypes,varnames,\
                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)       
    else:
        [Y,X]=DataSet
    if not bool_strclass:
        Y=init.mergeCategories(Y)
    class_counts=np.bincount(Y)
    min_class_count=np.min(class_counts)
    if min_class_count>5:
        k_neighbors=5
    else:
        k_neighbors=min_class_count-1
    if tar_ratio==-1:
        sm=SMOTE(kind=method,k_neighbors=k_neighbors,n_jobs=nthread)
    else:
        smoteratiodict=_calc_smoteratio(Y,class_counts,tar_ratio=tar_ratio)
        sm=SMOTE(ratio=smoteratiodict,kind=method,k_neighbors=k_neighbors,n_jobs=nthread)
    [X_res,Y_res]=sm.fit_sample(X,Y)
    if not bool_strclass:
        if bool_pandas:
            X_res_pd=pd.DataFrame(X_res,columns=varnames)
            Y_indi=init.expandCategories(Y_res,num_class=len(VegeTypes))
            Y_res_pd=pd.DataFrame(Y_indi,columns=VegeTypes)
            SMOTEDataSet=pd.concat([Y_res_pd,X_res_pd],axis=1)
        else:
            Y_indi=init.expandCategories(Y_res,num_class=len(VegeTypes))
            SMOTEDataSet=[Y_indi,X_res]
    else:
        X_res_pd=pd.DataFrame(X_res,columns=varnames)
        Y_indi=init.classNumToStr(Y_res,VegeTypes)
        Y_res_pd=pd.DataFrame(Y_indi,columns=[labelHeaderName])
        SMOTEDataSet=pd.concat([Y_res_pd,X_res_pd],axis=1)
    return SMOTEDataSet

#Note: DataSet1 is target dataset
def rmvRepRecord(DataSet1,DataSet2,varnames):
    bool_reprec=np.ones(len(DataSet1),dtype=np.bool)
    for varname in varnames:
        cmpDS2=init.getListFrompdDataSet(DataSet2,varname)
        repIds=np.array(init.findPDRowIndex(DataSet1,varname,cmpDS2,bool_list=True))
        bool_reprec=bool_reprec & repIds
    nrepRecIds=np.argwhere(bool_reprec==False)[:,0]
    rmvDataSet=DataSet1.loc[nrepRecIds,:]
    return rmvDataSet