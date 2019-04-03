#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:29:11 2018

@author: ecology
"""

import os
import numpy as np

from src import xgb_functions as xgbf
from src import initialize as init

def _calcInstanceWeights(Y):
    counts=np.bincount(Y)
    labelweights=np.zeros_like(counts,dtype=np.float)
    totalcounts=np.sum(counts)
    for i in range(len(labelweights)):
        labelweights[i]=(totalcounts-counts[i])/totalcounts
    weights=np.zeros(totalcounts,dtype=np.float)
    for i in range(totalcounts):
        weights[i]=labelweights[Y[i]]
    sumweights=np.sum(weights)
    weights=weights/sumweights
    return weights

def trainMulticlassSoftmaxModel(DataSet,VegeTypes,varnames,params,runtime=-1,bool_weight=False,bool_pandas=True,\
                                bool_strclass=False,labelHeaderName="",bool_save=False,savedir=""):
    ModelList=[]  
    if bool_pandas:
        [Y,X]=xgbf.trainingDataSet(DataSet,VegeTypes,varnames,\
                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)       
    else:
        [Y,X]=DataSet
    if not bool_strclass and len(Y.shape)>1:
        Y=init.mergeCategories(Y)
    if bool_weight:
        weights=_calcInstanceWeights(Y)
        model=xgbf.TrainModel(X,Y,params,weight=weights)
    else:
        model=xgbf.TrainModel(X,Y,params)
    if bool_save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if runtime==-1:
            modelName='softmax_multiclass.model'
        else:
            modelName='softmax_multiclass_run'+str(runtime)+'.model'
        modeldir=savedir+os.sep+modelName
        model.save_model(modeldir)
    ModelList.append(model)
    return ModelList

def trainMulticlassCategoryModel(DataSet,VegeTypes,varnames,params,runtime=-1,bool_weight=False,bool_pandas=True,\
                                 bool_strclass=False,labelHeaderName="",bool_save=False,savedir=""):
    ModelList=[]    
    if bool_pandas:
        [Y,X]=xgbf.trainingDataSet(DataSet,VegeTypes,varnames,\
                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)       
    else:
        [Y,X]=DataSet
    if not bool_strclass and len(Y.shape)>1:
        Y=init.mergeCategories(Y)

    num_class=len(VegeTypes)
    if bool_weight:
#        oriweights=CalcInstanceWeights(Y)
#        weights=np.zeros(num_instance*num_class,dtype=np.float)
#        for rec in range(num_instance):
#            weights[rec*num_class:(rec+1)*num_class]=np.ones(num_class,dtype=np.float)*oriweights[rec]/num_class
        [Y,X]=init.formatMulticlassCategoryInput(Y,X,num_class,1)        
        ratio=np.float(np.sum(Y==0))/np.sum(Y==1)
        params['scale_pos_weight']=ratio
        model=xgbf.TrainModel(X,Y,params)
    else:
        [Y,X]=init.formatMulticlassCategoryInput(Y,X,num_class,1)
        model=xgbf.TrainModel(X,Y,params)            
    if bool_save:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if runtime==-1:
            modelName='category_multiclass.model'
        else:
            modelName='category_multiclass_run'+str(runtime)+'.model'
        modeldir=savedir+os.sep+modelName
        model.save_model(modeldir)
    ModelList.append(model)
    return ModelList

def testMulticlassSoftmaxModel(ModelList,TestDataSet,VegeTypes,varnames,params,runtime=-1,bool_pandas=True,\
                               bool_strclass=False,labelHeaderName="",bool_save=False,savedir=""):
    num_class=len(VegeTypes)
    if not len(ModelList):
        if runtime==-1:
            modelName='softmax_multiclass.model'
        else:
            modelName='softmax_multiclass_run'+str(runtime)+'.model'
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params)
    else:
        model=ModelList[0]
    
    if bool_pandas:
        [test_Y,test_X]=xgbf.trainingDataSet(TestDataSet,VegeTypes,varnames,\
                                        bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)    
    else:
        [test_Y,test_X]=TestDataSet
    if len(test_X.shape)==1:
        t=np.zeros([1,len(varnames)])
        t[0,:]=test_X
        test_X=t
        t=np.zeros([1,num_class])
        t[0,:]=test_Y
        test_Y=t
    if not bool_strclass and len(test_Y.shape)>1:
        test_Y=init.mergeCategories(test_Y)
    pred_pY=xgbf.Predict(model,test_X,bool_binary=False)
    pred_Y=np.argmax(pred_pY,axis=1)
    return [pred_Y,pred_pY,test_Y]

def testMulticlassCategoryModel(ModelList,TestDataSet,VegeTypes,varnames,params,runtime=-1,bool_pandas=True,\
                                bool_strclass=False,labelHeaderName="",bool_save=False,savedir=""):
    num_class=len(VegeTypes)
    if not len(ModelList):
        if runtime==-1:
            modelName='category_multiclass.model'
        else:
            modelName='category_multiclass_run'+str(runtime)+'.model'
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params)
    else:
        model=ModelList[0]
    if bool_pandas:
        [test_Y,test_X]=xgbf.trainingDataSet(TestDataSet,VegeTypes,varnames,\
                                        bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)   
    else:
        [test_Y,test_X]=TestDataSet
    xshape=test_X.shape
    flag=len(xshape)
    if flag==1:
        t=np.zeros([1,len(varnames)])
        t[0,:]=test_X
        test_X=t  
        t=np.zeros([1,num_class])
        t[0,:]=test_Y
        test_Y=t
    if not bool_strclass and len(test_Y.shape)>1:
        test_Y=init.mergeCategories(test_Y)
    num_instance=test_X.shape[0]    
    test_X=init.formatMulticlassCategoryInput([],test_X,num_class,0)
    pred_pY=xgbf.Predict(model,test_X,bool_binary=False)
    if flag==1:  
        t=np.zeros([1,num_class])
        t[0,:]=pred_pY
        pred_pY=t
    else:
        pred_pY_reshape=np.zeros([num_instance,num_class])
        for i in range(num_instance):
            pred_pY_reshape[i,:]=pred_pY[i*num_class:(i+1)*num_class]    
        pred_pY=pred_pY_reshape
    pred_Y=np.argmax(pred_pY,axis=1)
    return [pred_Y,pred_pY,test_Y]

def predictMulticlassSoftmaxModelMatrix(ModelList,MatX,params,bool_save=False,savedir=""):
    matshape=MatX.shape
    if bool_save:
        modelName='softmax_multiclass.model'
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params)
    else:
        model=ModelList[0]
    pred_pY=xgbf.predictMultiMatrix(model,MatX,bool_binary=False)
    pred_Y=np.argmax(pred_pY,axis=1)
    pred_Y=pred_Y.reshape(matshape[0],matshape[1])
    prob_Y=np.zeros([matshape[0],matshape[1],pred_pY.shape[1]],dtype=np.float32)
    for i in range(pred_pY.shape[1]):
        prob_Y[:,:,i]=pred_pY[:,i].reshape(matshape[0],matshape[1])   
    pred_pY=prob_Y
    return [pred_Y,pred_pY]

def predictMulticlassCategoryModelMatrix(ModelList,MatX,num_class,params,bool_save=False,savedir=""):
    matshape=MatX.shape    
    pred_X=np.zeros([matshape[0]*matshape[1],matshape[2]],dtype=np.float32)
    for i in range(matshape[2]):
        pred_X[:,i]=MatX[:,:,i].flatten()
    if bool_save:
        modelName='category_multiclass.model'
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params)
    else:
        model=ModelList[0]
    pred_X=init.formatMulticlassCategoryInput([],pred_X,num_class,0)
    print("Predicting......")
    pred_pY=xgbf.Predict(model,pred_X,bool_binary=False)
    pred_pY_reshape=np.zeros([matshape[0]*matshape[1],num_class])
    for i in range(matshape[0]*matshape[1]):
        pred_pY_reshape[i,:]=pred_pY[i*num_class:(i+1)*num_class]
    pred_Y=np.argmax(pred_pY_reshape,axis=1)
    pred_Y=pred_Y.reshape(matshape[0],matshape[1])
    prob_Y=np.zeros([matshape[0],matshape[1],num_class],dtype=np.float32)
    for i in range(pred_pY_reshape.shape[1]):
        prob_Y[:,:,i]=pred_pY_reshape[:,i].reshape(matshape[0],matshape[1])   
    pred_pY=prob_Y
    return [pred_Y,pred_pY]

def predictMulticlassSoftmaxModelCvted(ModelList,pred_X,params,runtime=-1,bool_save=False,savedir=""):
    if bool_save:
        if runtime==-1:
            modelName='softmax_multiclass.model'
        else:
            modelName='softmax_multiclass_run'+str(runtime)+'.model'
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params)
    else:
        model=ModelList[0]
    pred_pY=xgbf.Predict(model,pred_X,bool_binary=False)
    return pred_pY

def predictMulticlassCategoryModelCvted(ModelList,pred_X,params,runtime=-1,bool_retlabel=False,num_instance=-1,num_class=-1,bool_save=False,savedir=""):
    if bool_save:
        if runtime==-1:
            modelName='category_multiclass.model'
        else:
            modelName='category_multiclass_run'+str(runtime)+'.model'
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params)
    else:
        model=ModelList[0]
    pred_pY=xgbf.Predict(model,pred_X,bool_binary=False)
    if bool_retlabel:        
        pred_pY_reshape=np.zeros([num_instance,num_class])
        for i in range(num_instance):
            pred_pY_reshape[i,:]=pred_pY[i*num_class:(i+1)*num_class]    
        pred_pY=pred_pY_reshape
        pred_Y=np.argmax(pred_pY,axis=1)
        return pred_Y
    else:
        return pred_pY