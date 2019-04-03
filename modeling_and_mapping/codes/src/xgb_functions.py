#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 23:32:24 2018

@author: ecology
"""

import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import src.initialize as init

#Format pandas dataframe to numpy matrix, suitable for XGBoost model input
def trainingDataSet(DataSet,vtnames,varnames,bool_strclass=False,labelHeaderName="",bool_binary=False):
#Input DataSet must be pandas dataframe
    if not bool_strclass:
        Y=np.array(DataSet[vtnames])
    else:
        Y=init.classStrToNum(DataSet,labelHeaderName,vtnames,bool_binary=bool_binary)  
    X=np.array(DataSet[varnames])
    return [Y,X]

#Split a given data set (numpy format) to train and test set 
def splitTrainTestData(DataSet,train_percent,bool_stratify=False):
    [Y,X]=DataSet
    if not bool_stratify:
        [train_x,test_x,train_y,test_y]=train_test_split(X,Y,test_size=1-train_percent,shuffle=True)
    else:
        if len(Y.shape)>1:
            if Y.shape[1]>1:
                class_labels=init.mergeCategories(Y)
            else:
                class_labels=Y[:,0]
        else:
            class_labels=Y
        [train_x,test_x,train_y,test_y]=train_test_split(X,Y,test_size=1-train_percent,\
                                        shuffle=True,stratify=class_labels)
    return [train_x,test_x,train_y,test_y]

#set XGBoost parameters
def setParams(bool_gpu,tree_method,num_class,eval_metric,max_depth,lamb,alpha,gamma,\
              subsample,colsample_bytree,min_child_weight,scale_pos_weight,eta,nthread,max_delta_step=0,gpu_id=0):
    params={'eval_metric':eval_metric,
            'max_depth':max_depth,
            'lambda':lamb,
            'alpha':alpha,
            'gamma':gamma,
            'subsample':subsample,
            'colsample_bytree':colsample_bytree,
            'min_child_weight':min_child_weight,
            'scale_pos_weight':scale_pos_weight,
            'eta':eta,
            'silent':True}  

    if tree_method==0:                    #gbtree logistic     
        params_single={'booster':'gbtree',
                       'objective':'binary:logistic'} 
        params.update(params_single) 
    elif tree_method==1:                  #softmax method
        params_multi={'objective':'multi:softprob',
                      'num_class':num_class}
        params.update(params_multi) 
    elif tree_method==2:                  #dart tree logistic
        params_single={'booster': 'dart',
                       'objective':'binary:logistic',
                       'sample_type':'weighted',
                       'normalize_type':'forest',
                       'rate_drop':0.2,
                       'skip_drop': 0.4}  
        params.update(params_single)  
    elif tree_method==3:                  #dart tree multiclass
        params_single={'booster': 'dart',
                       'objective':'multi:softprob',
                      'num_class':num_class,
                       'sample_type':'weighted',
                       'normalize_type':'forest',
                       'rate_drop':0.2,
                       'skip_drop': 0.4}  
        params.update(params_single)  
    elif tree_method==4:
        params_single={'booster':'gbtree',
                       'objective':'reg:logistic'} 
        params.update(params_single) 
    if not max_delta_step==0:
        params_delta={'max_delta_step':max_delta_step}
        params.update(params_delta)  
    if bool_gpu:
        params_gpu={'tree_method':'gpu_hist',
                    'updater':'grow_gpu',
                    'gpu_id':gpu_id}
        params.update(params_gpu)
    else:
        params_cpu={'tree_method':'exact',
                    'n_jobs':nthread}
        params.update(params_cpu)   
    if nthread>=0:
        os.putenv("OMP_NUM_THREADS",str(nthread))
    return params

def TrainModel(train_x,train_y,params,weight=[]):
    if not len(weight):
        dtrain=xgb.DMatrix(train_x,label=train_y) 
    else:
        dtrain=xgb.DMatrix(train_x,label=train_y,weight=weight) 
    watchlist=[(dtrain,'train')]
    model=xgb.train(params,dtrain,num_boost_round=3600,evals=watchlist,\
                    early_stopping_rounds=10,verbose_eval=False)
    return model

def Predict(model,pred_X,bool_binary,threshold=0.5):
    dX=xgb.DMatrix(pred_X)
    pred_pY=model.predict(dX)
    if bool_binary:
        pred_Y=(pred_pY>=threshold)*1
        return [pred_Y,pred_pY]
    else:
        return pred_pY

def Evaluate(test_Y,pred_Y,pred_pY,method):
    if len(test_Y.shape)>1:
        test_Y=test_Y[:,0]
    if len(pred_Y.shape)>1:
        pred_Y=pred_Y[:,0]
    if len(pred_pY.shape)>1:
        pred_pY=pred_pY[:,0]
    if method=='auc':
        evalue=metrics.roc_auc_score(test_Y,pred_pY)
    elif method=='accuracy':
        evalue=metrics.accuracy_score(test_Y,pred_Y)
    elif method=='recall':
        evalue=metrics.recall_score(test_Y,pred_Y)
    elif method=='precision':
        evalue=metrics.precision_score(test_Y,pred_Y)
    elif method=='f1score':
        evalue=metrics.f1_score(test_Y,pred_Y)
    elif method=='kappa':
        evalue=metrics.cohen_kappa_score(test_Y,pred_Y)
    elif method=='confmat':
        evalue=metrics.confusion_matrix(test_Y,pred_Y)
    elif method=='linear-accuracy':
        class_counts=np.bincount(test_Y)
        class_num=len(class_counts)
        class_accuracies=np.zeros(class_num)
        class_accuracy_weights=class_counts/np.sum(class_counts)
        class_accuracy_weights=1/4+3/4*class_accuracy_weights
        class_accuracy_weights=class_accuracy_weights/np.sum(class_accuracy_weights)
        for i in range(class_num):
            class_idxs=np.argwhere(test_Y==i)[:,0]
            test_Y_class=test_Y[class_idxs]
            pred_Y_class=pred_Y[class_idxs]
            class_accuracies[i]=metrics.accuracy_score(test_Y_class,pred_Y_class)
        evalue=np.sum(class_accuracy_weights*class_accuracies)
    elif method=='balanced-accuracy':
        class_counts=np.bincount(test_Y)
        class_num=len(class_counts)
        class_accuracies=np.zeros(class_num)
        class_accuracy_weights=np.ones(class_num)
        class_accuracy_weights=class_accuracy_weights/class_num
        for i in range(class_num):
            class_idxs=np.argwhere(test_Y==i)[:,0]
            test_Y_class=test_Y[class_idxs]
            pred_Y_class=pred_Y[class_idxs]
            class_accuracies[i]=metrics.accuracy_score(test_Y_class,pred_Y_class)
        evalue=np.sum(class_accuracy_weights*class_accuracies)        
    else:
        print("Input method is not in the availale list!")
    return evalue

def biEvalAndWriteResult(EvalueFolder,pred_Y,pred_pY,test_Y):
    if not os.path.exists(EvalueFolder):
        os.makedirs(EvalueFolder)
    evalArray=np.zeros([1,2])
    evalArray[0,0]=Evaluate(test_Y,pred_Y,pred_pY,'accuracy')
    evalArray[0,1]=Evaluate(test_Y,pred_Y,pred_pY,'kappa')
    evalFiledirto=EvalueFolder+os.sep+"Model_Evaluation.csv"
    init.writeArrayToCSV(evalArray,['accuracy','kappa'],evalFiledirto)
    YArray=np.zeros([len(test_Y),3])
    YArray[:,0]=test_Y
    YArray[:,1]=pred_Y
    YArray[:,2]=pred_pY
    YFiledirto=EvalueFolder+os.sep+"Real_and_Predicted_Results.csv"
    init.writeArrayToCSV(YArray,['real','predict','xgb_probs'],YFiledirto)

def mlcEvalAndWriteResult(EvalueFolder,pred_Y,pred_pY,test_Y):
    if not os.path.exists(EvalueFolder):
        os.makedirs(EvalueFolder)
    evalArray=np.zeros([1,4])
    evalArray[0,0]=Evaluate(test_Y,pred_Y,pred_pY,'accuracy')
    evalArray[0,1]=Evaluate(test_Y,pred_Y,pred_pY,'kappa')
    evalArray[0,2]=Evaluate(test_Y,pred_Y,pred_pY,'balanced-accuracy')
    evalArray[0,3]=Evaluate(test_Y,pred_Y,pred_pY,'linear-accuracy')
    print("Evaluation Result, ACC = %f, KAPPA = %f, B-ACC = %f, L-ACC = %f"%(evalArray[0,0],evalArray[0,1],evalArray[0,2],evalArray[0,3]))
    evalFiledirto=EvalueFolder+os.sep+"Model_Evaluation.csv"
    init.writeArrayToCSV(evalArray,['accuracy','kappa','balanced-accuracy','linear-accuracy'],evalFiledirto)
    YArray=np.zeros([len(test_Y),2])
    YArray[:,0]=test_Y
    YArray[:,1]=pred_Y
    YFiledirto=EvalueFolder+os.sep+"Real_and_Predicted_Results.csv"
    init.writeArrayToCSV(YArray,['real','predict'],YFiledirto)  

def evaluateMatrix(test_Y,pred_Y,pred_pY,method):
    test_Y=test_Y.flatten()
    pred_Y=pred_Y.flatten()
    pred_pY=pred_pY.flatten()
    evalue=Evaluate(test_Y,pred_Y,pred_pY,method)
    return evalue

#confusion index
def calcUncertainty(pred_pY):
    pred_sorted=np.sort(pred_pY,axis=2)
    mu_1=pred_sorted[:,:,-1]
    mu_2=pred_sorted[:,:,-2]
    CI=1-(mu_1-mu_2)
    return CI

def loadModel(modeldir,params):
    if 'gpu' in params.get('tree_method'):
        model= xgb.Booster({'predictor':'gpu_predictor','gpu_id':params['gpu_id']}) 
    else:
        model= xgb.Booster({'n_jobs':params['n_jobs'],'predictor':'cpu_predictor'}) 
    model.load_model(modeldir)
    return model
    