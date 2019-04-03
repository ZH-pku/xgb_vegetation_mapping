#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:56:58 2018

@author: ecology
"""

import os
import copy
import pandas as pd
import numpy as np

def readCSVasPandas(filedir):
    pdData=pd.DataFrame(pd.read_csv(filedir,header=0))
    return pdData
    
def getListFromPandas(filedir,header):
    pdList=readCSVasPandas(filedir)
    List=list(pdList[header])
    return List

def getListFrompdDataSet(pdData,header):
    List=list(pdData[header])
    return List

def classStrToNum(DataSet,headerName,vegetypes,bool_binary=False):
    Y=np.zeros(len(DataSet),dtype=np.int32)
    if not bool_binary:
        for i in range(len(vegetypes)):
            idx=DataSet[headerName].isin([vegetypes[i]])
            Y[idx==True]=i
    else:
        idx=DataSet[headerName].isin(vegetypes)
        Y[idx==True]=1
    Y=np.int32(Y)
    return Y

def classNumToStr(labels,vegetypes):
    Y=[]
    total=len(labels)
    for i in range(total):
        Y.append(vegetypes[labels[i]])
    return Y

def mergeCategories(Y):
    Y=np.int32(Y)
    labels=np.argwhere(Y==1)[:,1]
    return labels

def expandCategories(labels,num_class=-1):
    if num_class==-1:
        num_class=len(np.bincount(labels))
    total=len(labels)
    Y=np.zeros([total,num_class],dtype=np.int32)
    for i in range(total):
        Y[i,labels[i]]=1
    return Y

def writeArrayToCSV(Array,ArrayNames,filedirto):
    if not Array.shape[1]==len(ArrayNames):
        print("Array Dim != len(ArrayNames)")
        return
    save=pd.DataFrame({"ID":np.linspace(1,Array.shape[0],Array.shape[0]).astype(np.int32)})
    for i in range(Array.shape[1]):
        pdtmp=pd.DataFrame({ArrayNames[i]:Array[:,i]})
        save=pd.concat([save,pdtmp],axis=1)
    save.to_csv(filedirto,index=False,header=True)
    
def writeArrayListToCSV(ArrayList,ArrayNames,filedirto):
    if not len(ArrayList)==len(ArrayNames):
        print("Array Dim != len(ArrayNames)")
        return
    save=pd.DataFrame({"ID":np.linspace(1,len(ArrayList[0]),len(ArrayList[0])).astype(np.int32)})
    for i in range(len(ArrayList)):
        pdtmp=pd.DataFrame({ArrayNames[i]:ArrayList[i]})
        save=pd.concat([save,pdtmp],axis=1)
    save.to_csv(filedirto,index=False,header=True)

def getMask(MatX):
    nan=MatX[0,0,0]
    mask=copy.deepcopy(MatX[:,:,0])
    mask[MatX[:,:,0]==nan]=0
    mask[MatX[:,:,0]!=nan]=1
    mask=mask.astype(np.bool)
    return mask    
  
def formatMulticlassCategoryInput(Y,X,num_class,bool_label):
    xshape=X.shape
    InputArray=np.zeros([xshape[0]*num_class,xshape[1]+1])
    for rec in range(xshape[0]):
        for i in range(rec*num_class,(rec+1)*num_class):
            InputArray[i,0:xshape[1]]=X[rec,:]
    CateLabel=np.zeros(xshape[0]*num_class)
    label=np.linspace(0,num_class-1,num_class)
    for i in range(0,xshape[0]*num_class,num_class):
        CateLabel[i:i+num_class]=label
    InputArray[:,xshape[1]]=CateLabel
    if bool_label:
        InputLabel=np.zeros(xshape[0]*num_class,dtype=np.int32)
        for rec in range(xshape[0]):
            InputLabel[rec*num_class+Y[rec]]=1
        return [InputLabel,InputArray]
    else:
        return InputArray

def fomatMulticlassSoftmaxMatrix(MatX):
    matshape=MatX.shape
    pred_X=np.zeros([matshape[0]*matshape[1],matshape[2]],dtype=np.float32)
    for i in range(matshape[2]):
        pred_X[:,i]=MatX[:,:,i].flatten()    
    return pred_X
    
def formatMulticlassCategoryMatrix(MatX,num_class):
    matshape=MatX.shape    
    pred_X=np.zeros([matshape[0]*matshape[1],matshape[2]],dtype=np.float32)
    for i in range(matshape[2]):
        pred_X[:,i]=MatX[:,:,i].flatten()
    pred_X=formatMulticlassCategoryInput([],pred_X,num_class,0)
    return pred_X

def _probStretch(pred_pY,mask):
    for j in range(pred_pY.shape[1]):
        probs=pred_pY[:,j]
        p_max=np.max(probs[mask])
        pred_pY[:,j]=probs*(1/p_max)
    return pred_pY

def reshapeMulticlassMatrix(pred_pY,nrow,ncol,num_class,bool_onearray=False,bool_stretch=False,mask=-1):
    if bool_onearray:
        pred_pY_reshape=np.zeros([nrow*ncol,num_class])
        for i in range(nrow*ncol):
            pred_pY_reshape[i,:]=pred_pY[i*num_class:(i+1)*num_class]
    else:
        pred_pY_reshape=pred_pY
    if bool_stretch:
        pred_pY_reshape=_probStretch(pred_pY_reshape,mask)
    pred_Y=np.argmax(pred_pY_reshape,axis=1)
    pred_Y=pred_Y.reshape(nrow,ncol)
    prob_Y=np.zeros([nrow,ncol,num_class],dtype=np.float32)
    for i in range(num_class):
        prob_Y[:,:,i]=pred_pY_reshape[:,i].reshape(nrow,ncol)   
    pred_pY=prob_Y    
    return [pred_Y,pred_pY]

def generateVarialeTiffList(variableFolderdir,varnames,postfix):  
    TiffList=[]
    for varname in varnames:
        TiffList.append(variableFolderdir+os.sep+varname+postfix)
    Total=len(TiffList)
    print("TiffList Get!  Total = %d\n"%Total)
    return [TiffList,Total]
  
def rmvPDRow(DataSet,recid,bool_list=False):
    if bool_list:
        SelectDataSet=DataSet.drop(recid,axis=0,inplace=False)
    else:
        SelectDataSet=DataSet.drop([recid],axis=0,inplace=False)
    return SelectDataSet

def findPDRowIndex(DataSet,header,value,bool_list=False):
    if bool_list:
        RIDs=DataSet[header].isin(value)
    else:
        RIDs=DataSet[header].isin([value])
    return RIDs

