# -*- coding: utf-8 -*-
"""
Created on Fri May 25 00:47:39 2018

@author: ZH
"""

import os
import copy
import numpy as np

import src.initialize as init
from src import xgb_functions as xgbf
import src.multiclassbagging as mbag
import src.smote as smote

def getHierRelation(VTs1,VTs2):             #VTs1 is hierachically upper class
    if not len(VTs1)==len(VTs2):
        print("Set length error! Please make sure len(VTs1)=len(VTs2)!\n")
        return []
    VTs1_set=sorted(set(VTs1))
    VTs2_set=sorted(set(VTs2))
    class_num_VTs1=len(VTs1_set)
    HierRelations=[[] for i in range(class_num_VTs1)]
    for i in range(len(VTs2)):
        vt1=VTs1[i]
        vt2=VTs2[i]
        vt1_id=VTs1_set.index(vt1)
        if not vt2 in HierRelations[vt1_id]:
            HierRelations[vt1_id].append(vt2)
    return [VTs1_set,VTs2_set,HierRelations]

def _formatHierDataSet(DataSet,header,VTs1):
    RIDs=init.findPDRowIndex(DataSet,header,VTs1)
    HierDataSet=DataSet[RIDs]
    return HierDataSet

def estHierMulticlassModel(TrainDataSet,TestDataSet,baseMapTestResult,VegeTypes1,VegeTypes2,HierRelations,labelHeaderName_H,labelHeaderName_L,varnames,params,multiclassmethod,n_gpus,n_parallels,bool_parallel,\
                             baggingmetric,baggingweightindex,baggingmetricthres,varlabelweights,colsamplerate,train_percent,runtimes,\
                             bool_autolabel,varlabels,n_varlabels,bool_weight,bool_smote,bool_strclass,bool_save,hiermappingfolder):
    print("Begin Hierarchical Mapping...")
    baseMapTestResult=np.array(baseMapTestResult,dtype=np.int)
    pred_Y=np.zeros(len(TestDataSet),dtype=np.int)
    [test_Y,test_X]=xgbf.trainingDataSet(TestDataSet,VegeTypes2,[],\
                                    bool_strclass=bool_strclass,labelHeaderName=labelHeaderName_L) 
    for i in range(len(VegeTypes1)):
        VT1=VegeTypes1[i]
        VTs2=HierRelations[i]
        VTs2_IDs=[VegeTypes2.index(vt) for vt in VTs2]
        Y_IDs=np.argwhere(baseMapTestResult==i)[:,0]
        if len(VTs2)==1:
            print("VegeType %s has only one subtype: %s!\n"%(VT1,VTs2[0]))            
            pred_Y[Y_IDs]=VTs2_IDs[0]  
            continue
        print("Start training model: VegeType %s..."%VT1)
        HierTrainDataSet=_formatHierDataSet(TrainDataSet,labelHeaderName_H,VT1)
        HierTestDataSet=TestDataSet.loc[Y_IDs,:]
        #Set Parameters
        params_hier=copy.deepcopy(params)
        num_class=len(VTs2)        
        params_hier['num_class']=num_class
    
        #SMOTE for balanced dataset
        if bool_smote:
            HierTrainDataSet=smote.createSMOTEDataSet(HierTrainDataSet,VTs2,varnames,method='regular',tar_ratio=-1,\
                                                      bool_strclass=bool_strclass,labelHeaderName=labelHeaderName_L)
        #Train model
        multiclassFolderName="Hierarchical_Mapping_"+multiclassmethod+"_Models_"+VT1
        savedirbase=hiermappingfolder+os.sep+multiclassFolderName
        if bool_parallel:
            ModelList=mbag.trainMulticlassBaggingModel_parallel(HierTrainDataSet,VTs2,varnames,params_hier,multiclassmethod,n_gpus,n_parallels,\
                                                                baggingmetric=baggingmetric,baggingweightindex=baggingweightindex,baggingmetricthres=baggingmetricthres,\
                                                                varlabelweights=varlabelweights,colsamplerate=colsamplerate,train_percent=train_percent,runtimes=runtimes,\
                                                                bool_autolabel=bool_autolabel,varlabels=varlabels,n_varlabels=n_varlabels,bool_weight=bool_weight,\
                                                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName_L,bool_save=bool_save,savedirbase=savedirbase)
            [pred_Y_hier,pred_pY_hier,test_Y_hier]=mbag.testMulticlassBaggingModel_parallel(HierTestDataSet,VTs2,params_hier,multiclassmethod,n_gpus,n_parallels,\
                                                                runtimes=runtimes,bool_strclass=bool_strclass,labelHeaderName=labelHeaderName_L,\
                                                                bool_save=bool_save,savedirbase=savedirbase)
        else:
            ModelList=mbag.trainMulticlassBaggingModel(HierTrainDataSet,VTs2,varnames,params_hier,multiclassmethod,\
                                                       baggingmetric=baggingmetric,baggingweightindex=baggingweightindex,baggingmetricthres=baggingmetricthres,\
                                                       varlabelweights=varlabelweights,colsamplerate=colsamplerate,train_percent=train_percent,runtimes=runtimes,\
                                                       bool_autolabel=bool_autolabel,varlabels=varlabels,n_varlabels=n_varlabels,bool_weight=bool_weight,\
                                                       bool_strclass=bool_strclass,labelHeaderName=labelHeaderName_L,bool_save=bool_save,savedirbase=savedirbase)
            [pred_Y_hier,pred_pY_hier,test_Y_hier]=mbag.testMulticlassBaggingModel(HierTestDataSet,VTs2,params_hier,multiclassmethod,runtimes=runtimes,\
                                                        bool_strclass=bool_strclass,labelHeaderName=labelHeaderName_L,bool_save=bool_save,savedirbase=savedirbase)
        for vti in range(len(VTs2)):
            pred_Y_hier[pred_Y_hier==vti]=VTs2_IDs[vti]
        pred_Y[Y_IDs]=pred_Y_hier
        print("VegeType %s model finished!\n"%VT1)
    print("Hierarchical mapping models all established!\n")
    return [pred_Y,test_Y]
    
def predictHierDownMapping(basemaplayer,VegeTypes1,VegeTypes2,HierRelations,MatX,nrow,ncol,varnames,params,n_gpus,n_parallels,bool_parallel,\
                               multiclassmethod,runtimes,bool_save,hiermappingfolder):
    #Read Base Map Layer
    probs=np.zeros([nrow,ncol,len(VegeTypes2)],dtype=np.float32)
    for i in range(len(VegeTypes1)):
        VT1=VegeTypes1[i]
        VTs2=HierRelations[i]
        print("Predicting Upper VegeType: %s..."%VT1)
        VT1_id=VegeTypes1.index(VT1)
        num_class=len(VTs2)
        if num_class==1:
            print("Upper: %s ---> Lower: %s\n"%(VT1,VTs2[0]))
            VT2_id=VegeTypes2.index(VTs2[0])
            pred_pVT2=np.zeros_like(basemaplayer)
            pred_pVT2[basemaplayer==VT1_id]=1
            probs[:,:,VT2_id]=pred_pVT2
        else:
            multiclassFolderName="Hierarchical_Mapping_"+multiclassmethod+"_Models_"+VT1
            savedirbase=hiermappingfolder+os.sep+multiclassFolderName
            
            if bool_parallel:
                [pred_Y,pred_pY]=mbag.predictMulticlassBaggingModel_parallel(MatX,nrow,ncol,varnames,num_class,params,multiclassmethod,\
                                                   n_gpus,n_parallels,runtimes=runtimes,bool_save=bool_save,savedirbase=savedirbase)
            else:
                [pred_Y,pred_pY]=mbag.predictMulticlassBaggingModel(MatX,nrow,ncol,varnames,num_class,params,\
                                multiclassmethod,runtimes=runtimes,bool_save=bool_save,savedirbase=savedirbase)
            for j in range(num_class):
                VT2_id=VegeTypes2.index(VTs2[j])
                pred_pVT2=pred_pY[:,:,j]
                pred_pVT2[basemaplayer!=VT1_id]=0
                probs[:,:,VT2_id]=pred_pVT2
    pred_Y=np.argmax(probs,axis=2)
    pred_pY=probs
    return [pred_Y,pred_pY]

def predictHierUpMapping(basemaplayer,VegeTypes1,VegeTypes2,HierRelations): 
    basemaplayer=np.array(basemaplayer,dtype=np.int)
    pred_Y=np.zeros_like(basemaplayer,dtype=np.int)
    for i in range(len(VegeTypes1)):
        VT1=VegeTypes1[i]
        VTs2=HierRelations[i]
        print("Predicting Upper VegeType: %s..."%VT1)
        VT1_id=VegeTypes1.index(VT1)
        for vt2 in VTs2:
            print("Lower: %s ---> Upper: %s"%(vt2,VT1))
            VT2_id=VegeTypes2.index(vt2)
            pred_Y[basemaplayer==VT2_id]=VT1_id
        print("\n")
    return pred_Y