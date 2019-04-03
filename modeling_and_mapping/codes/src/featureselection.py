# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 19:46:25 2018

@author: ZH
"""

#featureselection.py contains functions relevant to feature selection procedures

import os
import copy
import numpy as np
from multiprocessing import Pool, Manager
from sklearn.model_selection import StratifiedKFold

import src.xgb_functions as xgbf
import src.processgeotiff as ptf
import src.initialize as init
import src.multiclass as mtc

def locateFeatureScores(feature_names,feature_scores):
    total=len(feature_names)
    fscores=np.zeros(total)
    for i in range(total):
        fname='f'+str(i)
        try:
            fscores[i]=feature_scores[fname]
        except:
            fscores[i]=0
    return [feature_names,fscores]

def removeTrivialFeature(feature_names,fscores):
    trivial_feature_idx=np.argmin(fscores)
    trivial_feature_name=feature_names[trivial_feature_idx]
    feature_names.remove(trivial_feature_name)
    fscores=np.delete(fscores,trivial_feature_idx,axis=0)
    return [feature_names,fscores,trivial_feature_name]

def removeAppointedFeature(feature_names,fscores,app_feature_name):
    app_feature_idx=feature_names.index(app_feature_name)
    feature_names.remove(app_feature_name)
    fscores=np.delete(fscores,app_feature_idx,axis=0)   
    return [feature_names,fscores]

def removeIdenticalFeatures(DataSet,varnames,rm_thres=0.98):
    [fslabels,fsdata]=xgbf.trainingDataSet(DataSet,[],varnames)
    n=len(varnames)
    icld_features=[varnames[0]]
#    print("feature: %s selected!"%varnames[0])
    R=np.corrcoef(fsdata.T)
    flag_rm=False
    for i in range(1,n):
        fname=varnames[i]
        n_in=len(icld_features)
        for j in range(n_in):
            fId=varnames.index(icld_features[j])
            if R[i,fId]>rm_thres:
                flag_rm=True
                break
        if not flag_rm:
            icld_features.append(fname)
#            print("feature: %s selected!"%fname)
        flag_rm=False
    print("Identical Features Removed!\n")
    return icld_features

#evaluate a given feature under certain feature set, using stratified k-fold cross validation or direct validation on valid set.
def evalFeature(CPIDs,evaluate_feature,TrainDataSet,ValidDataSet,VegeTypes,feature_names,multiclassmethod,params,evalue_method,\
                    bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save,savedir):
    print("Trying to evalute feature: %s"%evaluate_feature)
    params_parallel=copy.deepcopy(params)
    process_pid=os.getpid()
    if len(CPIDs)<n_parallels:
        CPIDs.append(process_pid)
    process_pid_index=CPIDs.index(process_pid)
    print("Worker #%d: PID = %d"%(process_pid_index,process_pid))
    if bool_gpu:
        params_parallel['gpu_id']=process_pid_index%n_gpus    
    if bool_cv==1:
        [Y,X]=xgbf.trainingDataSet(TrainDataSet,VegeTypes,feature_names,\
                                    bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)
        if not bool_strclass:
            class_labels=init.mergeCategories(Y)
        else:
            class_labels=Y
        pred_Y_cv=np.zeros(len(class_labels)*cv_num,dtype=np.int32)
        pred_pY_cv=np.zeros(len(class_labels)*cv_num)
        test_Y_cv=np.zeros(len(class_labels)*cv_num,dtype=np.int32)
        last_cv_idx=0
        current_cv_idx=0
        for cv_i in range(cv_num):
            skf=StratifiedKFold(n_splits=skf_split,shuffle=True)
            cv_j=0
            for train, test in skf.split(X,class_labels):
                train_x=X[train]
                train_y=Y[train]
                test_x=X[test]
                test_y=Y[test]    
                if multiclassmethod=='softmax':
                    ModelList=mtc.trainMulticlassSoftmaxModel([train_y,train_x],VegeTypes,feature_names,params_parallel,bool_weight=bool_weight,bool_pandas=False,\
                                                              bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
                    [pred_Y,pred_pY,test_Y]=mtc.testMulticlassSoftmaxModel(ModelList,[test_y,test_x],VegeTypes,feature_names,params_parallel,bool_pandas=False,\
                                                            bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
                elif multiclassmethod=='category':
                    ModelList=mtc.trainMulticlassCategoryModel([train_y,train_x],VegeTypes,feature_names,params_parallel,bool_weight=bool_weight,bool_pandas=False,\
                                                               bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
                    [pred_Y,pred_pY,test_Y]=mtc.testMulticlassCategoryModel(ModelList,[test_y,test_x],VegeTypes,feature_names,params_parallel,bool_pandas=False,\
                                                            bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
                else:
                    print("Invalid Multiclass Method Input!")
                current_cv_idx=len(test_Y)+last_cv_idx
                pred_Y_cv[last_cv_idx:current_cv_idx]=pred_Y
#                    pred_pY_cv[last_cv_idx:current_cv_idx]=pred_pY
                test_Y_cv[last_cv_idx:current_cv_idx]=test_Y
                last_cv_idx=current_cv_idx
#                    evalues_runtime[cv_i,cv_j]=xgbf.Evaluate(test_Y,pred_Y,pred_pY,evalue_method)                    
                cv_j=cv_j+1
        evalue=xgbf.Evaluate(test_Y_cv,pred_Y_cv,pred_pY_cv,evalue_method) 
    else:
        if multiclassmethod=='softmax':
            ModelList=mtc.trainMulticlassSoftmaxModel(TrainDataSet,VegeTypes,feature_names,params_parallel,bool_weight=bool_weight,bool_pandas=True,\
                                                      bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
            [pred_Y,pred_pY,test_Y]=mtc.testMulticlassSoftmaxModel(ModelList,ValidDataSet,VegeTypes,feature_names,params_parallel,bool_pandas=True,\
                                                    bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
        elif multiclassmethod=='category':
            ModelList=mtc.trainMulticlassCategoryModel(TrainDataSet,VegeTypes,feature_names,params_parallel,bool_weight=bool_weight,bool_pandas=True,\
                                                       bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
            [pred_Y,pred_pY,test_Y]=mtc.testMulticlassCategoryModel(ModelList,ValidDataSet,VegeTypes,feature_names,params_parallel,bool_pandas=True,\
                                                    bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
        else:
            print("Invalid Multiclass Method Input!")
        evalue=xgbf.Evaluate(test_Y,pred_Y,pred_pY,evalue_method) 
    print("Feature: %s partial evalue = %f\n"%(evaluate_feature,evalue))
    return evalue

#establish model and predict results
def _estabModelAndPred(TrainDataSet,ValidDataSet,VegeTypes,feature_names,multiclassmethod,params,evalue_method,EvalueFolder,variableFolderdir,postfix,\
                           bool_predictmap,bool_weight,bool_strclass,labelHeaderName,bool_save,savedir):
    num_class=len(VegeTypes)
    #Establish Training Model
    if multiclassmethod=='softmax':
        ModelList=mtc.trainMulticlassSoftmaxModel(TrainDataSet,VegeTypes,feature_names,params,bool_weight=bool_weight,\
                                                  bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)            
        [pred_Y,pred_pY,test_Y]=mtc.testMulticlassSoftmaxModel(ModelList,ValidDataSet,VegeTypes,feature_names,params,\
                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
    elif multiclassmethod=='category':
        ModelList=mtc.trainMulticlassCategoryModel(TrainDataSet,VegeTypes,feature_names,params,bool_weight=bool_weight,\
                                                   bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY,test_Y]=mtc.testMulticlassCategoryModel(ModelList,ValidDataSet,VegeTypes,feature_names,params,\
                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
    else:
        print("Invalid Multiclass Method Input!")
    
    #Write Test Results
    YArray=np.zeros([len(test_Y),2])
    YArray[:,0]=test_Y
    YArray[:,1]=pred_Y
    YFiledirto=EvalueFolder+os.sep+"Best_Feature_Real_and_Predicted_Results.csv"
    init.writeArrayToCSV(YArray,['real','predict'],YFiledirto)     
    
    #Evaluate Model and Write Result
    evalArray=np.zeros([1,2])
    evalArray[0,0]=xgbf.Evaluate(test_Y,pred_Y,pred_pY,'accuracy')
    evalArray[0,1]=xgbf.Evaluate(test_Y,pred_Y,pred_pY,'kappa')
    evalFiledirto=EvalueFolder+os.sep+"Best_Feature_Model_Evaluation_ValidDataSet.csv"
    init.writeArrayToCSV(evalArray,['accuracy','kappa'],evalFiledirto)
    
    #Find XGBoost Feature Scores
    featureScoreFiledirto=EvalueFolder+os.sep+"Feature_Scores.csv"
    model=ModelList[0]
    feature_scores=model.get_fscore()
    [feature_names,fscores]=locateFeatureScores(feature_names,feature_scores)
    init.writeArrayListToCSV([feature_names,fscores],['VariableName','FeatureScore'],featureScoreFiledirto)

    if bool_predictmap:
        #Predict Mapping Results
        print("Predict region...")
        nanDefault=-9999
        [TiffList,Total]=init.generateVarialeTiffList(variableFolderdir,feature_names,postfix)
        [MatX,Driver,GeoTransform,Proj,nrow,ncol]=ptf.readTiffAsNumpy(TiffList)
        BestFeatureProductFolder=EvalueFolder+os.sep+"Best_Features_Mapping_Results"
        if multiclassmethod=='softmax':
            pred_X=init.fomatMulticlassSoftmaxMatrix(MatX)
            pred_pY=mtc.predictMulticlassSoftmaxModelCvted(ModelList,pred_X,params,bool_save=bool_save,savedir=savedir)
            [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY,nrow,ncol,num_class,bool_onearray=False)
        elif multiclassmethod=='category':
            pred_X=init.formatMulticlassCategoryMatrix(MatX,num_class)
            pred_pY=mtc.predictMulticlassCategoryModelCvted(ModelList,pred_X,params,bool_save=bool_save,savedir=savedir)
            [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY,nrow,ncol,num_class,bool_onearray=True)
        for i in range(len(VegeTypes)):
            vtname=VegeTypes[i]
            ProductFolder=BestFeatureProductFolder+os.sep+vtname
            if not os.path.exists(ProductFolder):
                os.makedirs(ProductFolder)
            Filename1=vtname+"_xgboost_"+multiclassmethod+postfix
            ProductFiledirto1=ProductFolder+os.sep+Filename1 
            ptf.writeNumpyToTiff(pred_pY[:,:,i],Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto1,datatype='Float32')
        Filename2="VegeMap_XGBoost_multiclass_"+multiclassmethod+postfix
        ProductFolder=BestFeatureProductFolder
        ProductFiledirto2=ProductFolder+os.sep+Filename2
        ptf.writeNumpyToTiff(pred_Y,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto2,datatype='Int16')    
    return fscores

def estabModelAndPred(TrainDataSet,ValidDataSet,VegeTypes,feature_names,multiclassmethod,params,evalue_method,EvalueFolder,variableFolderdir,postfix,\
                           bool_predictmap,bool_weight,bool_strclass,labelHeaderName,bool_save=True,savedirbase=""):
    P=Pool(1)
    multiclassFolderName="Multiclass_XGBoost_"+multiclassmethod+"_Model"
    savedir=savedirbase+os.sep+multiclassFolderName
    fscores=P.apply(_estabModelAndPred,(TrainDataSet,ValidDataSet,VegeTypes,feature_names,multiclassmethod,params,evalue_method,EvalueFolder,variableFolderdir,postfix,\
                                            bool_predictmap,bool_weight,bool_strclass,labelHeaderName,bool_save,savedir))
    P.close()
    P.join()
    return fscores

#evaluate a given feature under certain feature set, in parallel computing
def evalFeature_parallel(bool_forward,feature_selected,features_pool,TrainDataSet,ValidDataSet,VegeTypes,multiclassmethod,params,evalue_method,\
                    bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save=False,savedirbase=""):
    #bool_forward=True represents appending a new feature from features_pool
    if bool_forward:
        feature_number=len(features_pool)
        evalues=np.zeros(feature_number)
        P=Pool(n_parallels)
        results_parallel=[]
        manager=Manager()
        CPIDs=manager.list()
        for i in range(feature_number):
            appending_feature=features_pool[i]
            feature_names=copy.deepcopy(feature_selected)
            feature_names.append(features_pool[i])
            
            multiclassFolderName="Multiclass_XGBoost_"+multiclassmethod+"_Model"
            savedir=savedirbase+os.sep+multiclassFolderName
            #stratified cross validation to evaluate features
            results_parallel.append(P.apply_async(evalFeature,(CPIDs,appending_feature,TrainDataSet,ValidDataSet,VegeTypes,feature_names,multiclassmethod,params,evalue_method,\
                                                                   bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save,savedir)))
        P.close()
        P.join()
        del CPIDs
        for i in range(feature_number):
            temp=results_parallel[i]
            evalues[i]=temp.get()
    else:
    #bool_forward=False represents deleting a selected feature from feature_selected
        P=Pool(n_parallels)
        results_parallel=[]
        manager=Manager()
        CPIDs=manager.list()
        feature_number=len(feature_selected)
        evalues=np.zeros(feature_number)        
        for i in range(feature_number):
            deleting_feature=feature_selected[i]
            feature_names_delete=copy.deepcopy(feature_selected)
            feature_names_delete.remove(feature_selected[i])
            
            #stratified cross validation to evaluate features
            multiclassFolderName="Multiclass_XGBoost_"+multiclassmethod+"_Model"
            savedir=savedirbase+os.sep+multiclassFolderName
            results_parallel.append(P.apply_async(evalFeature,(CPIDs,deleting_feature,TrainDataSet,ValidDataSet,VegeTypes,feature_names_delete,multiclassmethod,params,evalue_method,\
                                                                   bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save,savedir)))
        P.close()
        P.join()
        del CPIDs
        for i in range(feature_number):
            evalues[i]=results_parallel[i].get()           
    return evalues

def assemFinalFeatureSet(dirto,Feature_Metrics_Values,top_percent=0.95):
    #Find feature metric local maximum
    localmax_ids=[]
    for i in range(1,len(Feature_Metrics_Values)-1):
        if Feature_Metrics_Values[i-1]<Feature_Metrics_Values[i] and Feature_Metrics_Values[i+1]<Feature_Metrics_Values[i]:
            localmax_ids.append(i)
    localmax_ids=np.array(localmax_ids)
    #Find the top num_iter feature set and aggregate the found sets
    fm_sort_indexes=np.argsort(-Feature_Metrics_Values)
    overmax_id=fm_sort_indexes[0]
    overmax_value=np.max(Feature_Metrics_Values)
    Select_Variables=[]
    for i in range(len(localmax_ids)):
        n_iter=localmax_ids[i]
        if Feature_Metrics_Values[n_iter]>top_percent*overmax_value and n_iter<=overmax_id:
            EvalueFolder=dirto+os.sep+"SelectVariables_run"+str(n_iter)
            featureScoreFiledirto=EvalueFolder+os.sep+"Feature_Scores.csv"
            variables_runtime=init.getListFrompdDataSet(init.readCSVasPandas(featureScoreFiledirto),"VariableName")
            for varname in variables_runtime:
                Select_Variables.append(varname)
    Final_Feature_Set=sorted(list(set(Select_Variables)))
    return Final_Feature_Set