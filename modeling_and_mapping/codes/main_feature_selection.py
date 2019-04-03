#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:35:40 2018

@author: Heng Zhang
"""

#Please CITE the following article when using the codes

#H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
#emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

#Required Packages: numpy, pandas, imbalanced-learn, scikit-learn, xgboost, gdal
#Feature Selection: revised Improved Forward Floating Selection (revised IFFS)

import os
import copy
import numpy as np

import src.featureselection as fs
import src.xgb_functions as xgbf
import src.initialize as init
import src.smote as smote


#%%
if __name__=="__main__":
    #Set workspace and folder names
    root=r"XXX"                             #please change to your local computer workspace (the parent directory of 'codes' folder)
    productName="NZ_VegeMap"
    variableFolderName="NZ_Features"
    mapResultFolderName="RIFFS_VEG_CV_01"
    dataFolderName="data"+os.sep+productName
    resultFolderName="result"+os.sep+productName
    modelFolderName="models"                #folder to store the built XGB models temporarily
    
    dirfrom=root+os.sep+dataFolderName
    dirto=root+os.sep+resultFolderName+os.sep+mapResultFolderName
    variableFolderdir=root+os.sep+dataFolderName+os.sep+variableFolderName
    #Set train and test set files, extensive feature pool file, and vegetation class name file
    trainDataSetName="NZ_vegedata_train.csv"
    validDataSetName="NZ_vegedata_test.csv"
    variableIDName="Variable_ID_all.csv"
    vegetypeNames="VegeTypeNames_Habitat.csv"  #vegetation class name file
    postfix='.tif'                          #postfix of feature raster files in variable folder
    
    #Set train and test set parameters
    bool_strclass=True                      #whether the label is string (True) or number (False, have to start with 0 without vacancy)
    labelHeaderName="Habitat"               #header of targeted classes in train and test set (pandas dataframe)
    
    #Set processing parameters
    bool_smote=True                         #whether to use SMOTE for a balanced train set
    bool_weight=False                       #equal to bool_balance in single model when using 'category' method
    bool_save=False                         #whether to save built base models (recommend False in feature selection)
    bool_gpu=False                          #whether to use GPU computing
    
    n_parallels=24                          #num. of parallel processes, equals to num. of cores of CPUs (CPU computing) or not exceed memory limitation (GPU computing) recommended.
    n_gpus=2                                #num. of GPU devices.
    nthread=1

    #Set XGBoost parameters (see XGBoost online document: https://xgboost.readthedocs.io/en/latest/)     
    tree_method=1                           #0: gbtree logistic; 1: softmax multiclass; 2: dart tree logistic; 3: dart tree multiclass (see xgb_functions.py)
                                            #when using multi-category method, please set it to 0
    multiclassmethod='softmax'              #multiclassifier choices: 'softmax' or 'category'
                                            #   'softmax' uses xgboost inner softmax function as the multiclass classifier
                                            #   'category' converts a multi-classification problem to a binary classification problem (see src.multiclass.py file)
    eval_metric='merror' #mlogloss          #loss function in xgboost
    max_depth=6
    lamb=1.5
    alpha=1.5
    gamma=0.0005
    subsample=1
    colsample_bytree=1
    min_child_weight=1
    scale_pos_weight=1
    max_delta_step=1
    eta=0.7
    threshold=0.5
    
    #Set revised IFFS parameters
    rm_eq_thres=0.98                        #features with correlations>rm_eq_thres are seen as identical features which will not be repeatedly added to the feature pool.
    bool_cv=True                            #whether to use k-fold cross validation on train set to evaluate a feature. (if False, use validDataSet instead.)
    bool_predictmap=False                   #whether to produce a map in each cycle. (False recommended)
    n_max_iter=360                          #the program will stop if feature selection cycle exceeds n_max_iter.
    min_evalue_gain=0.0                     #
    max_backtrack_times=70                  #max times for evalue gain less than 0
    rm_itvl=5                               #remove trival features in the selected set after every rm_itvl runtimes 
                                            #(evaluation calculated by xgboost feature importance)
    cv_num=1                                #repeat times of k-fold cross validation
    skf_split=10                            #k-fold (stratified)
    evalue_method='kappa'                   #evaluation matric
#%%
    #Read datasets
    trainDataSetFiledir=dirfrom+os.sep+trainDataSetName
    validDataSetFiledir=dirfrom+os.sep+validDataSetName
    variableIDFiledir=dirfrom+os.sep+variableIDName
    vegetypeFiledir=dirfrom+os.sep+vegetypeNames
    TrainDataSet=init.readCSVasPandas(trainDataSetFiledir)
    ValidDataSet=init.readCSVasPandas(validDataSetFiledir)
    varnames=init.getListFromPandas(variableIDFiledir,'VariableName')
    VegeTypes=init.getListFromPandas(vegetypeFiledir,'VegeName')   

    num_class=len(VegeTypes)
    #Set XGBoost parameters
    params=xgbf.setParams(bool_gpu,tree_method,num_class,eval_metric,max_depth,lamb,alpha,gamma,subsample,colsample_bytree,\
                  min_child_weight,scale_pos_weight,eta,nthread,max_delta_step=max_delta_step,gpu_id=1)
    
#%%
    #Remove Identical Features
    features_included=fs.removeIdenticalFeatures(TrainDataSet,varnames,rm_thres=rm_eq_thres)
    print("%d features remained.\n"%len(features_included))
#%%
    #SMOTE for balanced dataset
    #tar_ratio is max(num. of classes)/min(num. of classes). A large number recommended if using stratified k-fold cross validation on train set (bool_cv=True)
    if bool_smote:
        TrainDataSet=smote.createSMOTEDataSet(TrainDataSet,VegeTypes,varnames,method='regular',tar_ratio=50,\
                                              bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)

#%%
    #Initialize
    n_iter=0
    backtracktimes=0
    evalue_selected=0.0
    evalue_gain=9999
    n_iter_final=0
    max_evalue_final=0.0
    max_FMS_Id=-1
    
    Variables_Selected=[]
    Variables_Selected_Final=[]
    Feature_Metrics_Values=np.zeros(n_max_iter)
    flag_backtrack=False
    savedirbase=root+os.sep+modelFolderName
    #%%
    while n_iter<n_max_iter and backtracktimes<max_backtrack_times:
        #Find if stop selecting loop (whether MAX ID (max_FMS_Id) in sequence changes)
        if n_iter%25==0:
            max_id=np.argmax(Feature_Metrics_Values)
            if max_id==max_FMS_Id:
                break           
            else:
                max_FMS_Id=max_id
        #Continue appending features
        if not flag_backtrack:
            print("Starting including iteration: %d"%n_iter)
            EvalueFolder=dirto+os.sep+"SelectVariables_run"+str(n_iter)
            if not os.path.exists(EvalueFolder):
                os.makedirs(EvalueFolder)
            #Generate current feature pool
            features_pool=copy.deepcopy(features_included)
            for varselect in Variables_Selected:
                print("feature selected: %s"%varselect)
                features_pool.remove(varselect)  
            #Search for best appending feature
            evalues=fs.evalFeature_parallel(True,Variables_Selected,features_pool,TrainDataSet,ValidDataSet,VegeTypes,multiclassmethod,params,evalue_method,\
                    bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)
            #Write all feature pool evaluation values
            partialevaluesFiledirto=EvalueFolder+os.sep+"Feature_Partial_Evalues.csv"
            init.writeArrayListToCSV([features_pool,evalues],['VariableName','PartialEvalues'],partialevaluesFiledirto)   
            
            best_feature_idx=np.argmax(evalues)
            best_feature_name=features_pool[best_feature_idx]
#            feature_names=copy.deepcopy(Variables_Selected)
#            feature_names.append(best_feature_name) 
            evalue_iter=evalues[best_feature_idx]
            evalue_gain=evalue_iter-evalue_selected
            print("Best feature: %s, Evalue Gain = %f\n"%(best_feature_name,evalue_gain))
            Variables_Selected.append(best_feature_name)
            evalue_selected=copy.deepcopy(evalue_iter)
            Feature_Metrics_Values[n_iter]=evalue_iter
            
            #Re-run XGBoost model on the current best feature set for evaluation results on ValidDataSet
            #Establish model and predict the results
            fscores=fs.estabModelAndPred(TrainDataSet,ValidDataSet,VegeTypes,Variables_Selected,multiclassmethod,params,evalue_method,EvalueFolder,variableFolderdir,postfix,\
                           bool_predictmap,bool_weight,bool_strclass,labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)
            #Update the final selected feature set
            if evalue_gain>min_evalue_gain:
                if evalue_selected>max_evalue_final:
                    n_iter_final=copy.deepcopy(n_iter)
                    max_evalue_final=copy.deepcopy(evalue_selected)
                    Variables_Selected_Final=copy.deepcopy(Variables_Selected)
            n_iter=n_iter+1
            #Add L remove r search to drop trival feature in XGBoost
            if n_iter%rm_itvl==rm_itvl-1:
                [Variables_Selected,fscores,trivial_feature_name]=fs.removeTrivialFeature(Variables_Selected,fscores)
                print("Iteration: %d   feature %s is removed!"%(n_iter,trivial_feature_name))
        #Backtracking step
        print("Backtracking Step: searching for the least significant feature to delete...     backtracktimes = %d\n"%backtracktimes)
        #Search in feature_names for feature to delete
        if len(Variables_Selected)<2:
            evalues_delete=[0.0]
        else:
            evalues_delete=fs.evalFeature_parallel(False,Variables_Selected,[],TrainDataSet,ValidDataSet,VegeTypes,multiclassmethod,params,evalue_method,\
                    bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)           

        feature_delete_idx=np.argmax(evalues_delete)             #delete the least significant feature (SFFS)
#            feature_delete_idx=np.argmin(evalues_delete)             #delete the most significant feature (SFFS)
        print("This evaluation = %f, Last evaluation = %f\n"%(evalues_delete[feature_delete_idx],evalue_iter))
        if evalues_delete[feature_delete_idx]>=evalue_iter:
            backtracktimes=backtracktimes+1
            feature_delete_name=Variables_Selected[feature_delete_idx]
            Variables_Selected.remove(feature_delete_name)
            fscores=np.delete(fscores,feature_delete_idx,axis=0)
            evalue_iter=evalues_delete[feature_delete_idx]
            flag_backtrack=True
            print("Feature %s of the last run is removed!\n"%(feature_delete_name))
        else:
            evalue_selected=evalue_iter
            flag_backtrack=False                  
            print("No need to backtrack! Continue...\n")
        #Replacing step
        if not flag_backtrack:
            print("Replacing Step: delete the most significant feature and add a new one...\n")
    #                feature_delete_idx=np.argmax(evalues_delete)             #delete the least significant feature (SFFS)
            feature_delete_idx=np.argmin(evalues_delete)             #delete the most significant feature (SFFS)
            feature_delete_name=Variables_Selected[feature_delete_idx]
            
            Variables_Selected_Temp=copy.deepcopy(Variables_Selected)
            Variables_Selected_Temp.remove(feature_delete_name)
            fscores_temp=copy.deepcopy(fscores)
            fscores=np.delete(fscores_temp,feature_delete_idx,axis=0)          
            features_pool_temp=copy.deepcopy(features_included)
            for varselect in Variables_Selected_Temp:
                print("feature selected: %s"%varselect)
                features_pool_temp.remove(varselect)                  
            evalues=fs.evalFeature_parallel(True,Variables_Selected_Temp,features_pool_temp,TrainDataSet,ValidDataSet,VegeTypes,multiclassmethod,params,evalue_method,\
                    bool_cv,cv_num,skf_split,bool_gpu,n_gpus,n_parallels,bool_weight,bool_strclass,labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)
            best_feature_idx=np.argmax(evalues)
            best_feature_name=features_pool_temp[best_feature_idx]
            evalue_gain_temp=evalues[best_feature_idx]-evalue_selected            
            Variables_Selected_Temp.append(best_feature_name)
            if evalue_gain_temp>0:
                print("Feature %s of the last run is removed!\n"%(feature_delete_name))
                print("Best new feature: %s, Evalue Gain = %f\n"%(best_feature_name,evalue_gain_temp))
                backtracktimes=backtracktimes+1
                Variables_Selected=copy.deepcopy(Variables_Selected_Temp)
                evalue_selected=copy.deepcopy(evalue_iter)
                #Update the Final Variable Set
                if evalue_gain>min_evalue_gain:
                    if evalue_selected>max_evalue_final:
                        n_iter_final=copy.deepcopy(n_iter)
                        max_evalue_final=copy.deepcopy(evalue_selected)
                        Variables_Selected_Final=copy.deepcopy(Variables_Selected)
            else:
                print("No gain in replacing feature.          backtracktimes = %d\n"%backtracktimes)

            
#%%     
    #Final feature set    
    print("\nFeature Selecting Finished!\nMaximum: final iteration times = %d, Evalue = %f"%(n_iter_final,max_evalue_final))
    for varselect in Variables_Selected_Final:
        print("feature selected: %s"%varselect)      
    print("\n")
    #Write final results
    bestfeaturesetFiledir=dirto+os.sep+"Maximum_Evalue_Feature_Set.csv"
    init.writeArrayListToCSV([Variables_Selected_Final],['VariableName'],bestfeaturesetFiledir)
    runtimefeaturemetricsFiledir=dirto+os.sep+"Runtime_Feature_Metrics_Values.csv"
    runtime_ids=np.linspace(0,n_max_iter-1,n_max_iter).astype(np.int32)
    init.writeArrayListToCSV([runtime_ids,Feature_Metrics_Values],["Runtime","Feature_Metrics_Value"],runtimefeaturemetricsFiledir)
    Final_Feature_Set=fs.assemFinalFeatureSet(dirto,Feature_Metrics_Values,top_percent=0.85)
    finalfeaturesetFiledir=dirto+os.sep+"Final_Feature_Set.csv"
    init.writeArrayListToCSV([Final_Feature_Set],['VariableName'],finalfeaturesetFiledir)
    print("Final Feature Set:")
    for varselect in Final_Feature_Set:
        print("final feature: %s"%varselect)
    print("\n")
    print("ALL DONE.")