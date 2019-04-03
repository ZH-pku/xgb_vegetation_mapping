#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 01:04:51 2018

@author: Heng Zhang
"""

#Please CITE the following article when using the codes

#H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
#emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

#Required Packages: numpy, pandas, imbalanced-learn, scikit-learn, xgboost, gdal
#Vegetation Mapping Using Bagging Ensemble Framework


import os

import src.xgb_functions as xgbf
import src.processgeotiff as ptf
import src.initialize as init
import src.multiclassbagging as mbag
import src.smote as smote
import src.ploting as plot


if __name__=="__main__":
    #Set workspace and folder names
    root=r"XXX"                             #please change to your local computer workspace (the parent directory of 'codes' folder).
    productName="NZ_VegeMap"
    variableFolderName="NZ_Features"
    mapResultFolderName="MLC_XGB_BAG_Habitat_01"
    dataFolderName="data"+os.sep+productName
    resultFolderName="result"+os.sep+productName
    modelFolderName="models"                #the folder to store the built XGB models temporarily 
    
    dirfrom=root+os.sep+dataFolderName
    dirto=root+os.sep+resultFolderName+os.sep+mapResultFolderName
    variableFolderdir=root+os.sep+dataFolderName+os.sep+variableFolderName
    #Set train and test set files, extensive feature pool file, and vegetation class name file
    trainDataSetName="NZ_vegedata_train.csv"
    testDataSetName="NZ_vegedata_test.csv"
    selectVariableName="Variable_ID_rmEnv_Habitat.csv"
    vegetypeNames="VegeTypeNames_Habitat.csv"  #vegetation class name file
    postfix='.tif'                          #postfix of feature raster files in variable folder
    
    #Set train and test set parameters
    bool_strclass=True                      #whether the label is string (True) or number (False, have to start with 0 without vacancy)
    labelHeaderName="Habitat"               #header of targeted classes in train and test set (pandas dataframe)
    
    #Set processing parameters
    bool_smote=True                         #whether to use SMOTE for a balanced train set
    bool_weight=False                       #equal to bool_balance in single model when using 'category' method
    bool_save=True                          #Please set True in this program
    bool_parallel=True                      #whether to use parallel computing (process based).
    bool_gpu=True                           #whether to use GPU computing
    
    n_parallels=24                          #num. of parallel processes, equals to num. of cores of CPUs (CPU computing) or not exceed memory limitation (GPU computing) recommended.
    n_gpus=1                                #num. of GPU devices.
    nthread=1                               #num. of thread(s) in one parallel process

    #Set XGBoost parameters (see XGBoost online document: https://xgboost.readthedocs.io/en/latest/)    
    tree_method=1                           #0: gbtree logistic; 1: softmax multiclass; 2: dart tree logistic; 3: dart tree multiclass (see xgb_functions.py)
                                            #when using multi-category method, please set it to 0
    multiclassmethod='softmax'              #multiclassifier choices: 'softmax' or 'category'
                                            #   'softmax' uses xgboost inner softmax function as the multiclass classifier
                                            #   'category' converts a multi-classification problem to a binary classification problem (see src.multiclass.py file)
    eval_metric='merror'                    #loss function in xgboost
    max_depth=6
    lamb=1.25
    alpha=1.5
    gamma=0.0
    subsample=1
    colsample_bytree=1
    min_child_weight=2
    scale_pos_weight=1
    max_delta_step=1
    eta=0.25
  
    #Set bagging framework parameters
    bool_autolabel=True                     #whether to use K-Means clustering to group variables (use stratified column sampling in bagging framework)
    n_varlabels=5                           #num. of variable clusters
    colsamplerate=0.7                       #column sample rate
    runtimes=610                            #num. of base models
    train_percent=0.7                       #row sample rate
    varlabelweights=[-1]                    #weights of the variable clusters. [-1] is default, indicating no difference
    baggingmetric='kappa'                   #metric for ensembling the built base models (weighted voting)                
    baggingweightindex=1                    #weight index for ensembling the built base models (baggingmetric^baggingweightindex as weight of each base model)
    baggingmetricthres=0.75                 #the threshold to filter out model performance < baggingmetricthres


#%%    
    #Read datasets
    trainDataSetFiledir=dirfrom+os.sep+trainDataSetName
    testDataSetFiledir=dirfrom+os.sep+testDataSetName
    selectVariableFiledir=dirfrom+os.sep+selectVariableName
    vegetypeFiledir=dirfrom+os.sep+vegetypeNames
    TrainDataSet=init.readCSVasPandas(trainDataSetFiledir)
    TestDataSet=init.readCSVasPandas(testDataSetFiledir)
    varnames=init.getListFromPandas(selectVariableFiledir,'VariableName')
    varlabels=init.getListFromPandas(selectVariableFiledir,'VariableClass')
#    varmeanings=init.getListFromPandas(selectVariableFiledir,'VariableMeaning')
    VegeTypes=init.getListFromPandas(vegetypeFiledir,'VegeName')

    num_class=len(VegeTypes)
    #Set XGBoost parameters
    params=xgbf.setParams(bool_gpu,tree_method,num_class,eval_metric,max_depth,lamb,alpha,gamma,subsample,colsample_bytree,\
                          min_child_weight,scale_pos_weight,eta,nthread,max_delta_step=max_delta_step)
#%%
    #SMOTE for balanced dataset
    #tar_ratio is max(num. of classes)/min(num. of classes). -1 represents full balance, recommended here.
    if bool_smote:
        TrainDataSet=smote.createSMOTEDataSet(TrainDataSet,VegeTypes,varnames,method='regular',tar_ratio=-1,\
                                              bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)
#%%
    #Train model
    multiclassFolderName="Multiclass_XGBoost_Bagging_"+multiclassmethod+"_Models"
    savedirbase=root+os.sep+modelFolderName+os.sep+multiclassFolderName
    if bool_parallel:
        #Train XGBoost bagging ensemble model on train set, using parallel computing (process based).
        ModelList=mbag.trainMulticlassBaggingModel_parallel(TrainDataSet,VegeTypes,varnames,params,multiclassmethod,n_gpus,n_parallels,\
                                                            baggingmetric=baggingmetric,baggingweightindex=baggingweightindex,baggingmetricthres=baggingmetricthres,\
                                                            varlabelweights=varlabelweights,colsamplerate=colsamplerate,train_percent=train_percent,runtimes=runtimes,\
                                                            bool_autolabel=bool_autolabel,varlabels=varlabels,n_varlabels=n_varlabels,bool_weight=bool_weight,\
                                                            bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)
        #Predict on test set
        [pred_Y,pred_pY,test_Y]=mbag.testMulticlassBaggingModel_parallel(TestDataSet,VegeTypes,params,multiclassmethod,n_gpus,n_parallels,\
                                                            runtimes=runtimes,bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,\
                                                            bool_save=bool_save,savedirbase=savedirbase)
    else:
        #Train XGBoost bagging ensemble model on train set
        ModelList=mbag.trainMulticlassBaggingModel(TrainDataSet,VegeTypes,varnames,params,multiclassmethod,\
                                                   baggingmetric=baggingmetric,baggingweightindex=baggingweightindex,baggingmetricthres=baggingmetricthres,\
                                                   varlabelweights=varlabelweights,colsamplerate=colsamplerate,train_percent=train_percent,runtimes=runtimes,\
                                                   bool_autolabel=bool_autolabel,varlabels=varlabels,n_varlabels=n_varlabels,bool_weight=bool_weight,\
                                                   bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)
        #Predict on test set
        [pred_Y,pred_pY,test_Y]=mbag.testMulticlassBaggingModel(TestDataSet,VegeTypes,params,multiclassmethod,runtimes=runtimes,\
                                                    bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedirbase=savedirbase)
#%%
    #Evaluate results
    evalueFolder=dirto
    xgbf.mlcEvalAndWriteResult(evalueFolder,pred_Y,pred_pY,test_Y)

    #Plot confusion matrix
    plotfiledirto=evalueFolder+os.sep+"conf_mat.png"
    plot.plot_confusion_matrix(xgbf.Evaluate(test_Y,pred_Y,pred_pY,'confmat'),VegeTypes,title='Confusion Matrix',cmap=None,normalize=False,\
                          figsize=(8, 6),fontsize=11,labelsize=11,savedir=plotfiledirto)    
#%%
    #Predict mapping results
    print("Predict region...")
    nanDefault=-9999
    [TiffList,Total]=init.generateVarialeTiffList(variableFolderdir,varnames,postfix)
    [MatX,Driver,GeoTransform,Proj,nrow,ncol]=ptf.readTiffAsNumpy(TiffList)
    multiclassFolderName="Multiclass_XGBoost_Bagging_"+multiclassmethod+"_Models"
    savedirbase=root+os.sep+modelFolderName+os.sep+multiclassFolderName
    
    if bool_parallel:
        [pred_Y,pred_pY]=mbag.predictMulticlassBaggingModel_parallel(MatX,nrow,ncol,varnames,num_class,params,multiclassmethod,\
                                           n_gpus,n_parallels,runtimes=runtimes,bool_save=bool_save,savedirbase=savedirbase)
    else:
        [pred_Y,pred_pY]=mbag.predictMulticlassBaggingModel(MatX,nrow,ncol,varnames,num_class,params,\
                        multiclassmethod,runtimes=runtimes,bool_save=bool_save,savedirbase=savedirbase)
    #Write probability map of each vegetation class
    for i in range(len(VegeTypes)):
        vtname=VegeTypes[i]
        ProductFolder=dirto+os.sep+vtname
        if not os.path.exists(ProductFolder):
            os.makedirs(ProductFolder)
        Filename1=vtname+"_xgb_"+multiclassmethod+postfix
        ProductFiledirto1=ProductFolder+os.sep+Filename1 
        ptf.writeNumpyToTiff(pred_pY[:,:,i],Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto1,datatype='Float32')
    #Write mapping result map
    Filename2="VegeMap_XGB_BAG_"+multiclassmethod+postfix
    ProductFolder=dirto
    ProductFiledirto2=ProductFolder+os.sep+Filename2
    ptf.writeNumpyToTiff(pred_Y,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto2,datatype='Int16')

    #Calculate uncertainty
    pred_uncert=xgbf.calcUncertainty(pred_pY)
    Filename3="model_uncertainty"+postfix
    ProductFolder=dirto
    ProductFiledirto3=ProductFolder+os.sep+Filename3
    ptf.writeNumpyToTiff(pred_uncert,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto3,datatype='Float32')    
    print("ALL DONE.\n")
