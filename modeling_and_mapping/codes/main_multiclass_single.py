#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:05:47 2018

@author: Heng Zhang
"""

#Please CITE the following article when using the codes

#H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
#emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

#Required Packages: numpy, pandas, imbalanced-learn, scikit-learn, xgboost, gdal
#Vegetation Mapping Using a Single XGBoost Model


import os
import xgboost as xgb

import src.xgb_functions as xgbf
import src.processgeotiff as ptf
import src.initialize as init
import src.multiclass as mlc
import src.smote as smote
import src.ploting as plot

    
if __name__=="__main__":
    #set workspace and folder names
    root=r"XXX"                             #please change to your local computer workspace (the parent directory of 'codes' folder).
    productName="NZ_VegeMap"
    variableFolderName="NZ_Features"
    mapResultFolderName="MLC_XGB_SIG_Habitat"
    dataFolderName="data"+os.sep+productName
    resultFolderName="result"+os.sep+productName
    modelFolderName="models"                #the folder to store the built XGB models temporarily 
    
    dirfrom=root+os.sep+dataFolderName
    dirto=root+os.sep+resultFolderName+os.sep+mapResultFolderName
    variableFolderdir=root+os.sep+dataFolderName+os.sep+variableFolderName
    #Set train and test set files, extensive feature pool file, and vegetation class name file
    trainDataSetName="NZ_vegedata_train.csv"
    testDataSetName="NZ_vegedata_test.csv"
    selectVariableName="Variable_ID_all.csv"
    vegetypeNames="VegeTypeNames_Habitat.csv"  #vegetation class name file
    postfix='.tif'                          #postfix of feature raster files in variable folder
    
    #Set train and test set parameters
    bool_strclass=True                      #whether the label is string (True) or number (False, have to start with 0 without vacancy)
    labelHeaderName="Habitat"               #header of targeted classes in train and test set (pandas dataframe)
    
    #Set processing parameters
    bool_smote=True                         #whether to use SMOTE for a balanced train set
    bool_weight=False                       #equal to bool_balance in single model when using 'category' method
    bool_save=False                         #whether to save built base models
    bool_gpu=True                           #whether to use GPU computing

    #Set XGBoost parameters (see XGBoost online document: https://xgboost.readthedocs.io/en/latest/)  
    tree_method=1                           #0: gbtree logistic; 1: softmax multiclass; 2: dart tree logistic; 3: dart tree multiclass (see xgb_functions.py)
                                            #when using multi-category method, please set it to 0
    multiclassmethod='softmax'              #multiclassifier choices: 'softmax' or 'category'
                                            #   'softmax' uses xgboost inner softmax function as the multiclass classifier
                                            #   'category' converts a multi-classification problem to a binary classification problem (see src.multiclass.py file)
    eval_metric='merror'                    #loss function in xgboost
    max_depth=6
    lamb=1.25
    alpha=1.25
    gamma=0.0
    subsample=0.52
    colsample_bytree=0.75
    min_child_weight=2
    scale_pos_weight=1
    max_delta_step=2    
    eta=0.05
    nthread=1
    threshold=0.5
#%%
    #Read datasets
    trainDataSetFiledir=dirfrom+os.sep+trainDataSetName
    testDataSetFiledir=dirfrom+os.sep+testDataSetName
    selectVariableFiledir=dirfrom+os.sep+selectVariableName
    vegetypeFiledir=dirfrom+os.sep+vegetypeNames
    TrainDataSet=init.readCSVasPandas(trainDataSetFiledir)
    TestDataSet=init.readCSVasPandas(testDataSetFiledir)
    varnames=init.getListFromPandas(selectVariableFiledir,'VariableName')
    varmeanings=init.getListFromPandas(selectVariableFiledir,'VariableMeaning')
    VegeTypes=init.getListFromPandas(vegetypeFiledir,'VegeName')   

    num_class=len(VegeTypes)
    #Set XGBoost parameters
    params=xgbf.setParams(bool_gpu,tree_method,num_class,eval_metric,max_depth,lamb,alpha,gamma,subsample,colsample_bytree,\
                          min_child_weight,scale_pos_weight,eta,nthread,max_delta_step=max_delta_step,gpu_id=0)
#%%
    #SMOTE for balanced dataset
    #tar_ratio is max(num. of classes)/min(num. of classes). -1 represents full balance, recommended here.
    if bool_smote:        
        TrainDataSet=smote.createSMOTEDataSet(TrainDataSet,VegeTypes,varnames,method='regular',tar_ratio=-1,\
                                              bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)
#%%
    #Train model
    multiclassFolderName="Multiclass_XGBoost_"+multiclassmethod+"_Model"
    savedir=root+os.sep+modelFolderName+os.sep+multiclassFolderName
    print("Start training model...  method: %s"%multiclassmethod)
    if multiclassmethod=='softmax':
        ModelList=mlc.trainMulticlassSoftmaxModel(TrainDataSet,VegeTypes,varnames,params,bool_weight=bool_weight,\
                                                  bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY,test_Y]=mlc.testMulticlassSoftmaxModel(ModelList,TestDataSet,VegeTypes,varnames,params,\
                                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
    elif multiclassmethod=='category':
        ModelList=mlc.trainMulticlassCategoryModel(TrainDataSet,VegeTypes,varnames,params,bool_weight=bool_weight,\
                                                   bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY,test_Y]=mlc.testMulticlassCategoryModel(ModelList,TestDataSet,VegeTypes,varnames,params,\
                                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_save=bool_save,savedir=savedir)
    else:
        print("Invalid Multiclass Method Input!")
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
    multiclassFolderName="Multiclass_XGBoost_"+multiclassmethod+"_Model"
    savedir=root+os.sep+modelFolderName+os.sep+multiclassFolderName
    if multiclassmethod=='softmax':
        pred_X=init.fomatMulticlassSoftmaxMatrix(MatX)
        pred_pY=mlc.predictMulticlassSoftmaxModelCvted(ModelList,pred_X,params,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY,nrow,ncol,num_class,bool_onearray=False)
    elif multiclassmethod=='category':
        pred_X=init.formatMulticlassCategoryMatrix(MatX,num_class)
        pred_pY=mlc.predictMulticlassCategoryModelCvted(ModelList,pred_X,params,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY,nrow,ncol,num_class,bool_onearray=True)
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
    Filename2="VegeMap_XGB_SIG_"+multiclassmethod+postfix
    ProductFolder=dirto
    ProductFiledirto2=ProductFolder+os.sep+Filename2
    ptf.writeNumpyToTiff(pred_Y,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto2,datatype='Int16')

    #Plot XGBoost inner feature importance scores
    model=ModelList[0]
    xgb.plot_importance(model)

    #Calculate uncertainty
    pred_uncert=xgbf.calcUncertainty(pred_pY)
    Filename3="model_uncertainty"+postfix
    ProductFiledirto3=ProductFolder+os.sep+Filename3
    ptf.writeNumpyToTiff(pred_uncert,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto3,datatype='Float32')  
    print("ALL DONE.\n")