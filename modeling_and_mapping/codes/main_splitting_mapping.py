# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:07:27 2018

@author: Heng Zhang
"""

#Please CITE the following article when using the codes

#H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
#emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

#Required Packages: numpy, pandas, imbalanced-learn, scikit-learn, xgboost, gdal
#Splitting Mapping: according to Hierarchical Vegetation Classification System (HVCS)


import os
import numpy as np

import src.xgb_functions as xgbf
import src.processgeotiff as ptf
import src.initialize as init
import src.hiermapping as hmap
import src.ploting as plot
        
if __name__=="__main__":
    #Set workspace and folder names
    root=r"XXX"                             #please change to your local computer workspace (the parent directory of 'codes' folder)
    productName="DzB_VegeMap"
    variableFolderName="DzB_Features"
    mapResultFolderName="Splitting_Mapping_Vege_Types_01"
    dataFolderName="data"+os.sep+productName
    resultFolderName="result"+os.sep+productName
    modelFolderName="models"+os.sep+"Splitting_Hierarchical_Mapping_Models"
    
    dirfrom=root+os.sep+dataFolderName
    dirto=root+os.sep+resultFolderName+os.sep+mapResultFolderName
    variableFolderdir=root+os.sep+dataFolderName+os.sep+variableFolderName
    #Set train and test set files, extensive feature pool file, and vegetation class name file
    trainDataSetName="DzB_vegedata_train.csv"
    testDataSetName="DzB_vegedata_test.csv"
    selectVariableName="Variable_ID_all.csv"
    hierRelationsName="VegeClass_HVCS_DzB.csv"  #HVCS file
    postfix='.tif'
    
    #Set base map layer directories
    baseMapLayerFolderdir=r"XXX"
    baseMapLayerFiledir=baseMapLayerFolderdir+os.sep+"VegeMap_XGBoost_multiclass_softmax.tif"
    baseMapTestResultFiledir=baseMapLayerFolderdir+os.sep+"Real_and_Predicted_Results.csv"
    baseMapUncertFiledir=baseMapLayerFolderdir+os.sep+"model_uncertainty.tif"
    #Set class headers
    bool_strclass=True
    labelHeaderName_H="sub"                 #upper system of HVCS
    labelHeaderName_L="assoc"               #lower system of HVCS

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
    max_depth=8                             #recommended a large number (>6) if the targeted vegetation classes have fewer observations
    lamb=1.25
    alpha=1.5
    gamma=0.0
    subsample=1
    colsample_bytree=1
    min_child_weight=1
    scale_pos_weight=1
    max_delta_step=1  
    eta=0.05
  
    #Set bagging framework parameters
    bool_autolabel=True                     #whether to use K-Means clustering to group variables (use stratified column sampling in bagging framework)
    n_varlabels=5                           #num. of variable clusters
    colsamplerate=0.7                       #column sample rate
    runtimes=300                            #num. of base models
    train_percent=0.7                       #row sample rate
    varlabelweights=[-1]                    #weights of the variable clusters. [-1] is default, indicating no difference
    baggingmetric='accuarcy'                #metric for ensembling the built base models (weighted voting)
    baggingweightindex=1                    #weight index for ensembling the built base models (baggingmetric^baggingweightindex as weight of each base model)
    baggingmetricthres=0.6                  #the threshold to filter out model performance < baggingmetricthres
    

#%%
    #Read datasets
    trainDataSetFiledir=dirfrom+os.sep+trainDataSetName
    testDataSetFiledir=dirfrom+os.sep+testDataSetName
    selectVariableFiledir=dirfrom+os.sep+selectVariableName
    TrainDataSet=init.readCSVasPandas(trainDataSetFiledir)
    TestDataSet=init.readCSVasPandas(testDataSetFiledir)
    varnames=init.getListFromPandas(selectVariableFiledir,'VariableName')
    varlabels=init.getListFromPandas(selectVariableFiledir,'VariableClass')
    
    #Read and format HVCS
    HierRelationsFiledir=dirfrom+os.sep+hierRelationsName
    baseMapTestResult=init.getListFromPandas(baseMapTestResultFiledir,'predict')
    [VegeTypes1,VegeTypes2,HierRelations]=hmap.getHierRelation(init.getListFromPandas(HierRelationsFiledir,labelHeaderName_H),\
                                                init.getListFromPandas(HierRelationsFiledir,labelHeaderName_L))
    

    num_class=len(VegeTypes1)               #This parameter will be changed in model training process
    #Set XGBoost parameters
    params=xgbf.setParams(bool_gpu,tree_method,num_class,eval_metric,max_depth,lamb,alpha,gamma,subsample,colsample_bytree,\
                          min_child_weight,scale_pos_weight,eta,nthread,max_delta_step=max_delta_step)
#%%
    #Processing splitting mapping
    #Establish models
    hiermappingfolder=root+os.sep+modelFolderName
    [pred_Y,test_Y]=hmap.estHierMulticlassModel(TrainDataSet,TestDataSet,baseMapTestResult,VegeTypes1,VegeTypes2,HierRelations,labelHeaderName_H,labelHeaderName_L,varnames,params,multiclassmethod,n_gpus,n_parallels,bool_parallel,\
                             baggingmetric,baggingweightindex,baggingmetricthres,varlabelweights,colsamplerate,train_percent,runtimes,\
                             bool_autolabel,varlabels,n_varlabels,bool_weight,bool_smote,bool_strclass,bool_save,hiermappingfolder)
#%%    
    #Evaluate results
    evalueFolder=dirto
    xgbf.mlcEvalAndWriteResult(evalueFolder,pred_Y,np.zeros_like(pred_Y),test_Y)

    #Plot confusion matrix
    plotfiledirto=evalueFolder+os.sep+"conf_mat.png"
    plot.plot_confusion_matrix(xgbf.Evaluate(test_Y,pred_Y,np.zeros_like(pred_Y),'confmat'),VegeTypes2,title='Confusion Matrix',cmap=None,normalize=False,\
                          figsize=(8, 6),fontsize=6,labelsize=6,savedir=plotfiledirto)    

#%%   
    #Predict mapping results
    print("Predict region...")
    nanDefault=-9999
    [baseMapLayer,Driver,GeoTransform,Proj,nrow,ncol]=ptf.readTiffAsNumpy([baseMapLayerFiledir])
    baseMapLayer=baseMapLayer[:,:,0].astype(np.int32)    
    [baseMapUncert,Driver,GeoTransform,Proj,nrow,ncol]=ptf.readTiffAsNumpy([baseMapUncertFiledir])
    baseMapUncert=baseMapUncert[:,:,0]
    [TiffList,Total]=init.generateVarialeTiffList(variableFolderdir,varnames,postfix)
    [MatX,Driver,GeoTransform,Proj,nrow,ncol]=ptf.readTiffAsNumpy(TiffList)

    [pred_Y,pred_pY]=hmap.predictHierDownMapping(baseMapLayer,VegeTypes1,VegeTypes2,HierRelations,MatX,nrow,ncol,varnames,params,\
                                                    n_gpus,n_parallels,bool_parallel,multiclassmethod,runtimes,bool_save,hiermappingfolder)
    #Write probability map of each vegetation class
    for i in range(len(VegeTypes2)):
        vtname=VegeTypes2[i]
        ProductFolder=dirto+os.sep+vtname
        if not os.path.exists(ProductFolder):
            os.makedirs(ProductFolder)
        Filename1=vtname+"_xgb_"+multiclassmethod+postfix
        ProductFiledirto1=ProductFolder+os.sep+Filename1 
        ptf.writeNumpyToTiff(pred_pY[:,:,i],Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto1,datatype='Float32')
    #Write mapping result map
    Filename2="Splitting_Mapping_multiclass_"+multiclassmethod+postfix
    ProductFolder=dirto
    if not os.path.exists(ProductFolder):
        os.makedirs(ProductFolder)
    ProductFiledirto2=ProductFolder+os.sep+Filename2
    ptf.writeNumpyToTiff(pred_Y,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto2,datatype='Int16')
    
    #Calculate uncertainty
    pred_uncert=1-(1-baseMapUncert)*(1-xgbf.calcUncertainty(pred_pY))
    Filename3="model_uncertainty"+postfix
    ProductFolder=dirto
    ProductFiledirto3=ProductFolder+os.sep+Filename3
    ptf.writeNumpyToTiff(pred_uncert,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto3,datatype='Float32')    
    print("ALL DONE.\n")       
