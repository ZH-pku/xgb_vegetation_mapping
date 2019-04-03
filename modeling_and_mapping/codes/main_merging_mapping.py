# -*- coding: utf-8 -*-
"""
Created on Fri May 25 02:37:50 2018

@author: Heng Zhang
"""

#Please CITE the following article when using the codes

#H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
#emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

#Required Packages: numpy, pandas, imbalanced-learn, scikit-learn, xgboost, gdal
#Merging Mapping: according to Hierarchical Vegetation Classification System (HVCS)

import os
import numpy as np

import src.xgb_functions as xgbf
from src import processgeotiff as ptf
from src import initialize as init
from src import hiermapping as hmap
import src.ploting as plot


if __name__=="__main__":
    #Set workspace and folder names
    root=r"XXX"                             #please change to your local computer workspace (the parent directory of 'codes' folder)
    productName="DzB_VegeMap"
    mapResultFolderName="Merging_Mapping_Vege_Types_01"
    dataFolderName="data"+os.sep+productName
    resultFolderName="result"+os.sep+productName
    
    dirfrom=root+os.sep+dataFolderName
    dirto=root+os.sep+resultFolderName+os.sep+mapResultFolderName

    hierRelationsName="VegeClass_HVCS_DzB.csv"  #HVCS file
    postfix='.tif'

    #Set class headers
    labelHeaderName_H="type"                #upper system of HVCS
    labelHeaderName_L="sub"                 #lower system of HVCS
    
    #Set base map layer directories
    baseMapLayerFolderdir=r"XXX"
    baseMapLayerFileName="VegeMap_XGB_BAG_softmax.tif"
    baseMapLayerFiledir=baseMapLayerFolderdir+os.sep+baseMapLayerFileName
    baseMapTestResultFileName="Real_and_Predicted_Results.csv"
    baseMapTestResultFiledir=baseMapLayerFolderdir+os.sep+baseMapTestResultFileName

    #Read and format HVCS
    HierRelationsFiledir=dirfrom+os.sep+hierRelationsName
    baseMapTestResult=init.getListFromPandas(baseMapTestResultFiledir,'predict')
    realTestY=init.getListFromPandas(baseMapTestResultFiledir,'real')
    [VegeTypes1,VegeTypes2,HierRelations]=hmap.getHierRelation(init.getListFromPandas(HierRelationsFiledir,labelHeaderName_H),\
                                                init.getListFromPandas(HierRelationsFiledir,labelHeaderName_L))
#%%
    #Produce merged predicted test set
    pred_Y=hmap.predictHierUpMapping(baseMapTestResult,VegeTypes1,VegeTypes2,HierRelations)
    test_Y=hmap.predictHierUpMapping(realTestY,VegeTypes1,VegeTypes2,HierRelations)
    #Evaluate
    EvalueFolder=dirto
    xgbf.mlcEvalAndWriteResult(EvalueFolder,pred_Y,np.zeros_like(pred_Y),test_Y)

    #Plot confusion matrix
    plotfiledirto=EvalueFolder+os.sep+"conf_mat.png"
    chsize=5
    plot.plot_confusion_matrix(xgbf.Evaluate(test_Y,pred_Y,np.zeros_like(pred_Y),'confmat'),VegeTypes1,title='Confusion Matrix',cmap=None,normalize=False,\
                          figsize=(8, 6),fontsize=chsize,labelsize=chsize,savedir=plotfiledirto)        
    
#%%
    #Produce merged map
    print("Predict region...")
    nanDefault=-9999
    [baseMapLayer,Driver,GeoTransform,Proj,nrow,ncol]=ptf.readTiffAsNumpy([baseMapLayerFiledir])
    baseMapLayer=baseMapLayer[:,:,0].astype(np.int32)    
    
    pred_Y=hmap.predictHierUpMapping(baseMapLayer,VegeTypes1,VegeTypes2,HierRelations)
    
    #Write mapping results
    Filename2="Merging_Mapping_result"+postfix
    ProductFolder=dirto
    if not os.path.exists(ProductFolder):
        os.makedirs(ProductFolder)
    ProductFiledirto2=ProductFolder+os.sep+Filename2
    ptf.writeNumpyToTiff(pred_Y,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,ProductFiledirto2,datatype='Int16')
    print("ALL DONE.")
    