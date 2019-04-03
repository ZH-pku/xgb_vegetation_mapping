# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 20:13:05 2018

@author: Heng Zhang
"""

#Please CITE the following article when using the codes

#H. Zhang, A. Eziz, J. Xiao, S. Tao, S. Wang, Z. Tang, J. Zhu and J. Fang, 2019. High-resolution Vegetation Mapping Using eXtreme Gradient Boosting Based on Extensive Features. Remote Sensing.(submitted)
#emails: heng.zhang@pku.edu.cn, hengzhang.zhh@gmail.com; anwareziz@pku.edu.cn

#Required Packages: gdal, scikit-fmm, numpy, 
#Calculating the Distance to Waterbody


import os
import skfmm
import numpy as np

import processgeotiff as ptf

#solving eikonal equation via fast marching method (using scikit-fmm package)
def skfmm_Dist(mask,dx=0.1,narrow=30):
    d=skfmm.distance((1-mask),dx=dx,narrow=narrow)
    return d

if __name__=="__main__":
    #set workspace and folder names
    root=r"XXX"                             #please change to your local computer workspace (the parent directory of 'codes' folder).
    productName="NZ_WaterMasks"
    dataFolderName="data"+os.sep+productName
    resultFolderName="result"+os.sep+productName
    
    dirfrom=root+os.sep+dataFolderName
    dirto=root+os.sep+resultFolderName
    if not os.path.exists(dirto):
        os.makedirs(dirto)
    
    #set fresh water mask filename and coastline mask filename
    freshWaterMaskName="freshwatermask.tif"
    coastlineMaskName="coastline.tif"
    freshWaterMaskFiledir=dirfrom+os.sep+freshWaterMaskName
    coastlineMaskFiledir=dirfrom+os.sep+coastlineMaskName    

    #calculate distance to fresh water
    [OriData,Driver,GeoTransform,Proj,nrow,ncol]=ptf.ReadTiffAsNumpy([freshWaterMaskFiledir])
    freshWaterMask=OriData[:,:,0]
    freshWaterMask[freshWaterMask>0]=1
    freshWaterMask[freshWaterMask<=0]=0
    freshWaterMask=freshWaterMask.astype(np.bool)
    del OriData
    print("calculating distance to fresh water...")
    freshWaterDistance=skfmm_Dist(freshWaterMask,dx=0.2,narrow=800000)    
    freshWaterDistFileName="dist_to_freshwater.tif"  
    freshwaterfiledir=dirto+os.sep+freshWaterDistFileName
    ptf.WriteNumpyToTiff(freshWaterDistance,Driver,GeoTransform,Proj,nrow,ncol,-9999,freshwaterfiledir,datatype='Float32')
    print("computing done.\n")
    
    #calculate distance to sea
    [OriData,Driver,GeoTransform,Proj,nrow,ncol]=ptf.ReadTiffAsNumpy([coastlineMaskFiledir])
    landMask=OriData[:,:,0]
    landMask[landMask>0]=1
    landMask[landMask<=0]=0
    landMask=landMask.astype(np.bool)
    landMask=~landMask
    del OriData
    print("calculating distance to Sea...")
    seaDistance=skfmm_Dist(landMask,dx=0.2,narrow=800000)
    seaDistFileName="dist_to_sea.tif"  
    seadistfiledir=dirto+os.sep+seaDistFileName
    ptf.WriteNumpyToTiff(seaDistance,Driver,GeoTransform,Proj,nrow,ncol,-9999,seadistfiledir,datatype='Float32')
    print("computing done.\n")
    print("ALL DONE.\n")