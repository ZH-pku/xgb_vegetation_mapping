#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 01:35:52 2018

@author: ecology
"""

import numpy as np
from osgeo import gdal

def readTiffAsNumpy(TiffList):
    print("Reading GeoTiff files...")
    total=len(TiffList)
    tmpfiledir=TiffList[0]
    tmp=gdal.Open(tmpfiledir)
    ncol=tmp.RasterXSize
    nrow=tmp.RasterYSize
    Driver=tmp.GetDriver()
    GeoTransform=tmp.GetGeoTransform()
    Proj=tmp.GetProjection()
    OriData=np.zeros([nrow,ncol,total],dtype=np.float32)
    for i in range(total):
#        print("reading: %s"%TiffList[i])
        data=gdal.Open(TiffList[i])
        OriData[:,:,i]=data.ReadAsArray().astype(np.float32)
    return [OriData,Driver,GeoTransform,Proj,nrow,ncol]

def writeNumpyToTiff(TargetData,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,filedirto,datatype='Float32'):
    if datatype=='Int16':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Int16)
        TargetData=TargetData.astype(np.int16)
    elif datatype=='Int32':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Int32)
        TargetData=TargetData.astype(np.int32)  
    elif datatype=='UInt16':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_UInt16)
        TargetData=TargetData.astype(np.uint16)
    elif datatype=='UInt32':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_UInt32)
        TargetData=TargetData.astype(np.uint32)  
    elif datatype=='Float32':        
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Float32)
        TargetData=TargetData.astype(np.float32)
    elif datatype=='Float64':
        output=Driver.Create(filedirto,ncol,nrow,1,gdal.GDT_Float64)
        TargetData=TargetData.astype(np.float64)        
    else:
        print("Input data type unavailable! Please choose one from below:")
        print("Int16  Int32  UInt16  UInt32  Float32  Float64")
    output.SetGeoTransform(GeoTransform)
    output.SetProjection(Proj)
    outBand=output.GetRasterBand(1)
#    outBand.SetNoDataValue(nanDefault)    
    outBand.WriteArray(TargetData,0,0)
    outBand.FlushCache()
    