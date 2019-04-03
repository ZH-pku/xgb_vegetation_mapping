#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 01:35:52 2018

@author: ecology
"""

import subprocess
import numpy as np
from osgeo import gdal

def GetUTMTiffCorners(Filedirs):
    if type(Filedirs)==list:
        filedir=Filedirs[0]
    if type(Filedirs)==str:
        filedir=Filedirs
    gdalCMD="gdalinfo "+filedir
    p= subprocess.Popen(gdalCMD,stdout=subprocess.PIPE,shell=True)
    out,err= p.communicate()
    out=str(out)
    upperleft= out[out.find("Upper Left")+15:out.find("Upper Left")+38]
    loweright= out[out.find("Lower Right")+15:out.find("Lower Right")+38]
    sul=upperleft.split(', ')
    slr=loweright.split(', ')
    top=float(sul[1])
    left=float(sul[0])
    right=float(slr[0])
    bottom=float(slr[1])
    print("Image data extent get!")
    return np.array([top,left,right,bottom])

def ReadTiffAsNumpy(TiffList):
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
        data=gdal.Open(TiffList[i])
        OriData[:,:,i]=data.ReadAsArray().astype(np.float32)
    return [OriData,Driver,GeoTransform,Proj,nrow,ncol]

def WriteNumpyToTiff(TargetData,Driver,GeoTransform,Proj,nrow,ncol,nanDefault,filedirto,datatype='Float32'):
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
        print("Data type not listed! Please choose from the bellowing:")
        print("Int16  Int32  UInt16  UInt32  Float32  Float64")
    output.SetGeoTransform(GeoTransform)
    output.SetProjection(Proj)
    outBand=output.GetRasterBand(1)
#    outBand.SetNoDataValue(nanDefault)    
    outBand.WriteArray(TargetData,0,0)
    outBand.FlushCache()
    