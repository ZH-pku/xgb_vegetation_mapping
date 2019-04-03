# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 01:05:32 2018

@author: ZH
"""

import os
import copy
import time
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager

import src.xgb_functions as xgbf
import src.initialize as init
import src.multiclass as mlc
import src.varclustering as vc

def _calDueTime(time_start,time_stop,done_now,done_lastrun):
    time_use=time_stop-time_start
    time_due=round((1-done_now)*(time_use/(done_now-done_lastrun)))
    sec_per_day=24*60*60
    sec_per_hour=60*60
    sec_per_min=60
    num_day=math.floor(time_due/sec_per_day)
    num_hour=math.floor((time_due-sec_per_day*num_day)/sec_per_hour)
    num_min=math.floor((time_due-sec_per_day*num_day-sec_per_hour*num_hour)/sec_per_min)
    return num_day,num_hour,num_min

def _calWeight(evalues,weightindex,evaluethreshold):
    weights=np.power(evalues,weightindex)
    weights[evalues<evaluethreshold]=0
    wsum=np.sum(weights)
    weights=weights/wsum
    return weights  

def _findListSubsetIndexes(subsetlist,totalsetlist):
    idx=np.zeros(len(subsetlist),dtype=np.int32)
    for i in range(len(subsetlist)):
        idx[i]=totalsetlist.index(subsetlist[i])
    return idx

def _stratifiedRandomChoice_column(varnames,varlabels,varlabelweights,colsamplerate,runtimes):
    varlabels=np.array(varlabels)
    varlabel_count=np.bincount(varlabels)
    varlabelweights=np.array(varlabelweights)
    if varlabelweights[0]==-1:                #No Column Sample Weights
        colsamplerates=np.ones(len(varlabel_count))*colsamplerate
    elif varlabelweights[0]==-2:              #Category Balance Weights
        colsamplerates=np.zeros(len(varlabel_count))
        totalcounts=np.sum(varlabel_count)
        for i in range(len(colsamplerates)):
            colsamplerates[i]=(totalcounts-varlabel_count[i])/totalcounts    
        colsamplerates=colsamplerates/np.sum(colsamplerates)
        colsamplerates=colsamplerates*colsamplerate*len(colsamplerates)
    else:
        varlabelweights=varlabelweights/np.sum(varlabelweights)
        colsamplerates=varlabelweights*colsamplerate*len(varlabel_count)
    varclass_count=np.ceil((varlabel_count/np.sum(varlabel_count))*np.sum(varlabel_count)*colsamplerates).astype(np.int32)
    selectruntimesvarnames=[]
    for runtime in range(runtimes):
        selectvarnames=[]
        for i in range(len(varlabel_count)):
            varindexes=np.linspace(0,varlabel_count[i]-1,varlabel_count[i]).astype(np.int32)
            selectindexes=np.random.choice(varindexes,size=varclass_count[i],replace=False)
            varclassindexes=np.argwhere(varlabels==i)[:,0]
            selectclassindexes=varclassindexes[selectindexes].astype(np.int32)
            for j in range(len(selectclassindexes)):
                selectvarnames.append(varnames[selectclassindexes[j]])
        selectruntimesvarnames.append(selectvarnames)
    return selectruntimesvarnames

def trainMulticlassBaggingModel(DataSet,VegeTypes,varnames,params,multiclassmethod,baggingmetric='kappa',baggingweightindex=1,\
                                baggingmetricthres=0.7,varlabelweights=[-1],colsamplerate=0.7,train_percent=0.75,runtimes=300,\
                                bool_autolabel=True,varlabels=[],n_varlabels=5,bool_weight=False,bool_strclass=False,labelHeaderName="",\
                                bool_save=False,savedirbase=""):
    if bool_autolabel:
        varlabels=vc.KMeansLabel(DataSet,varnames,n_varlabels=n_varlabels)
    selectruntimesvarnames=_stratifiedRandomChoice_column(varnames,varlabels,varlabelweights,colsamplerate,runtimes)
    evalValues=np.zeros(runtimes)
    weights=np.zeros(runtimes)
    for runtime in range(runtimes):
        savedir=savedirbase+os.sep+"runtime_"+str(runtime)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        RuntimeDataSet=xgbf.trainingDataSet(DataSet,VegeTypes,selectruntimesvarnames[runtime],\
                                            bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)
        [train_x,test_x,train_y,test_y]=xgbf.splitTrainTestData(RuntimeDataSet,train_percent,bool_stratify=True)
        try:
            if multiclassmethod=='softmax':
                ModelList=mlc.trainMulticlassSoftmaxModel([train_y,train_x],VegeTypes,varnames,params,runtime=runtime,bool_weight=bool_weight,\
                                                          bool_pandas=False,bool_save=bool_save,savedir=savedir)
                [pred_Y,pred_pY,test_Y]=mlc.testMulticlassSoftmaxModel(ModelList,[test_y,test_x],VegeTypes,varnames,params,runtime=runtime,\
                                                            bool_pandas=False,bool_save=bool_save,savedir=savedir)
            elif multiclassmethod=='category':
                ModelList=mlc.trainMulticlassCategoryModel([train_y,train_x],VegeTypes,varnames,params,runtime=runtime,bool_weight=bool_weight,\
                                                           bool_pandas=False,bool_save=bool_save,savedir=savedir)
                [pred_Y,pred_pY,test_Y]=mlc.testMulticlassCategoryModel(ModelList,[test_y,test_x],VegeTypes,varnames,params,runtime=runtime,\
                                                            bool_pandas=False,bool_save=bool_save,savedir=savedir)
            else:
                print("Invalid Multiclass Method Input!")
            evalValues[runtime]=xgbf.Evaluate(test_Y,pred_Y,pred_pY,baggingmetric)
            print("Runtime: %d model done. Evaluation Value = %f"%(runtime,evalValues[runtime]))
        except:
            print("Model not established!")
            evalValues[runtime]=0.0
    weights=_calWeight(evalValues,baggingweightindex,baggingmetricthres)
    evalFiledirto=savedirbase+os.sep+"Runtime_Model_Evaluation_Weights.csv"
    init.writeArrayListToCSV([evalValues,weights],[baggingmetric,'weight'],evalFiledirto)
    #Write Each Runtime Model Variables Names
    selectvarnamesfiledir=savedirbase+os.sep+"Runtime_Model_Select_Variables.csv"
    save=pd.DataFrame({})
    for runtime in range(runtimes):
        pdtmp=pd.DataFrame({"SelectVarName_run"+str(runtime):selectruntimesvarnames[runtime]})
        save=pd.concat([save,pdtmp],axis=1)
    save.to_csv(selectvarnamesfiledir,index=False,header=True)

def testMulticlassBaggingModel(TestDataSet,VegeTypes,params,multiclassmethod,runtimes=300,bool_strclass=False,labelHeaderName="",\
                               bool_save=True,savedirbase=""):
    if not bool_save:
        print("Bagging Method has to save models!")
        return
    num_class=len(VegeTypes)
    evalweightsFileName="Runtime_Model_Evaluation_Weights.csv"
    selectvarnamesfiledir=savedirbase+os.sep+"Runtime_Model_Select_Variables.csv"
    evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
    baggingweights=init.getListFromPandas(evalweightsFiledirto,'weight')
    selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
    selectruntimesvarnames=[]
    for runtime in range(runtimes):
        selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
    del selrunvarspdData

    pred_pY_ense=np.zeros([len(TestDataSet),num_class])
    for runtime in range(runtimes):
        if baggingweights[runtime]==0:
            print("Model not established!")
            continue
        print("Predicting runtime = %d"%runtime)
        savedir=savedirbase+os.sep+"runtime_"+str(runtime)
        if multiclassmethod=='softmax':        
            [pred_Y,pred_pY,test_Y]=mlc.testMulticlassSoftmaxModel([],TestDataSet,VegeTypes,selectruntimesvarnames[runtime],\
                                    params,runtime=runtime,bool_pandas=True,bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,\
                                    bool_save=bool_save,savedir=savedir)
        elif multiclassmethod=='category':
            [pred_Y,pred_pY,test_Y]=mlc.testMulticlassCategoryModel([],TestDataSet,VegeTypes,selectruntimesvarnames[runtime],\
                                    params,runtime=runtime,bool_pandas=True,bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,\
                                    bool_save=bool_save,savedir=savedir)
        else:
            print("Invalid Multiclass Method Input!")
#        pred_pY_ense=pred_pY_ense+pred_pY*baggingweights[runtime]
        pred_Y_epd=init.expandCategories(pred_Y,num_class=num_class)
        pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_Y_epd.astype(np.float32)
    pred_Y=np.argmax(pred_pY_ense,axis=1)
    return [pred_Y,pred_pY_ense,test_Y]

def predictMulticlassBaggingModel(MatX,nrow,ncol,varnames,num_class,params,multiclassmethod,runtimes=300,bool_save=True,savedirbase=""):
    count=0.0
    if not bool_save:
        print("Bagging Method has to save models!")
        return
    evalweightsFileName="Runtime_Model_Evaluation_Weights.csv"
    selectvarnamesfiledir=savedirbase+os.sep+"Runtime_Model_Select_Variables.csv"
    evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
    selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
    baggingweights=init.getListFromPandas(evalweightsFiledirto,'weight')
    selectruntimesvarnames=[]
    for runtime in range(runtimes):
        selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
    del selrunvarspdData
    bool_mask=init.getMask(MatX)
    time_start=time.time()
    if multiclassmethod=='softmax':
        pred_pY_ense=np.zeros([nrow*ncol,num_class],dtype=np.float32)
        pred_X=init.fomatMulticlassSoftmaxMatrix(MatX)
        for runtime in range(runtimes):
            if baggingweights[runtime]==0:
                print("Model not established!")
                continue
            selruntimevarstr=selectruntimesvarnames[runtime]
            selruntimevaridx=_findListSubsetIndexes(selruntimevarstr,varnames)
            pred_X_runtime=pred_X[:,selruntimevaridx]
            print("Predicting Bagging Model...    runtime = %d"%runtime)            
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            pred_pY=mlc.predictMulticlassSoftmaxModelCvted([],pred_X_runtime,params,\
                                                           runtime=runtime,bool_save=bool_save,savedir=savedir)
            pred_Y=np.argmax(pred_pY,axis=1)
            pred_Y_epd=init.expandCategories(pred_Y,num_class=num_class)
            pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_Y_epd.astype(np.float32)
            time_stop=time.time()
            count=count+1
            done=count/runtimes
            remain=(runtimes-count)/runtimes
            num_day,num_hour,num_min=_calDueTime(time_start,time_stop,done,0.0)
            print("Model: %d Calculating Finished!      Done: %.2f%%, Remaining: %.2f%%"%(runtime,100*done,100*remain))
            print("Calculating will finish in %d Days %d Hours %d Minutes\n"%(num_day,num_hour,num_min))            
        [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY_ense,nrow,ncol,num_class,bool_onearray=False,mask=bool_mask.flatten())
    elif multiclassmethod=='category':
        pred_pY_ense=np.zeros([nrow*ncol,num_class],dtype=np.float32)
        pred_X=init.formatMulticlassCategoryMatrix(MatX,num_class)
        for runtime in range(runtimes):
            if baggingweights[runtime]==0:
                print("Model not established!")
                continue
            selruntimevarstr=selectruntimesvarnames[runtime]
            selruntimevaridx=_findListSubsetIndexes(selruntimevarstr,varnames)
            pred_X_runtime=pred_X[:,selruntimevaridx]
            print("Predicting Bagging Model...    runtime = %d"%runtime)            
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            pred_Y=mlc.predictMulticlassCategoryModelCvted([],pred_X_runtime,params,runtime=runtime,bool_retlabel=True,num_instance=nrow*ncol,num_class=num_class,\
                                                            bool_save=bool_save,savedir=savedir)
            pred_Y_epd=init.expandCategories(pred_Y,num_class=num_class)
            pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_Y_epd.astype(np.float32)
            time_stop=time.time()
            count=count+1
            done=count/runtimes
            remain=(runtimes-count)/runtimes
            num_day,num_hour,num_min=_calDueTime(time_start,time_stop,done,0.0)
            print("Model: %d Calculating Finished!      Done: %.2f%%, Remaining: %.2f%%"%(runtime,100*done,100*remain))
            print("Calculating will finish in %d Days %d Hours %d Minutes\n"%(num_day,num_hour,num_min))
        [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY_ense,nrow,ncol,num_class,bool_onearray=False,mask=bool_mask.flatten())
    return [pred_Y,pred_pY]

def _trainMulticlassBaggingModel(CPIDs,DataSet,VegeTypes,varnames,params,multiclassmethod,bool_gpu,n_gpus,n_parallels,\
                                 selectruntimesvarnames,runtime,train_percent,baggingmetric,bool_weight,bool_strclass,labelHeaderName,\
                                 bool_save,savedirbase):
    #Assign task to worker
    print("Training #%d model..."%runtime)
    params_parallel=copy.deepcopy(params)
    process_pid=os.getpid()
    if len(CPIDs)<n_parallels:
        CPIDs.append(process_pid)
    process_pid_index=CPIDs.index(process_pid)
    print("Worker #%d: PID = %d"%(process_pid_index,process_pid))
    if bool_gpu:
        params_parallel['gpu_id']=process_pid_index%n_gpus    
    
    #Execute model training process
    savedir=savedirbase+os.sep+"runtime_"+str(runtime)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    RuntimeDataSet=xgbf.trainingDataSet(DataSet,VegeTypes,selectruntimesvarnames[runtime],\
                                        bool_strclass=bool_strclass,labelHeaderName=labelHeaderName)
    [train_x,test_x,train_y,test_y]=xgbf.splitTrainTestData(RuntimeDataSet,train_percent,bool_stratify=1)
    if multiclassmethod=='softmax':
        ModelList=mlc.trainMulticlassSoftmaxModel([train_y,train_x],VegeTypes,varnames,params_parallel,runtime=runtime,\
                                                  bool_weight=bool_weight,bool_pandas=False,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY,test_Y]=mlc.testMulticlassSoftmaxModel(ModelList,[test_y,test_x],VegeTypes,varnames,params_parallel,\
                                                    runtime=runtime,bool_pandas=False,bool_save=bool_save,savedir=savedir)
    elif multiclassmethod=='category':
        ModelList=mlc.trainMulticlassCategoryModel([train_y,train_x],VegeTypes,varnames,params_parallel,runtime=runtime,\
                                                   bool_weight=bool_weight,bool_pandas=False,bool_save=bool_save,savedir=savedir)
        [pred_Y,pred_pY,test_Y]=mlc.testMulticlassCategoryModel(ModelList,[test_y,test_x],VegeTypes,varnames,params_parallel,\
                                                    runtime=runtime,bool_pandas=False,bool_save=bool_save,savedir=savedir)
    else:
        print("Invalid Multiclass Method Input!")
    evalValue=xgbf.Evaluate(test_Y,pred_Y,pred_pY,baggingmetric)
#    evalValues[runtime]=xgbf.Evaluate(test_Y,pred_Y,pred_pY,access_method)
    print("Runtime: %d model training finished. Evaluation Value = %f\n"%(runtime,evalValue))
    return evalValue

def trainMulticlassBaggingModel_parallel(DataSet,VegeTypes,varnames,params,multiclassmethod,n_gpus,n_parallels,baggingmetric='kappa',baggingweightindex=1,\
                                         baggingmetricthres=0.7,varlabelweights=[-1],colsamplerate=0.7,train_percent=0.75,runtimes=300,\
                                         bool_autolabel=True,varlabels=[],n_varlabels=5,bool_weight=False,bool_strclass=False,labelHeaderName="",\
                                         bool_save=False,savedirbase=""):
    if bool_autolabel:
        varlabels=vc.KMeansLabel(DataSet,varnames,n_varlabels=n_varlabels)
    selectruntimesvarnames=_stratifiedRandomChoice_column(varnames,varlabels,varlabelweights,colsamplerate,runtimes)
    evalValues=np.zeros(runtimes)
    weights=np.zeros(runtimes)
    #Judge bool_gpu
    if 'gpu' in params.get('tree_method'):
        bool_gpu=True
    else:
        bool_gpu=False        
    #Open multiprocessing parallel pools    
    P=Pool(n_parallels)
    results_parallel=[]
    manager=Manager()
    CPIDs=manager.list()
    #Execute
    for runtime in range(runtimes):
        results_parallel.append(P.apply_async(_trainMulticlassBaggingModel,(CPIDs,DataSet,VegeTypes,varnames,params,multiclassmethod,\
                                                                            bool_gpu,n_gpus,n_parallels,selectruntimesvarnames,runtime,\
                                                                            train_percent,baggingmetric,bool_weight,bool_strclass,labelHeaderName,\
                                                                            bool_save,savedirbase)))
    P.close()
    P.join()
    del CPIDs
    for runtime in range(runtimes):
        temp=results_parallel[runtime]
        try:
            evalValues[runtime]=temp.get()
        except:
            evalValues[runtime]=0.0
    #Calculate ensemble weights
    weights=_calWeight(evalValues,baggingweightindex,baggingmetricthres)
    evalFiledirto=savedirbase+os.sep+"Runtime_Model_Evaluation_Weights.csv"
    init.writeArrayListToCSV([evalValues,weights],[baggingmetric,'weight'],evalFiledirto)
    #Write Each Runtime Model Variables Names
    selectvarnamesfiledir=savedirbase+os.sep+"Runtime_Model_Select_Variables.csv"
    save=pd.DataFrame({})
    for runtime in range(runtimes):
        pdtmp=pd.DataFrame({"SelectVarName_run"+str(runtime):selectruntimesvarnames[runtime]})
        save=pd.concat([save,pdtmp],axis=1)
    save.to_csv(selectvarnamesfiledir,index=False,header=True)

def _testMulticlassBaggingModel(CPIDs,RuntimeList,TestDataSet,VegeTypes,params,multiclassmethod,bool_gpu,n_gpus,n_parallels,\
                                selectruntimesvarnames,baggingweights,bool_strclass,labelHeaderName,bool_save,savedirbase):
    print("Predicting Multiclass Bagging Ensemble Models...")
    params_parallel=copy.deepcopy(params)
    process_pid=os.getpid()
    if len(CPIDs)<n_parallels:
        CPIDs.append(process_pid)
    process_pid_index=CPIDs.index(process_pid)
    print("Worker #%d: PID = %d"%(process_pid_index,process_pid))
    if bool_gpu:
        params_parallel['gpu_id']=process_pid_index%n_gpus    
    num_class=len(VegeTypes)
    pred_pY_ense=np.zeros([len(TestDataSet),num_class])
    for runtime in RuntimeList:
        if baggingweights[runtime]==0:
            print("Model not established!")
            continue
        print("Predicting runtime = %d"%runtime)
        savedir=savedirbase+os.sep+"runtime_"+str(runtime)
        if multiclassmethod=='softmax':        
            [pred_Y,pred_pY,test_Y]=mlc.testMulticlassSoftmaxModel([],TestDataSet,VegeTypes,selectruntimesvarnames[runtime],\
                                    params_parallel,runtime=runtime,bool_pandas=True,bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,\
                                    bool_save=bool_save,savedir=savedir)
        elif multiclassmethod=='category':
            [pred_Y,pred_pY,test_Y]=mlc.testMulticlassCategoryModel([],TestDataSet,VegeTypes,selectruntimesvarnames[runtime],\
                                    params_parallel,runtime=runtime,bool_pandas=True,bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,\
                                    bool_save=bool_save,savedir=savedir)
        else:
            print("Invalid Multiclass Method Input!")
        pred_Y_epd=init.expandCategories(pred_Y,num_class=num_class)
        pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_Y_epd.astype(np.float32)
    return [pred_Y,pred_pY_ense,test_Y]

def testMulticlassBaggingModel_parallel(TestDataSet,VegeTypes,params,multiclassmethod,n_gpus,n_parallels,runtimes=300,\
                                        bool_strclass=False,labelHeaderName="",bool_save=True,savedirbase=""):
    if not bool_save:
        print("Bagging Method has to save models!")
        return
    evalweightsFileName="Runtime_Model_Evaluation_Weights.csv"
    selectvarnamesfiledir=savedirbase+os.sep+"Runtime_Model_Select_Variables.csv"
    evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
    baggingweights=init.getListFromPandas(evalweightsFiledirto,'weight')
    selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
    selectruntimesvarnames=[]
    for runtime in range(runtimes):
        selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
    del selrunvarspdData
    #Assign task to worker
    RuntimeLists=[[] for i in range(n_parallels)]
    for runtime in range(runtimes):
        worker_id=runtime%n_parallels
        RuntimeLists[worker_id].append(runtime)
    #Judge bool_gpu
    if 'gpu' in params.get('tree_method'):
        bool_gpu=True
    else:
        bool_gpu=False
        
    P=Pool(n_parallels)
    results_parallel=[]
    manager=Manager()
    CPIDs=manager.list()
    for i in range(n_parallels):
        results_parallel.append(P.apply_async(_testMulticlassBaggingModel,(CPIDs,RuntimeLists[i],TestDataSet,VegeTypes,params,multiclassmethod,bool_gpu,n_gpus,n_parallels,\
                                    selectruntimesvarnames,baggingweights,bool_strclass,labelHeaderName,bool_save,savedirbase)))
    P.close()
    P.join()
    del CPIDs
    
    pred_pY_ense=np.zeros([len(TestDataSet),len(VegeTypes)])
    for i in range(n_parallels):
        temp=results_parallel[i]
        [pred_Y,pred_pY_ense_para,test_Y]=temp.get()
        pred_pY_ense=pred_pY_ense+pred_pY_ense_para    
    
    pred_Y=np.argmax(pred_pY_ense,axis=1)
    return [pred_Y,pred_pY_ense,test_Y]

def _predictMulticlassBaggingModel(CPIDs,RuntimeList,MatX,nrow,ncol,varnames,num_class,params,selectruntimesvarnames,baggingweights,multiclassmethod,\
                                   bool_gpu,n_gpus,n_parallels,bool_save,savedirbase):

    print("Predicting Multiclass Bagging Ensemble Models...")
    params_parallel=copy.deepcopy(params)
    process_pid=os.getpid()
    if len(CPIDs)<n_parallels:
        CPIDs.append(process_pid)
    process_pid_index=CPIDs.index(process_pid)
    print("Worker #%d: PID = %d"%(process_pid_index,process_pid))
    if bool_gpu:
        params_parallel['gpu_id']=process_pid_index%n_gpus    
        
    #Execute predicting process    
    if multiclassmethod=='softmax':
        pred_pY_ense=np.zeros([nrow*ncol,num_class],dtype=np.float32)
        pred_X=init.fomatMulticlassSoftmaxMatrix(MatX)
        for runtime in RuntimeList:
            if baggingweights[runtime]==0:
                print("Model not established!")
                continue
            selruntimevarstr=selectruntimesvarnames[runtime]
            selruntimevaridx=_findListSubsetIndexes(selruntimevarstr,varnames)
            pred_X_runtime=pred_X[:,selruntimevaridx]
            print("Predicting Multiclass Model...    runtime = %d"%runtime)            
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            pred_pY=mlc.predictMulticlassSoftmaxModelCvted([],pred_X_runtime,params_parallel,\
                                                           runtime=runtime,bool_save=bool_save,savedir=savedir)
            pred_Y=np.argmax(pred_pY,axis=1)
            pred_Y_epd=init.expandCategories(pred_Y,num_class=num_class)
            pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_Y_epd.astype(np.float32)
            print("Model: %d Calculating Finished!\n"%(runtime))
  
    elif multiclassmethod=='category':
        pred_pY_ense=np.zeros([nrow*ncol,num_class],dtype=np.float32)
        pred_X=init.formatMulticlassCategoryMatrix(MatX,num_class)
        for runtime in RuntimeList:
            selruntimevarstr=selectruntimesvarnames[runtime]
            selruntimevaridx=_findListSubsetIndexes(selruntimevarstr,varnames)
            pred_X_runtime=pred_X[:,selruntimevaridx]
            print("Predicting Multiclass Model...    runtime = %d"%runtime)            
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            pred_Y=mlc.predictMulticlassCategoryModelCvted([],pred_X_runtime,params_parallel,runtime=runtime,bool_retlabel=True,num_instance=nrow*ncol,num_class=num_class,\
                                                            bool_save=bool_save,savedir=savedir)
            pred_Y_epd=init.expandCategories(pred_Y,num_class=num_class)
            pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_Y_epd.astype(np.float32)
            print("Model: %d Calculating Finished!\n"%(runtime))
    return pred_pY_ense

def predictMulticlassBaggingModel_parallel(MatX,nrow,ncol,varnames,num_class,params,multiclassmethod,\
                                           n_gpus,n_parallels,runtimes=300,bool_save=True,savedirbase=""):
    if not bool_save:
        print("Bagging Method has to save models!")
        return
    evalweightsFileName="Runtime_Model_Evaluation_Weights.csv"
    selectvarnamesfiledir=savedirbase+os.sep+"Runtime_Model_Select_Variables.csv"
    selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
    evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
    baggingweights=init.getListFromPandas(evalweightsFiledirto,'weight')
    selectruntimesvarnames=[]
    for runtime in range(runtimes):
        selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
    del selrunvarspdData
    bool_mask=init.getMask(MatX)
    #Assign task to worker
    RuntimeLists=[[] for i in range(n_parallels)]
    for runtime in range(runtimes):
        worker_id=runtime%n_parallels
        RuntimeLists[worker_id].append(runtime)
    #Judge bool_gpu
    if 'gpu' in params.get('tree_method'):
        bool_gpu=True
    else:
        bool_gpu=False

    P=Pool(n_parallels)
    results_parallel=[]
    manager=Manager()
    CPIDs=manager.list()
    for i in range(n_parallels):
        results_parallel.append(P.apply_async(_predictMulticlassBaggingModel,(CPIDs,RuntimeLists[i],MatX,nrow,ncol,varnames,num_class,params,\
                                        selectruntimesvarnames,baggingweights,multiclassmethod,bool_gpu,n_gpus,n_parallels,bool_save,savedirbase)))
    P.close()
    P.join()
    del CPIDs
    
#    if  multiclassmethod=='softmax':
    pred_pY_ense=np.zeros([nrow*ncol,num_class],dtype=np.float32)
#    elif multiclassmethod=='category':
#        pred_pY_ense=np.zeros(nrow*ncol*num_class,dtype=np.float32)
    
    for i in range(n_parallels):
        temp=results_parallel[i]
        pred_pY_ense_para=temp.get()
        pred_pY_ense=pred_pY_ense+pred_pY_ense_para
    
#    if multiclassmethod=='softmax':
    [pred_Y,pred_pY]=init.reshapeMulticlassMatrix(pred_pY_ense,nrow,ncol,num_class,bool_onearray=False,mask=bool_mask.flatten())    
#    elif multiclassmethod=='category':
#        [pred_Y,pred_pY]=init.ReshapeMulticlassMatrix(pred_pY_ense,nrow,ncol,num_class,1,bool_stretch=bool_stretch,mask=bool_mask.flatten())   
    return [pred_Y,pred_pY]