# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:10:14 2018

@author: ecology
"""

import os
import copy
import time
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool, Manager

import src.initialize as init
import src.xgb_functions as xgbf
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

def _calWeight(evalues,runtime,weightindex,evaluethreshold):
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

def trainSingleclassBaggingModel(DataSet,vtname,varnames,params,baggingmetric='auc',baggingweightindex=1,\
                       baggingmetricthres=0.7,single_thres=0.5,varlabelweights=[-1],colsamplerate=0.7,\
                       train_percent=0.75,runtimes=300,bool_autolabel=True,varlabels=[],n_varlabels=5,bool_balance=True,\
                       bool_strclass=False,labelHeaderName="",bool_save=False,savedirbase=""):
    ModelList=[]
    if bool_autolabel:
        varlabels=vc.KMeansLabel(DataSet,varnames,n_varlabels=n_varlabels)
    selectruntimesvarnames=_stratifiedRandomChoice_column(varnames,varlabels,varlabelweights,colsamplerate,runtimes)
    evalValues=np.zeros(runtimes)
    for runtime in range(runtimes):
        RuntimeDataSet=xgbf.trainingDataSet(DataSet,[vtname],selectruntimesvarnames[runtime],\
                                            bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_binary=True)
        [train_x,test_x,train_y,test_y]=xgbf.splitTrainTestData(RuntimeDataSet,train_percent,bool_stratify=1)
        if bool_balance:
            if len(train_y.shape)>1:
                ratio=np.float(np.sum(train_y[:,0]==0))/np.sum(train_y[:,0]==1)
            else:
                ratio=np.float(np.sum(train_y[:]==0))/np.sum(train_y[:]==1)
            params['scale_pos_weight']=ratio
        model=xgbf.TrainModel(train_x,train_y,params)
        if bool_save:  
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            modelName=vtname+'_xgboost_singleclass_run'+str(runtime)+".model"
            modeldir=savedir+os.sep+modelName
            model.save_model(modeldir)
        else:
            ModelList.append(model)
        [pred_Y,pred_pY]=xgbf.Predict(model,test_x,bool_binary=1,threshold=single_thres)
        evalValues[runtime]=xgbf.Evaluate(test_y,pred_Y,pred_pY,baggingmetric)
        print("Runtime: %d model done. Evaluation Value = %f"%(runtime,evalValues[runtime]))
    baggingweights=_calWeight(evalValues,runtimes,baggingweightindex,baggingmetricthres)
    if bool_save:
        #Save Weights
        evalweightsFileName=vtname+"_Runtime_Evaluation_Weight.csv"
        evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
        evalweightsarray=np.zeros([runtimes,2])
        evalweightsarray[:,0]=evalValues
        evalweightsarray[:,1]=baggingweights
        evalweightarrayname=[baggingmetric,'weight']
        init.writeArrayToCSV(evalweightsarray,evalweightarrayname,evalweightsFiledirto)
        #Save Used Parameters
        selectvarnamesfiledir=savedirbase+os.sep+vtname+"_Runtime_Model_Select_Variables.csv"
        save=pd.DataFrame({})
        for runtime in range(runtimes):
            pdtmp=pd.DataFrame({"SelectVarName_run"+str(runtime):selectruntimesvarnames[runtime]})
            save=pd.concat([save,pdtmp],axis=1)
        save.to_csv(selectvarnamesfiledir,index=False,header=True)
        return []
    else:
        return [ModelList,selectruntimesvarnames,baggingweights]

def _trainSingleclassBaggingModel(CPIDs,DataSet,vtname,params,baggingmetric,bool_gpu,n_gpus,n_parallels,selectruntimesvarnames,\
                                  runtime,train_percent,single_thres,bool_balance,bool_strclass,labelHeaderName,bool_save,savedirbase):
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
    RuntimeDataSet=xgbf.trainingDataSet(DataSet,[vtname],selectruntimesvarnames[runtime],\
                                    bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_binary=True)
    [train_x,test_x,train_y,test_y]=xgbf.splitTrainTestData(RuntimeDataSet,train_percent,bool_stratify=1)
    if bool_balance:
        if len(train_y.shape)>1:
            ratio=np.float(np.sum(train_y[:,0]==0))/np.sum(train_y[:,0]==1)
        else:
            ratio=np.float(np.sum(train_y[:]==0))/np.sum(train_y[:]==1)
        params_parallel['scale_pos_weight']=ratio
    model=xgbf.TrainModel(train_x,train_y,params_parallel)
    savedir=savedirbase+os.sep+"runtime_"+str(runtime)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    modelName=vtname+'_xgboost_singleclass_run'+str(runtime)+".model"
    modeldir=savedir+os.sep+modelName
    model.save_model(modeldir)
    [pred_Y,pred_pY]=xgbf.Predict(model,test_x,bool_binary=1,threshold=single_thres)
    evalValue=xgbf.Evaluate(test_y,pred_Y,pred_pY,baggingmetric)    
    print("Runtime: %d model training finished. Evaluation Value = %f\n"%(runtime,evalValue))
    return evalValue
    
    
def trainSingleclassBaggingModel_parallel(DataSet,vtname,varnames,params,n_gpus,n_parallels,baggingmetric='auc',baggingweightindex=1,\
                                          baggingmetricthres=0.7,single_thres=0.5,varlabelweights=[-1],colsamplerate=0.7,\
                                          train_percent=0.75,runtimes=300,bool_autolabel=True,varlabels=[],n_varlabels=5,bool_balance=True,\
                                          bool_strclass=False,labelHeaderName="",bool_save=True,savedirbase=""):

    if not bool_save:
        print("Single Bagging Ensemble Only for bool_save = True!")
        return []
    if bool_autolabel:
        varlabels=vc.KMeansLabel(DataSet,varnames,n_varlabels=n_varlabels)
    selectruntimesvarnames=_stratifiedRandomChoice_column(varnames,varlabels,varlabelweights,colsamplerate,runtimes)
    evalValues=np.zeros(runtimes)
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
        results_parallel.append(P.apply_async(_trainSingleclassBaggingModel,(CPIDs,DataSet,vtname,params,baggingmetric,bool_gpu,n_gpus,n_parallels,selectruntimesvarnames,\
                                  runtime,train_percent,single_thres,bool_balance,bool_strclass,labelHeaderName,bool_save,savedirbase)))
    P.close()
    P.join()
    del CPIDs
    for runtime in range(runtimes):
        temp=results_parallel[runtime]
        evalValues[runtime]=temp.get()  
    #Calculate ensemble weights
    baggingweights=_calWeight(evalValues,runtimes,baggingweightindex,baggingmetricthres)
    #Save Weights
    evalweightsFileName=vtname+"_Runtime_Evaluation_Weight.csv"
    evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
    evalweightsarray=np.zeros([runtimes,2])
    evalweightsarray[:,0]=evalValues
    evalweightsarray[:,1]=baggingweights
    evalweightarrayname=[baggingmetric,'weight']
    init.writeArrayToCSV(evalweightsarray,evalweightarrayname,evalweightsFiledirto)
    #Save Used Parameters
    selectvarnamesfiledir=savedirbase+os.sep+vtname+"_Runtime_Model_Select_Variables.csv"
    save=pd.DataFrame({})
    for runtime in range(runtimes):
        pdtmp=pd.DataFrame({"SelectVarName_run"+str(runtime):selectruntimesvarnames[runtime]})
        save=pd.concat([save,pdtmp],axis=1)
    save.to_csv(selectvarnamesfiledir,index=False,header=True)
    return []

def testSingleclassBaggingModel(Models,TestDataSet,vtname,params,single_thres=0.5,runtimes=300,\
                                bool_strclass=False,labelHeaderName="",bool_save=False,savedirbase=""):
    ModelList=[]
    if bool_save:
        evalweightsFileName=vtname+"_Runtime_Evaluation_Weight.csv"
        selectvarnamesfiledir=savedirbase+os.sep+vtname+"_Runtime_Model_Select_Variables.csv"
        evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
        ense_weights=init.getListFromPandas(evalweightsFiledirto,'weight')
        selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
        selectruntimesvarnames=[]
        for runtime in range(runtimes):
            selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
        del selrunvarspdData
    else:
        [ModelList,selectruntimesvarnames,ense_weights]=Models

    pred_pY_ense=np.zeros(len(TestDataSet))
    for runtime in range(runtimes):
        print("Predicting runtime = %d"%runtime)
        if bool_save:
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            modelName=vtname+'_xgboost_singleclass_run'+str(runtime)+".model"
            modeldir=savedir+os.sep+modelName
            model=xgbf.loadModel(modeldir,params)
        else:
            model=ModelList[runtime]
        varnames=selectruntimesvarnames[runtime]
        [test_Y,test_X]=xgbf.trainingDataSet(TestDataSet,[vtname],varnames,\
                                bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_binary=True)
        [pred_Y,pred_pY]=xgbf.Predict(model,test_X,bool_binary=1,threshold=single_thres)
        pred_pY_ense=pred_pY_ense+pred_pY*ense_weights[runtime]
    pred_Y_ense=(pred_pY_ense>=single_thres)*1
    pred_Y=pred_Y_ense
    pred_pY=pred_pY_ense
    if len(test_Y.shape)>1:
        test_Y=test_Y[:,0]
    return [pred_Y,pred_pY,test_Y]    


def _testSingleclassBaggingModel(CPIDs,RuntimeList,TestDataSet,vtname,runtime,params,ModelList,bool_gpu,n_gpus,n_parallels,\
                                 selectruntimesvarnames,baggingweights,single_thres,bool_strclass,labelHeaderName,\
                                 bool_save,savedirbase):
    print("Predicting Singleclass Bagging Ensemble Models...")
    params_parallel=copy.deepcopy(params)
    process_pid=os.getpid()
    if len(CPIDs)<n_parallels:
        CPIDs.append(process_pid)
    process_pid_index=CPIDs.index(process_pid)
    print("Worker #%d: PID = %d"%(process_pid_index,process_pid))
    if bool_gpu:
        params_parallel['gpu_id']=process_pid_index%n_gpus    
        
    pred_pY_ense=np.zeros(len(TestDataSet))
    for runtime in RuntimeList:
        print("Predicting runtime = %d"%runtime)
        if bool_save:
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            modelName=vtname+'_xgboost_singleclass_run'+str(runtime)+".model"
            modeldir=savedir+os.sep+modelName
            model=xgbf.loadModel(modeldir,params_parallel)
        else:
            model=ModelList[runtime]
        varnames=selectruntimesvarnames[runtime]
        [test_Y,test_X]=xgbf.trainingDataSet(TestDataSet,[vtname],varnames,\
                        bool_strclass=bool_strclass,labelHeaderName=labelHeaderName,bool_binary=True)
        [pred_Y,pred_pY]=xgbf.Predict(model,test_X,bool_binary=1,threshold=single_thres)
        pred_pY_ense=pred_pY_ense+pred_pY*baggingweights[runtime]
    pred_Y_ense=(pred_pY_ense>=single_thres)*1
    pred_Y=pred_Y_ense
    pred_pY=pred_pY_ense
    if len(test_Y.shape)>1:
        test_Y=test_Y[:,0]
    return [pred_Y,pred_pY,test_Y]


def testSingleclassBaggingModel_parallel(Models,TestDataSet,vtname,params,n_gpus,n_parallels,\
                                         single_thres=0.5,runtimes=300,bool_strclass=False,labelHeaderName="",\
                                         bool_save=False,savedirbase=""):
    ModelList=[]
    if bool_save:
        evalweightsFileName=vtname+"_Runtime_Evaluation_Weight.csv"
        selectvarnamesfiledir=savedirbase+os.sep+vtname+"_Runtime_Model_Select_Variables.csv"
        evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
        baggingweights=init.getListFromPandas(evalweightsFiledirto,'weight')
        selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
        selectruntimesvarnames=[]
        for runtime in range(runtimes):
            selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
        del selrunvarspdData
    else:
        [ModelList,selectruntimesvarnames,ense_weights]=Models
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
    #Open multiprocessing parallel pools        
    P=Pool(n_parallels)
    results_parallel=[]
    manager=Manager()
    CPIDs=manager.list()
    for i in range(n_parallels):
        results_parallel.append(P.apply_async(_testSingleclassBaggingModel,(CPIDs,RuntimeLists[i],TestDataSet,vtname,runtime,params,ModelList,\
                                bool_gpu,n_gpus,n_parallels,selectruntimesvarnames,baggingweights,single_thres,bool_strclass,labelHeaderName,bool_save,savedirbase)))
    P.close()
    P.join()
    del CPIDs

    pred_pY_ense=np.zeros(len(TestDataSet))
    for i in range(n_parallels):
        temp=results_parallel[i]
        [pred_Y,pred_pY_ense_para,test_Y]=temp.get()
        pred_pY_ense=pred_pY_ense+pred_pY_ense_para
    pred_Y_ense=(pred_pY_ense>=single_thres)*1
    pred_Y=pred_Y_ense
    return [pred_Y,pred_pY_ense,test_Y]

def predictSingleclassBaggingModelMatrix(Models,MatX,vtname,varnames,params,single_thres=0.5,runtimes=300,filter_percent=0,\
                                         bool_save=False,savedirbase=""):
    count=0.0
    if bool_save:
        evalweightsFileName=vtname+"_Runtime_Evaluation_Weight.csv"
        selectvarnamesfiledir=savedirbase+os.sep+vtname+"_Runtime_Model_Select_Variables.csv"
        evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
        ense_weights=init.getListFromPandas(evalweightsFiledirto,'weight')
        selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
        selectruntimesvarnames=[]
        for runtime in range(runtimes):
            selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
        del selrunvarspdData
    else:
        [ModelList,selectruntimesvarnames,ense_weights]=Models
    matshape=MatX.shape
    bool_mask=init.getMask(MatX)
    pred_X=np.zeros([matshape[0]*matshape[1],matshape[2]],dtype=np.float32)
    for i in range(matshape[2]):
        pred_X[:,i]=MatX[:,:,i].flatten()
    pred_pY_ense=np.zeros(matshape[0]*matshape[1],dtype=np.float32)
    time_start=time.time()
    for runtime in range(runtimes):
        print("Predicting runtime = %d..."%(runtime))
        if bool_save:
            savedir=savedirbase+os.sep+"runtime_"+str(runtime)
            modelName=vtname+'_xgboost_singleclass_run'+str(runtime)+".model"
            modeldir=savedir+os.sep+modelName
            model=xgbf.loadModel(modeldir,params)
        else:
            model=ModelList[runtime]
        selruntimevarstr=selectruntimesvarnames[runtime]
        selruntimevaridx=_findListSubsetIndexes(selruntimevarstr,varnames)
        pred_X_runtime=pred_X[:,selruntimevaridx]
        [pred_Y,pred_pY]=xgbf.Predict(model,pred_X_runtime,bool_binary=1,threshold=single_thres)
        pred_pY_ense=pred_pY_ense+ense_weights[runtime]*pred_pY
        time_stop=time.time()
        count=count+1
        done=count/runtimes
        remain=(runtimes-count)/runtimes
        num_day,num_hour,num_min=_calDueTime(time_start,time_stop,done,0.0)
        print("Model: %d Calculating Finished!      Done: %.2f%%, Remaining: %.2f%%"%(runtime,100*done,100*remain))
        print("Calculating will finish in %d Days %d Hours %d Minutes\n"%(num_day,num_hour,num_min))
    pred_Y_ense=(pred_pY_ense>=single_thres)*1
    pred_pY_ense=pred_pY_ense.reshape(matshape[0],matshape[1])
    pred_Y_ense=pred_Y_ense.reshape(matshape[0],matshape[1])
    if filter_percent>0:
        p_max=np.max(np.max(pred_pY_ense[bool_mask]))
        pred_pY_ense[pred_pY_ense<p_max*filter_percent]=0
    return [pred_Y_ense,pred_pY_ense]

def _predictSingleclassBaggingModelMatrix(CPIDs,RuntimeList,vtname,pred_X,varnames,selectruntimesvarnames,params,matshape,baggingweights,\
                                          single_thres,bool_gpu,n_gpus,n_parallels,bool_save,savedirbase):
    print("Predicting Singleclass Bagging Ensemble Models...")
    params_parallel=copy.deepcopy(params)
    process_pid=os.getpid()
    if len(CPIDs)<n_parallels:
        CPIDs.append(process_pid)
    process_pid_index=CPIDs.index(process_pid)
    print("Worker #%d: PID = %d"%(process_pid_index,process_pid))
    if bool_gpu:
        params_parallel['gpu_id']=process_pid_index%n_gpus    
    #Execute tasks
    pred_pY_ense=np.zeros(matshape[0]*matshape[1],dtype=np.float32) 
    for runtime in RuntimeList:
        print("Predicting Singleclass Model...    runtime = %d"%runtime)        
        savedir=savedirbase+os.sep+"runtime_"+str(runtime)
        modelName=vtname+'_xgboost_singleclass_run'+str(runtime)+".model"
        modeldir=savedir+os.sep+modelName
        model=xgbf.loadModel(modeldir,params_parallel)
        selruntimevarstr=selectruntimesvarnames[runtime]
        selruntimevaridx=_findListSubsetIndexes(selruntimevarstr,varnames)
        pred_X_runtime=pred_X[:,selruntimevaridx]
        [pred_Y,pred_pY]=xgbf.Predict(model,pred_X_runtime,bool_binary=1,threshold=single_thres)
        pred_pY_ense=pred_pY_ense+baggingweights[runtime]*pred_pY
        print("Model: %d Calculating Finished!\n"%(runtime))
    return pred_pY_ense


def predictSingleclassBaggingModelMatrix_parallel(Models,MatX,vtname,varnames,params,n_gpus,n_parallels,\
                                                  single_thres=0.5,runtimes=300,filter_percent=0,bool_save=True,savedirbase=""):
    if not bool_save:
        print("Single Bagging Ensemble Only for bool_save=True!")
        return []
    #Read weights and features file
    evalweightsFileName=vtname+"_Runtime_Evaluation_Weight.csv"
    selectvarnamesfiledir=savedirbase+os.sep+vtname+"_Runtime_Model_Select_Variables.csv"
    evalweightsFiledirto=savedirbase+os.sep+evalweightsFileName
    baggingweights=init.getListFromPandas(evalweightsFiledirto,'weight')
    selrunvarspdData=init.readCSVasPandas(selectvarnamesfiledir)
    selectruntimesvarnames=[]
    for runtime in range(runtimes):
        selectruntimesvarnames.append(init.getListFrompdDataSet(selrunvarspdData,"SelectVarName_run"+str(runtime)))
    del selrunvarspdData
    
    matshape=MatX.shape
    bool_mask=init.getMask(MatX)
    pred_X=np.zeros([matshape[0]*matshape[1],matshape[2]],dtype=np.float32)
    for i in range(matshape[2]):
        pred_X[:,i]=MatX[:,:,i].flatten()
    
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
    #Open multiprocessing parallel pools        
    P=Pool(n_parallels)
    results_parallel=[]
    manager=Manager()
    CPIDs=manager.list()
    
    for i in range(n_parallels):
        results_parallel.append(P.apply_async(_predictSingleclassBaggingModelMatrix,(CPIDs,RuntimeLists[i],vtname,pred_X,varnames,\
                                        selectruntimesvarnames,params,matshape,baggingweights,single_thres,bool_gpu,n_gpus,n_parallels,bool_save,savedirbase)))
    P.close()
    P.join()
    del CPIDs
    #Collect the multiprocessing results
    pred_pY_ense=np.zeros(matshape[0]*matshape[1],dtype=np.float32)
    for i in range(n_parallels):
        temp=results_parallel[i]
        pred_pY_ense_para=temp.get()
        pred_pY_ense=pred_pY_ense+pred_pY_ense_para
        
    pred_Y_ense=(pred_pY_ense>=single_thres)*1
    pred_pY_ense=pred_pY_ense.reshape(matshape[0],matshape[1])
    pred_Y_ense=pred_Y_ense.reshape(matshape[0],matshape[1])
    if filter_percent>0:
        p_max=np.max(np.max(pred_pY_ense[bool_mask]))
        pred_pY_ense[pred_pY_ense<p_max*filter_percent]=0
    return [pred_Y_ense,pred_pY_ense]
