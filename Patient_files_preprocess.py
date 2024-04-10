import os
from os.path import exists
import numpy as np
import pandas as pd
from EEGUtility_segmented_V4 import RawDataLoader
from EEGUtility_segmented_V4 import Clusterizer
from sklearn.decomposition import PCA,IncrementalPCA
import pickle
import pickletools, gzip
import gc
#Dump all files of trainers
class PCA_trainer():
    def __init__(self, basenames, numEEGChannelPCAs, reducedPCADim):
        self.basenames = basenames
        self.numChannels = numEEGChannelPCAs
        self.reducedPCADim=reducedPCADim
        self.pca=[None]*(self.numChannels + 4)
         
    def run(self):
        
        for i, ii in enumerate(list(range(1,self.numChannels+1)) + [-3,-2,-1]):
                suffix = '/PCA_Dump_'+str(ii)+'.pkl'
                  #for each channel we have to aggregate the data
                data = []
                for basename in self.basenames:
                    fname = basename+suffix
                    print("In PCAtrainer,fname",fname)
                    print("Which folder am I running from",os.getcwd())
                    #read the corresponding channel eg:7 data from everyone
                    if exists(fname):
                        print("file exists")
                        with gzip.open(fname, 'rb') as f:
                            p = pickle.Unpickler(f)
                            clf = p.load()                 
                            data.append(clf)
                    else:
                        print("file does not exist")
                if len(data) ==0:     
                    data=[np.zeros((7546000,138))]  #give a large number       
                data = np.concatenate(data)
                flatdataTrain= data.reshape(-1,138)  
                print("flatdataTrain")
                print(flatdataTrain.shape)
                self.pca[i+1] = PCA(self.reducedPCADim)    
                print("PCA of channel is about to be fitted ",i+1)
                print("self.reducedPCADim",self.reducedPCADim)
                self.pca[i+1].fit(flatdataTrain[:,self.reducedPCADim:] )                
                del data
                gc.collect()
        return self.pca        

class ProviderPatient():
    #List of RawData classes 
    def __init__(self, rawDataList,num_clusters,clusterizer=False):
        self.rawDataList = rawDataList
        self.pca = None
        self.num_clusters=num_clusters
        self.clusterizer=clusterizer
        
    def setPCA(self, v_train,index_forPCA_transferred, reducedPCADim):
         
        flatdataTrain= v_train.reshape(-1,138)  
        #if None use that index
         
        if self.pca[index_forPCA_transferred] is None:
            self.pca[index_forPCA_transferred] =PCA(reducedPCADim)
            # it figures out coefficients of pca transform only on the train set
            self.pca[index_forPCA_transferred].fit(flatdataTrain[:,reducedPCADim:] )


   
    def clusterAndPrepareAllData(self):
        #Read from files
        for rawData in self.rawDataList:
            print("Calling EEGUtility combineData")
            rawData.combineData()
        if self.clusterizer:
            print("Calling EEGUtility Clusterizer")
            clusterizer=Clusterizer(self.rawDataList, self.num_clusters)   
            clusterizer.train()
        else:
            clusterizer=None
        x_rawDataList=[]
        y_rawDataList=[]
        for rawData in self.rawDataList:
            print("Calling EEGUtility prepareData")
            x_rawData, y_rawData = rawData.prepareData(clusterizer)

            #List of all patients
            x_rawDataList.append(x_rawData)
            y_rawDataList.append(y_rawData)
               
    
        x_train = np.concatenate([x[0] for x in x_rawDataList], axis=0)
        x_valid = np.concatenate([x[1] for x in x_rawDataList], axis=0)
        x_test  = np.concatenate([x[2] for x in x_rawDataList], axis=0)
        y_train = np.concatenate([y[0] for y in y_rawDataList], axis=0)
        y_valid = np.concatenate([y[1] for y in y_rawDataList], axis=0)
       
        y_test  = np.concatenate([y[2] for y in y_rawDataList], axis=0)
        
        return [x_train, x_valid, x_test], [y_train, y_valid, y_test]



    def loadRMTMulti_Implicit_ExplicitPreloaded(self,mode='Run'): 
        
        for rawData in self.rawDataList:
            rawData.allvaluesTrain = []
            rawData.allvaluesValid = []
            rawData.allvaluesTest = []

        if self.rawDataList[0].TimeSegmentation is False:
            #Empty list of PCAs initialized
            if self.pca is None:
                self.pca=[None]*(self.rawDataList[0].numChannels + 4)
            for j in range(1, self.rawDataList[0].numChannels + 1):
                print("In loadRMTMulti_Implicit_Explicit, Channel........" + str(j))
                all_train=[]
                for rawData in self.rawDataList:
                    vtrain= rawData.pre_singleChannel(rawData.EEGRMTpath, j,mode)
                     
                    all_train.append(vtrain)
                     
                if mode != 'Run':
                    continue
     
                 
                #all_train is a list of vtrains
                if len(all_train)!=1:
                    v_train=np.concatenate(all_train,axis=0)
                else:
                    v_train=all_train[0]
                self.setPCA(v_train,j,self.rawDataList[0].reducedPCADim)            
                #post Single Channel
                for rawData in self.rawDataList:
                            
                     
                     
                    rawData.pca = self.pca
                    featuresChannel = rawData.post_singleChannel(j)  
                    #print("featuresChannel[0] shape")
                    #print(featuresChannel[0].shape)
                    rawData.allvaluesTrain.append(featuresChannel[0])
                    rawData.allvaluesValid.append(featuresChannel[1])
                    rawData.allvaluesTest.append(featuresChannel[2])    

        else:
            # ALl EEG TimeSegments share a common PCA considered as one,ICP,ECG and ABP
            # Collapsing all TimeSegments to a flat,columnwise,j starts at 1
            dfs=[]
            rawData = self.rawDataList[0]
            for j in range(1,self.rawDataList[0].TimeSegments+1):
                print("Chunks........"+str(j))
                filename=rawData.EEGRMTpath.format(j)
                print('filename',filename)
                df=pd.read_csv(filename,header=None)
                print("Features from each 500 timesteps")
                print(df.shape)
                 
                df.iloc[1,:]+=(j-1)*500  #500 is hard-coded chunck/segment length for RMT
                 
                dfs.append(df)
               
            #Glue all time segments together   
            df=pd.concat(dfs,axis=1)
            print("TS done",df.values.shape)
            print("df.head() of features",df.head)
            if self.pca is None:
                self.pca=[None,None,None,None,None]
                rawData.pca=self.pca
            
            index_forPCA_transferred=1
            vtrain = rawData.preloaded_preSingleChannel(df, 1,mode)
            self.setPCA(vtrain,1,rawData.reducedPCADim)
             
            featuresChannel = rawData.preloaded_postSingleChannel(1)    
            print("featuresChannel[0] shape")
            print(featuresChannel[0].shape)
            rawData.allvaluesTrain.append(featuresChannel[0])
            rawData.allvaluesValid.append(featuresChannel[1])
            rawData.allvaluesTest.append(featuresChannel[2])    

                    
        print("ICP about to start")

        for rawData in self.rawDataList:
            if rawData.ICPRMTpath is not None:
                vtrain= rawData.pre_singleChannel(rawData.ICPRMTpath, -3,mode)
                if self.rawDataList[0].TimeSegmentation is False:
                    all_train.append(vtrain)
                else:
                    continue

        if mode == 'Run':
            if self.rawDataList[0].TimeSegmentation is False:
                if len(all_train)!=1:
                    v_train=np.concatenate(all_train,axis=0)
                else:
                    v_train=all_train[0]
                self.setPCA(v_train,-3,self.rawDataList[0].reducedPCADim)  
 
            for rawData in self.rawDataList:
                rawData.pca = self.pca
                if rawData.ICPRMTpath is not None:                   
                    featuresChannel = rawData.post_singleChannel(-3)  
                    rawData.allvaluesTrain.append(featuresChannel[0])
                    rawData.allvaluesValid.append(featuresChannel[1])
                    rawData.allvaluesTest.append(featuresChannel[2])
                else:
                    rawData.allvaluesTrain.append(np.zeros(rawData.allvaluesTrain[-1].shape))
                    rawData.allvaluesValid.append(np.zeros(rawData.allvaluesValid[-1].shape))
                    rawData.allvaluesTest.append(np.zeros(rawData.allvaluesTest[-1].shape))

        print("ECG about to start")
        all_train=[]
        for rawData in self.rawDataList:
            if rawData.ECGRMTpath is not None:
                vtrain= rawData.pre_singleChannel(rawData.ECGRMTpath, -2,mode)
                if self.rawDataList[0].TimeSegmentation is False:
                    all_train.append(vtrain)
                else:
                    continue

        if mode == 'Run':
            if self.rawDataList[0].TimeSegmentation is False:
                print("length of all_train",len(all_train))#here it is zero 
                 
                if len(all_train)>1:
                    v_train=np.concatenate(all_train,axis=0)
                    self.setPCA(v_train,-2,self.rawDataList[0].reducedPCADim)  
                elif len(all_train)==0:
                    pass #dont do anything
                else:
                    v_train=all_train[0]
                    self.setPCA(v_train,-2,self.rawDataList[0].reducedPCADim)                

            for rawData in self.rawDataList:
                rawData.pca = self.pca
                if rawData.ECGRMTpath is not None:                   
                    featuresChannel = rawData.post_singleChannel(-2)  
                    rawData.allvaluesTrain.append(featuresChannel[0])
                    rawData.allvaluesValid.append(featuresChannel[1])
                    rawData.allvaluesTest.append(featuresChannel[2])
                else:
                    rawData.allvaluesTrain.append(np.zeros(rawData.allvaluesTrain[-1].shape))
                    rawData.allvaluesValid.append(np.zeros(rawData.allvaluesValid[-1].shape))
                    rawData.allvaluesTest.append(np.zeros(rawData.allvaluesTest[-1].shape))

        print("ABP about to start")
        all_train=[]
        for rawData in self.rawDataList:
            if rawData.ABPRMTpath is not None:
                vtrain= rawData.pre_singleChannel(rawData.ABPRMTpath, -1,mode)
                if self.rawDataList[0].TimeSegmentation is False:
                    all_train.append(vtrain)
                else:
                    continue
 
        if mode == 'Run':
            if self.rawDataList[0].TimeSegmentation is False:
                if len(all_train)!=1:
                    v_train=np.concatenate(all_train,axis=0)
                else:
                    v_train=all_train[0]
                self.setPCA(v_train,-1,self.rawDataList[0].reducedPCADim)                  

            for rawData in self.rawDataList:
                rawData.pca = self.pca
                if rawData.ABPRMTpath is not None:                   
                    featuresChannel = rawData.post_singleChannel(-1)  
                    rawData.allvaluesTrain.append(featuresChannel[0])
                    rawData.allvaluesValid.append(featuresChannel[1])
                    rawData.allvaluesTest.append(featuresChannel[2])
                else:
                    rawData.allvaluesTrain.append(np.zeros(rawData.allvaluesTrain[-1].shape))
                    rawData.allvaluesValid.append(np.zeros(rawData.allvaluesValid[-1].shape))
                    rawData.allvaluesTest.append(np.zeros(rawData.allvaluesTest[-1].shape))

        if mode == 'Run':          
                allvaluesTrain = np.concatenate([rawData.allvaluesTrain for rawData in self.rawDataList], axis=1)
                allvaluesValid = np.concatenate([rawData.allvaluesValid for rawData in self.rawDataList], axis=1)
                allvaluesTest  = np.concatenate([rawData.allvaluesTest for rawData in self.rawDataList], axis=1)
                for rawData in self.rawDataList:
                    print("Checking axis error",mode)
                    print(np.asarray(rawData.allvaluesTrain).shape)  
                    print(np.asarray(rawData.allvaluesValid).shape) 
                    print(np.asarray(rawData.allvaluesTest).shape)


                    del rawData.allvaluesTrain
                    del rawData.allvaluesValid
                    del rawData.allvaluesTest
                print("channelDataTrain before",np.array(allvaluesTrain).shape)
                #first dimension needs to be chunks
                channelDataTrain = np.swapaxes(np.array(allvaluesTrain), 0, 1)  # Example size:126,26,500,75,10
                channelDataValid = np.swapaxes(np.array(allvaluesValid), 0, 1)  # Example size:42,26,500,75,10
                channelDataTest = np.swapaxes(np.array(allvaluesTest), 0, 1)    # Example size:43,26,500,75,10

                channelDataTrain = channelDataTrain[:, :, :self.rawDataList[0].sequenceLength - (self.rawDataList[0].max_shift),:, :]
                 
                channelDataValid = channelDataValid[:, :, :self.rawDataList[0].sequenceLength - (self.rawDataList[0].max_shift),:, :]
                channelDataTest = channelDataTest[:, :, :self.rawDataList[0].sequenceLength - (self.rawDataList[0].max_shift), :, :]

                del allvaluesTrain
                del allvaluesValid
                del allvaluesTest
                return  channelDataTrain,channelDataValid,channelDataTest   
        else:

                return None,None,None    



