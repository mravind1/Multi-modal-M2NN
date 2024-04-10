#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM 
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from math import floor
from sklearn.decomposition import PCA
import pickle
import pickletools, gzip
from itertools import product
from sklearn.cluster import KMeans
from collections import defaultdict
# alternative to load files    
def load_stuff(fnames):
    train = []
    valid = []
    test = []
    
    for fname in fnames:
 
        with gzip.open(fname, 'rb') as f:
            p = pickle.Unpickler(f)
            clf = p.load() 
        train.append(clf[0])
        valid.append(clf[1])
        test.append(clf[2])
    train = np.concatenate(train)
    valid = np.concatenate(valid)
    test = np.concatenate(test)
    return [train, valid, test]

def save_stuff(data,base_filename,filePart):
      print("base_filename", base_filename )  
      fname = base_filename  + filePart  + '.pkl'
      print("In save_Stuff",fname)
      with gzip.open(fname, "wb") as f:
        pickled = pickle.dumps(data,protocol=4)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)
      print("Write file",fname)
class Clusterizer():
    def __init__(self, rawDataList,num_clusters):
         
        self.rawDataList = rawDataList
        self.num_clusters=num_clusters 
        self.kmeans = {"EEG": KMeans(num_clusters),
                       "ICP": KMeans(num_clusters),
                       "ECG": KMeans(num_clusters),
                       "ABP": KMeans(num_clusters)
                       }
               
    def train(self):
        train_data = {"EEG": [],
                      "ICP": [],
                      "ECG": [],
                      "ABP": [],
                    }
        for patient in self.rawDataList:
        
            patient_seq = patient.x.shape[0]
            channelDim=patient.channelDim
            print("Clusterizer,patient.x.shape combined",patient.x.shape)
             
            data = patient.x.reshape(patient_seq, -1, channelDim)
            #print(data.shape)
            train_data['EEG'].append(data[:,:-3,:].reshape(patient_seq, -1))
            train_data['ICP'].append(data[:,-3,:])
            train_data['ECG'].append(data[:,-2,:])
            train_data['ABP'].append(data[:,-1,:])

        for typeOfFile, kmeans in self.kmeans.items():
          print("Clusterizer,typeOfFile")  
          print(typeOfFile)  
          data = np.concatenate(train_data[typeOfFile])
          #given training set define clusters
          kmeans.fit(data)
 
class RawDataLoader():

    
    def __init__(self,baseName=None,
                       labelFile=None,
                       ICPFile=None,
                       ECGFile=None,
                       ABPFile=None,
                       EEGRMTpath=None,
                       ICPRMTpath=None,
                       ECGRMTpath=None,
                       ABPRMTpath=None,
                       OUTPUTbase_filename=None,                        
                       TimeSegmentation=False, 
                       TimeSegments=211,
                       numChannels=None,
                       channelDim=None,
                       min_shift=33,
                       max_shift=38,
                       minimumPoints=1,
                       sequenceLength=500,
                       maxRMTFeaturesPerChunk=11000,
                       maxRMTFeaturesPerTimeStep=75,
                       reducedPCADim=10,
                       valid_fraction=0.2,
                       test_fraction=0.2,
                       forcedNonSeizure=False,
                       pca=None,                    
                       pcaDump_filename=None,
                       noSplitToNegPos=False,
                       shuffleTraining=False
                
                ):
        self.baseName=baseName
        self.labelFile=labelFile
        self.ICPFile=ICPFile
        self.ECGFile=ECGFile
        self.ABPFile=ABPFile
        self.EEGRMTpath=EEGRMTpath
        self.ICPRMTpath=ICPRMTpath
        self.ECGRMTpath=ECGRMTpath
        self.ABPRMTpath=ABPRMTpath
        self.numChannels=numChannels
        self.channelDim=channelDim
        self.min_shift=min_shift
        self.max_shift=max_shift
        self.minimumPoints=minimumPoints
        self.sequenceLength=sequenceLength
        self.EEGlength=None
        self.valid_fraction=valid_fraction
        self.test_fraction=test_fraction
        self.pca=pca
        self.shuffleTraining=shuffleTraining
        self.maxRMTFeaturesPerChunk=maxRMTFeaturesPerChunk
        self.maxRMTFeaturesPerTimeStep=maxRMTFeaturesPerTimeStep
        self.reducedPCADim=reducedPCADim
        self.OUTPUTbase_filename=OUTPUTbase_filename
        self.forcedNonSeizure=forcedNonSeizure 
        self.noSplitToNegPos=noSplitToNegPos
        self.TimeSegmentation=TimeSegmentation
        self.TimeSegments=TimeSegments
        self.pcaDump_filename=pcaDump_filename
        self.activeEEGChannelsAtTimesteps={}
        self.activeEEGChannelsAtTimestepsAtOctave={}
        if baseName is None:
            raise ValueError("basename not defined")
        if labelFile is None:
            raise ValueError("label file not defined")
        if numChannels is None:
            raise ValueError("numChannels not defined")
        if channelDim is None:
            raise ValueError("channelDim not defined")

        self.yTensor=None
        

     
    def combineData(self):

        print("Combining Power Spectrum data,in combineData!!!")
        x, y = self.loadData()
        self.EEGlength=len(x)
        self.nchuncks = floor(self.EEGlength / self.sequenceLength)  # 211
        print("EEG in combineData",x.shape)
         
        if self.ICPFile is not None:
            ICP = self.load_external_vars( self.ICPFile)
            print("ICP in combineData",ICP.shape)
        else:
            ICP = np.zeros([self.EEGlength, self.channelDim])
        if self.ECGFile is not None:
            ECG = self.load_external_vars( self.ECGFile)
            print("ECG in combineData",ECG.shape)
        else:
            ECG = np.zeros([self.EEGlength, self.channelDim])
        if self.ABPFile is not None:
            ABP = self.load_external_vars( self.ABPFile)
            print("ABP in combineData",ABP.shape)
        else:
            ABP = np.zeros([self.EEGlength, self.channelDim])    
        x = np.concatenate([x, ICP, ECG, ABP], axis=1)
        print("x.shape for EEG,ICP,ECG and ABP,in combineData",x.shape)
        self.x=x
        self.y=y
         
         
        print("self.pca",self.pca)
        
    def prepareData(self,clusterizer=None):
        if clusterizer is not None:
            print("In prepareData,data.shape",self.x.shape)
            data = self.x.reshape(len(self.x), -1, self.channelDim) 
            print("data.shape after reshape",data.shape)
            EEGData=np.eye(clusterizer.num_clusters)[clusterizer.kmeans["EEG"].predict(data[:,:-3,:].reshape(len(self.x), -1))]
            ICPData=np.eye(clusterizer.num_clusters)[clusterizer.kmeans["ICP"].predict(data[:,-3,:])]
            ECGData=np.eye(clusterizer.num_clusters)[clusterizer.kmeans["ECG"].predict(data[:,-2,:])]
            ABPData=np.eye(clusterizer.num_clusters)[clusterizer.kmeans["ABP"].predict(data[:,-1,:])]
            self.x=np.concatenate([EEGData,ICPData,ECGData,ABPData],axis=1)
         
        xTensor = self.chunk_data(self.x)
        self.yTensor = self.chunk_data(self.y)
        print(xTensor.shape)
        print(self.yTensor.shape)
        #Forcing first six to be nonseizure if this arg is True
        if self.forcedNonSeizure:
            self.yTensor[:,:6,:]=0

        xTensor,self.yTensor  = self.create_lag(xTensor )
        x_pos, x_neg, y_pos, y_neg = self.anomaly_detectionAndSplit(xTensor)
        # test train split
        # Shuffle data if needed
        length_neg = len(x_neg)
        length_pos = len(x_pos)
        #
        # if there is a shuffling do so else just keep ascending order of chunks
        if self.shuffleTraining:
            self.splittingIndices_neg = self.buildRandomIndex(length_neg)
            self.splittingIndices_pos = self.buildRandomIndex(length_pos)
        else:#first 60% of neg+ first 60% of pos


            self.splittingIndices_neg = np.arange(length_neg)
            self.splittingIndices_pos = np.arange(length_pos)
        # this doesnt do anything,put chunks in regular ascending order
        x_neg = self.shuffle(x_neg, self.splittingIndices_neg)
        y_neg = self.shuffle(y_neg, self.splittingIndices_neg)

        x_pos = self.shuffle(x_pos, self.splittingIndices_pos)
        y_pos = self.shuffle(y_pos, self.splittingIndices_pos)

        # x=[x_train, x_valid, x_test]
        #  [(i,j) for i,j in zip([1,2,3,4,5],['a','b','c','d','e'])
        # [(1,'a') , (2,'b'), (3,'c'), (4,'d'), (5,'e')]
        # zip pairs 1,a 2,b 3,c 4,d and 5,e
        # concatenate_neg_pos_chunks(data_neg,data_pos,combinedWindow,numChannels,channelDim,seqLength)
        # test_train_split(y_neg) = [y_train_neg, y_valid_neg, y_test_neg]
        # test_train_split(y_pos) = [y_train_pos, y_valid_pos, y_test_pos]
        # for neg, pos in zip(test_train_split(y_neg), test_train_split(y_pos)):
        #   print(neg, pos)

        # 1st iteration:neg,pos=y_train_neg, y_train_pos
        # 2nd iteration:neg,pos=y_valid_neg, y_valid_pos
        # 3rd iteration:neg,pos=y_test_neg, y_test_pos
        # concatenate_neg_pos_chunks(neg,pos,max_shift,numChannels,channelDim,sequence_length)
        # 1st iteration: concatenate_neg_pos_chunks(y_train_neg, y_train_pos, ...) -> y_train
        # 2nd iteration: concatenate_neg_pos_chunks(y_valid_neg, y_valid_pos, ...) -> y_valid
        # 3rd iteration: concatenate_neg_pos_chunks(y_test_neg, y_test_pos, ...) -> y_test

        #test_train_split(x_neg) = [stuff1, stuff2, stuff3]
        #test_train_split(x_pos) = [stuff4, stuff5, stuff6]
        #Concatenate
        #[[stuff1, stuff4],-->first time in for loop x_neg_train,x_pos_train
        # [stuff2, stuff5],-->second time in for loop x_neg_valid,x_pos_valid
        # [stuff3, stuff6]],-->third time in for loop x_neg_test,x_pos_test
        #[train_x,
        # valid_x
        # test_x]
        # final result
        # y = [y_train, y_valid, y_test]
        # x = [x_train, x_valid, x_test]

        if clusterizer is not None:
            self.pseudoDatachannels=1+3#4
            self.pseudoDim=clusterizer.num_clusters
        else:
            self.pseudoDatachannels=self.numChannels+3
            self.pseudoDim=self.channelDim
        print("In prepareData,self.pseudoDataChannels",self.pseudoDatachannels)   
        x_shape = (-1, self.sequenceLength - (self.max_shift), self.pseudoDatachannels, self.pseudoDim)
        y_shape = (-1, self.sequenceLength - (self.max_shift), 1)
        # for each thing in for loop concatenate_neg_pos_chunks(neg, pos, y_shape) is called
        y = [self.concatenate_neg_pos_chunks(neg, pos, y_shape) for neg, pos in
             zip(self.test_train_split(y_neg), self.test_train_split(y_pos))]


        x = [self.concatenate_neg_pos_chunks(neg, pos, x_shape) for neg, pos in
         zip(self.test_train_split(x_neg), self.test_train_split(x_pos))]
        return x, y
    
    def loadData(self):
        dfs = []
        if self.TimeSegmentation is not True:
             
            for i in range(1,self.numChannels+1):
                fname = self.baseName.format(str(i))
                #print(fname)
                dfs.append(pd.read_csv(fname, header=None))
            df = pd.concat(dfs,axis=1)


            x=df.values
            
        else:
             
            for i in range(1,self.TimeSegments+1):
                fname=self.baseName.format(str(i))
                df=pd.read_csv(fname,header=None)
                print(df.shape)
                print(df.head())
                dfs.append(df)
            df = pd.concat(dfs, axis=0)
            x=df.values
            print("x shape",x.shape)
        y=pd.read_csv(self.labelFile,header=None)
        y=y.values.reshape(-1,1)
        print("y shape",y.shape)
        return x,y

    def load_external_vars(self,fname):
          fname1 = os.path.join(fname)
           
          df1=pd.read_csv(fname1, header=None)
          external_var=np.zeros((self.EEGlength,self.channelDim))
          available_length = min(self.EEGlength, len(df1))
          external_var[0:available_length,:self.channelDim]=df1.values[0:available_length,:self.channelDim]
          return external_var
   
    def chunk_data(self,data ):
        last_input = self.sequenceLength*self.nchuncks  
         
        shape=[self.nchuncks, self.sequenceLength]+list(data.shape[1:] )
        dataTensor=data[:last_input].reshape(shape)
        return dataTensor
     
    def create_lag(self,xTensor):
        # interval between max and min shift
        lag_window = self.max_shift-self.min_shift +1

        #lag_shift is shift or the start of the lag_window - minimum warning of onset of anomaly 48seconds
        #lag_window is number of timestep to include, is the duration of how long we have to look in future,from 48sec
        if (lag_window  == 0):
            if (self.min_shift == 0):
                return xTensor.copy(), self.yTensor.copy()
            else:
                new_x = xTensor[:, :-self.min_shift, :]
                new_y = self.yTensor[:, self.min_shift:, :]
                return new_x, new_y
        #cutting last values in x
        #new_x  = xTensor[:, :-(max_shift), :]
        new_x  = xTensor[:, :-(lag_window-1+self.min_shift), :]
        #cutting first values in y
        #result = self.yTensor[:, max_shift :, :]
        result = self.yTensor[:, lag_window-1 + self.min_shift :, :]
        # is steps from now an anomaly
        for lag in range(0,lag_window-1):
            #print('tmp@lag={}: y[{}:{}]'.format(lag, (lag + self.min_shift),-(lag_window-1 -lag)))
            #                  ( 0 +33):-(6-1-0)
            tmp = self.yTensor[:, (lag + self.min_shift):-(lag_window-1 -lag), :]

            result = result+tmp #count of anomalies in the window
         # this is equivalent to a logical OR if minimumPoints is 1
        # this is equivalent to a logical AND if minimumPoints is size of window
        result=np.where(result >= self.minimumPoints,1,0)
        return new_x, result

    def anomaly_detectionAndSplit(self,xTensor ):

        x_neg = []
        x_pos = []
        y_neg=[]
        y_pos=[]
        countofZeros=0
        countofOnes=0
        countofOnesInAPosChunk=[]
        local_chunck_size=self.yTensor.shape[1]
        for i in range(self.nchuncks):
          countofOnesInAPosChunk.append(0)
          flag = False
          for j in range(local_chunck_size):
            if self.yTensor[i,j,0] == 1:
              countofOnes +=1
              countofOnesInAPosChunk[-1]+=1
              flag = True
            else:
                countofZeros +=1
#if no split everything goes to positive chunk
          if flag or self.noSplitToNegPos:


              x_pos.append(xTensor[i])
              y_pos.append(self.yTensor[i,:,:])

          else:

              x_neg.append(xTensor[i])
              y_neg.append(self.yTensor[i,:,:])
              del countofOnesInAPosChunk[-1]

        print("anomaly_detectionAndSplit")
        print(np.asarray(x_neg).shape)  
        print(np.asarray(x_pos).shape)  
        print(np.asarray(y_neg).shape)  
        print(np.asarray(y_pos).shape)  
        return x_pos,x_neg,y_pos,y_neg

    def buildRandomIndex(self,length):
        splittingIndices = np.arange(length)
        # print(splittingIndices)
        np.random.shuffle(splittingIndices)
        return splittingIndices


    def shuffle(self,data, splittingIndices):
        shuffled = [data[i] for i in splittingIndices]
        return shuffled


    def test_train_split(self,data):
        local_nchuncks = len(data)
        trainFraction=1-self.valid_fraction-self.test_fraction
        train_cutoff = floor((1-self.valid_fraction-self.test_fraction)*local_nchuncks )
        valid_cutoff = floor((1-self.test_fraction)*local_nchuncks )
        test_cutoff = local_nchuncks
        try:
            print("test_train_split in train",np.max(data[:train_cutoff]))
            print("test_train_split in valid",np.max(data[train_cutoff:valid_cutoff]))
            print("test_train_split in test",np.max(data[valid_cutoff:test_cutoff]))
        except ValueError:
            pass
        return data[:train_cutoff], data[train_cutoff:valid_cutoff],data[valid_cutoff:test_cutoff]


    def concatenate_neg_pos_chunks(self,data_neg,data_pos,shape):
        print("Target shape in concatenate",shape)
        
        print("data_neg shape")
        print(np.asarray(data_neg).shape)
        print("data_pos shape")
        print(np.asarray(data_pos).shape)
        #if data_neg does not have same no of dimensions as data_pos
        #make data_neg the same shape as data_pos
       	if len(data_neg)!=0 and len(data_pos)!=0:
          x=np.concatenate([data_neg,data_pos])
        elif len(data_neg)==0 and len(data_pos)!=0:
          x=data_pos
        elif len(data_neg)!=0 and len(data_pos)==0:
          raise ValueError('no positive samples in data')
        else:
          x=np.zeros((0,shape[1],shape[2]))
        
         
        x=np.asarray(x).reshape(*shape)
        try:
            print("concatenate_neg_pos_chunks",np.max(x))
        except ValueError:
            pass
        return x

 
    
    #Go through each patient
    #Go through each channel 
    #Make a dictionary and store the link between a timestep and list of EEG channels/corresponding RMT features stacked/active there 
    def pre_singleChannel(self, path, channelNum=None,mode='Run'):
        # dataframe starts from 0
        if channelNum is not None:
            df = pd.read_csv(path.format(channelNum), header=None)
        else:
            df = pd.read_csv(path, header=None)

        chunck_cols, timestamps_cols = self.plan_RMT_chunks(df,channelNum)
        #print("pre_singleChannel",chunck_cols) 
        values, timestamps = self.parseRMT(df, chunck_cols, timestamps_cols)

        self.valueCollection = self.allSplitting(values)
        self.timeStampCollection = self.allSplitting(timestamps)
        if mode != 'Run':
            
            #print("self.pcaDump_filename",self.pcaDump_filename)
            #Save in pklformat as it is more compact
            save_stuff(self.valueCollection[0],self.pcaDump_filename,f'/PCA_Dump_{channelNum}')
             
             
        return self.valueCollection[0] 

    #Per patient per channel
    def post_singleChannel(self, index_forPCA_transferred):
        print("post_singleChannel")
        reducedvalueCollection = self.parameterReduction(index_forPCA_transferred, self.valueCollection)
        
        valuesStackedTrain = self.stack_features_ontimestep(reducedvalueCollection[0], self.timeStampCollection[0][:, :, 0])
        valuesStackedValid = self.stack_features_ontimestep(reducedvalueCollection[1], self.timeStampCollection[1][:, :, 0])
        valuesStackedTest = self.stack_features_ontimestep(reducedvalueCollection[2], self.timeStampCollection[2][:, :, 0])

        del self.valueCollection
        del self.timeStampCollection
        
        return valuesStackedTrain, valuesStackedValid, valuesStackedTest
    # For explicit case,all time segments are considered a single channel    
    def preloaded_preSingleChannel(self,df,index_forPCA_transferred,mode='Run'):
        chunck_cols, timestamps_cols=self.plan_RMT_chunks(df)
        values,timestamps=self.parseRMT(df,chunck_cols,timestamps_cols)
        self.valueCollection=self.allSplitting(values)
        self.timeStampCollection=self.allSplitting(timestamps)
        return self.valueCollection[0] 
    # For explicit case,all time segments are considered a single channel
    def preloaded_postSingleChannel(self,index_forPCA_transferred):
        reducedvalueCollection=self.parameterReduction(index_forPCA_transferred,self.valueCollection)
        
        valuesStackedTrain=self.stack_features_ontimestep(reducedvalueCollection[0],self.timeStampCollection[0][:,:,0])
        valuesStackedValid=self.stack_features_ontimestep(reducedvalueCollection[1],self.timeStampCollection[1][:,:,0])
        valuesStackedTest=self.stack_features_ontimestep(reducedvalueCollection[2],self.timeStampCollection[2][:,:,0])


        return valuesStackedTrain,valuesStackedValid,valuesStackedTest


        
     


    def multihot_to_dense(self,vecs, shape):
      output = np.zeros(shape)
       
      for i in range(shape[0]):#no of chunks 
        for j in range(shape[1]): #462 lists of which channels are active
           for pos in vecs[i][j]: 
             output[i,j,pos] = 1
                 
                
      return output
    # Can be used separately to find active Channels at an Octave
    def get_activeChannelsAtATimestepAtOctave(self,octaveKeys):
          
        octave_results = {}
        for octave in octaveKeys:
            results = []# Return a 3d list:  chunk, timestep, channels
            for chunck in range(len(self.activeEEGChannelsAtTimestepsAtOctave[1][octave])):#look at first channel,first octavebin to see no of chuncks
                 
                chunck_results = defaultdict(list)  #keys are times and values are list of active channels
                for chan, octave_bin_dict in self.activeEEGChannelsAtTimestepsAtOctave.items():#1:[[]],2:[[]]go through each channel
                    #print("chan",chan)#[[]]
                #print("chuncks",chuncks)
                    times = octave_bin_dict[octave][chunck]
                    for time in times:
                    #Finally add
                        chunck_results[time].append(chan)

                flat_chuncks_results = []
                     
                for time in range(self.sequenceLength-self.max_shift):
                    flat_chuncks_results.append(chunck_results[time])
                print("results")    
                print(np.asarray(results).shape)    
                results.append(flat_chuncks_results)
                                    
            v_pos,v_neg,y_pos,y_neg=self.anomaly_detectionAndSplit(results)

            v_neg=self.shuffle(v_neg,self.splittingIndices_neg)
            v_pos=self.shuffle(v_pos,self.splittingIndices_pos)
             
            shape_neg=(len(v_neg), len(v_neg[0]), self.numChannels+3+1)#we dont use zero index
            shape_pos=(len(v_pos), len(v_pos[0]), self.numChannels+3+1)


            #from a list of channels to multionehot
            v_neg = self.multihot_to_dense(v_neg, shape_neg)
            v_pos = self.multihot_to_dense(v_pos,shape_pos )

            v_shape = [-1] + list(np.asarray(v_pos).shape[1:])
            #print(v_shape)
            # # Verbose Version
            v_neg_train, v_neg_valid,v_neg_test=self.test_train_split(v_neg)
            v_pos_train, v_pos_valid,v_pos_test=self.test_train_split(v_pos)
            v_train = self.concatenate_neg_pos_chunks(v_neg_train, v_pos_train, v_shape)
            v_valid = self.concatenate_neg_pos_chunks(v_neg_valid, v_pos_valid, v_shape)
            v_test  = self.concatenate_neg_pos_chunks(v_neg_test,  v_pos_test,  v_shape)
            active = [v_train, v_valid, v_test] 
            octave_results[octave] = active
        return octave_results

  
    # Can be used separately to find active Channels
    def get_activeChannelsAtATimestep(self):
          # Return a 3d list:  chunk, timestep, channels

        results = []
        
          
        for chunck in range(len(self.activeEEGChannelsAtTimesteps[1])):
            
            #print("chunck",chunck)#0..211
            chunck_results = defaultdict(list)  #keys are times and values are list of active channels
            for chan, chuncks_dict in self.activeEEGChannelsAtTimesteps.items():#1:[[]],2:[[]]
                #print("chan",chan)#[[]]
                #print("chuncks",chuncks)
                times_dict = chuncks_dict[chunck]
                for time_dict in times():
                    #print("time_dict",time_dict)
                    octave_results=defaultdict(list)
                    for key,times in time_dict.items():
                        #print("key",key)
                        for time in times:  
                            #print("time",time)
                            chunck_results[time].append(chan)
                    octave_results[key].append(chunck_results)  
            for octave, chunck_results in octave_results:  
                #print("octave",octave)
                flat_chuncks_results = []
                 
                for time in range(self.sequenceLength-self.max_shift):
                    #print("time flattened",time)
                    flat_chuncks_results.append(chunck_results[time])
                results.append(flat_chuncks_results)
        
        v_pos,v_neg,y_pos,y_neg=self.anomaly_detectionAndSplit(results)
          

        v_neg=self.shuffle(v_neg,self.splittingIndices_neg)
        v_pos=self.shuffle(v_pos,self.splittingIndices_pos)
        #from a list of channels to multionehot
        v_neg = self.multihot_to_dense(v_neg, (len(v_neg), len(v_neg[0]), self.numChannels+3+1))
        v_pos = self.multihot_to_dense(v_pos, (len(v_pos), len(v_pos[0]), self.numChannels+3+1))
                
        v_shape=(-1,v_pos.shape[-2],v_pos.shape[-1])
        #print(v_pos)
         
        v_neg_train, v_neg_valid,v_neg_test=self.test_train_split(v_neg)
        v_pos_train, v_pos_valid,v_pos_test=self.test_train_split(v_pos)
        v_train = self.concatenate_neg_pos_chunks(v_neg_train, v_pos_train, v_shape)
        v_valid = self.concatenate_neg_pos_chunks(v_neg_valid, v_pos_valid, v_shape)
        v_test  = self.concatenate_neg_pos_chunks(v_neg_test,  v_pos_test,  v_shape)
        v = [v_train, v_valid, v_test] 
        return v


    #Plan RMT chunks
    def plan_RMT_chunks(self,df,channelNum=None):
         
        chunck_cols = [[] for _ in range(self.nchuncks)]
        timestamps_cols = [[] for _ in range(self.nchuncks)]
        #chunk is the outer list, timesteps is the inner list where channel is active

        
        activeEEGChannel_at_timestamps = [[] for _ in range(self.nchuncks)]
        def get_holder():
            return [[] for _ in range(self.nchuncks)]
        #a dict of 2D arrays for each octave bin 1:{1}{[ [297],[379], [464]]}
        
        new_dict = defaultdict(get_holder)
        #enumerate through list of RMT columns at a time, to see which chunck or timestep does this go to
        for i, col in enumerate(df.columns):
            time = int(df.iloc[1,i]) #find the time corresponding to each RMT column
            standardDevTime=df.iloc[3,i]# find the standard deviation to calculate length of feature in time
            standardDevTime=int(standardDevTime) #
            octaveTime_bin=int(df.iloc[5,i])# find which octave the RMT feature column belongs to
             
            # chunk 0 runs from 0 to 499,500 to 1000 time etc need to find which chunck does the columns belong to
            if octaveTime_bin in [1,2,3]:
                for t in range(time - 3*standardDevTime, time + 3*standardDevTime + 1):
                    z = (t-time)/standardDevTime#3
                    p = np.exp(-z**2) # t==time: p=1,  t far from time, p~=0.05
                    chunk_id = floor(t/self.sequenceLength)
                     
                     
                    if (t >= self.sequenceLength * self.nchuncks or t<0 ) :
                        continue
   
                    randNum=np.random.rand(1)
                    if randNum < p:
                        timestamps_cols[chunk_id].append(t%self.sequenceLength)
                        
                        chunck_cols[chunk_id].append(col)
                         
                for t in range(time -0*standardDevTime, time + 0*standardDevTime + 1):    
                    chunk_id = floor(t/self.sequenceLength)
                    if (t >= self.sequenceLength * self.nchuncks or t<0 ) :
                        continue


                    activeEEGChannel_at_timestamps[chunk_id].append(t%self.sequenceLength)
                    new_dict[octaveTime_bin][chunk_id].append(t%self.sequenceLength)
        #For each channel we have this dictionary        
        self.activeEEGChannelsAtTimesteps[channelNum]=activeEEGChannel_at_timestamps
        self.activeEEGChannelsAtTimestepsAtOctave[channelNum] = new_dict
         
        return chunck_cols,timestamps_cols    
    


    # go through top level list of RMT columns corresponding to chunks
    def parseRMT(self,df,chunck_cols,timestamps_cols):
        values = np.zeros((self.nchuncks, 138, self.maxRMTFeaturesPerChunk))  
        # timestamps has one value for each RMT column
        timestamps= -np.ones((self.nchuncks ,1,self.maxRMTFeaturesPerChunk)).astype(int)  
         
        
         
        for i, cols in enumerate(chunck_cols):
          # get the col names corresponding to the chunks
          # put selected feature information to chunk(1,138,400)
          # columns corresponding to a chunk column 4 and column 5 for example,make a df to nparray with df.values


            if len(cols) >self.maxRMTFeaturesPerChunk:
                cols=cols[0:self.maxRMTFeaturesPerChunk]
                timestamps_cols[i]=timestamps_cols[i][0:self.maxRMTFeaturesPerChunk]
            values[i,:,:len(cols)] =  df.iloc[:,cols].values 
             
            timestamps[i,0,:len(cols)] = np.array(timestamps_cols[i]).astype(int)
        values=np.swapaxes(values,2,1)   
        timestamps=np.swapaxes(timestamps,2,1)  

        return values,timestamps

    def allSplitting(self,values):
        v_pos,v_neg,y_pos,y_neg=self.anomaly_detectionAndSplit(values)
        #test train split into three regions

        v_neg=self.shuffle(v_neg,self.splittingIndices_neg)
        v_pos=self.shuffle(v_pos,self.splittingIndices_pos)
        
        print("v_pos shape", np.asarray(v_pos).shape)
        print("v_neg shape",np.asarray(v_neg).shape)
        v_shape = [-1] + list(np.asarray(v_pos).shape[1:])
        
        # test train split
        v_neg_train, v_neg_valid,v_neg_test=self.test_train_split(v_neg)
        v_pos_train, v_pos_valid,v_pos_test=self.test_train_split(v_pos)
        v_train = self.concatenate_neg_pos_chunks(v_neg_train, v_pos_train, v_shape)
        v_valid = self.concatenate_neg_pos_chunks(v_neg_valid, v_pos_valid, v_shape)
        v_test  = self.concatenate_neg_pos_chunks(v_neg_test,  v_pos_test,  v_shape)
        v = [v_train, v_valid, v_test]
        

        return v
    def parameterReduction(self,index_forPCA_transferred,valueCollection):

        v_train,v_valid,v_test=valueCollection
        flatdataTrain=v_train.reshape(-1,138)
        flatdataValid=v_valid.reshape(-1,138)
        flatdataTest=v_test.reshape(-1,138)
        #if None use that index
        #print("parameterReduction,self.pca[index_forPCA_transferred",self.pca )   
        if self.pca[index_forPCA_transferred] is None:
            
            self.pca[index_forPCA_transferred] =PCA(self.reducedPCADim)
            # it figures out coefficients of pca transform only on the train set
            self.pca[index_forPCA_transferred].fit(flatdataTrain[:,10:] )
         
        # pca transform uses the coefficients to give output
        if flatdataTrain.shape[0]==0:
            #print("index_forPCA_transferred",index_forPCA_transferred)
            #print("flatdataTrain shape",flatdataTrain.shape)
            reduced_dataTrain =flatdataTrain
        else:    
            #print("flatdataTrain shape",flatdataTrain.shape)
            reduced_dataTrain =self.pca[index_forPCA_transferred].transform(flatdataTrain[:,10:])
        if  flatdataValid.shape[0]==0:  
            reduced_dataValid =flatdataValid
        else:    
            reduced_dataValid =self.pca[index_forPCA_transferred].transform(flatdataValid[:,10:])
        if flatdataTest.shape[0]==0:    
            reduced_dataTest =flatdataTest
        else:    
            reduced_dataTest =self.pca[index_forPCA_transferred].transform(flatdataTest[:,10:])



        # undoing the flattening
        reduced_dataTrain=reduced_dataTrain.reshape(-1,self.maxRMTFeaturesPerChunk,self.reducedPCADim)#126,400,10
        reduced_dataValid=reduced_dataValid.reshape(-1,self.maxRMTFeaturesPerChunk,self.reducedPCADim)
        reduced_dataTest=reduced_dataTest.reshape(-1,self.maxRMTFeaturesPerChunk,self.reducedPCADim)
        return reduced_dataTrain,reduced_dataValid,reduced_dataTest

    def stack_features_ontimestep(self,channelData,timesteps):
        from collections import defaultdict
        #print("channelData.shape",channelData.shape)
        values=np.zeros((channelData.shape[0],self.sequenceLength,self.maxRMTFeaturesPerTimeStep,self.reducedPCADim))
        for i in range(channelData.shape[0]):

            depthCount=defaultdict(int)
             
            for k in range(self.maxRMTFeaturesPerChunk):
                # timestep1, timestep2 etc upto timestep 500
                 
                time=timesteps[i,k].astype(int)
                if time== -1:
                    continue
                values[i,time,depthCount[time],:]=channelData[ i,k,:]
                if depthCount[time]==self.maxRMTFeaturesPerTimeStep-1:
                    pass
                else:    
                    depthCount[time]+=1
        return values
    
    def rank_signals(self,train, valid, test, data_configuration ):
        # For ICP, ECG and ABP sizes initialize to 1
        second_size = 1
        third_size = 1
        fourth_size = 1
      # data_configuration is either "x" or "channel"
        if data_configuration == 'channel':
             
            print("self.numChannels",self.numChannels)
            first_size=self.numChannels
             
            if self.TimeSegmentation is True:                 
                first_size=1
                
            sizes = (first_size, second_size, third_size, fourth_size)
            assert train.shape[1] == sum(sizes)
        else:# if it is the input data,sizes is 15,1,1,1 or 1,1,1,1 with and without clustering for example
            
             
             
            first_size=self.pseudoDatachannels-3
            sizes = (first_size, second_size, third_size, fourth_size)
            
        #all combinations of length of sizes,-1 is to escape all False,False,False,False condition
        fnames=[]
         
        combinations = list(product([True,False], repeat=len(sizes)))[:-1]  #e.g. [[True, True, True, True], [True, True, True, False]] etc upto all fifteen combinations.
        print("combinations",combinations)
        # For each combination/setting apply mask to train, valid and test data
        for combination in combinations:
            
          mask = sum([ [boolean]*size for boolean, size in zip(combination, sizes) ],[])
            
          if (data_configuration == 'channel'):
            mask = np.array(mask).reshape([1,-1,1,1,1])
          else:
            mask = np.array(mask).reshape([1,1,-1,1])
          
            
          # Apply mask to train, valid and test regions   
          train_prune = train*mask            
          valid_prune = valid*mask
          test_prune  = test*mask
          data = [train_prune, valid_prune, test_prune]

          comb_name = ''.join([word[1] for word in str(combination).split(',')])
          fname = self.OUTPUTbase_filename + '_' + data_configuration + '_' + comb_name + '.pkl'
          #print(fname) 
          fnames.append(fname)
          with gzip.open(fname, "wb") as f:
            pickled = pickle.dumps(data,protocol=4)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
        return fnames  




