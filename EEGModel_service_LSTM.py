import numpy as np
import pandas as pd
from EEGUtility_segmented_V4 import load_stuff
from keras.layers import Concatenate, Input, Reshape, Bidirectional,Dense,Lambda,RNN, Flatten
from keras.models import Model
from keras.layers import Input, Reshape,Concatenate,Dot,Permute ,RepeatVector,Multiply,LSTM
from keras.optimizers import Adam, SGD
import scipy.sparse as sp
from keras import backend as K
import tensorflow as tf
import keras

import os
import gc
def noneReader():
    while True:
        yield None
def reader(split, fnames, batchsize):
     
    while True:
        for fname in fnames:
             
            train, valid, test = load_stuff([fname])
            if split == 'train':
                subset = train
                del valid
                del test
            elif split == 'valid':
                del train
                subset = valid
                del test
            elif split == 'test':
                del train
                del valid
                subset = test
            gc.collect()
            #n is the number of full batches initially 
            n = len(subset)//batchsize
            print("In reader,len(subset)",len(subset))
            print("In reader,batchsize",batchsize)
            print("In reader,number of full batches",n)
            for i in range(n):
                 
                yield subset[i*batchsize:(i+1)*batchsize]
            #Returns partial batch
            if n*batchsize < len(subset):
                n=n+1         
                yield subset[(n-1)*batchsize:]  
            if n==0:
                raise ValueError(f"No batches available in {fname}")
            #for the print out how many batches 
            print(f"file {fname} in {split} has {n} batches") 

def data_generator(split, x_fnames, y_fnames, channel_fnames, batchsize, reducedsequenceLength,  nChannels_EEG,nDim,zeroed_channel_list=None):
    
    x_reader = reader(split, x_fnames, batchsize)
    if y_fnames is None:
        y_reader=noneReader()
    else: 
         
        y_reader = reader(split, y_fnames, batchsize)
    if channel_fnames is None:
        channel_reader=noneReader()
    else:    
         
        channel_reader = reader(split, channel_fnames, batchsize)
    # Give the next batch of data for training
    for x, y, chan in zip(x_reader, y_reader, channel_reader):
         
        if zeroed_channel_list is not None:
             
            for zeroed_channel in zeroed_channel_list:
                x[:,:,zeroed_channel,:] = 0
                chan[:,zeroed_channel,:,:,:] = 0
        x_EEG, x_ICP, x_ECG, x_ABP = x[:,:,:-3,:], x[:,:,-3,:], x[:,:,-2,:], x[:,:,-1,:]
         
        x_EEG = x_EEG.reshape(-1, reducedsequenceLength, nChannels_EEG, nDim )
        x_ICP = x_ICP.reshape(-1, reducedsequenceLength, 1, nDim )
        x_ECG = x_ECG.reshape(-1, reducedsequenceLength, 1, nDim )
        x_ABP = x_ABP.reshape(-1, reducedsequenceLength, 1, nDim )   
        if y_fnames is None:
            if channel_fnames is None:
                 
                yield ([x_EEG, x_ICP, x_ECG, x_ABP])     
                
            else:
                yield ([x_EEG, x_ICP, x_ECG, x_ABP, chan]) 
        else:    
            
            if channel_fnames is None:
                 
                yield ([x_EEG, x_ICP, x_ECG, x_ABP], y)     
                
            else:
                yield ([x_EEG, x_ICP, x_ECG, x_ABP, chan],y) 
         
    
class RunModel():

    def __init__(self,
                 seed=None,
                 folder=None,
                 base_filename=None,
                 datasets=['TTTT', 'TTTF', 'TTFT', 'TTFF',
                          'TFTT', 'TFTF', 'TFFT', 'TFFF', 'FTTT',
                          'FTTF', 'FTFT', 'FTFF', 'FFTT', 'FFTF', 'FFFT'
                          ],
                 output_test=[True]+[False]*14,#corresponds to the datasets
                 OUTPUT_folder=None,
                 pretrainedweights=None, 
                 
                 hidden_dim_EEG = None,
                 max_shift=None ,
                 TimeSegmentation=False,
                 hidden_dim_extVars = None,
                 sequenceLength = None,#  same as number of timesteps in a chunk
                 nChannels_EEG=None,
                 nDim=None,
                 channelShape=None,
                 num_heads=None,
                 **kwargs 
                                
                 ):
        self.seed=seed
        self.folder=folder
        self.base_filename=base_filename
        self.datasets=datasets
        self.output_test=output_test
        self.OUTPUT_folder=OUTPUT_folder
        self.pretrainedweights=pretrainedweights
        self.hidden_dim_EEG =  hidden_dim_EEG
        self.max_shift=max_shift
        self.TimeSegmentation=TimeSegmentation
        self.hidden_dim_extVars=hidden_dim_extVars
        self.sequenceLength = sequenceLength 
        self.reducedsequenceLength=self.sequenceLength-(self.max_shift)
         
        self.nChannels_EEG=nChannels_EEG
        self.nDim=nDim
        self.channelShape=channelShape,
        self.num_heads=num_heads 
        self.fnames_modelweights = os.path.join(self.folder, self.base_filename + '_model_run_muti_ConnectivityMatrix'+str(self.seed)+".h5")
 
    def loaded_shuffler(self, x_train_full, channel_train_full, y_train):
        #shuffling the order of training chunks
        print("loaded shuffler")
        index_train_full = np.arange(len(x_train_full))
        np.random.shuffle(index_train_full)
        x_train_full = x_train_full[index_train_full]
        channel_train_full = channel_train_full[index_train_full]
        # how many times we have repeated data
         
         
        n_repeats=int(len(self.datasets))
        y_train_full = np.concatenate(n_repeats * [y_train], axis=0)[index_train_full]
        return x_train_full, channel_train_full, y_train_full

    def loadFiles(self):
        fnames_yTensor = [os.path.join(self.folder, self.base_filename + '_y.pkl')]
        fnames_rawData = [os.path.join(self.folder, self.base_filename + '_x_' + setname + '.pkl') for setname in
                          self.datasets]
        fnames_channelData = [os.path.join(self.folder, self.base_filename + '_channel_' + setname + '.pkl') for setname in
                              self.datasets]
 

 
        # y_train,y_valid,y_test  
         
        y = load_stuff(fnames_yTensor)
        # 
        self.ntrainsamples=len(y[0])
        self.nvalidationsamples=len(y[1])
        self.ntestsamples=len(y[2])
        
         
        x_full = load_stuff(fnames_rawData)
        print("Loading channels")
        
        channel_full = load_stuff(fnames_channelData)
     
        x_full[0], channel_full[0], y[0] = self.loaded_shuffler(x_full[0], channel_full[0], y[0])
        self.nChannels_EEG = np.asarray(x_full[0]).shape[2] - 3
        self.nDim=np.asarray(x_full[0]).shape[3]
        
        
        x_EEG_train = x_full[0][:, :, :self.nChannels_EEG, :]
        x_EEG_valid = x_full[1][:, :, :self.nChannels_EEG, :]
        x_EEG_test = x_full[2][:, :, :self.nChannels_EEG, :]
            
        x_ICP_train = np.expand_dims(x_full[0][:, :, self.nChannels_EEG, :], 2)
        x_ICP_valid = np.expand_dims(x_full[1][:, :, self.nChannels_EEG, :], 2)
        x_ICP_test = np.expand_dims(x_full[2][:, :, self.nChannels_EEG, :], 2)

        x_ECG_train = np.expand_dims(x_full[0][:, :, self.nChannels_EEG + 1, :], 2)
        x_ECG_valid = np.expand_dims(x_full[1][:, :, self.nChannels_EEG + 1, :], 2)
        x_ECG_test = np.expand_dims(x_full[2][:, :, self.nChannels_EEG + 1, :], 2)

        x_ABP_train = np.expand_dims(x_full[0][:, :, self.nChannels_EEG + 2, :], 2)
        x_ABP_valid = np.expand_dims(x_full[1][:, :, self.nChannels_EEG + 2, :], 2)
        x_ABP_test = np.expand_dims(x_full[2][:, :, self.nChannels_EEG + 2, :], 2)
        self.y = {'train':y[0],'valid':y[1],'test':y[2]}
        self.channel = {'train':channel_full[0],'valid':channel_full[1],'test':channel_full[2]}
        self.channelShape=self.channel['train'].shape[1:]
        self.x = {'EEG_train': x_EEG_train, 'EEG_valid': x_EEG_valid, 'EEG_test': x_EEG_test,
                  'ICP_train': x_ICP_train, 'ICP_valid': x_ICP_valid, 'ICP_test': x_ICP_test,
                  'ECG_train': x_ECG_train, 'ECG_valid': x_ECG_valid, 'ECG_test': x_ECG_test,
                  'ABP_train': x_ABP_train, 'ABP_valid': x_ABP_valid, 'ABP_test': x_ABP_test}
        
    #LSTM-MMA attention     
    def buildModel_timeOnly_SplitAttention(self):
          
            # Take your regular input x of all EEG channels combined, ICP, ECG and ABP
             
            input_EEG = Input(shape=(self.reducedsequenceLength,self.nChannels_EEG,self.nDim)) 
            input_ICP = Input(shape=(self.reducedsequenceLength,1,self.nDim)) 
            input_ECG=Input(shape=(self.reducedsequenceLength,1,self.nDim))
            input_ABP=Input(shape=(self.reducedsequenceLength,1,self.nDim))
            # 
            flat_EEG = Reshape((-1,self.nChannels_EEG*self.nDim)) (input_EEG)
            flat_ICP = Reshape((-1,1*self.nDim)) (input_ICP)
            flat_ECG = Reshape((-1,1*self.nDim)) (input_ECG)
            flat_ABP = Reshape((-1,1*self.nDim)) (input_ABP)

            flat=Concatenate()([flat_EEG,flat_ICP,flat_ECG,flat_ABP])
           
            print("self.channelShape",self.channelShape)
            aspect_=Input(shape=self.channelShape,dtype=np.float32)
             
             
            permutedAspect=Permute((2,1,3,4))(aspect_)  
            flat_aspect = Reshape((self.reducedsequenceLength,-1)) (permutedAspect) # 2D
            
            # standard LSTMs takes inputs
            rnn_EEG = LSTM(self.hidden_dim_EEG, return_sequences=True) (flat_EEG)
            rnn_ICP = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ICP)
            rnn_ECG = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ECG)
            rnn_ABP = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ABP)

            rnn= Concatenate()([rnn_EEG,rnn_ICP,rnn_ECG,rnn_ABP])
            # we are putting the area around RMT features as query to match with data as key
            A0 = PreAttention(self.num_heads,10)([flat_aspect, flat])
            #V0 we have to do linear transformation,taking a vector and multiplying by a coefficient matrix 
             
             
            V0 = Dense(self.num_heads*10)(flat) #it increases the size of the model,might be helpful
            V0 = Reshape((-1, self.num_heads, 10))(V0)#10 is the dimension of the output of attention per multihead attention
            V0 = Permute((2,1,3))(V0)
            # if we have L1 weights to be trained change to trainable=True
            weighted0 = PostAttention(trainable=False)([A0,V0])

            rnn_with_aspect=Concatenate()([rnn, weighted0])
            #RMT data via weighted0 is telling you which part of LSTM is useful
            A1 = PreAttention(self.num_heads,10)([rnn_with_aspect, rnn])
            V1 = Dense(self.num_heads*10)(rnn)
            V1 = Reshape((-1,self.num_heads,10))(V1)
            V1 = Permute((2,1,3))(V1)
            weighted1 = PostAttention(trainable=False)([A1, V1])
            print("weighted1",weighted1.shape) 
            # concatenating all the intermediate hidden states to y
            output = Dense(1, activation='sigmoid')(weighted1)

            self.model = Model(inputs=[input_EEG,input_ICP,input_ECG,input_ABP,aspect_], outputs=output)
            #Regular model takes inputs and give outputs after training
            #attentionView model is built from same layers and hence shares actual coefficients 
            #When we train the regular model,this model's coefficients are also updated
            #attentionView model is going to output attention values instead
            #
            self.attentionViewModel=Model(inputs=[input_EEG,input_ICP,input_ECG,input_ABP,aspect_], outputs=A0)
             
            #optimizer=Adam(lr=0.001,beta_1=0.9, beta_2=0.999)
             
             
            # build method
             
            self.model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
            print(self.model.summary())
    
    #Competitor self-attention
    def buildModel_timeOnly_SplitAttention_NoRMT(self):
          
            # Take your regular input x of all EEG channels combined, ICP, ECG and ABP
             
            input_EEG = Input(shape=(self.reducedsequenceLength,self.nChannels_EEG,self.nDim)) 
            input_ICP = Input(shape=(self.reducedsequenceLength,1,self.nDim)) 
            input_ECG=Input(shape=(self.reducedsequenceLength,1,self.nDim))
            input_ABP=Input(shape=(self.reducedsequenceLength,1,self.nDim))
            # 
            flat_EEG = Reshape((-1,self.nChannels_EEG*self.nDim)) (input_EEG)
            flat_ICP = Reshape((-1,1*self.nDim)) (input_ICP)
            flat_ECG = Reshape((-1,1*self.nDim)) (input_ECG)
            flat_ABP = Reshape((-1,1*self.nDim)) (input_ABP)

            flat=Concatenate()([flat_EEG,flat_ICP,flat_ECG,flat_ABP])
           
            print("self.channelShape",self.channelShape)
            
            # standard LSTMs takes inputs
            rnn_EEG = LSTM(self.hidden_dim_EEG, return_sequences=True) (flat_EEG)
            rnn_ICP = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ICP)
            rnn_ECG = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ECG)
            rnn_ABP = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ABP)

            rnn= Concatenate()([rnn_EEG,rnn_ICP,rnn_ECG,rnn_ABP])
            # we are putting the area around RMT features as query to match with data as key
            A0 = PreAttention_noscaling()([flat, rnn])
            #V0 we have to do linear transformation,taking a vector and multiplying by a coefficient matrix 
             
             
            V0 = Dense(1*1)(rnn) #it increases the size of the model,might be helpful
            V0 = Reshape((-1, 1, 1))(V0)#10 is the dimension of the output of attention per multihead attention
            V0 = Permute((2,1,3))(V0)
            # if we have L1 weights to be trained change to trainable=True
            weighted0 = PostAttention_noscaling(trainable=False)([A0,V0])

            print("weighted0",weighted0.shape) 
            # concatenating all the intermediate hidden states to y
            output = Dense(1, activation='sigmoid')(weighted0)

            self.model = Model(inputs=[input_EEG,input_ICP,input_ECG,input_ABP], outputs=output)
            #Regular model takes inputs and give outputs after training
            #attentionView model is built from same layers and hence shares actual coefficients 
            #When we train the regular model,this model's coefficients are also updated
            #attentionView model is going to output attention values instead
            #
            self.attentionViewModel=Model(inputs=[input_EEG,input_ICP,input_ECG,input_ABP], outputs=A0)
             
             
            # build method
             
            self.model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
            print(self.model.summary())

    

    #Competitor no attention
    def buildModel_timeOnly_NoSplitAttention_NoRMT(self):
          
            # Take your regular input x of all EEG channels combined, ICP, ECG and ABP
             
            input_EEG = Input(shape=(self.reducedsequenceLength,self.nChannels_EEG,self.nDim)) 
            input_ICP = Input(shape=(self.reducedsequenceLength,1,self.nDim)) 
            input_ECG=Input(shape=(self.reducedsequenceLength,1,self.nDim))
            input_ABP=Input(shape=(self.reducedsequenceLength,1,self.nDim))
            # 
            flat_EEG = Reshape((-1,self.nChannels_EEG*self.nDim)) (input_EEG)
            flat_ICP = Reshape((-1,1*self.nDim)) (input_ICP)
            flat_ECG = Reshape((-1,1*self.nDim)) (input_ECG)
            flat_ABP = Reshape((-1,1*self.nDim)) (input_ABP)

            flat=Concatenate()([flat_EEG,flat_ICP,flat_ECG,flat_ABP])
           
            
            # standard LSTMs takes inputs
            rnn_EEG = LSTM(self.hidden_dim_EEG, return_sequences=True) (flat_EEG)
            rnn_ICP = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ICP)
            rnn_ECG = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ECG)
            rnn_ABP = LSTM(self.hidden_dim_extVars, return_sequences=True) (flat_ABP)

            rnn= Concatenate()([rnn_EEG,rnn_ICP,rnn_ECG,rnn_ABP])
            # concatenating all the intermediate hidden states to y
            #rnn_flat=Flatten()(rnn)
            output = Dense(1, activation='sigmoid')(rnn)

            self.model = Model(inputs=[input_EEG,input_ICP,input_ECG,input_ABP], outputs=output)
             
            self.model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
            print(self.model.summary())

 
    


    def trainModel(self):

             
            #validation is only first file as we are giving self.nvalidationsamples
            print("self.x['EEG_train']",self.x['EEG_train'].shape)
            print("self.x['EEG_valid']",self.x['EEG_valid'].shape)
            print("self.channel['train']",self.channel['train'].shape)
            print("self.channel['test']",self.channel['test'].shape)
            if(self.pretrainedweights is None):
                if len(self.x['EEG_valid'])==0:
                       validationdata=None 
                else:
                       validationdata=([self.x['EEG_valid'][:self.nvalidationsamples],
                                                                          self.x['ICP_valid'][:self.nvalidationsamples], 
                                                                          self.x['ECG_valid'][:self.nvalidationsamples],
                                                                          self.x['ABP_valid'][:self.nvalidationsamples], 
                                                               self.channel['valid'][:self.nvalidationsamples]], self.y['valid'])
                self.history = self.model.fit([self.x['EEG_train'],self.x['ICP_train'],self.x['ECG_train'],self.x['ABP_train'],self.channel['train']], self.y['train'],batch_size=60,
                                            epochs=17    , validation_data=validationdata, verbose=2, shuffle=False )
                                             
                                   
                #model_json = self.model.to_json()
                #with open("model_json", "w") as json_file:
                 #   json_file.write(model_json)
                outputfnames_modelweights = os.path.join(self.OUTPUT_folder, self.base_filename + '_model_run_muti_ConnectivityMatrix'+str(self.seed)+".h5")
                self.model.save_weights(outputfnames_modelweights)
                print("Saved model to disk",outputfnames_modelweights)
            else:    
              print("Using pretrained weights")


              self.model.load_weights(self.pretrainedweights )




    from keras import backend as K

    def aggregateResults(self):

 
            for layer in self.model.layers:
                print(layer.name)

            get_wt_output = K.function([self.model.layers[0].input, self.model.layers[1].input,self.model.layers[2].input,
                                       self.model.layers[3].input,
                                        self.model.layers[8].input],
                                             [self.model.layers[-1].output])


            self.predicted_probability = {}
            self.y_pred = {}

            for i, (setname, do_output) in enumerate(zip(self.datasets, self.output_test)):
                print("setname",setname)
                print("do_output",do_output)
                if do_output:
                        print("do_output again",do_output)
                        self.predicted_probability[setname] = get_wt_output([self.x['EEG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.x['ICP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.x['ECG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.x['ABP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.channel['test'][i*self.ntestsamples:(i+1)*self.ntestsamples]
                                                 ])[0]
                         
                        self.y_pred[setname] = self.model.predict([self.x['EEG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.x['ICP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.x['ECG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.x['ABP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.channel['test'][i*self.ntestsamples:(i+1)*self.ntestsamples]
                                                                  ]) 
                        
                        
                         
                        bestF1=self.evaluate_cutoffs(setname)
                         
                        print("best onsetF1",bestF1)
                        #if we want to use a double cutoff
                        if np.isnan(bestF1 ) or (bestF1 < 1.00):
                            bestF1=self.evaluate_cutoffs(setname,True)


    def trainModel_NoAttention_NoRMT(self):

             
            #validation is only first file as we are giving self.nvalidationsamples
            print("self.x['EEG_train']",self.x['EEG_train'].shape)
            print("self.x['EEG_valid']",self.x['EEG_valid'].shape)
             
            if(self.pretrainedweights is None):
                if len(self.x['EEG_valid'])==0:
                       validationdata=None 
                else:
                       validationdata=([self.x['EEG_valid'][:self.nvalidationsamples],
                                                                          self.x['ICP_valid'][:self.nvalidationsamples], 
                                                                          self.x['ECG_valid'][:self.nvalidationsamples],
                                                                          self.x['ABP_valid'][:self.nvalidationsamples], 
                                                               ], self.y['valid'])
                self.history = self.model.fit([self.x['EEG_train'],self.x['ICP_train'],self.x['ECG_train'],self.x['ABP_train']], self.y['train'],batch_size=60,
                                            epochs=17    , validation_data=validationdata, verbose=2, shuffle=False )
                                             
                                   
                #model_json = self.model.to_json()
                #with open("model_json", "w") as json_file:
                 #   json_file.write(model_json)
                outputfnames_modelweights = os.path.join(self.OUTPUT_folder, self.base_filename + '_model_run_muti_ConnectivityMatrix'+str(self.seed)+".h5")
                self.model.save_weights(outputfnames_modelweights)
                print("Saved model to disk",outputfnames_modelweights)
            else:    
              print("Using pretrained weights")


              self.model.load_weights(self.pretrainedweights )




    from keras import backend as K

    def aggregateResults_NoAttention_NoRMT(self):

 
            for layer in self.model.layers:
                print(layer.name)

            get_wt_output = K.function([self.model.layers[0].input, self.model.layers[1].input,self.model.layers[2].input,
                                       self.model.layers[3].input],
                                         
                                             [self.model.layers[-1].output])


            self.predicted_probability = {}
            self.y_pred = {}

            for i, (setname, do_output) in enumerate(zip(self.datasets, self.output_test)):
                print("setname",setname)
                print("do_output",do_output)
                if do_output:
                        print("do_output again",do_output)
                        self.predicted_probability[setname] = get_wt_output([self.x['EEG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.x['ICP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.x['ECG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                 self.x['ABP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                # self.channel['test'][i*self.ntestsamples:(i+1)*self.ntestsamples]
                                                 ])[0]
                         
                        self.y_pred[setname] = self.model.predict([self.x['EEG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.x['ICP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.x['ECG_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                     self.x['ABP_test'][i*self.ntestsamples:(i+1)*self.ntestsamples],
                                                   #  self.channel['test'][i*self.ntestsamples:(i+1)*self.ntestsamples]
                                                                  ]) 
                        
                        
                         
                        bestF1=self.evaluate_cutoffs(setname)
                         
                        print("best onsetF1",bestF1)
                        #if we want to use a double cutoff
                        if np.isnan(bestF1 ) or (bestF1 < 0.50):
                            bestF1=self.evaluate_cutoffs(setname,True)




       
    def build_onsets2D(self,y):
 
              new_y = np.ones(np.asarray(y).shape)
              #print("y.shape")
              #print(y.shape)
              new_y[0:6, :] = y[0:6, :]
              # Have to take in y in 2D shape
              for i in range(6, y.shape[0]):
                #prev-->i=6, y[0:6,:]  
                #prev-->i=7, y[1:7,:]
                #prev-->i=8, y[2:8,:]
                #......................
                #prev-->i=488, y[482:488,:]
                prev = y[i-6:i, :]

                invalid = np.all(prev==1, axis=0)
                new_y[i,:] = np.where(invalid, 0, y[i,:])


              return new_y
          
    from sklearn.metrics import confusion_matrix  
    def get_rates(self,setname,cutoff,cutoff2=None,writeToFile=False):
        #print("Get_rates method for !!*************************************",setname,cutoff,cutoff2 )
        xx_local_pred = self.predicted_probability[setname]
        
        
            # Defines onset/cont/not cases for both pred and true  
        if (cutoff2 is not None):
                  
                     mask=np.logical_and(xx_local_pred > cutoff,xx_local_pred <cutoff2)

                     y_predrounded=np.where(mask, 1,0 ) 
        else:
                     y_predrounded=np.where(xx_local_pred > cutoff ,1,0) 
          
        y_test_flat=self.y['test'].reshape(-1,1).astype('int') 
        y_testcopy=y_test_flat.copy() 
        true_onsets = self.build_onsets2D(y_testcopy) 
        print("true_onsets")
        print(true_onsets.shape)
        true_multi_=np.where(true_onsets,1,2*y_testcopy)
        y_predrounded=y_predrounded.reshape(-1,1) 
        predicted_onsets = self.build_onsets2D(y_predrounded)      
        predict_multi_=np.where(predicted_onsets,1,2*y_predrounded) 
        # Ground Truth/Predicted,
        # a is NonSeizure /NonSeizure, b is NonSeizure/Onset, c is NonSeizure/Continuation
        # d is Onset/NonSeizure, e is Onset/Onset, f is Onset/Continuation
        # g is Continuation/NonSeizure, h is Continuation/Onset, i is Continuation/Continuation
        full_cm = np.zeros(true_multi_.shape).astype(str)
        full_cm = np.where(np.logical_and(predict_multi_==0, true_multi_==0), 'a', full_cm) # NonSeizure /NonSeizure  
        full_cm = np.where(np.logical_and(predict_multi_==1, true_multi_==0), 'b', full_cm) # NonSeizure/Onset  
        full_cm = np.where(np.logical_and(predict_multi_==2, true_multi_==0), 'c', full_cm) #NonSeizure/Continuation  
        full_cm = np.where(np.logical_and(predict_multi_==0, true_multi_==1), 'd', full_cm) # Onset/NonSeizure
        full_cm = np.where(np.logical_and(predict_multi_==1, true_multi_==1), 'e', full_cm) # Onset/Onset
        full_cm = np.where(np.logical_and(predict_multi_==2, true_multi_==1), 'f', full_cm) # Onset/Continuation
        full_cm = np.where(np.logical_and(predict_multi_==0, true_multi_==2), 'g', full_cm) # Continuation/NonSeizure
        full_cm = np.where(np.logical_and(predict_multi_==1, true_multi_==2), 'h', full_cm) # Continuation/Onset
        full_cm = np.where(np.logical_and(predict_multi_==2, true_multi_==2), 'i', full_cm) #Continuation/Continuation
        # zip function is iterating row wise values(effective transpose from columnwise)
        if writeToFile:
            if hasattr(self,'attentionViewModel'):
                if self.attn_outputs is None :
                    self.attn_outputs=self.viewAttention()

                attn = self.attn_outputs[setname].transpose((0,2,1,3)).reshape(-1,self.num_heads,self.reducedsequenceLength)
                
                attncols = [f"Head {i} attn" for i in range(1,self.num_heads+1)]
                df = pd.DataFrame(attn.argmax(axis=-1), columns=attncols)
            else:
                df=pd.DataFrame()
                attncols=[]
            df['True'] = true_multi_.reshape(-1)
            df['Predict'] = predict_multi_.reshape(-1)
            df['Result'] = full_cm.reshape(-1)
            df = df[['True', 'Predict', 'Result']+attncols]
            print("Writing report for TL*******************************************************************")
            if cutoff2 is None:
                 df.to_csv(f'Long_SingleReportGaussian_{self.seed}_{cutoff}_{setname}_{self.folder}.csv', index=False)
            else:    
                 df.to_csv(f'Long_SingleReportGaussian_{self.seed}_{cutoff}_{cutoff2}_{setname}_{self.folder}.csv', index=False)
        return full_cm


    def confusion_matrix_with_threshold_report(self,setname,cutoff,cutoff2=None):
         
        full_cm = self.get_rates(setname,cutoff,cutoff2,False)
        cm = {}
        #for letter in ['a', 'b', 'c ....] sum the points belonging to each category
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
            cm[letter] = (full_cm==letter).sum()
        print(cm)    
        return cm



    def evaluate_cutoffs(self,setname, double_cutoff=False):
        print("evaluate_cutoffs", setname)
        
        results = []
          
        # looping through all possible cutoffs and for each cutoff calculate CO Recall and Precision
        for i in range(0,101):
            if (not double_cutoff):

                  print("Single cutoff for ",setname)
                  cutoff = i/100
                  cm = self.confusion_matrix_with_threshold_report(setname,cutoff)
                  summary = []
                  summary.append((cutoff))
                  TP=  cm['e']+cm['f']+cm['h']+cm['i']
                  FN=  cm['d']+cm['g'] 
                  FP=  cm['b']+cm['c']  
                  TN=  cm['a']
                  # Recall  
                  recall=TP/(TP+FN)
                  print("TP",TP)
                  precision=TP/(TP+FP)  
                  summary.append(recall)
                  summary.append(precision)
                  F1 = 2 *recall*precision/(recall+precision)
                  summary.append(F1)
   
                  summary.append(cm['e'] / (cm['e'] + cm['d'])) #recall_crit_ver_1a: ignore prediction of continuation completely
                  summary.append(cm['e'] / (cm['e'] + cm['d']+ cm['f']))  
                  summary.append((cm['e']+cm['f']) / (cm['e'] + cm['d']+ cm['f'])) 
                  
                  summary.append(cm['e'] / (cm['e'] + cm['b'])) #precision_crit_ver_1
                  # No of events predicted All TP/TP+FP ignoring continuation  
                  summary.append((cm['e']+cm['f']) / (cm['e']+cm['f'] + cm['b']+cm['c'])) #precision_crit_ver_2: no distinction between predicted onsets and continuation
                  summary.append(cm['a'])
                  summary.append(cm['b'])
                  summary.append(cm['c'])
                  summary.append(cm['d'])
                  summary.append(cm['e'])
                  summary.append(cm['f'])
                  summary.append(cm['g'])
                  summary.append(cm['h'])
                  summary.append(cm['i'])
                  results.append(summary)
            elif(double_cutoff):
                for j in range(0,101):
                  cutoff = i/100
                  cutoff2= j/100
                  print("Double cutoff",setname,cutoff,cutoff2)  
                  cm = self.confusion_matrix_with_threshold_report(setname,cutoff,cutoff2)
                  TP=  cm['e']+cm['f']+cm['h']+cm['i']
                  FN=  cm['d']+cm['g'] 
                  FP=  cm['b']+cm['c']  
                  TN=  cm['a']
                  # Recall  
                  recall=TP/(TP+FN)
                  precision=TP/(TP+FP)  
                  F1 = 2 *recall*precision/(recall+precision)
                   
                  recall_crit_ver_1a = cm['e'] / (cm['e'] + cm['d'])
                  recall_crit_ver_1b = cm['e'] / (cm['e'] + cm['d']+ cm['f'])
                  recall_crit_ver_2 = (cm['e']+cm['f']) / (cm['e'] + cm['d']+ cm['f'])
                   
                  precision_crit_ver_1 = cm['e'] / (cm['e'] + cm['b'])
                  precision_crit_ver_2 = (cm['e']+cm['f']) / (cm['e']+cm['f'] + cm['b']+cm['c'])

                  summary = [(cutoff,cutoff2), recall, precision,F1, recall_crit_ver_1a, recall_crit_ver_1b, recall_crit_ver_2, 
                             precision_crit_ver_1,  precision_crit_ver_2,
                             cm['a'], cm['b'], cm['c'], cm['d'], cm['e'], cm['f'], cm['g'], cm['h'], cm['i']]          
                  results.append(summary)
        
        df_cm = pd.DataFrame(data = results,
                             columns = ['cutoff','recall','precision','F1', "recall_crit_ver_1a","recall_crit_ver_1b", "recall_crit_ver_2", "precision_crit_ver_1", "precision_crit_ver_2","a","b","c","d","e","f","g","h","i"]
                            ) 

        df_cm['onsetF1'] = 2*df_cm['recall_crit_ver_1a']*df_cm["precision_crit_ver_1"] / (df_cm['recall_crit_ver_1a']+df_cm["precision_crit_ver_1"])
        for col in ["recall_crit_ver_1a", "precision_crit_ver_1"]:
                col_max = np.max(df_cm[col])
                print("col")
                print(col)
                print("Max value of col ",col)
                print(col_max)
                print((df_cm[col]==col_max).sum())
                #if (df_cm[col]==col_max).sum() != 0:
                 #   print("(df_cm[col]==col_max).sum() != 0")
                     
         
        best_onsetf1 = np.max(df_cm["onsetF1"]) # best value
        subset = df_cm[df_cm["onsetF1"]==best_onsetf1] # all rows with best f1 score
        if len(subset)!=0:
             best_row = subset.iloc[0]
             print("For best onset F1,best row")   
             print(best_row)
             print("best row shape")
             print(best_row.shape)
             print("best_row[cutoff]")   
             print(best_row["cutoff"])    
             print("seed")
             print(self.seed)
             if double_cutoff is True:
                 print("best_row[cutoff][0]")
                 print(best_row["cutoff"][0])
                 print("best_row[cutoff][1]")
                 print(best_row["cutoff"][1])   
                 self.get_rates(setname,best_row["cutoff"][0],best_row["cutoff"][1],False)   
                 outputfile=os.path.join(self.OUTPUT_folder,'Result_' +self.base_filename+str(self.seed)+'_' +setname+'_UC2.xls')   
                 best_row.to_excel( outputfile, sheet_name="onsetF1_Usecase2")
             else:   
                self.get_rates(setname,best_row["cutoff"],None,False)
                outputfile=os.path.join(self.OUTPUT_folder,'Result_' +self.base_filename+str(self.seed)+'_' +setname+'_UC1.xls') 
                best_row.to_excel( outputfile, sheet_name="onsetF1_Usecase1")
        else:
            print("For best onset F1,No valid F1 score ")
             

        return df_cm['onsetF1'].max()

# Scalable version of RunModel using an iterator    
class RunStreamModel(RunModel):
    def __init__(self, **kwargs):
         
        RunModel.__init__(self, **kwargs)
        self.nChannels_EEG = kwargs['nChannels_EEG']
        self.nDim = kwargs['nDim']
        self.channelShape=kwargs['channelShape']
        self.batchsize=kwargs['batchsize']
        self.steps_per_epoch=kwargs['steps_per_epoch']
        self.validation_steps=kwargs['validation_steps']
        self.num_batches_test=kwargs['num_batches_test']
        self.num_heads=kwargs['num_heads']
        self.pretrainedweights=kwargs['pretrainedweights']
        self.attn_outputs=None 
    def _set_names(self,datasets):
        self.y_fnames = [os.path.join(self.folder, self.base_filename + '_y.pkl')]
        self.x_fnames = [os.path.join(self.folder, self.base_filename + '_x_' + setname + '.pkl') for setname in
                          datasets]
        self.channel_fnames = [os.path.join(self.folder, self.base_filename + '_channel_' + setname + '.pkl') for setname in
                              datasets]
        
    def trainModel(self):
        self._set_names(self.datasets)
         
        if(self.pretrainedweights is None):
             
            train_data = data_generator('train', self.x_fnames, self.y_fnames, self.channel_fnames, self.batchsize, 
                                        self.reducedsequenceLength, self.nChannels_EEG, self.nDim) 
            # add check if no validation
            if self.validation_steps is None:
                valid_data = None
            else:
                valid_data = data_generator('valid', self.x_fnames, self.y_fnames, self.channel_fnames, self.batchsize, 
                                            self.reducedsequenceLength, self.nChannels_EEG, self.nDim) 

            self.history = self.model.fit_generator(train_data, epochs=17, validation_data=valid_data,
                                          steps_per_epoch = self.steps_per_epoch,
                                          validation_steps = self.validation_steps,
                                           verbose=2, shuffle=False )
                                             
                                   
            outputfnames_modelweights = os.path.join(self.OUTPUT_folder, self.base_filename + '_model_run_muti_ConnectivityMatrix'+str(self.seed)+".h5")
            self.model.save_weights(outputfnames_modelweights)
            print("Saved model to disk",outputfnames_modelweights)
        else:    
            print("Using pretrained weights")
            self.model.load_weights(self.pretrainedweights )

            
    def aggregateResults(self,zeroed_channel_list=None):
        
        for layer in self.model.layers:
            print(layer)
         
            get_wt_output = K.function([self.model.layers[0].input, self.model.layers[1].input,
                                      self.model.layers[2].input,
                                      self.model.layers[3].input,
                                      self.model.layers[8].input, 
                                      
                                      
                                        
                                   ],
                                            [self.model.layers[-1].output])

        


        self.predicted_probability = {}
        self.y_pred = {}
        
         
        for i, (setname, do_output) in enumerate(zip(self.datasets, self.output_test)):
            print("RunStreamModel setname",setname)
            print("RunStreamModel do_output",do_output)
            if do_output:
                self._set_names([setname])           
                #Return data results of size upto 99,999
                test_data = data_generator('test', self.x_fnames, None, self.channel_fnames, 99999, 
                                           self.reducedsequenceLength, self.nChannels_EEG, self.nDim,
                                          )               

                pred_results = []
                prob_results = []
                #For loop loops over batches of test data
                for i, (x_EEG, x_ICP, x_ECG, x_ABP, chan) in enumerate(test_data):
                #for each batch we append the results to prob_results    
                    
                    #pred_results is if it is a seizure or not
                    pred_results.append(self.model.predict([x_EEG,x_ICP, x_ECG, x_ABP, chan]))
                    # prob_results is if it is 60% or 90%
                    prob_results.append(get_wt_output([x_EEG, x_ICP, x_ECG, x_ABP,chan])[0])
                    if i+1==self.num_batches_test:
                          break
                #we are concatenating results from all batches into a single list,default is row wise
                self.predicted_probability[setname] = np.concatenate(prob_results)
                self.y_pred[setname] = np.concatenate(pred_results)
                #This will break if you have multiple y files for diff setnames
                train, valid, test = load_stuff(self.y_fnames)
                del train
                del valid
                self.y={'test':test}        
                        
                # Single cutoff 
                bestF1=self.evaluate_cutoffs(setname,False)
                #if we want to use a double cutoff
                if np.isnan(bestF1 ) or (bestF1 < 0.50):
                   bestF1=self.evaluate_cutoffs(setname,True)
                    
                         
                print("RunStreamModel got best onsetF1, going to delete",bestF1)
                 
                del self.y_pred[setname]
                del self.predicted_probability[setname]
                del self.y

    def viewAttention(self):
        self.attn_outputs = {}          
         
        for i, (setname, do_output) in enumerate(zip(self.datasets, self.output_test)):
            print("RunStreamModel setname",setname)
            print("RunStreamModel do_output",do_output)
            if do_output:
                self._set_names([setname])           
                #Return data results of size upto 99,999
                test_data = data_generator('test', self.x_fnames, None, self.channel_fnames, 99999, 
                                           self.reducedsequenceLength, self.nChannels_EEG, self.nDim)               

                pred_results = []
                for i, (x_EEG, x_ICP, x_ECG, x_ABP, chan) in enumerate(test_data):
                    pred_results.append(self.attentionViewModel.predict([x_EEG,x_ICP, x_ECG, x_ABP, chan]))
                    #Stop after one iteration,hack
                    if i+1==self.num_batches_test:
                          break
                self.attn_outputs[setname] = np.concatenate(pred_results)
                #This will break if you have multiple y files for diff setnames
        return self.attn_outputs
    

#Google Transformer online implementation code split as two modules and adapted to add L1 regularization in the post module    #Attention is all you need:https://arxiv.org/abs/1706.03762   
from keras import backend as K # caps are layers
from keras.engine.topology import Layer
from keras.layers import Concatenate, Input, Reshape
from keras import regularizers
class PostAttention(Layer):
    def __init__(self, **kwargs):
        super(PostAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_shape=(A,V_seq)
        self.nb_head = input_shape[0][1]
        self.size_per_head = input_shape[1][3]
        #8 coefficients , each head is multiplied by a coefficient
        #Model will use which heads have higher coefficient
        #Uncomment for with L1 use
        #self.wt_per_head=self.add_weight(name='WH',
                                  #shape=(1,1, self.nb_head,1),# it needs to be 4D
                                  #initializer='Ones', #no effect value 
                                   
                                  # regularizer= tf.keras.regularizers.l1(0.001),       
                                   
                                   #trainable=True)
        super(PostAttention, self).build(input_shape)
        
    def call(self, x):
        # Query, Key, Value 
        if len(x) == 2:
            A, V_seq = x
            # A shape = batch, nheads, Q_len, K_len
            # V_seq shape input=  batch, nheads, V_len, size_per_head
            V_len = K.shape(V_seq)[2]
            Q_len = K.shape(A)[2]
            K_len = K.shape(A)[3]
            #if V_len != K_len:
             # raise ValueError("Inconsistent size between V_len and K_len")
        else:
          raise NotImplementedError("Only support [A, V_seq] input")
        
        A = K.reshape(A, (-1, Q_len, K_len)) #batch*nheads,Q_len,K_len
        V_seq = K.reshape(V_seq, (-1, V_len, self.size_per_head)) #batch*nheads,V_len,size_per_head
        #axes 2 is K_len in A,axes 1 is V_len in V_seq
        O_seq = K.batch_dot(A, V_seq, axes=[2, 1]) #batch *nheads,Q_len,size_per_head
        O_seq = K.reshape(O_seq,(-1, self.nb_head, Q_len, self.size_per_head))        
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3)) #batchlength, output seq length Q_len, numheads, size_per_head 
        # Uncomment in case of L1
        #O_seq=  O_seq * self.wt_per_head
        # reshape smashes numheads and last hidden dim into output dim, Concatenate results of all heads
        O_seq = K.reshape(O_seq, (-1, Q_len, self.size_per_head*self.nb_head))
        return O_seq
        
    def compute_output_shape(self, input_shape):
        print("calling compute output shape************************")
        print(input_shape[0][0], input_shape[0][1], input_shape[0][1]*input_shape[1][3])
        return (input_shape[0][0], input_shape[0][2], input_shape[0][1]*input_shape[1][3])

        
 
class PreAttention(Layer):
    # Number of heads running in parallel, no of kinds of information to pass
    # Size per head, dimensionality of transformed keys and values, no of nodes for building Attention matrix
    
    def __init__(self, nb_head, size_per_head, **kwargs):
        print("init**********************************")
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        print("Inside init,output_dim=head*size per head")
        print(self.output_dim)
        super(PreAttention, self).__init__(**kwargs)
    
    # Three separate weights
    def build(self, input_shape):
        #Query weight 15*75*10 for one head, 1125*20 for two heads
        #Query is input_shape[0],dim of how many input frequencies (20) thats coming in 
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        #Key is input_shape[1],dim of how many input frequencies (20) thats coming in,shape=15*20+20+20+20
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(PreAttention, self).build(input_shape)


    def call(self, x):
        # Query, Key, Value 
        if len(x) == 2:
            Q_seq, K_seq = x
            # the value in dimension 1 gets attended to
            # Qlen how many outputs it is creating, Klen how many inputs is choosing
            Q_len = K.shape(Q_seq)[1]
            K_len = K.shape(K_seq)[1]
            mask=None 

        else:
          raise NotImplementedError("Only support [Q, K] input")
         
         
        Q_seq = K.dot(Q_seq, self.WQ) #batch length, sequence length, output dim
        #print("Q_seq after", Q_seq.shape)                                    
        # split Q_seq into multiple heads, make it 4D, Q is splitting the heads apart
        Q_seq = K.reshape(Q_seq, (-1, Q_len, self.nb_head, self.size_per_head))
        # batch length,num heads,sequence length,size per head
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
         
        K_seq = K.reshape(K_seq, (-1, K_len, self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        #print("K_seq after permute", K_seq)
                                     
        ### start new code -1 is batch size*no of heads,seq length,head size
        K_seq = K.reshape(K_seq, (-1, K_len, self.size_per_head))
        Q_seq = K.reshape(Q_seq, (-1, Q_len, self.size_per_head))
        A = K.batch_dot(Q_seq, K_seq, axes=[2, 2]) / self.size_per_head ** 0.5
        A = K.reshape(A, (-1, self.nb_head, Q_len, K_len))
        ### end new code
        #batch length,num heads,output seq length, input seq length 
        #A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        #print("A shape ", np.shape(A))
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        #A = self.Mask(A, mask, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        #softmax gets applied to input seqlength dimension(gone away) which is the last dimension
        A = K.softmax(A)
        print("A shape after softmax", np.shape(A))
        A = K.reshape(A, (-1, self.nb_head, Q_len, K_len )) # batch,nheads,Q_len,K_len
        
        return A

    def compute_output_shape(self, input_shape):
        print("calling compute output shape************************")
        print(input_shape[0][0], self.nb_head, input_shape[0][1], input_shape[1][1])
        return (input_shape[0][0], self.nb_head, input_shape[0][1], input_shape[1][1])  
class PreAttention_noscaling(Layer):
    # Number of heads running in parallel, no of kinds of information to pass
    # Size per head, dimensionality of transformed keys and values, no of nodes for building Attention matrix
    
    def __init__(self,  **kwargs):
        print("init**********************************")
        #self.nb_head = nb_head
        #self.size_per_head = size_per_head
        self.output_dim = 1
        print("Inside init,output_dim=head*size per head")
        print(self.output_dim)
        super(PreAttention_noscaling, self).__init__(**kwargs)
    
    # Three separate weights
    def build(self, input_shape):
        #Query weight 15*75*10 for one head, 1125*20 for two heads
        #Query is input_shape[0],dim of how many input frequencies (20) thats coming in 
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        #Key is input_shape[1],dim of how many input frequencies (20) thats coming in,shape=15*20+20+20+20
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(PreAttention_noscaling, self).build(input_shape)


    def call(self, x):
        # Query, Key, Value 
        if len(x) == 2:
            Q_seq, K_seq = x
            # the value in dimension 1 gets attended to
            # Qlen how many outputs it is creating, Klen how many inputs is choosing
            Q_len = K.shape(Q_seq)[1]
            K_len = K.shape(K_seq)[1]
            mask=None 

        else:
          raise NotImplementedError("Only support [Q, K] input")
         
         
        Q_seq = K.dot(Q_seq, self.WQ) #batch length, sequence length, output dim
                                             
        # split Q_seq into multiple heads, make it 4D, Q is splitting the heads apart
        Q_seq = K.reshape(Q_seq, (-1, Q_len, 1, 1))
        # batch length,num heads,sequence length,size per head
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
         
        
        K_seq = K.dot(K_seq, self.WK)
         
        K_seq = K.reshape(K_seq, (-1, K_len, 1, 1))
        # print("K_seq after reshape")
        # print(K_seq)
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        #print("K_seq after permute", K_seq)
                                     
        ### start new code -1 is batch size*no of heads,seq length,head size
        K_seq = K.reshape(K_seq, (-1, K_len, 1))
        Q_seq = K.reshape(Q_seq, (-1, Q_len, 1))
       # A = K.batch_dot(Q_seq, K_seq, axes=[2, 2]) / 1 ** 1
        A = K.batch_dot(Q_seq, K_seq, axes=[2, 2]) / 1 ** 1
        A = K.reshape(A, (-1, 1, Q_len, K_len))
        ### end new code
        #batch length,num heads,output seq length, input seq length 
        #A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        #print("A shape ", np.shape(A))
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        #A = self.Mask(A, mask, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        #softmax gets applied to input seqlength dimension(gone away) which is the last dimension
        A = K.softmax(A)
        print("A shape after softmax", np.shape(A))
        A = K.reshape(A, (-1, 1, Q_len, K_len )) # batch,nheads,Q_len,K_len
        
        return A

    def compute_output_shape(self, input_shape):
        print("calling compute output shape************************")
        print(input_shape[0][0], 1, input_shape[0][1], input_shape[1][1])
        return (input_shape[0][0], 1, input_shape[0][1], input_shape[1][1]) 
from keras import backend as K # caps are layers
from keras.engine.topology import Layer
from keras.layers import Concatenate, Input, Reshape
from keras import regularizers
class PostAttention_noscaling(Layer):
    def __init__(self, **kwargs):
        super(PostAttention_noscaling, self).__init__(**kwargs)

    def build(self, input_shape):
       #input_shape=(A,V_seq)
        #8 coefficients , each head is multiplied by a coefficient
        #Model will use which heads have higher coefficient
        #Uncomment for with L1 use
        #self.wt_per_head=self.add_weight(name='WH',
                                  #shape=(1,1, self.nb_head,1),# it needs to be 4D
                                  #initializer='Ones', #no effect value 
                                   
                                  # regularizer= tf.keras.regularizers.l1(0.001),       
                                   
                                   #trainable=True)
        super(PostAttention_noscaling, self).build(input_shape)
        
    def call(self, x):
        # Query, Key, Value 
        if len(x) == 2:
            A, V_seq = x
            # A shape = batch, nheads, Q_len, K_len
            # V_seq shape input=  batch, nheads, V_len, size_per_head
            V_len = K.shape(V_seq)[2]
            Q_len = K.shape(A)[2]
            K_len = K.shape(A)[3]
            #if V_len != K_len:
             # raise ValueError("Inconsistent size between V_len and K_len")
        else:
          raise NotImplementedError("Only support [A, V_seq] input")
        
        A = K.reshape(A, (-1, Q_len, K_len)) #batch*nheads,Q_len,K_len
        V_seq = K.reshape(V_seq, (-1, V_len, 1)) #batch*nheads,V_len,size_per_head
        #axes 2 is K_len in A,axes 1 is V_len in V_seq
        O_seq = K.batch_dot(A, V_seq, axes=[2, 1]) #batch *nheads,Q_len,size_per_head
        O_seq = K.reshape(O_seq,(-1, 1, Q_len, 1))        
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3)) #batchlength, output seq length Q_len, numheads, size_per_head 
        # Uncomment in case of L1
        #O_seq=  O_seq * self.wt_per_head
        # reshape smashes numheads and last hidden dim into output dim, Concatenate results of all heads
        O_seq = K.reshape(O_seq, (-1, Q_len, 1*1))
        return O_seq

        
    def compute_output_shape(self, input_shape):
        print("calling compute output shape************************")
        print(input_shape[0][0], input_shape[0][1], input_shape[0][1]*input_shape[1][3])
        return (input_shape[0][0], input_shape[0][2], input_shape[0][1]*input_shape[1][3])

    

class RunModel_DCRNN(RunModel):
        def __init__(self, **kwargs):
            #kwargs is a dictonary of keyword arguments,** unpacks it
            self.noOfHops=kwargs['noOfHops']
            self.adjacencyMatrix_DCRNN=kwargs['adjacencyMatrix_DCRNN']
            self.nChannels_EEG = kwargs['nChannels_EEG']
            self.nDim = kwargs['nDim']
             
            self.batchsize=kwargs['batchsize']
            self.steps_per_epoch=kwargs['steps_per_epoch']
            self.validation_steps=kwargs['validation_steps']
            self.num_batches_test=kwargs['num_batches_test']
             

            
            super().__init__(**kwargs)
        def _set_names(self,datasets):
            self.y_fnames = [os.path.join(self.folder, self.base_filename + '_y.pkl')]
            self.x_fnames = [os.path.join(self.folder, self.base_filename + '_x_' + setname + '.pkl') for setname in
                          datasets]
    
        def buildModel_DCRNN(self):

                # Take your regular input x of all EEG channels combined, ICP, ECG and ABP

                hidden_dim_EEG=self.nDim
                input_EEG = Input(shape=(self.reducedsequenceLength,self.nChannels_EEG,self.nDim)) 
                input_ICP = Input(shape=(self.reducedsequenceLength,1,self.nDim)) 
                input_ECG=Input(shape=(self.reducedsequenceLength,1,self.nDim))
                input_ABP=Input(shape=(self.reducedsequenceLength,1,self.nDim))
                
                 
                permuted_EEG=Permute((1,3,2))(input_EEG)
                # In Every single channel for EEG,each frequency gets a node
                wide_EEG=Reshape((self.reducedsequenceLength,self.nDim*self.nChannels_EEG,1)) (permuted_EEG)
                #462,1,20
                wide_ICP=Reshape((self.reducedsequenceLength,1,1*self.nDim)) (input_ICP)
                wide_ECG=Reshape((self.reducedsequenceLength,1,1*self.nDim)) (input_ECG)
                wide_ABP=Reshape((self.reducedsequenceLength,1,1*self.nDim)) (input_ABP)
                # every EEG channel/frequency is getting info from of all frequencies of ICP, ECG and ABP to go with it
                # a tile function makes self.nChannels_EEG*self.nDim copies and putting in 3rd dimension
                # it makes a copy for every one of the 300 nodes.
                repeated_ICP=Lambda(lambda x: tf.tile(x, [1,1,self.nChannels_EEG*self.nDim,1]))(wide_ICP)
                repeated_ECG=Lambda(lambda x: tf.tile(x, [1,1,self.nChannels_EEG*self.nDim,1]))(wide_ECG)
                repeated_ABP=Lambda(lambda x: tf.tile(x, [1,1,self.nChannels_EEG*self.nDim,1]))(wide_ABP)
                flat=Concatenate()([wide_EEG,repeated_ICP,repeated_ECG,repeated_ABP])
                 
                # Flattening all data out within a node
                # In last dimension 1 is EEG descriptor,3 is ICP, ECG and ABP * 20 descriptor/number of frequencies seqlength
                flat_EEG = Reshape((self.reducedsequenceLength,self.nChannels_EEG*self.nDim*(1+3*self.nDim))) (flat)  


                # DCRNN cell is definition for one timestep
                #Return a vector of descriptors at each location at one single timestep 
                cell = DCRNNCell(hidden_dim_EEG,self.adjacencyMatrix_DCRNN,self.noOfHops,self.nChannels_EEG*self.nDim, (1+3*self.nDim) )
                #Apply the DCRNN cell for all timesteps
                rnn_EEG = RNN(cell,return_sequences=True)(flat_EEG )

                #Use the vector description from all local locations/predictions at all timesteps to make a single prediction of seizure at a time
                output = Dense(1, activation='sigmoid')(rnn_EEG)
                self.model = Model(inputs=[input_EEG,input_ICP,input_ECG,input_ABP], outputs=output)

                 
                self.model.compile(loss='binary_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
                print(self.model.summary())
        def trainModel(self):
            self._set_names(self.datasets)
            if(self.pretrainedweights is None):
                train_data = data_generator('train', self.x_fnames, self.y_fnames,None,  self.batchsize, 
                                            self.reducedsequenceLength, self.nChannels_EEG, self.nDim) 
                # add check if no validation
                if self.validation_steps is None:
                    valid_data = None
                else:
                    valid_data = data_generator('valid', self.x_fnames, self.y_fnames, None,self.batchsize, 
                                                self.reducedsequenceLength, self.nChannels_EEG, self.nDim) 

                self.history = self.model.fit_generator(train_data, epochs=17, validation_data=valid_data,
                                              steps_per_epoch = self.steps_per_epoch,
                                              validation_steps = self.validation_steps,
                                               verbose=2, shuffle=False )


                outputfnames_modelweights = os.path.join(self.OUTPUT_folder, self.base_filename + '_model_run_muti_ConnectivityMatrix'+str(self.seed)+".h5")
                self.model.save_weights(outputfnames_modelweights)
                print("Saved model to disk",outputfnames_modelweights)
            else:    
                print("Using pretrained weights")
                self.model.load_weights(self.pretrainedweights )

            
        def aggregateResults(self,zeroed_channel_list=None):
        
            for layer in self.model.layers:
                print(layer)
                get_wt_output = K.function([self.model.layers[0].input, self.model.layers[1].input,
                                       self.model.layers[2].input,self.model.layers[3].input],
                                       [self.model.layers[-1].output])






            self.predicted_probability = {}
            self.y_pred = {}


            for i, (setname, do_output) in enumerate(zip(self.datasets, self.output_test)):
                print("RunStreamModel setname",setname)
                print("RunStreamModel do_output",do_output)
                if do_output:
                    self._set_names([setname])           
                    #Return data results of size upto 99,999
                    test_data = data_generator('test', self.x_fnames, None, None, 99999, 
                                               self.reducedsequenceLength, self.nChannels_EEG, self.nDim,
                                              )               

                    pred_results = []
                    prob_results = []
                    #For loop loops over batches of test data
                    for i, (x_EEG, x_ICP, x_ECG, x_ABP) in enumerate(test_data):
                    #for each batch we append the results to prob_results    

                        #pred_results is if it is a seizure or not
                        pred_results.append(self.model.predict([x_EEG,x_ICP, x_ECG, x_ABP]))
                        # prob_results is if it is 60% or 90%
                        prob_results.append(get_wt_output([x_EEG, x_ICP, x_ECG, x_ABP])[0])
                        if i+1==self.num_batches_test:
                              break
                    #we are concatenating results from all batches into a single list,default is row wise
                    self.predicted_probability[setname] = np.concatenate(prob_results)
                    self.y_pred[setname] = np.concatenate(pred_results)
                    #This will break if you have multiple y files for diff setnames
                    train, valid, test = load_stuff(self.y_fnames)
                    del train
                    del valid
                    self.y={'test':test}        

                     
                    bestF1=self.evaluate_cutoffs(setname,False)
                    print("bestF1.........................%%%%%%%%%%%%%%%%%%%%%%",bestF1)
                    if np.isnan(bestF1 ) or (bestF1 < 0.50):
                        bestF1=self.evaluate_cutoffs(setname,True)


                    print("RunStreamModel got best onsetF1, going to delete",bestF1)
                     
                    del self.y_pred[setname]
                    del self.predicted_probability[setname]
                    del self.y                

# DCRNN paper code implementation is used here.
#https://arxiv.org/abs/1707.01926
class DCRNNCell(keras.layers.Layer):
      #max_diffusion_step is 1 hops
      def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes,descriptor_dim, **kwargs):

        self._hidden_dim = num_units # hidden state/descriptor vector 
        self._maxK = max_diffusion_step
        self.state_size=num_units *num_nodes   
        self.output_size= num_units *num_nodes  
        self._num_nodes = num_nodes # 300 channels which is channel frequency pair
        self._descriptor_dim=descriptor_dim #inputs=21, 1+20*3, 20 for ICP, ECG and ABP data so we have all data everywhere.
        self.supports = [] 
        #calculate normalized transition state matrix 
        # one towards traffic and away from traffic
        self.supports.append(K.transpose(self.calculate_random_walk_matrix(adj_mx)))
        self.supports.append(K.transpose(self.calculate_random_walk_matrix(adj_mx.T)))

        super(self.__class__, self).__init__(**kwargs)

      # calculate normalized transition state matrix based on adjacency matrix
      # adjacency matrix is list of connections between nodes
      # random walk matrix is the probability of going from node i to node j if you are walking randomly along adjacency matrix  
      def calculate_random_walk_matrix(self, adj_mx):
        # make sparse matrix
        adj_mx = sp.coo_matrix(adj_mx)
        #input matrix is int,normalizing and change to float
        d = np.array(adj_mx.sum(1)).astype(float)
        # reciprocal
        d_inv = np.power(d, -1).flatten()
        # if divide by 0 case, set it to 0.
        d_inv[np.isinf(d_inv)] = 0.
        # creates a new matrix with d_inv on diagonal
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx).todense()
        return tf.convert_to_tensor(random_walk_mx,dtype='float32')
        
      def call(self, inputs, state):
              r = tf.nn.sigmoid(self.gConv(inputs, state, 'r'))

              u = tf.nn.sigmoid(self.gConv(inputs, state, 'u'))
               
              print("shape of r",r.shape) 
              print("shape of u",u.shape)

              print("shape of state",state[-1].shape)
              c = tf.nn.tanh(self.gConv(inputs, r * state, 'c'))
              print("shape of c",c.shape)
              output = u * state[-1] + (1 - u) * c
              print("shape of output.......",output.shape)
              # return output is the horizontal,[output] is a list of states/output from a cell  
              return output,[output]  
      def get_config(self):
         return {'num_nodes': self._num_nodes, # num graph nodes
                'hidden_dim': self._hidden_dim,
                'max_K': self._maxK 
               }

      def build(self, input_shapes):
        print("input_shapes",input_shapes)
        #x_shape = input_shapes[0]
        #h_shape = input_shapes[1]
        # K is the no of hops
        # Q is the output shape
        # P is the input shape
        # states=no of nodes/frequencies * how big is description per node(hidden_dim) 
        len_Q = self._hidden_dim# 1,same as lengthoutput
        # times two for forward and backward step
        len_K = self._maxK*2 + 1
        # we are concatenating the hidden_dim/state information from previous node with new input/descriptor_dim
        len_P = self._hidden_dim+self._descriptor_dim    #input 15+15
        # pernodes/neighbors in 1 to 2 hops
        # connectivity matrix should not have self referencing connection
        # it creates a lot of unnecessary cumulative connections 
        # For every input P and number of steps K we have to predict the value of Q.
        # We are multiplying P and k as we need to look at all combinations of P and k.
        # theta is the weight per node,goal of theta is the weight of local linear transformation  
        # len_P information is multiplied with no of hops 0,1,2,-1,-2 hops etc
        self.theta_r = self.add_weight(shape=[len_P * len_K, len_Q], initializer='uniform', name='theta_r')
        self.theta_u = self.add_weight(shape=[len_P * len_K, len_Q], initializer='uniform', name='theta_u')
        self.theta_c = self.add_weight(shape=[len_P * len_K, len_Q], initializer='uniform', name='theta_c')

        # for every hidden dim in the output
        self.bias_r = self.add_weight(
            shape=[1, len_Q], initializer='Zeros', name='bias_r')
        self.bias_u = self.add_weight(
            shape=[1, len_Q], initializer='Zeros', name='bias_u')
        self.bias_c = self.add_weight(
            shape=[1, len_Q], initializer='Zeros', name='bias_c')

      def _append(self, x, x_):
            x_ = K.expand_dims(x_, axis=0)
            return K.concatenate([x, x_], axis=0)  

      def gConv(self, inputs, states,weight_name): 
        # inputs is the descriptor=features  
        # a descriptor is a vector used to describe a node
        # we get info for one sequence position at a time
        #  state.shape=(batch, nodes, hidden) 
        # inputs.shape=(batch, nodes, descriptors) where descriptors is channel or feature for a node
        # self._num_nodes is per node which is the frequency
        # self._descriptor_dim=all EEG sensors
        #get the last state from states

        state=states[-1]
        print("type of inputs",type(inputs))    
        batch_size = inputs.shape[0]

        #combined info of inputs and state 
        # batch size is the first thing
        inputs = tf.reshape(inputs, (-1, self._num_nodes, self._descriptor_dim))
        state = tf.reshape(state, (-1, self._num_nodes, self._hidden_dim))
        x_raw = K.concatenate([inputs, state], axis = -1)
        print("inputs shape",inputs.shape)
        print("state shape",state.shape)
        print("x_raw shape",x_raw.shape)
        # len_P=self.descriptor_dim + self._hidden_dim is how much stuff we have describing a node/sensor
        #len_Q=self._hidden_dim or the output per node
        len_P = x_raw.shape[2]
        len_Q = self._hidden_dim

        x_raw = K.permute_dimensions(x_raw, (1, 2, 0)) # (nodes, len_P, batch_size)
        x_raw = K.reshape(x_raw, (self._num_nodes, -1)) # convert to 2D,last dim is len_P * batch_size
        x0 = x_raw # x0 = W^0 * x_raw # nodes, len_P*batch_size 
        # before:nodes,len_P*batch_size 
        # after:1, nodes, len_P*batch_size
        all_x = K.expand_dims(x0, axis=0)  #all_x.shape = 1, nodes, len_P*batch_size 
        #Loop through neighboring information
        if self._maxK > 0: 
            for support in self.supports: # dO_W, dI_W = self.supports, W normalized and Wtransposed normalized
              print("support",type(support))  
              x1 = K.dot(support, x0)  
              #possible K values
              #K=0 result is x0
              #K=1 result is x1=W.x0  

              all_x = self._append(all_x, x1)
              #xK-1=x1
              x_prev = x1
              for k in range(2, self._maxK+1):
                #K=2 result is x2=W.x1=W.W.x0=W^2x0
                x_next = K.dot(support, x_prev)
                all_x = self._append(all_x, x_next)
                x_prev = x_next
        # all the values will be between 0 and 1(dO_W@dO_W etc)        
        #all_x = [x_raw or x0, dO_W@x0, dO_W@dO_W@x0, dO_W@dO_W@dO_W@x0, ..., dI_W@x0, dI_W@dI_W@x0 ...]
        # how many different hops or K in both directions are there in all_x ,+1 is for x0 
        #num_matrices=2K+1
        #num_matrices=we build x0, x1 for Wi and Wo,x2 for Wi and Wo
        # all_x= num_matrices,num_nodes,-1
        num_matrices = all_x.shape[0]

        # unpack length_P and batch_size
        all_x = K.reshape(all_x, (num_matrices, self._num_nodes, len_P, -1))
        # batch size has to be the first
        all_x = tf.transpose(all_x, (3, 1, 2, 0))  # (batch_size, num_nodes, len_P, num_matrices)
        print("len_P",len_P)
        print("num_matrices",num_matrices)
        # (batch_size*num_nodes, len_P*num_matrices)
        all_x = K.reshape(all_x, (-1, len_P * num_matrices))          
        if weight_name =='r':
             theta = self.theta_r
             bias=self.bias_r   
        if weight_name =='u':
             theta = self.theta_u
             bias=self.bias_u    
        if weight_name =='c':
             theta = self.theta_c
             bias=self.bias_c   

        # local information of node i @theta
        all_x = K.dot(all_x, theta)  
        all_x = all_x + bias
        # 20 * 15
        # batch size, num_nodes,hidden_dim
        all_x = K.reshape(all_x, (-1, self._num_nodes * self._hidden_dim))
        print("call output shape",all_x.shape)
        return all_x             

    