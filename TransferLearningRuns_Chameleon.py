#Test automatically
#Training patient A, to transfer to patient B

from EEGUtility_segmented_V4 import load_stuff
from EEGModel_service_V6_LSTM import data_generator
from EEGModel_service_V6_LSTM import RunStreamModel 
def runs():
    
    #seeds=[601,602,603,604,605,606,607,608,609,610]
    seeds=[42,43,41,46,44]
    for seed in seeds:
            print("seed",seed)
            params={
                  'seed':seed,
                    
                   'folder': 'ExplicitFeatureContext_from_PatientA_withcluster_redseqlength',
                   'base_filename':'experiment1_combined', 
                   'hidden_dim_EEG': 100,
                   'max_shift': 38 ,
                   'hidden_dim_extVars': 20,
                   'sequenceLength' : 50,                     
                   'pretrainedweights' : None,
                   'TimeSegmentation':True, 
                    #if clustering this is 1,else the number of EEG channels
                    'nChannels_EEG':1,
                    'nDim':200,
                    'channelShape':[4,462,80,10],
                    'batchsize':60,
                    'steps_per_epoch':3*3,  
                    'validation_steps':3*1,  
                    'num_batches_test':1,
                     
                    'num_heads':8,
                   'datasets':['TFFF','FTFF','TTFF','TTTF','TTTT'],
                   'OUTPUT_folder':'ExplicitFeatureContext_from_PatientA_withcluster_redseqlength/Results_TL/', 
                    # If you want to see the accuracy/recall/prec on the corresponding test data
                   'output_test':[True,True,True,True,True] 
                   }
            runModel=RunStreamModel(**params)
            runModel.buildModel_timeOnly_SplitAttention()  
            runModel.trainModel()
            fname = f'ExplicitFeatureContext_from_PatientA_withcluster_redseqlength/Results_TL/experiment1_combined_model_run_muti_ConnectivityMatrix{seed}.h5'
            runModel.model.save_weights(fname)
             

            # Transfer to patient B
            params_B={'seed':seed,
                    
                   'folder': 'ExplicitFeatureContext_from_PatientB_cluster_redseqlength',
                   'base_filename':'experiment1_combined',# which usecase
                   'hidden_dim_EEG': 100,
                   'max_shift': 38 ,
                   'hidden_dim_extVars': 20,
                   'sequenceLength' : 50, 
                    #if clustering this is 1,else the number of EEG channels
                   'pretrainedweights' : fname,
                   'TimeSegmentation':True, 
                    'nChannels_EEG':1,
                    'nDim':200,
                    'channelShape':[4,462,80,10],
                    'batchsize':60,
                    'steps_per_epoch':3*3,  
                    'validation_steps':3*1,  
                    'num_batches_test':1,
                     
                    'num_heads':8,
                   'datasets':['TFFF','FTFF','TTFF','TTTF','TTTT'],
                   'OUTPUT_folder':'ExplicitFeatureContext_from_PatientB_cluster_redseqlength/Results_TL/', 
                   # If you want to see the accuracy/recall/prec on the corresponding test data      
                   'output_test':[True,True,True,True,True] }
            runModel=RunStreamModel(**params_B)
            runModel.buildModel_timeOnly_SplitAttention()  
            runModel.trainModel()
            runModel.aggregateResults()
        
def main():
    runs()
if __name__ == "__main__":
    main()