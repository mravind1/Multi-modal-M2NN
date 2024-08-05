MMA-Metadata supported Multivariate attention for onset detection and prediction
0.Python 3.8.5 and Keras 2.3.1 code is used for binary classification problem for EEG, ICP, ECG and ABP datasets described in paper.

1. Code for EEG is made available in Multimodal_M2NN_files\Code\EEG.

Patient_files_preprocess.py calls EEGUtility_segmented_V4.py for preprocessing patient files.An example is shown in EEGModel-SeizurePrediction-TransferLearning-WithNoCluster-WithContext.ipynb and TransferLearningRuns_Chameleon.py how to preprocess and further to build the healthcare model respectively.

TransferLearningRuns_Chameleon.py is a wrapper class calling EEGModel_Service_V6_LSTM.py and is used for training the MMA model. EEGModel_Service_V6_CNN.py is used for CNN baselines.

RMT_feature_extraction folder has Matlab wrapper classes to call RMT feature extraction algorithm.

2. Code for COVID is kept in  Multimodal_M2NN_files\Code\Covid and dataset is kept in Multimodal_M2NN_files\Data\Covid\CovidRMTdata. Python3.8.5 and Keras 2.3.1 code is used for the regression problem as well.

Coviddata_prediction_Ranking-TimeSegmentation-Rateoflogofcases.ipynb- This file has code for a COVID dataset prediction problem using MMA described in paper.

Coviddata_prediction_Ranking-TimeSegmentation-DCRNN.ipynb-- This file has code for Covid dataset prediction problem using DCRNN Competitor model.


2.1 Dataset:  us-states.csv joined with a separate data source namely nst-est2019-alldata.csv


2.2 Connectivity_matrix: implicitvariate_connectivity_matrix_States.mat is the connectivity matrix for implicit frequency context case.
fullvariate_connectivity_matrix_States.csv is the connectivity matrix for explicit spatial context case.

3. Code for Traffic is kept in  Multimodal_M2NN_files\Code\Traffic and dataset is kept in Multimodal_M2NN_files\Data\Traffic. Python3.8.5 and Keras 2.3.1 code is used for the regression problem as well.

4. Code for Bitcoin is kept in  Multimodal_M2NN_files\Code\Bitcoin and dataset is kept in Multimodal_M2NN_files\Data\Bitcoin. Python3.8.5 and Keras 2.3.1 code is used for the regression problem as well.

