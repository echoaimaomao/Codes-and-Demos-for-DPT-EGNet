# Codes-and-Demos-for-DPT-EGNet
This repository provides an implementation of the " Dual-Path Transformer based Network with Equalization-Generation Components Prediction " (DPT-EGNet) for Flexible Vibrational Sensor Speech Enhancement in the Time-domain. And the demos of speech enhanced by different models are provided.

# Details about implementing the code
This implementation is based on [TSTNN] (https://github.com/key2miao/TSTNN), thanks Kai Wang for sharing.
0. It is not difficult to install the package for implementing the code. Just follow the tips of the remindings of lacked package, you can make it. In our environment, the pytorch version is 1.5.1 and the python version is 3.6.

1.Preparation of the data set.
 a.Prepare the parallel data and put the data into the folder like: train_ac_data, train_fvs_data, test_ac_data, test_fvs_data
 b.gen_pair.py ------ generate the h5py data and file list for training and testing. the data set path in it should be changed
 
2.Training the dataset
 train.py ------ start training, the file_list_path should be changed
 (DPT_EG.py is the code of the proposed model) 
 
