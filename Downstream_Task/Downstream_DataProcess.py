#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import array, split
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import heartpy as hp
import pickle
import pandas as pd
import os

def SamplePerClass_Index(dataset_array, num_samples):
    ''' Takes the data and number of samples 
    to return the index of equall number of samples per each value in the data'''
    
    df = pd.DataFrame(dataset_array)
    np.random.seed(42)
    df = df.groupby(df[0]).apply(lambda s: s.sample(num_samples)).droplevel(level=0)
    np.random.seed(42)
    indexes = np.array((df.index))
    return indexes



# split the signal
def split_dataset(data, n_input):
    ''' Takes the signal data with the number of data point for windows 
    and returns the split non-overlapped windows '''

    remainder = len(data) % n_input
    if remainder != 0:
        data = data[:-remainder]
    data = array(split(data, len(data)/n_input))
    return data


#Creating the inputs based on the overlapping / This finction givs us two same windows as x and y
def to_supervised(data, n_input, shift ):
    ''' Takes the split non-overlapped windows with the number of data point
    and number of data point for overlapping which returns the overlapping windows'''

    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))  #flatten the split non-overlapped data
    X = list()
    in_start = 0
    #step over the entire history one time step at a time
    for i in range(len(data)):
        #define the end of the input sequence
        in_end = in_start + n_input
        # ensure we have enough data for this instance
        if in_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
        # move along one time step
        in_start += shift
    X = array(X)
    return X


def all_equal(data):
    ''' takes the windows and check if all datapoint are equal to return True '''
    
    iterator = iter(data)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)



def label_maker_Dalia(data_ppg, data_activity):
    ''' Takes both ppg and activity window data to label the activity signal / 
    This function removes the activity window with not equal data point in the window
    besides the corosponding ppg window data and label the ppg window based on the activity window '''

    labels_activity = []
    Not_Equal_id = []
    for i in range(data_activity.shape[0]):
        Check = all_equal(data_activity[i,:])
        if Check == True :
            labels_activity.append(data_activity[i,:][0])
        else:
            Not_Equal_id.append(i)
    
    Final_data_ppg = np.delete(data_ppg, Not_Equal_id, axis=0)
    labels_activity = (np.array(labels_activity)).reshape(-1,1)
    return Final_data_ppg, labels_activity



def load_and_preprocess_data(config, id_data, scaler=None, fit_scaler=True, removing=True):

    """
    Load and preprocess signal data from a given file path.

    Parameters:
    - config: config file including all parameters.
    - id_data: test person id. 
    - scaler: Instance of a scaler (e.g., StandardScaler). If None, a new scaler will be created.
    - fit_scaler: Boolean flag indicating whether to fit the scaler to the data or use it for transformation only.

    Returns:
    - Preprocessed signal data.
    - Scaler used for the data normalization.
    """
    
    
    params = config["prepare_datasets_params"]
    n_input = params["n_input"]
    shift = params["shift"]
    signal_freq = params["signal_freq"]
    low_freq = params["low_freq"]
    high_freq = params["high_freq"]
    filter_order = params["filter_order"]
    n_input_activity_sig = params["n_input_activity_sig"]
    shift_activity_sig = params["shift_activity_sig"]
    selected_acctivities = config["selected_activities"]
    
    data_directory = config.get("data_directory", "")
    data_path = os.path.join(data_directory, f'S{id_data}.pkl')

    
    try:
        with open(data_path, 'rb') as handle:
            data_dic = pickle.load(handle, encoding='latin1')
        dataset = data_dic['signal']['wrist']['BVP'].astype('float32')
        activity_sig = data_dic['activity'].astype('float32')
        
        filtered_data = hp.filter_signal(dataset[:,0], cutoff=[low_freq, high_freq], sample_rate=signal_freq, order=filter_order, filtertype='bandpass')

        if scaler is None:
            scaler = StandardScaler()
        
        if fit_scaler:
            normalized_data = scaler.fit_transform(filtered_data.reshape(-1, 1))
        else:
            normalized_data = scaler.transform(filtered_data.reshape(-1, 1))

        NormData = normalized_data
        split_data_ppg = split_dataset(NormData, n_input)
        train_x = to_supervised(split_data_ppg, n_input , shift) 

        split_data_activity_label = split_dataset(activity_sig, n_input_activity_sig)
        train_activity_label = to_supervised(split_data_activity_label, n_input_activity_sig, shift_activity_sig)  
            
        train_x, train_activity_label = label_maker_Dalia(train_x, train_activity_label)  

        if removing:
            train_activity_full_list = [0,1,2,3,4,5,6,7,8]
            Not_selected_activities = [x for x in train_activity_full_list if x not in selected_acctivities]
            index_removing = np.where(train_activity_label == Not_selected_activities)
            train_x = np.delete(train_x, index_removing[0], axis=0)
            train_activity_label = np.delete(train_activity_label, index_removing[0], axis=0)
            for i in range(5):
                train_activity_label[train_activity_label == selected_acctivities[i]] = i            
                
        return train_x , train_activity_label, scaler
    
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None, None, None




def prepare_datasets(config, training_persons, test_persons):

    """
    Prepare training and testing datasets.

    """
    
    # Load and preprocess training data
    train_data = []
    label_data = []
    for id_data in training_persons:
        train_x, train_activity_label, scaler = load_and_preprocess_data(config, id_data, scaler=None, fit_scaler=True)
        train_data.append(train_x)
        label_data.append(train_activity_label)
        
    # Concatenate and shuffle training data
    train_data = np.concatenate(train_data)
    label_data = np.concatenate(label_data)

    train_data, label_data = shuffle(train_data, label_data, random_state = 42)
    
    
    # Load and preprocess testing data using the fitted scaler
    test_data = []
    test_label_data = []
    for id_data in test_persons:
        test_x, test_activity_label, _ = load_and_preprocess_data(config, id_data, scaler=scaler, fit_scaler=False)
        test_data.append(test_x)
        test_label_data.append(test_activity_label)

    # Concatenate testing data
    test_data = np.concatenate(test_data)
    test_label_data = np.concatenate(test_label_data)
    test_labels_categorical =  to_categorical(test_label_data) 

    return train_data, label_data, test_data, test_label_data, test_labels_categorical



