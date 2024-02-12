import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import heartpy as hp
from numpy import array, split
import os

# split the signal
def split_dataset(data, n_input):
    # restructure into windows of weekly data
    remainder1 = len(data) % n_input
    if remainder1 != 0:
        data = data[:-remainder1]
    data = array(split(data, len(data)/n_input))
    return data


#Creating the inputs based on the overlapping / This finction givs us two same windows as x and y
def to_supervised(train, n_input, shift ):
    #flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X,y = list(), list()
    in_start = 0
    #step over the entire history one time step at a time
    for i in range(len(data)):
        #define the end of the input sequence
        in_end = in_start + n_input
        # ensure we have enough data for this instance
        if in_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
            y.append(x_input)
        # move along one time step
        in_start += shift
    X = array(X)
    X = X.reshape((len(X), np.prod(X.shape[1:])))
    y = array(y)
    y = y.reshape((len(y), np.prod(y.shape[1:])))
    return X, y


def load_and_preprocess_data(config, id_data, scaler=None, fit_scaler=True):
    """
    Load and preprocess signal data from a given file path.

    Parameters:
    - config: config file including all paramters/ 
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
    
    data_directory = config.get("data_directory", "")
    data_path = os.path.join(data_directory, f'S{id_data}.pkl')


    try:
        with open(data_path, 'rb') as handle:
            data_dic = pickle.load(handle, encoding='latin1')
        dataset = data_dic['signal']['wrist']['BVP'].astype('float32')
        filtered_data = hp.filter_signal(dataset[:,0], cutoff=[low_freq, high_freq], sample_rate=signal_freq, order=filter_order, filtertype='bandpass')

        if scaler is None:
            scaler = StandardScaler()
        
        if fit_scaler:
            normalized_data = scaler.fit_transform(filtered_data.reshape(-1, 1))
        else:
            normalized_data = scaler.transform(filtered_data.reshape(-1, 1))

        NormData = normalized_data
        SplitData = split_dataset(NormData, n_input)
        train_x, train_y = to_supervised(SplitData, n_input , shift)
            
        return train_x , scaler
    
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None, None


def prepare_datasets(config, training_persons, test_prsons):
    """
    Prepare training and testing datasets.

    """
    # Load and preprocess training data
    train_data = []
    for id_data in training_persons:
        train_x, scaler = load_and_preprocess_data(config, id_data, scaler=None, fit_scaler=True)
        train_data.append(train_x)

    # Concatenate and shuffle training data
    train_data = np.concatenate(train_data)
    train_data = shuffle(train_data, random_state=42)


    # Load and preprocess testing data using the fitted scaler
    test_data = []
    for id_data in test_prsons:
        test_x, _ = load_and_preprocess_data(config, id_data, scaler=scaler, fit_scaler=False)
        test_data.append(test_x)

    # Concatenate testing data
    test_data = np.concatenate(test_data)

    return train_data, test_data




