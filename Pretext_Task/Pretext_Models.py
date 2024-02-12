#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:30:52 2024

@author: raminghorbani
"""


#Load the libraries

import os

from tensorflow.keras.layers import Activation, Input, Reshape, BatchNormalization, UpSampling1D, Conv1D, MaxPooling1D
from tensorflow.keras import optimizers 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from Pretext_Utils import input_gen, step_for_epoch

##############################################################################
from numpy.random import seed
seed(42)
from tensorflow.random import set_seed
set_seed(42)
##############################################################################


def CNNAE_model(config, train_data, test_data, repeat):


    test_num = config["test_person"]
    CNN_AE_params = config["CNN_AE_params"]

    rep_dimension = CNN_AE_params["rep_dimension"]
    n_steps = CNN_AE_params["n_steps"]
    epochs = CNN_AE_params["epochs"]
    batch_size = CNN_AE_params["batch_size"]
    optimizer_params = CNN_AE_params["optimizer"]
    early_stopping_params = CNN_AE_params["early_stopping"]
    model_directory = config.get("model_directory", "")

        
    K.clear_session()
    
    # define parameters for Autoencoder
    verbose, epochs, batch_size = 1, epochs, batch_size
            
    # Model
    input_sig = Input(shape=(n_steps))
    x = Reshape((512,1))(input_sig)
    x = Conv1D(64, 32, padding="same")(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 32, padding="same")(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x) 
    x = Conv1D(1, 32, padding="same")(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    bottleneck = MaxPooling1D(2)(x)

    #Decoder
    x = Conv1D(64, 32, padding="same")(bottleneck)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)  
    x = Conv1D(128, 32, padding="same")(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)      
    x = Conv1D(1, 32, padding="same")(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)   
    x = Reshape((512,))(x)  
    model = Model(input_sig, x)
    
    
    #Define the optimizer
    optimizer = optimizers.legacy.Adam(**optimizer_params)

    model.compile(optimizer= optimizer , loss='mse')
    print(model.summary())


    # Define an encoder model (without the decoder)
    encoder = Model(inputs=input_sig, outputs=bottleneck)
    
    # Save initial random weights of the encoder
    initial_weights_path = os.path.join(model_directory, f'initial_PPGCNNAE_W_{test_num}_{repeat}.h5')
    encoder.save_weights(initial_weights_path)
    
    es = EarlyStopping(**early_stopping_params, monitor='val_loss', mode='min', verbose=1)

    train_generator = input_gen(train_data, train_data, batch_size)
    valid_generator = input_gen(test_data, test_data, batch_size)
    size_per_epoch = step_for_epoch(train_data, batch_size)
    size_per_epoch_valid = step_for_epoch(test_data, batch_size)
    
    history = model.fit(train_generator,
              validation_data=valid_generator,
              steps_per_epoch= size_per_epoch,
              validation_steps = size_per_epoch_valid,
              epochs= epochs,
              verbose=verbose,
              callbacks=[es])

    his_info = [history.history]
    
    # Save the encoder model after training
    encoder_model_path = os.path.join(model_directory, f'CNNAE_encoder_Person{test_num}_D{rep_dimension}_Repeat{repeat}.h5')
    encoder.save(encoder_model_path)
    
    return model, his_info
    
    
    
    
    
    
    