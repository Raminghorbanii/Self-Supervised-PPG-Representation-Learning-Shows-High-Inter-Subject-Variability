#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################
#########################
#########################

import os
import gc
import json
import tensorflow as tf
from joblib import dump
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K

from Pretext_Models import CNNAE_model
from Pretext_DataProcess import prepare_datasets


# Load configuration
with open('Pretext_config.json') as config_file:
    config = json.load(config_file)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#########################
#########################
#########################

# Load data using parameters from config
test_num = config["test_person"]
test_persons = [config["test_person"]]
training_persons = config["training_persons"]
training_persons.remove(config["test_person"])

train_data, test_data = prepare_datasets(config, training_persons, test_persons)
print('Datasets are created')

#########################
#########################
#########################

# Training Model

Final_results_list = list()
repeat = config["repeat"]

for r in range(repeat):
    
    model, his_info = CNNAE_model(config, train_data, test_data, repeat = r)

    train_predictions = model.predict(train_data)
    train_mse = mean_squared_error(train_data, train_predictions)
    print('Train Da MSE is %s' %train_mse)
    gc.collect()
    
    test_predictions = model.predict(test_data)
    test_mse = mean_squared_error(test_data, test_predictions)
    print('Test Da MSE is %s' %test_mse)
    gc.collect()

    Final_results_list.append([train_mse, test_mse, his_info])
    
    del model
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    
    print('############## End of Run number %s' %r)
    
    
#########################
#########################
#########################
    
#joblib.dump(Final_results_list,'Final_results_list%s' %dimension +'_person%s' %test_num +'.pkl')

