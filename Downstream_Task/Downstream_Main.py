#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:05:08 2024

@author: raminghorbani
"""


#########################
#########################
#########################

import os
import gc
import json
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from joblib import dump

from Downstream_Models import KnnModel, LRModel, CnnLstmModel, Encoder_MLP
from Downstream_DataProcess import prepare_datasets, SamplePerClass_Index

# Load configuration
with open('Downstream_config.json') as config_file:
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
selected_activities = config["selected_activities"]

train_data, label_data, test_data, test_label_data, test_labels_categorical = prepare_datasets(config, training_persons, test_persons)


#Selecting sample per each class to make different size of balanced training data
Sample_Num_per_class_list = config["sample_num_per_class_list"]
selected_model = config["selected_model"]

for Sample_Num_per_class in Sample_Num_per_class_list:

    samples_index = SamplePerClass_Index(label_data, Sample_Num_per_class)  
    train_ppg_final, train_labels_final = train_data[samples_index], label_data[samples_index]
    train_ppg_final, train_labels_final = shuffle(train_ppg_final, train_labels_final, random_state = 42)
    train_labels_categorical_final = to_categorical(train_labels_final)  
            
    del samples_index
    gc.collect()
    
    
    if selected_model == 'kNN':
        #Knn Model
        results = KnnModel(config, train_ppg_final, train_labels_final, 
                                         test_data, test_label_data,test_labels_categorical,
                                         test_id = test_num, Sample_Num_per_class = Sample_Num_per_class ,learnedRep = True).evaluation()
              

    if selected_model == 'LR':
        #LR Model
        results = LRModel(config, train_ppg_final, train_labels_final, 
                                         test_data, test_label_data,test_labels_categorical,
                                         test_id = test_num, Sample_Num_per_class = Sample_Num_per_class ,learnedRep = True).evaluation()


    if selected_model == 'CNN-LSTM':
        #Cnn-Lstm Model
        results = [CnnLstmModel(config, train_ppg_final, train_labels_final, train_labels_categorical_final,
                                                   test_data, test_label_data, test_labels_categorical,
                                                   test_id = test_num).evaluation() for i in tf.range(5)]

    if selected_model == 'MLP':
        #Encoder_MLP Model
        results = [Encoder_MLP(config, train_ppg_final, train_labels_final, train_labels_categorical_final,
                                                   test_data, test_label_data, test_labels_categorical,
                                                   test_id = test_num, repeat = i).evaluation() for i in range(5)]
        
    results_filename = f'results_{selected_model}_person{config["test_person"]}_SampleNum{Sample_Num_per_class}.pkl'
    dump([results], results_filename)
    


