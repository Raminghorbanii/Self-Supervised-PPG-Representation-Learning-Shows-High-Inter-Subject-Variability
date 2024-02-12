#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:10:26 2022

@author: raminghorbani
"""

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, average_precision_score, f1_score, confusion_matrix

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from Downstream_Utils import input_gen, step_for_epoch


class KnnModel():
    
    def __init__(self, config, train_ppg, train_labels, test_ppg, test_labels 
                 , test_labels_categorical, test_id, Sample_Num_per_class, learnedRep):
        
        knn_params = config["knn_params"]
        self.rep_dimension = knn_params["rep_dimension"]
        
        self.saved_model_directory = config.get("saved_model_directory", "")
        self.person_id = test_id
        self.Sample_Num_per_class = Sample_Num_per_class
        self.train_ppg = train_ppg
        self.test_ppg = test_ppg
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.test_labels_categorical = test_labels_categorical
        
        #This is optimized for dimension 64.
        if self.Sample_Num_per_class == 2: self.K = 6
        elif self.Sample_Num_per_class == 5: self.K = 12
        elif self.Sample_Num_per_class == 10: self.K = 20
        elif self.Sample_Num_per_class == 50: self.K = 20            
        else : self.K = 20
        
        if learnedRep:
            encoder_model_path = os.path.join(self.saved_model_directory, f'CNNAE_encoder_Person{self.person_id}_D{self.rep_dimension}_Repeat0.h5')
            encoder = load_model(encoder_model_path, compile=False)
        
            train_ppg = encoder.predict(self.train_ppg)
            self.train_ppg = train_ppg.reshape(train_ppg.shape[0], train_ppg.shape[1]*train_ppg.shape[2])        
            test_ppg = encoder.predict(self.test_ppg)
            self.test_ppg = test_ppg.reshape(test_ppg.shape[0], test_ppg.shape[1]*test_ppg.shape[2])
        
        else:
            self.train_ppg = train_ppg.reshape(self.train_ppg.shape[0], self.train_ppg.shape[1]*self.train_ppg.shape[2])
            self.test_ppg = test_ppg.reshape(self.test_ppg.shape[0], self.test_ppg.shape[1]*self.test_ppg.shape[2])
        
        
        self.knn = self.model_fit()

    def model_fit(self):
        knn = KNeighborsClassifier(n_neighbors= self.K, weights = 'distance')
        knn.fit(self.train_ppg, self.train_labels.reshape(self.train_labels.shape[0],))
        return knn
        
    def evaluation(self):

        y_pred = self.knn.predict(self.test_ppg)        
        y_scores = self.knn.predict_proba(self.test_ppg)

        accuracy = accuracy_score(self.test_labels, y_pred)
        balanced_accuracy_test = balanced_accuracy_score(self.test_labels, y_pred)
        
        auc_each = roc_auc_score(self.test_labels_categorical,y_scores, multi_class ='ovr', average=None)
        auc = roc_auc_score(self.test_labels_categorical, y_scores, multi_class ='ovr',  average = 'weighted')
        print('Test AUC', auc)         
        pr_auc = average_precision_score(self.test_labels_categorical, y_scores, average = 'weighted')
        
        f1 = f1_score(self.test_labels, y_pred, average='weighted')
        cm = confusion_matrix(self.test_labels, y_pred)
        
        return auc, cm



class LRModel():
    
    def __init__(self, config, train_ppg, train_labels, test_ppg, test_labels 
                 , test_labels_categorical, test_id, Sample_Num_per_class, learnedRep):
        
        lr_params = config["lr_params"]
        self.rep_dimension = lr_params["rep_dimension"]
        
        self.saved_model_directory = config.get("saved_model_directory", "")
        self.person_id = test_id
        self.Sample_Num_per_class = Sample_Num_per_class
        self.train_ppg = train_ppg
        self.test_ppg = test_ppg
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.test_labels_categorical = test_labels_categorical
        

        if learnedRep:
            encoder_model_path = os.path.join(self.saved_model_directory, f'CNNAE_encoder_Person{self.person_id}_D{self.rep_dimension}_Repeat0.h5')
            encoder = load_model(encoder_model_path, compile=False)
        
            train_ppg = encoder.predict(self.train_ppg)
            self.train_ppg = train_ppg.reshape(train_ppg.shape[0], train_ppg.shape[1]*train_ppg.shape[2])        
            test_ppg = encoder.predict(self.test_ppg)
            self.test_ppg = test_ppg.reshape(test_ppg.shape[0], test_ppg.shape[1]*test_ppg.shape[2])
        
        else:
            self.train_ppg = train_ppg.reshape(self.train_ppg.shape[0], self.train_ppg.shape[1]*self.train_ppg.shape[2])
            self.test_ppg = test_ppg.reshape(self.test_ppg.shape[0], self.test_ppg.shape[1]*self.test_ppg.shape[2])
        
        
        self.lr = self.model_fit()

    def model_fit(self):
        lr = LogisticRegression(solver = 'liblinear')
        lr.fit(self.train_ppg, self.train_labels.reshape(self.train_labels.shape[0],))
        return lr
        
    def evaluation(self):

        y_pred = self.lr.predict(self.test_ppg)        
        y_scores = self.lr.predict_proba(self.test_ppg)

        accuracy = accuracy_score(self.test_labels, y_pred)
        balanced_accuracy_test = balanced_accuracy_score(self.test_labels, y_pred)
        
        auc_each = roc_auc_score(self.test_labels_categorical,y_scores, multi_class ='ovr', average=None)
        auc = roc_auc_score(self.test_labels_categorical, y_scores, multi_class ='ovr',  average = 'weighted')
        print('Test AUC', auc)         
        pr_auc = average_precision_score(self.test_labels_categorical, y_scores, average = 'weighted')
        
        f1 = f1_score(self.test_labels, y_pred, average='weighted')
        cm = confusion_matrix(self.test_labels, y_pred)
        
        return auc, cm





class CnnLstmModel():
    def __init__(self, config, train_ppg, train_labels, train_labels_categorical, test_ppg, test_labels, test_labels_categorical, test_id ):
        
        cnn_lstm_params = config["cnn_lstm_params"]

        self.saved_model_directory = config.get("saved_model_directory", "")
        self.rep_dimension = cnn_lstm_params["rep_dimension"]
        self.n_features = cnn_lstm_params["n_features"]
        self.n_steps = cnn_lstm_params["n_steps"]
        self.epochs = cnn_lstm_params["epochs"]
        self.batch_size = cnn_lstm_params["batch_size"]
        self.optimizer_params = cnn_lstm_params["optimizer"]
        self.dropout_rate = cnn_lstm_params.get("dropout_rate", 0.5)  # Default to 0.5 if not specified
        self.early_stopping_params = cnn_lstm_params["early_stopping"]
    
        
        
        self.person_id = test_id
        self.train_ppg = train_ppg
        self.test_ppg = test_ppg
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_labels_categorical = train_labels_categorical
        self.test_labels_categorical = test_labels_categorical
        self.model, self.learning_info = self.model_fit()
     
    def model_fit(self):
        
        K.clear_session()        
        input_rep = Input(shape=(self.n_steps, self.n_features))
        
        x = Conv1D(32, 64, activation='tanh',kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01))(input_rep)
        x = BatchNormalization()(x)
        x = MaxPooling1D(4)(x)
        x = Dropout(0.5)(x)
        x = LSTM(32, activation="tanh", recurrent_activation="sigmoid",
                 return_sequences= False,kernel_regularizer=l2(0.01),
                 bias_regularizer=l2(0.01))(x) 
        output = Dense(5,'softmax' , name = 'activity' )(x) 

        # this model maps the input to 5 output of activity labels
        model = Model(input_rep, output)
        
        #Compling the model 
        optimizer = optimizers.Adam(**self.optimizer_params)

        model.compile(optimizer= optimizer, loss= {'activity':'categorical_crossentropy'}, metrics={'activity':['AUC']})
        print(model.summary()) 
        
        #Fit
        train_generator = input_gen(self.train_ppg, self.train_labels_categorical, self.batch_size)
        valid_generator = input_gen(self.test_ppg, self.test_labels_categorical, self.batch_size)
        size_per_epoch = step_for_epoch(self.train_ppg, self.batch_size)
        size_per_epoch_valid = step_for_epoch(self.test_ppg, self.batch_size) 

        es = EarlyStopping(**self.early_stopping_params, monitor='val_loss', mode='min', verbose=1)

        mc = ModelCheckpoint('best_CnnLSTM_model%s' %self.person_id  +'_D%s' %self.rep_dimension +'_Complex.h5', monitor='val_loss', mode='min',
                             save_weights_only=True, verbose=1, save_best_only=True)


        history = model.fit(train_generator,validation_data=valid_generator,
                                   steps_per_epoch= size_per_epoch,
                                   validation_steps = size_per_epoch_valid,
                                   epochs= self.epochs, verbose=1,
                                   callbacks=[es, mc])


        learning_info = [history.history]
        model.load_weights('best_CnnLSTM_model%s' %self.person_id  +'_D%s' %self.rep_dimension +'_Complex.h5')
        
        return model, learning_info
    
    def evaluation(self):

        trainPredict = self.model.predict(self.train_ppg)
        testPredict = self.model.predict(self.test_ppg)
        
        #Returning the final label
        train_pred = np.argmax(trainPredict, 1)
        test_pred = np.argmax(testPredict, 1)
        
        auc_train = roc_auc_score(self.train_labels_categorical, trainPredict, multi_class = 'ovr', average = 'macro')
        auc_each_train = roc_auc_score(self.train_labels_categorical, trainPredict, multi_class ='ovr', average=None)
        auc_test = roc_auc_score(self.test_labels_categorical, testPredict, average = 'weighted')
        auc_each_test = roc_auc_score(self.test_labels_categorical, testPredict, multi_class ='ovr', average=None)
        conf_mat_train = confusion_matrix(self.train_labels, train_pred)
        conf_mat_test = confusion_matrix(self.test_labels, test_pred)

        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        return  auc_test, conf_mat_test        
            

            



class Encoder_MLP():
    def __init__(self, config, train_ppg, train_labels, train_labels_categorical,
                 test_ppg, test_labels, test_labels_categorical, test_id, repeat):
        
        encoder_mlp_params = config["encoder_mlp_params"]
        
        self.rep_dimension = encoder_mlp_params["rep_dimension"]
        self.n_features = encoder_mlp_params["n_features"]
        self.n_steps = encoder_mlp_params["n_steps"]
        self.epochs = encoder_mlp_params["epochs"]
        self.batch_size = encoder_mlp_params["batch_size"]
        self.optimizer_params = encoder_mlp_params["optimizer"]
        self.early_stopping_params = encoder_mlp_params["early_stopping"]
        
        # Determine model directory
        self.saved_model_directory = config.get("saved_model_directory", "")
        
        self.repeat = repeat
        self.person_id = test_id
        self.train_ppg = train_ppg
        self.test_ppg = test_ppg
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_labels_categorical = train_labels_categorical
        self.test_labels_categorical = test_labels_categorical

             
    
    def model(self):
        
        encoder_model_path = os.path.join(self.saved_model_directory, f'CNNAE_encoder_Person{self.person_id}_D{self.rep_dimension}_Repeat0.h5')
        encoder_weights_path = os.path.join(self.saved_model_directory, f'initial_PPGCNNAE_W_{self.person_id}_{self.repeat}.h5')

        # Load the encoder model and weights
        encoder = load_model(encoder_model_path, compile=False)
        encoder.load_weights(encoder_weights_path)
        
        # Freeze all the layers
        for layer in encoder.layers[:]:
            layer.trainable = True
        
        # Check the trainable status of the individual layers
        for layer in encoder.layers:
            print(layer, layer.trainable)
                
        K.clear_session()        
        input_rep = Input(shape=(self.n_steps, self.n_features))
        
        x = encoder(input_rep)
        x = Reshape((self.rep_dimension,))(x)
        output = Dense(5,'softmax' , name = 'activity' )(x) 

        # this model maps the input to 5 output of activity labels
        model = Model(input_rep, output)
        
        #Compling the model 
        optimizer = optimizers.Adam(**self.optimizer_params)

        model.compile(optimizer= optimizer, loss= {'activity':'categorical_crossentropy'}, metrics={'activity':['AUC']})
        print(model.summary()) 
        
        #Fit
        train_generator = input_gen(self.train_ppg, self.train_labels_categorical, self.batch_size)
        valid_generator = input_gen(self.test_ppg, self.test_labels_categorical, self.batch_size)
        size_per_epoch = step_for_epoch(self.train_ppg, self.batch_size)
        size_per_epoch_valid = step_for_epoch(self.test_ppg, self.batch_size) 

        es = EarlyStopping(**self.early_stopping_params, monitor='val_loss', mode='min', verbose=1)

        mc = ModelCheckpoint('best_EncoderMlp_model%s' %self.person_id  +'_D%s' %self.rep_dimension +'_Random.h5', monitor='val_loss', mode='min',
                             save_weights_only=True, verbose=1, save_best_only=True)


        history = model.fit(train_generator,validation_data=valid_generator,
                                   steps_per_epoch= size_per_epoch,
                                   validation_steps = size_per_epoch_valid,
                                   epochs= self.epochs, verbose=1,
                                   callbacks=[es, mc])


        learning_info = [history.history]
        model.load_weights('best_EncoderMlp_model%s' %self.person_id  +'_D%s' %self.rep_dimension +'_Random.h5')
        
        return model, learning_info
    
    def evaluation(self):
        final_model = self.model()[0]
        trainPredict = final_model.predict(self.train_ppg)
        testPredict = final_model.predict(self.test_ppg)
        
        #Returning the final label
        train_pred = np.argmax(trainPredict, 1)
        test_pred = np.argmax(testPredict, 1)
        
        auc_train = roc_auc_score(self.train_labels_categorical, trainPredict, multi_class = 'ovr', average = 'macro')
        auc_each_train = roc_auc_score(self.train_labels_categorical, trainPredict, multi_class ='ovr', average=None)
        auc_test = roc_auc_score(self.test_labels_categorical, testPredict, average = 'weighted')
        auc_each_test = roc_auc_score(self.test_labels_categorical, testPredict, multi_class ='ovr', average=None)
        conf_mat_train = confusion_matrix(self.train_labels, train_pred)
        conf_mat_test = confusion_matrix(self.test_labels, test_pred)

        K.clear_session()
        tf.compat.v1.reset_default_graph()
        
        return  auc_test, conf_mat_test        
            
