# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:31:47 2021

@author: Piggy
"""


from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dropout, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import os
import pandas as pd
import preprocess_data as pre
import telegram_send as letter
import tensorflow as tf



        
frames = 50
nodes = 25
channels = 2
data_shape = (frames, nodes*channels)
action_type = 0

optimizer = Adam(0.0002, 0.5)         

        
def build_action_classifier():
    model = Sequential()
    model.add(Conv1D(64,kernel_size=10,padding='same',activation="relu"))
    model.add(Conv1D(64,kernel_size=10,padding='same',activation="relu"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(BatchNormalization())
    
    model.add(Conv1D(128,kernel_size=3,padding='same',activation="relu"))
    model.add(Conv1D(128,kernel_size=3,padding='same',activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    
    X = Input(shape = (frames, nodes*channels))
    action_labels = model(X)
    
    return model       

def build_ID_classifier():
    
    model = Sequential()
    model.add(Conv1D(256,kernel_size=10,padding='same',activation="relu"))
    model.add(Conv1D(256,kernel_size=10,padding='same',activation="relu"))
    model.add(MaxPooling1D(pool_size=3))
    model.add(BatchNormalization())
    
    model.add(Conv1D(128,kernel_size=3,padding='same',activation="relu"))
    model.add(Conv1D(128,kernel_size=3,padding='same',activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    # X = Input(shape = (frames, nodes*channels+6))
    X = Input(shape = (frames, nodes*channels))
    action_labels = model(X)
    
    return model       

def make_action_data():
    X_train = np.load('./json_data/data_train.npy')
    Y_train = np.load('./json_data/labels_ID_train.npy')
    
    X_test = np.load('./json_data/data_test.npy')
    Y_test = np.load('./json_data/labels_ID_test.npy')
    
    return (X_train, X_test, Y_train, Y_test) 

def make_ID_data():
    
    X_train = np.load('./json_data/data_train.npy')
    Y_train = np.load('./json_data/labels_ID_train.npy')
    action_train = np.load('./json_data/labels_action_train.npy')
    idx = (np.argmax(action_train, axis=1)==action_type)
    X_train = X_train[idx]
    Y_train = Y_train[idx]
    X_train, Y_train = shuffle(X_train, Y_train, random_state = 0)
    
    X_test = np.load('./json_data/data_test.npy')
    Y_test = np.load('./json_data/labels_ID_test.npy')
    action_test = np.load('./json_data/labels_action_test.npy')
    idx = (np.argmax(action_test, axis=1)==action_type)
    X_test = X_test[idx]
    Y_test = Y_test[idx]
    X_test, Y_test = shuffle(X_test, Y_test, random_state = 0)
    
    return (X_train, X_test, Y_train, Y_test)       
    

def train_predict_action():    
    # train
    '''''''''''
    # for action classifier
    '''''''''''
    X_train_1, X_test_1, Y_train_1, Y_test_1 = make_action_data()
    action_classifier.fit(x = X_train_1, y = Y_train_1, validation_split = 0.2, epochs = 100, batch_size=50)
    
    Y_predict = action_classifier.predict(X_test_1)
    Y_predict = np.argmax(Y_predict, axis=1)
    
    Y_test = np.argmax(Y_test_1, axis=1)
    print('action_classifier accuracy:')
    print(pd.crosstab(Y_test, Y_predict, rownames=['label'], colnames=['predict'], normalize='index'))
    
    accuracy = accuracy_score(Y_test, Y_predict)
    
    base_dir = os.path.dirname(os.path.realpath('__file__'))
    detail = str(base_dir)+"\n"+"action_classifier\n"+str(accuracy)+"\n"+str(pd.crosstab(Y_test, Y_predict, rownames=['label'], colnames=['predict'], normalize='index'))
    letter.send_email(detail, True)  


def train_predict_ID():
    '''''''''''
    # for ID classifier
    '''''''''''
    X_train_2, X_test_2, Y_train_2, Y_test_2 = make_ID_data()
    
    '''
    # prepare training data
    Y_predict_train = action_classifier.predict(X_train_2)
    Y_predict_train = np.repeat(Y_predict_train, frames, axis=1)
    Y_predict_train = Y_predict_train.reshape(Y_predict_train.shape[0], frames, 6)
    X_train_2 = np.concatenate((X_train_2, Y_predict_train), axis = 2)    
    
    # prepare testing data
    Y_predict_test = action_classifier.predict(X_test_2)
    Y_predict_test = np.repeat(Y_predict_test, frames, axis=1)
    Y_predict_test = Y_predict_test.reshape(Y_predict_test.shape[0], frames, 6)
    X_test_2 = np.concatenate((X_test_2, Y_predict_test), axis = 2)
    '''
    
    ID_classifier.fit(x = X_train_2, y = Y_train_2, validation_split = 0.2, epochs = 1, batch_size=50)
    
    # 測試並傳送資料
    Y_predict = ID_classifier.predict(X_test_2)
    Y_predict = np.argmax(Y_predict, axis=1)
    
    Y_test = np.argmax(Y_test_2, axis=1)
    accuracy = accuracy_score(Y_test, Y_predict)
    print('ID_classifier accuracy:'+str(accuracy))
    print(pd.crosstab(Y_test, Y_predict, rownames=['label'], colnames=['predict'], normalize='index'))
    
    
    base_dir = os.path.dirname(os.path.realpath('__file__'))
    detail = str(base_dir)+"\n"+"ID_classifier: "+str(accuracy)+"\n"+str(pd.crosstab(Y_test, Y_predict, rownames=['label'], colnames=['predict'], normalize='index'))
    letter.send_email(detail, True)   
    

action_classifier = build_action_classifier()
'''
action_classifier.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
train_predict_action()
action_classifier.summary()
'''

ID_classifier = build_ID_classifier()

ID_classifier.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
ID_classifier.summary()
train_predict_ID()
