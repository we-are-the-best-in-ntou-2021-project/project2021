"""
distangling GAN
"""

import decodingJason as de
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.layers import Input, Dense, Activation, Dropout, Concatenate
from keras.optimizers import Adam
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import tensorflow as tf
from numpy import load
import pylab
import matplotlib.pyplot as plt
import preprocessData as pre
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #選擇哪一塊gpu
config = ConfigProto()
config.allow_soft_placement=True #如果你指定的設備不存在，允許TF自動分配設備
config.gpu_options.per_process_gpu_memory_fraction=0.9 #分配百分之七十的顯存給程序使用，避免內存溢出，可以自己調整
config.gpu_options.allow_growth = True #按需分配顯存，這個比較重要
session = InteractiveSession(config=config)
"""

class distanglingGAN():
    def __init__(self, frame = 70, dim = 50, num_labels = 6, num_class = 4):
        self.input_shape = (frame, dim)
        self.latent_dim = 128
        self.num_class = num_class
        self.num_labels = num_labels
        
        # Build the encoder(generator)
        self.enconder = self.__encoder()
        # Build the discriminator
        self.discriminator = self.__discriminator()
        # Build the classifier
        self.classifier = self.__classifier()
        # For the combined model we will only train the encoder(generator)
        self.combined = self.__combined(self.encoder, self.discriminator, self.classifier)
        
    


    def __encoder(self, ):
        
        model = Sequential()
        
        model.add(Conv1D(64,kernel_size=5,padding='same',activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=3))
        model.add(BatchNormalization())
        
        model.add(Conv1D(64,kernel_size=5,padding='same',activation="relu"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(BatchNormalization())
          
        model.add(Conv1D(128,kernel_size=5,padding='same',activation="relu"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(self.latent_dim))
        
        return model
        
    def __discriminator(self, ):
        
        model = Sequential()
        
        model.add(Dense(256,input_dim=self.latent_dim*2))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        #model.summary()
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        
        return model
 
 
    def __classifier(self, ):
        
        model = Sequential()
        
        model.add(Dense(256,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.num_class, activation='softmax'))
        #model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        
        return model
        
        
    def __combined(self, G, D, C):
        
        G.trainable = True
        D.trainable = False
        C.trainable = False
        
        input0 = Input(shape=self.input_shape) 
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)
        
        feature_C = G(input0)
        feature1 = G(input1)
        feature2 = G(input2)
        feature_D = Concatenate()([feature1, feature2])
        
        D_out = D(feature_D)
        C_out = C(feature_C)
        
        model = Model([input0, input1, input2], [C_out, D_out])
        model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[3, 1], 
                      optimizer=Adam(lr=0.0002, beta_1=0.5))
        
        return model
        
    
    def __predict(self, X_test, Y_test):
        pass
    
    def __random_select(self,  X, Y, Y_p, n_batch, idx_p, idx_a):
        
        # output0 for classifier (labels = rabbit:0, bear:1, haha:2, senior:3)
        idx = np.random.permutation(Y.shape[0])
        idx = idx[:n_batch]
        X_rand0 = X[idx]
        X_rand0 = X_rand.reshape((X_rand0.shape[0],X_rand0.shape[1],-1))
        # X_rand0: input0
        Y_rand0 = Y_p[idx]
        Y_rand0 = keras.utils.to_categorical(Y_rand0)
        
        # output1(same person different actions) for discriminator (labels=True)
        num_person_list = np.random.multinomial(n_batch//2, np.ones(4)/4,size=1)[0]
        idx = np.concatenate([np.random.choice(idx_p[i][0], num_person_list[i]*2)for i in range(4)])
        X_rand1 = X[idx]
        X_rand1 = X_rand1.reshape((2,X_rand1.shape[0],X_rand1.shape[1],X_rand1.shape[2],X_rand1.shape[3]))
        # X_rand1[0,:]: input1;  X_rand1[1,:]: input2
        #Y_rand1 = np.ones((n_batch//2))
    
        
        # output2(same actions different person) for discriminator (labels=False)
        action_num = 5
        num_action_list = np.random.multinomial(n_batch//2, np.ones(action_num)/action_num,size=1)[0]
        idx = np.concatenate([np.random.choice(idx_a[i][0], num_action_list[i]*2)for i in range(action_num)])
        X_rand2 = X[idx]
        X_rand2 = X_rand2.reshape((2, X_rand2.shape[0], X_rand2.shape[1], X_rand2.shape[2], X_rand2.shape[3]))
        # X_rand2[0,:]: input1; X_rand2[1,:]: input2
        #Y_rand2 = np.zeros((n_batch//2))

        return X_rand0, Y_rand0, X_rand1, X_rand2
        
    
    
    def train(self, X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, n_epochs = 50, n_batch = 64):
        
        people_list = [0,1,2,3]
        actions_list = [0,1,2,3,4]
        idx_p = [np.where(Y_p_train==i)for i in people_list]
        idx_a = [np.where(Y_train==i)for i in actions_list]
        Y_rand1 = np.ones((n_batch//2))
        Y_rand2 = np.zeros((n_batch//2))
        
        for epoch in n_epochs:
            
            # randomly select some data to train classifier and discriminator
            X_rand0, Y_rand0, X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, n_batch, idx_p, idx_a)
            X_rand0.reshape((X_rand0.shape[0],X_rand0.shape[1],-1))
            X_rand1.reshape((X_rand1.shape[0],X_rand1.shape[1],-1))
            X_rand2.reshape((X_rand2.shape[0],X_rand2.shape[1],-1))
            
            ## update classifier
            f0 = self.enconder.predict(X_rand0)
            self.classifier.train_on_batch(X_rand0, Y_rand0)
            
            ## update discriminator
            ### same
            same1 = self.enconder.predict(X_rand1[0])
            same2 = self.enconder.predict(X_rand1[1])
            same = np.hstack((same1,same2))
            self.classifier.train_on_batch(same, Y_rand1)
            ### different            
            diff1 = self.enconder.predict(X_rand2[0])
            diff2 = self.enconder.predict(X_rand2[1])
            diff = np.hstack((diff1,diff2))
            self.classifier.train_on_batch(diff, Y_rand2)
            
            # randomly select some data to train generator only
            input0, C_out, X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, n_batch, idx_p, idx_a)
            input0.reshape((input0.shape[0], input0.shape[1],-1))
            X_rand1.reshape((X_rand1.shape[0],X_rand1.shape[1],-1))
            X_rand2.reshape((X_rand2.shape[0],X_rand2.shape[1],-1))
            same1 = self.enconder.predict(X_rand1[0])
            same2 = self.enconder.predict(X_rand1[1])
            diff1 = self.enconder.predict(X_rand2[0])
            diff2 = self.enconder.predict(X_rand2[1])
            input1 = np.concatenate((same1,diff1), axis=0)
            input2 = np.concatenate((same2,diff2), axis=0)
            D_out = np.concatencate((Y_rand1,Y_rand2))           
            ## update generator
            self.combined.train_on_batch([input0, input1, input2], [C_out, D_out])     
        
            
            # predict model every 10 epochs
            if(epoch % 10 == 0):
                self.__predict(X_test, Y_test)  # unfinished
                



if __name__ == '__main__':
    pass
  
