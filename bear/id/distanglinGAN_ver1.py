# -*- coding: utf-8 -*-
"""
distangling GAN
No, this is not GAN
id_classifier + min
"""

import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.layers import Input, Dense, Activation, Dropout, Concatenate
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import tensorflow as tf
from numpy import load, save
import pylab
import matplotlib.pyplot as plt
import preprocessData as pre
import os

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
    def __init__(self, frame = 50, dim = 100, num_labels = 6, num_class = 4):
        self.input_shape = (frame, dim)
        self.latent_dim = 128
        self.num_class = num_class
        self.num_labels = num_labels
        
        # Build the encoder(generator)
        self.encoder = self.__encoder()
        #self.encoder.compile(loss='binary_crossentropy', optimizer='Adam')
        # Build the discriminator
        self.minD = self.__min(self.encoder)   
        # Build the classifier
        self.classifier = self.__classifier()
        self.classifier.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5],metrics=['accuracy'])
        # For the combined model we will only train the encoder(generator)
        self.combined = self.__combined(self.encoder, self.minD, self.classifier)
        self.combined.compile(loss=['categorical_crossentropy', 'mse'], loss_weights=[1, 0.1], 
                      optimizer=Adam(lr=0.0002, beta_1=0.5))

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
        
        #model.summary()
        return model
        
    def __min(self, G):
        
        def abs_min(pair):
            f1, f2 = pair
            sub = keras.layers.Subtract()([f1,f2])
            return keras.backend.abs(sub)
        
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)
        f1 = G(input1)
        f2 = G(input2)
        out = Lambda(abs_min)([f1,f2])
        model = Model([input1,input2], [out])
        #model.summary()
        
        return model
 
 
    def __classifier(self, ):
        
        model = Sequential()
        
        model.add(Dense(256,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.num_class, activation='softmax'))
        #model.summar()
        
        return model
        
        
    def __combined(self, G, minD, C):
        
        input0 = Input(shape=self.input_shape) 
        #pair same person
        input1 = Input(shape=self.input_shape)
        input2 = Input(shape=self.input_shape)
        
        f0 = G(input0)
        C_out = C(f0)

        minD_out = minD([input1,input2])
        
        model = Model([input0,input1,input2],[C_out,minD_out])
        
        return model
        
    
    def my_predict(self, X_test, Y_test):
        
        #reshape data
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],-1))
        Y_test = keras.utils.to_categorical(Y_test)        
        # predict
        Feature = self.encoder.predict(X_test)
        a = self.classifier.evaluate(Feature,Y_test,verbose=0)
        print(a)
        predictions = self.classifier.predict_classes(Feature)
        Y_test = np.argmax(Y_test,axis=1)
        print(pd.crosstab(Y_test,predictions,rownames=['label'],colnames=['predict']))
    
    
    def __random_select(self,  X, Y, Y_p, batch_size, idx_p, idx_a):
        
        # output0 for classifier (labels = rabbit:0, bear:1, haha:2, senior:3)
        idx = np.random.permutation(Y.shape[0])
        idx = idx[:batch_size]
        X_rand0 = X[idx]
        X_rand0 = X_rand0.reshape((X_rand0.shape[0],X_rand0.shape[1],-1))
        # X_rand0: input0
        Y_rand0 = Y_p[idx]
        Y_rand0 = keras.utils.to_categorical(Y_rand0)
        
        # output1(same person different actions) for discriminator (labels=True)
        num_person_list = np.random.multinomial(batch_size, np.ones(4)/4,size=1)[0]
        idx = np.concatenate([np.random.choice(idx_p[i][0], num_person_list[i]*2)for i in range(4)])
        #idx.shape= (batch_size*2, )
        X_rand1 = X[idx]
        #X_rand1.shape = (batch_size*2,frame,25,2)
        input1 = X_rand1[0::2].reshape((1,batch_size,X_rand1.shape[1],X_rand1.shape[2]*X_rand1.shape[3]))
        input2 = X_rand1[1::2].reshape((1,batch_size,X_rand1.shape[1],X_rand1.shape[2]*X_rand1.shape[3]))
        X_rand1 = np.concatenate((input1,input2),axis=0)
        # X_rand1[0,:]: input1;  X_rand1[1,:]: input2
        #Y_rand1 = np.ones((batch_size))
    

        return X_rand0, Y_rand0, X_rand1
        
    
    
    def train(self, X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, n_batch, batch_size):
        
        people_list = [0,1,2,3]
        actions_list = [0,1,2]
        idx_p = [np.where(Y_p_train==i)for i in people_list]
        idx_a = [np.where(Y_train==i)for i in actions_list]
        Y_rand1 = np.zeros((batch_size, self.latent_dim))
        
        for batch in range(n_batch):
            
            # randomly select some data to train classifier and discriminator
            X_rand0, Y_rand0, X_rand1 = self.__random_select(X_train, Y_train, Y_p_train, batch_size, idx_p, idx_a)
            
            ## update encoder and classfier with minD
            loss = self.combined.train_on_batch([X_rand0,X_rand1[0],X_rand1[1]],[Y_rand0,Y_rand1])
            
            #print ('batch: %d, [Discriminator :: d_loss: %f, accuracy: %f], [ Classifier :: loss: %f, acc: %f]' % (batch, d_loss, d_acc, c_loss[0],c_loss[1]))
            # predict model every 10 epochs
            if((batch+1) % 100 == 0):
                print ('batch: '+ str(batch) + '\ntrain loss: ' + str(loss)) 
                if((batch+1)%300==0):
                    self.my_predict(X_test, Y_p_test)  
                



if __name__ == '__main__':
    
    filename = 'gym_6kind_012_'
    X_train = load(filename+"data.npy")
    Y_train = load(filename+"labels.npy")
    Y_p_train = load(filename+"labels_p.npy")
    X_test = load(filename+"data_test.npy")
    Y_test = load(filename+"labels_test.npy")
    Y_p_test = load(filename+"labels_p_test.npy")
    
    
    tmp = pre.make_velocity(X_train)
    X_train = np.concatenate((X_train, tmp), axis=-1)
    tmp = pre.make_velocity(X_test)
    X_test = np.concatenate((X_test, tmp), axis=-1)
    #X_train = pre.make_velocity(X_train)
    #X_test = pre.make_velocity(X_test)
    
    GAN = distanglingGAN()
    GAN.train(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, n_batch=2100,batch_size=128)
    
    