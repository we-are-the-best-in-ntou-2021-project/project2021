# -*- coding: utf-8 -*-
"""
feature represetor
+
2 encoder(id and action)
+
2classifier(id and action)
"""


import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.layers import Input, Dense, Activation, Dropout, Concatenate, Dot, GlobalAveragePooling1D
from keras.optimizers import Adam
from tensorflow.keras import initializers
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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn import manifold, datasets
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from keras import backend as K
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" #選擇哪一塊gpu
config = ConfigProto()
config.allow_soft_placement=True #如果你指定的設備不存在，允許TF自動分配設備
config.gpu_options.per_process_gpu_memory_fraction=0.9 #分配百分之七十的顯存給程序使用，避免內存溢出，可以自己調整
config.gpu_options.allow_growth = True #按需分配顯存，這個比較重要
session = InteractiveSession(config=config)
"""

class IdActionClassifier():
    def __init__(self, frame, dim, num_actions, num_class):
        self.input_shape = (frame, dim)
        self.latent_dim = 128
        self.num_class = num_class
        self.num_actions = num_actions
        
        self.encoder, self.feature_representer, self.action_encoder, self.id_encoder = self.__encoder() 
        self.action_classifier = self.__action_classifier()
        self.action_classifier.compile(loss=['categorical_crossentropy'], loss_weights=[1], 
                              metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        self.id_classifier = self.__id_classifier()
        self.id_classifier.compile(loss=['categorical_crossentropy'], loss_weights=[1], 
                              metrics=['accuracy'], optimizer=Adam(lr=0.0001))        
        self.combined = self.__combined()
        self.combined.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1,1], 
                              metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        
        
        
    def myloss(self,y_true,y_pred):
        
        return tf.negative(K.binary_crossentropy(y_true,y_pred))
       
    
    def __encoder(self, ):
        
        # building feature representer
        input0 = Input(shape=self.input_shape)
        c0 = Conv1D(64,kernel_size=5,padding='same',activation="relu", input_shape=self.input_shape)(input0)
        c0 = MaxPooling1D(pool_size=2)(c0)
        c0 = BatchNormalization()(c0)
        
        c2 = Conv1D(64,kernel_size=5,padding='same',activation="relu")(c0)
        c2 = MaxPooling1D(pool_size=2)(c2)
        c2 = BatchNormalization()(c2)
        
        c1 = Conv1D(128,kernel_size=5,padding='same',activation="relu")(c2)
        c1 = MaxPooling1D(pool_size=2)(c1)
        c1 = BatchNormalization()(c1)              
        
        feature_representer = Model([input0],[c1]) 
        
        # building action encoder
        input1 = Input(shape=c1.get_shape()[1::])
        ac0 = Conv1D(256,kernel_size=5,padding='same',activation="relu", input_shape=c1.get_shape())(input1)
        ac0 = MaxPooling1D(pool_size=2)(ac0)
        ac0 = BatchNormalization()(ac0)
        
        ac1 = Conv1D(256,kernel_size=5,padding='same',activation="relu")(ac0)
        ac1 = BatchNormalization()(ac1)   
        
        af1 = GlobalAveragePooling1D()(ac1)
        A0 = Dense(self.latent_dim)(af1)
        action_encoder = Model([input1],[A0])
        
        # building id encoder
        input2 = Input(shape=c1.get_shape()[1::])
        ic0 = Conv1D(256,kernel_size=5,padding='same',activation="relu", input_shape=c1.get_shape())(input2)
        ic0 = MaxPooling1D(pool_size=2)(ic0)
        ic0 = BatchNormalization()(ic0)
        
        ic1 = Conv1D(256,kernel_size=5,padding='same',activation="relu")(ic0)
        ic1 = BatchNormalization()(ic1)   
        
        if1 = GlobalAveragePooling1D()(ic1)
        I0 = Dense(self.latent_dim)(if1)
        id_encoder = Model([input2],[I0])
        
        # combine models
        inp = Input(shape=self.input_shape)
        tmp = feature_representer(inp)
        A1 = action_encoder(tmp)
        I1 = id_encoder(tmp)
        model = Model([inp], [A1,I1])
                
        return model, feature_representer, action_encoder, id_encoder
    
    
    def __combined(self, ):
                
        input0 = Input(shape=self.input_shape)
        feature_a, feature_Id = self.encoder(input0)
  
        a_out = self.action_classifier(feature_a)
        Id_out = self.id_classifier(feature_Id)
        
        model = Model([input0],[a_out,Id_out])
        
        return model
        
    
    def __id_classifier(self, ):
        
        input0 = Input(shape=(self.latent_dim,))
        
        c = Sequential()
        c.add(Dense(self.latent_dim//2, activation='relu'))
        c.add(Dense(self.num_class,activation='softmax'))    
        out = c(input0)
        
        model = Model([input0],[out])
        
        return model
        
    
    def __action_classifier(self, ):
        
        input0 = Input(shape=(self.latent_dim,))
        
        c = Sequential()
        c.add(Dense(self.latent_dim//2, activation='relu'))
        c.add(Dense(self.num_actions,activation='softmax'))    
        out = c(input0)
        
        model = Model([input0],[out])
        
        return model
     
        
    
    def my_predict(self, X_test, Y_test, Y_p_test):
        
        #reshape data
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],-1))
        Y_test = keras.utils.to_categorical(Y_test)
        Y_p_test = keras.utils.to_categorical(Y_p_test)
        
        # predict
        feature_a, feature_id = self.encoder.predict(X_test)
        a = self.action_classifier.evaluate(feature_a, Y_test)
        b = self.id_classifier.evaluate(feature_id, Y_p_test)
        
        predict = self.action_classifier.predict(feature_a)
        predictions = np.argmax(predict,axis=1)
        Y_test = np.argmax(Y_test,axis=1)
        print("action:")
        print(a)
        print(pd.crosstab(Y_test,predictions,rownames=['label'],colnames=['predict']))
        
        predict = self.id_classifier.predict(feature_id)
        predictions = np.argmax(predict,axis=1)
        Y_p_test = np.argmax(Y_p_test,axis=1)
        print("id:")
        print(b)
        print(pd.crosstab(Y_p_test,predictions,rownames=['label'],colnames=['predict']))
        
        return a, b
    
    
    def __random_select(self, X, Y, Y_p, batch_size):
        
        # randomly select some training data to train
        idx = np.random.permutation(Y.shape[0])
        idx = idx[:batch_size]
        X_rand0 = X[idx]
        X_rand0 = X_rand0.reshape((X_rand0.shape[0],X_rand0.shape[1],-1))
        # Y is the label of actions
        Y_rand = Y[idx]
        Y_rand = keras.utils.to_categorical(Y_rand)
        # Y is the label of id
        Y_rand_p = Y_p[idx]
        Y_rand_p = keras.utils.to_categorical(Y_rand_p)
        

        return X_rand0, Y_rand, Y_rand_p
           
    
    def train(self, X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, action_start, n_batch, batch_size):
        
        train_a_acc = np.zeros((n_batch//50))
        train_id_acc = np.zeros((n_batch//50))
        
        test_a_acc = np.zeros((n_batch//50))
        test_id_acc = np.zeros((n_batch//50))
        #scr = np.zeros((n_batch//1000))
        #scr_dense = np.zeros((n_batch//1000))
        i = 0
        #j = 0
        
        for batch in range(n_batch):
            
            # randomly select some data to train the combined model(encoder+action_classifier)
            X_rand, Y_rand, Y_rand_p = self.__random_select(X_train, Y_train, Y_p_train, batch_size)
            
            ## update combined model
            combined_loss, a_loss, id_loss, a_acc, id_acc = self.combined.train_on_batch([X_rand], [Y_rand,Y_rand_p])
                      
            if((batch+1) % 50 == 0):
                print ('\nbatch %d:  [%f  %f  %f  %f  %f]' % (batch, combined_loss, a_loss, id_loss, a_acc, id_acc))
                a, b = self.my_predict(X_test, Y_test, Y_p_test)
                train_a_acc[i] = a_acc
                train_id_acc[i] = id_acc
                test_a_acc[i] = a[1]
                test_id_acc[i] = b[1]
                i = i+1      
        
        # plot acc and loss
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(train_a_acc,label='train_a_acc')
        plt.plot(test_a_acc,label='test_a_acc')
        plt.legend()
        plt.xlabel('batch(100batchs)')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(train_id_acc,label='train_id_acc')
        plt.plot(test_id_acc,label='test_id_acc')
        plt.legend()
        plt.xlabel('batch(100batchs)')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.show()  
        
        #feature representation tsne show
        self.tsneshow0(X_train, Y_p_train, X_test, Y_p_test, self.num_class)
        self.tsneshow0(X_train, Y_train, X_test, Y_test, self.num_actions)
        self.tsneshow(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test)
        
                      
    

    def tsneshow(self, x_train, y_train, y_p_train, x_test, y_test, y_p_test):
        
        # feature representor
        inp = self.feature_representer.layers[0].input                                       
        output = self.feature_representer.layers[-1].output        
        func = K.function([inp], [output])         
        
        # data preprocessing
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], -1))
        x_train = x_train.reshape((x_train.shape[0],x_test.shape[1],-1))
        
        # feature represent (testing)
        feature_train = func([x_train])
        feature_test = func([x_test])
        
        a = feature_train[0].reshape((feature_train[0].shape[0],-1))
        b = feature_test[0].reshape((feature_test[0].shape[0],-1))
        feature_represnet = np.vstack((a, b))
        
        X_tsne = manifold.TSNE(n_components=2, init='random', random_state=15, verbose=1).fit_transform(feature_represnet)
        y_test = y_test + self.num_actions
        y = np.concatenate((y_train, y_test), axis=0)
        y_p_test = y_p_test + self.num_class
        y_p = np.concatenate((y_p_train, y_p_test),axis=0)
        
        #print(type(X_tsne))
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne-x_min) / (x_max-x_min)
        plt.figure(figsize=(10, 10))
        
        color_yellow=(1.0, 1.0, 0.0, 0.4)
        color_red=(1.0, 0.0, 0.0, 0.4)
        color_blue=(0, 0, 1.0, 0.4)
        color_black=(0, 0, 0, 0.4)
        color_orange=(1.0,0.27,0.0,0.4)
        color_cyan = (0, 1.0, 1.0, 0.2)
        color_green=(0, 1.0, 0, 0.2)
        color_purple=(1.0, 0, 1.0, 0.2)
        color_skin=(0.9, 0.8, 0.7, 0.2)      
        color_gray=(0.5,0.5,0.41,0.4)     
        color_brown=(0.368, 0.149, 0.07, 0.2)
        
        colormap = np.array([color_yellow, color_red, color_blue, color_black, color_brown, 
                             color_green, color_purple, color_cyan, color_skin, color_gray])
        #colormap = np.array([color_yellow, color_red, color_blue, color_black, 
        #                     color_orange, color_cyan, color_purple, color_skin])
        plt.scatter(X_norm[:,0], X_norm[:,1], c=colormap[y],marker='.')
        plt.show()
        
        plt.scatter(X_norm[:,0], X_norm[:,1], c=colormap[y_p],marker='.')
        plt.show()
    
    def tsneshow0(self, x_train, y_train, x_test, y_test, shift):
        
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], -1))
        x_train = x_train.reshape((x_train.shape[0],x_test.shape[1],-1))
        if shift==self.num_actions:
            feature_test,_ = self.encoder.predict(x_test)
            feature_train,_ = self.encoder.predict(x_train)
        else:
            _,feature_test = self.encoder.predict(x_test)
            _,feature_train = self.encoder.predict(x_train)
        feature = np.concatenate((feature_train,feature_test),axis = 0)
        X_tsne = manifold.TSNE(n_components=2, init='random', random_state=15, verbose=1).fit_transform(feature)
        y_test = y_test + shift
        y = np.concatenate((y_train, y_test), axis=0)
        
        #print(type(X_tsne))
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne-x_min) / (x_max-x_min)
        plt.figure(figsize=(10, 10))
        
        color_yellow=(1.0, 1.0, 0.0, 0.4)
        color_red=(1.0, 0.0, 0.0, 0.4)
        color_blue=(0, 0, 1.0, 0.4)
        color_black=(0, 0, 0, 0.4)
        color_orange=(1.0,0.27,0.0,0.4)
        color_cyan = (0, 1.0, 1.0, 0.2)
        color_green=(0, 1.0, 0, 0.2)
        color_purple=(1.0, 0, 1.0, 0.2)
        color_skin=(0.9, 0.8, 0.7, 0.2)      
        color_gray=(0.5,0.5,0.41,0.4)     
        color_brown=(0.368, 0.149, 0.07, 0.2)
        
        colormap = np.array([color_yellow, color_red, color_blue, color_black, color_brown, 
                             color_green, color_purple, color_cyan, color_skin, color_gray])
        #colormap = np.array([color_yellow, color_red, color_blue, color_black, 
        #                     color_orange, color_cyan, color_purple, color_skin])
        plt.scatter(X_norm[:,0], X_norm[:,1], c=colormap[y],marker='.')
        plt.show()

        
    def meanAndMae(self, X_train, Y_train, Y_p_train, X_test, Y_p_test, id_check, action_start):
        
        filter_id_train = (Y_p_train==id_check)
        Y_train = Y_train[filter_id_train]
        X_train = X_train[filter_id_train]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],-1))
        feature_train = self.encoder.predict(X_train)
        mean_all_train = np.mean(feature_train,axis=0)
        mean_train = np.zeros((self.num_actions, feature_train.shape[1]))
        for i in range(self.num_actions):
            i = i+action_start
            tmp = feature_train[Y_train == i]
            mean_train[i-action_start] = np.mean(tmp,axis=0)
        
        filter_id_test = (Y_p_test==id_check)
        X_test = X_test[filter_id_test]        
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],-1))
        feature_test = self.encoder.predict(X_test)
        mean_test = np.mean(feature_test, axis=0)
        
        print("id check"+str(id_check))
        print("\n----test_alltrain mean distence mae----")
        print(self.mae(mean_all_train, mean_test))
        print("----train actions distence mae----")
        print("action0 action1:")
        print(self.mae(mean_train[0],mean_train[1]))
        print("action1 action2:")
        print(self.mae(mean_train[2],mean_train[1]))
        print("action0 action2:")
        print(self.mae(mean_train[2],mean_train[0]))
        print("------------------------------------------------------------")
    
    def mae(self, A, B):
        return np.sum(np.abs(A-B))
        
        
        
if __name__ == '__main__':

    
    label_test = 1
    id_start = 0
    
    X_train = load("D:/IndependentStudy/train_test_split/train_data.npy")
    Y_train = load("D:/IndependentStudy/train_test_split/train_action.npy")
    Y_p_train = load("D:/IndependentStudy/train_test_split/train_ID.npy")
    
    X_test = load("D:/IndependentStudy/train_test_split/test_data.npy")
    Y_test = load("D:/IndependentStudy/train_test_split/test_action.npy")
    Y_p_test = load("D:/IndependentStudy/train_test_split/test_ID.npy")

    tmp = pre.make_velocity(X_train)
    X_train = np.concatenate((X_train, tmp), axis=-1)
    
    tmp = pre.make_velocity(X_test)
    X_test = np.concatenate((X_test, tmp), axis=-1)
   
    
    GAN0 = IdActionClassifier(frame = 50, dim = 100, num_actions = 5, num_class = 4)
    GAN0.train(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, id_start, n_batch=5000,batch_size=64)
    
    

    
