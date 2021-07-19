"""
distangling GAN

id_classifier + actions_discriminator
"""


import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.layers import Input, Dense, Activation, Dropout, Concatenate
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
    def __init__(self, frame = 50, dim = 100, num_actions = 3, num_class = 4):
        self.input_shape = (frame, dim)
        self.latent_dim = 50
        self.num_class = num_class
        self.num_actions = num_actions
        
        # Build the encoder(generator)
        self.encoder = self.__encoder()
        #self.encoder.compile(loss='binary_crossentropy', optimizer='Adam')
        # Build the discriminator
        self.discriminator = self.__discriminator()   
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), loss_weights=[1],metrics=['accuracy'])
        # Build the classifier
        self.classifier = self.__classifier()
        self.classifier.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), loss_weights=[1],metrics=['accuracy'])
        # For the combined model we will only train the encoder(generator)
        self.combined = self.__combined(self.encoder, self.discriminator, self.classifier)
        self.combined.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1, 1], 
                      optimizer=Adam(lr=0.0001))

    def __encoder(self, ):
        
        model = Sequential()
        
        model.add(Conv1D(32,kernel_size=5,padding='same',activation="relu", input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(32,kernel_size=5,padding='same',activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
          
        model.add(Conv1D(64,kernel_size=5,padding='same',activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(self.latent_dim))
        
        
        return model
        
    def __discriminator(self, ):
        
        model = Sequential()
        
        model.add(Dense(50,input_dim=self.latent_dim*2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        #model.summary()
        
        return model
 
 
    def __classifier(self, ):
        
        model = Sequential()
        
        model.add(Dense(50))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(self.num_class, activation='softmax'))
        #model.summar()
        
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
        
        
        # output1(same person different actions) for discriminator 
        #num_person_list = np.random.multinomial(batch_size//2, np.ones(self.num_class)/self.num_class,size=1)[0]
        idx = np.array([])
        num_person_list = np.random.multinomial(batch_size//2, np.ones(self.num_class)/self.num_class,size=1)[0]
        for i in range(0, self.num_class):
            for _ in range(num_person_list[i]):
                while(True):
                    tmp = np.random.choice(idx_p[i][0], 2)
                    if(Y[tmp[0]] != Y[tmp[1]]):
                        idx = np.concatenate((idx,tmp)).astype(int)
                        break   
        #idx.shape= (batch_size, )
        X_rand1 = X[idx]
        #X_rand1.shape = (batch_size,frame,25,dim)
        input1 = X_rand1[0::2].reshape((1,batch_size//2,X_rand1.shape[1],X_rand1.shape[2]*X_rand1.shape[3]))
        input2 = X_rand1[1::2].reshape((1,batch_size//2,X_rand1.shape[1],X_rand1.shape[2]*X_rand1.shape[3]))
        X_rand1 = np.concatenate((input1,input2),axis=0)
        # X_rand1[0,:]: input1;  X_rand1[1,:]: input2
        # Y_rand1 = np.ones((batch_size//2))
           
        # output2(same actions different person) for discriminator 
        idx = np.array([])
        num_action_list = np.random.multinomial(batch_size//2, np.ones(self.num_actions)/self.num_actions,size=1)[0]
        for i in range(0, self.num_actions):
            for _ in range(num_action_list[i]):
                while(True):
                    tmp = np.random.choice(idx_a[i][0], 2)
                    if(Y_p[tmp[0]] != Y_p[tmp[1]]):
                        idx = np.concatenate((idx,tmp)).astype(int)
                        break                
        #idx.shape= (batch_size, )
        X_rand2 = X[idx]
        input1 = X_rand2[0::2].reshape((1,batch_size//2,X_rand1.shape[2],-1))
        input2 = X_rand2[1::2].reshape((1,batch_size//2,X_rand1.shape[2],-1))
        X_rand2 = np.concatenate((input1,input2),axis=0)


        return X_rand0, Y_rand0, X_rand1, X_rand2
        
    
    
    def train(self, X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, action_start, n_batch, batch_size):
        
        idx_p = [np.where(Y_p_train==i)for i in range(self.num_class)]
        idx_a = [np.where(Y_train==i+action_start)for i in range(self.num_actions)]
        Y_rand2 = np.ones((batch_size//2))
        Y_rand1 = np.zeros((batch_size//2))
        
        for batch in range(n_batch):
            
            # randomly select some data to train classifier and discriminator
            X_rand0, Y_rand0, X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, batch_size, idx_p, idx_a)
            
            ## update classifier
            f0 = self.encoder.predict(X_rand0)
            c_loss = self.classifier.train_on_batch(f0, Y_rand0)
            
            ## update actions_discriminator
            ### same person different actions (label=false)
            false1 = self.encoder.predict(X_rand1[0])
            false2 = self.encoder.predict(X_rand1[1])
            false = np.hstack((false1,false2))
            d_loss0,d_acc0 = self.discriminator.train_on_batch(false, Y_rand1)
            ### different person same actions (label=true)            
            true1 = self.encoder.predict(X_rand2[0])
            true2 = self.encoder.predict(X_rand2[1])
            true = np.hstack((true1,true2))
            d_loss1,d_acc1 = self.discriminator.train_on_batch(true, Y_rand2)
            
            # randomly select some data to train generator only
            input0, C_out, X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, batch_size, idx_p, idx_a)
            false1 = X_rand1[0]
            false2 = X_rand1[1]
            true1 = X_rand2[0]
            true2 = X_rand2[1]
            input1 = np.concatenate((false1, true1), axis=0)
            input2 = np.concatenate((false2, true2), axis=0)
            D_out = np.concatenate((Y_rand2,Y_rand1))           
            ## update generator
            self.combined.train_on_batch([input0, input1, input2], [C_out, D_out])     
            
            d_loss = (d_loss1+d_loss0)/2
            d_acc = (d_acc1+d_acc0)/2
            #print ('batch: %d, [Discriminator :: d_loss: %f, accuracy: %f], [ Classifier :: loss: %f, acc: %f]' % (batch, d_loss, d_acc, c_loss[0],c_loss[1]))
            # predict model every 10 epochs
            #if((batch+1)%50 == 0):
            print ('batch: %d, [Discriminator :: d_loss: %f, accuracy: %f], [ Classifier :: loss: %f, acc: %f]' % (batch, d_loss, d_acc, c_loss[0],c_loss[1]))
            if((batch+1) % 100 == 0):
                self.my_predict(X_test, Y_p_test)  
                



if __name__ == '__main__':
    
    filename = 'gym_6kind_012_5_'
    action_start = 0
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
    
    GAN0 = distanglingGAN()
    GAN0.train(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, action_start, n_batch=5000,batch_size=64)
    

    
