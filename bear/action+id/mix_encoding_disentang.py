# -*- coding: utf-8 -*-
"""

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
        self.latent_dim = 64
        self.num_class = num_class
        self.num_actions = num_actions
        
        self.encoder, self.feature_representer, self.action_encoder, self.id_encoder = self.__encoder() 
        self.encoder.compile(loss=['categorical_crossentropy','categorical_crossentropy'], loss_weights=[1,1],
                             metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        self.action_classifier = self.__action_classifier()
        self.action_classifier.compile(loss=['categorical_crossentropy'], loss_weights=[1], 
                              metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        self.id_classifier = self.__id_classifier()
        self.id_classifier.compile(loss=['categorical_crossentropy'], loss_weights=[1], 
                              metrics=['accuracy'], optimizer=Adam(lr=0.0001))        
        self.combined = self.__combined()
        self.combined.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1,1], 
                              metrics=['accuracy'], optimizer=Adam(lr=0.0001))
        self.discriminator_id = self.__discriminator_id()
        self.discriminator_id.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), loss_weights=[1],metrics=['accuracy'])
        self.discriminator_action = self.__discriminator_action()
        self.discriminator_action.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), loss_weights=[1],metrics=['accuracy'])
        self.stacked_id = self.__stacked_id()
        self.stacked_id.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1,1], 
                      optimizer=Adam(lr=0.0001))
        self.stacked_action = self.__stacked_action()
        self.stacked_action.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1,1], 
                      optimizer=Adam(lr=0.0001))
    

        
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
        
        self.encoder_input_shape = (int(c1.get_shape()[1::][0]),int(c1.get_shape()[1::][1]))
        
        # building action encoder   
        input1 = Input(shape=self.encoder_input_shape)
        ac0 = Conv1D(128,kernel_size=5,padding='same',activation="relu")(input1)
        ac0 = MaxPooling1D(pool_size=2)(ac0)
        ac1 = BatchNormalization()(ac0)
        
        af1 = Flatten()(ac1)
        A0 = Dense(self.latent_dim)(af1)
        action_encoder = Model([input1],[A0])
        
        # building id encoder
        input2 = Input(shape=self.encoder_input_shape)
        ic0 = Conv1D(128,kernel_size=5,padding='same',activation="relu", input_shape=self.encoder_input_shape)(input2)
        ic0 = MaxPooling1D(pool_size=2)(ic0)
        ic1 = BatchNormalization()(ic0)

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
     
        
    def __discriminator_id(self, ):
        
        f1 = Input(shape=(self.latent_dim,))
        f2 = Input(shape=(self.latent_dim,))
        
        classifier = Sequential()

        classifier.add(Dense(64))
        classifier.add(LeakyReLU(alpha=0.2))
        
        classifier.add(Dense(self.num_class,activation='softmax'))
        
        class1 = classifier(f1)
        class2 = classifier(f2)
        dot = Dot(axes=-1)([class1,class2])
        
        model = Model([f1,f2],[dot])
        #model.summary()
        return model
        
    def __discriminator_action(self, ):
        
        f1 = Input(shape=(self.latent_dim,))
        f2 = Input(shape=(self.latent_dim,))
        
        classifier = Sequential()

        classifier.add(Dense(64))
        classifier.add(LeakyReLU(alpha=0.2))
        
        classifier.add(Dense(self.num_actions,activation='softmax'))
        
        class1 = classifier(f1)
        class2 = classifier(f2)
        dot = Dot(axes=-1)([class1,class2])
        
        model = Model([f1,f2],[dot])
        #model.summary()
        return model   
    
    def __stacked_id(self, ):
        
        self.id_classifier.trainable = False
        self.discriminator_action.trainable = False
        self.id_encoder.trainable = True
        
        input0 = Input(shape=self.encoder_input_shape) 
        input1 = Input(shape=self.encoder_input_shape)
        input2 = Input(shape=self.encoder_input_shape)
        
        feature_C = self.id_encoder(input0)
        feature1 = self.id_encoder(input1)
        feature2 = self.id_encoder(input2)
        
        D_out = self.discriminator_action([feature1,feature2])
        C_out = self.id_classifier(feature_C)
        
        model = Model([input0, input1, input2], [C_out, D_out])
        
        return model
        
    
    def __stacked_action(self, ):
        
        self.action_classifier.trainable = False
        self.discriminator_id.trainable = False
        self.action_encoder.trainable = True
        
        input0 = Input(shape=self.encoder_input_shape) 
        input1 = Input(shape=self.encoder_input_shape)
        input2 = Input(shape=self.encoder_input_shape)
        
        feature_C = self.action_encoder(input0)
        feature1 = self.action_encoder(input1)
        feature2 = self.action_encoder(input2)
        
        D_out = self.discriminator_id([feature1,feature2])
        C_out = self.action_classifier(feature_C)
        
        model = Model([input0, input1, input2], [C_out, D_out])
        
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
    
    def __random_select0(self, X, Y, Y_p, batch_size):
        
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
    
    def __random_select(self, X, Y, Y_p, batch_size, idx_p, idx_a):
        
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
                    #print(idx_a)
                    tmp = np.random.choice(idx_a[i][0], 2)
                    if(Y_p[tmp[0]] != Y_p[tmp[1]]):
                        idx = np.concatenate((idx,tmp)).astype(int)
                        break                
        #idx.shape= (batch_size, )
        X_rand2 = X[idx]
        input1 = X_rand2[0::2].reshape((1,batch_size//2,X_rand1.shape[2],-1))
        input2 = X_rand2[1::2].reshape((1,batch_size//2,X_rand1.shape[2],-1))
        X_rand2 = np.concatenate((input1,input2),axis=0)


        return  X_rand1, X_rand2
    
    
    def train(self, X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, action_start, n_batch, batch_size):
        
        train_a_acc = np.zeros((n_batch//50))
        train_id_acc = np.zeros((n_batch//50))
        
        test_a_acc = np.zeros((n_batch//50))
        test_id_acc = np.zeros((n_batch//50))
        #scr = np.zeros((n_batch//1000))
        #scr_dense = np.zeros((n_batch//1000))
        i = 0
        #j = 0
        
        if not os.path.exists("./weight"): 
            for batch in range(n_batch):
                
                # randomly select some data to train the combined model(encoder+action_classifier)
                X_rand, Y_rand, Y_rand_p = self.__random_select0(X_train, Y_train, Y_p_train, batch_size)
                
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
            #self.tsneshow(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test)
            self.action_classifier.save_weights('./action_classifier_weights.h5')
            self.id_classifier.save_weights('./id_classifier_weights.h5')
            self.feature_representer.save_weights('./feature_representer_weights.h5') 
            self.action_encoder.save_weights('./action_encoder_weights.h5')
            self.id_encoder.save_weights('./id_encoder_weights.h5')
            #self.encoder.save_weights('./encoder.h5')
            
        else:
            self.action_classifier.load_weights('./weight/action_classifier_weights.h5')
            self.id_classifier.load_weights('./weight/id_classifier_weights.h5')
            self.feature_representer.load_weights('./weight/feature_representer_weights.h5') 
            self.action_encoder.load_weights('./weight/action_encoder_weights.h5')
            self.id_encoder.load_weights('./weight/id_encoder_weights.h5')

        
        # after training feature representer
        idx_p = [np.where(Y_p_train==i)for i in range(self.num_class)]
        idx_a = [np.where(Y_train==j)for j in range(self.num_actions)]
    
        Y_rand0 = np.zeros((batch_size//2))
        Y_rand1 = np.ones((batch_size//2))
        
        ## feature representer 
        inp = self.feature_representer.layers[0].input                                       
        output = self.feature_representer.layers[-1].output        
        func = K.function([inp], [output])
        
        train_a_acc = np.zeros((n_batch//50))
        train_id_acc = np.zeros((n_batch//50))
        i = 0
        test_a_acc = np.zeros((n_batch//50))
        test_id_acc = np.zeros((n_batch//50))
        
        for batch in range(n_batch):
            
        # randomly select some data to train classifier and discriminator
            X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, batch_size, idx_p, idx_a)    
            X_rand, Y_rand, Y_rand_p = self.__random_select0(X_train, Y_train, Y_p_train, batch_size)
            ## preparing input data for classifier
            f_a, f_i = self.encoder.predict(X_rand)
            ## same person different actions (label=false)
            spda0_a, spda0_i = self.encoder.predict(X_rand1[0])
            spda1_a, spda1_i = self.encoder.predict(X_rand1[1])
            ## different person same actions (label=true)            
            dpsa0_a, dpsa0_i = self.encoder.predict(X_rand2[0])
            dpsa1_a, dpsa1_i = self.encoder.predict(X_rand2[1])
            #print(1)
            ## updataing id/action discriminator
            ### id discriminator
            idd_loss0,idd_acc0 = self.discriminator_id.train_on_batch([spda0_a,spda1_a], Y_rand1)
            idd_loss1,idd_acc1 = self.discriminator_id.train_on_batch([dpsa0_a,dpsa1_a], Y_rand0)
            idd_loss = (idd_loss0+idd_loss1)/2
            idd_acc = (idd_acc0+idd_acc1)/2
            #print(2)
            ### action discriminator
            ad_loss0,ad_acc0 = self.discriminator_action.train_on_batch([spda0_i,spda1_i], Y_rand0)
            ad_loss1,ad_acc1 = self.discriminator_action.train_on_batch([dpsa0_i,dpsa1_i], Y_rand1)
            ad_loss = (ad_loss0+ad_loss1)/2
            ad_acc = (ad_acc0+ad_acc1)/2
            
            #print(3)
            ##updating id/aciton classifier
            ### id classifier
            _, id_acc = self.id_classifier.train_on_batch([f_i],Y_rand_p)
            ### action classifier
            _, a_acc = self.action_classifier.train_on_batch([f_a],Y_rand)
            
        # updataing action/id encoder
            ## preparing input data
            #print(4)
            spda0 = func([X_rand1[0]])[0]
            spda1 = func([X_rand1[1]])[0]
            dpsa0 = func([X_rand2[0]])[0]
            dpsa1 = func([X_rand2[1]])[0]
            X_rand = func([X_rand])[0]
            X_rand0 = X_rand[0:batch_size//2]
            X_rand1 = X_rand[batch_size//2:batch_size]
            Y_rand_p0 = Y_rand_p[0:batch_size//2]
            Y_rand_p1 = Y_rand_p[batch_size//2:batch_size]
            Y_rand_a0 = Y_rand[0:batch_size//2]
            Y_rand_a1 = Y_rand[batch_size//2:batch_size]
            #print(5)
            ## updating id encoder
            self.stacked_id.train_on_batch([X_rand0,spda0,spda1],[Y_rand_p0, Y_rand1])
            self.stacked_id.train_on_batch([X_rand1,dpsa0,dpsa1],[Y_rand_p1, Y_rand0])
            ## updating action encoder
            #print(6)
            self.stacked_action.train_on_batch([X_rand0,spda0,spda1],[Y_rand_a0, Y_rand0])
            self.stacked_action.train_on_batch([X_rand1,dpsa0,dpsa1],[Y_rand_a1, Y_rand1])
            #print(7)
            if((batch+1) % 50 == 0):
                print ('\nbatch %d:  stacked_id [classifier:%f, discriminator:%f], stacked_action [classifier:%f, discriminator:%f]' % (batch, id_acc, ad_acc, a_acc, idd_acc))
                a, b = self.my_predict(X_test, Y_test, Y_p_test)
                test_a_acc[i] = a[1]
                test_id_acc[i] = b[1]
                train_id_acc[i] = id_acc
                train_a_acc[i] = a_acc
                i=i+1
            
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
        
        self.tsneshow0(X_train, Y_p_train, X_test, Y_p_test, self.num_class)
        self.tsneshow0(X_train, Y_train, X_test, Y_test, self.num_actions)
        #self.tsneshow(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test)
            
            
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
    
    X_train = load("./train_test_split/train_data.npy")
    Y_train = load("./train_test_split/train_action.npy")
    Y_p_train = load("./train_test_split/train_ID.npy")
    
    X_test = load("./train_test_split/test_data.npy")
    Y_test = load("./train_test_split/test_action.npy")
    Y_p_test = load("./train_test_split/test_ID.npy")

    tmp = pre.make_velocity(X_train)
    X_train = np.concatenate((X_train, tmp), axis=-1)
    
    tmp = pre.make_velocity(X_test)
    X_test = np.concatenate((X_test, tmp), axis=-1)
   
    
    GAN0 = IdActionClassifier(frame = 50, dim = 100, num_actions = 5, num_class = 4)
    GAN0.train(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, id_start, n_batch=3000,batch_size=64)
    
    

    
