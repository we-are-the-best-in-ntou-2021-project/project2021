# -*- coding: utf-8 -*-

"""
distangling GAN

id_classifier + actions_discriminator(dot)
"""


import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.layers import Input, Dense, Activation, Dropout, Concatenate, Dot
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

class distanglingGAN():
    def __init__(self, frame, dim, num_actions, num_class):
        self.input_shape = (frame, dim)
        self.latent_dim = 50
        self.num_class = num_class
        self.num_actions = num_actions
        
        
        # Build the encoder(generator)
        self.encoder = self.__encoder()
        # self.encoder.compile(loss='binary_crossentropy', optimizer='Adam')
        # Build the discriminator
        self.discriminator = self.__discriminator()   
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), loss_weights=[1],metrics=['accuracy'])
        # Build the classifier
        self.classifier = self.__classifier()
        self.classifier.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), loss_weights=[1],metrics=['accuracy'])
        # For the combined model we will only train the encoder(generator)
        self.combined = self.__combined(self.encoder, self.discriminator, self.classifier)
        self.combined.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], loss_weights=[1,0], 
                      optimizer=Adam(lr=0.0001))
        
    def myloss(self,y_true,y_pred):
        return tf.negative(K.binary_crossentropy(y_true,y_pred))
    
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
 
 
    def __classifier(self, ):
        
        model = Sequential()
        """
        model.add(Dense(100,input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        """
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
        
        D_out = D([feature1,feature2])
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
        return a
    
    
    def __random_select(self, X, Y, Y_p, batch_size, idx_p, idx_a):
        
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


        return X_rand0, Y_rand0, X_rand1, X_rand2
           
    
    def train(self, X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, action_start, n_batch, batch_size):
        
        idx_p = [np.where(Y_p_train==i)for i in range(self.num_class)]
        idx_a = [np.where(Y_train==j+action_start)for j in range(self.num_actions)]

        Y_rand2 = np.ones((batch_size//2))
        Y_rand1 = np.zeros((batch_size//2))
        
        train_loss = np.zeros((n_batch//50))
        train_dis_loss = np.zeros((n_batch//50))
        train_acc = np.zeros((n_batch//50))
        train_dis_acc = np.zeros((n_batch//50))
        test_loss = np.zeros((n_batch//50))
        test_acc = np.zeros((n_batch//50))
        #scr = np.zeros((n_batch//1000))
        #scr_dense = np.zeros((n_batch//1000))
        i = 0
        #j = 0
        
        for batch in range(n_batch):
            
            # randomly select some data to train classifier and discriminator
            X_rand0, Y_rand0, X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, batch_size, idx_p, idx_a)
            
            ## update classifier
            f0 = self.encoder.predict(X_rand0)
            c_loss = self.classifier.train_on_batch(f0, Y_rand0)
          
            ## update actions_discriminator
            ### same person different actions (label=false)
            diff1 = self.encoder.predict(X_rand1[0])
            diff2 = self.encoder.predict(X_rand1[1])
            d_loss0,d_acc0 = self.discriminator.train_on_batch([diff1,diff2], Y_rand1)
            ### different person same actions (label=true)            
            same1 = self.encoder.predict(X_rand2[0])
            same2 = self.encoder.predict(X_rand2[1])
            d_loss1,d_acc1 = self.discriminator.train_on_batch([same1,same2], Y_rand2)
            
            # randomly select some data to train generator only
            input0, C_out, X_rand1, X_rand2 = self.__random_select(X_train, Y_train, Y_p_train, batch_size, idx_p, idx_a)           
            same1 = X_rand1[0]
            same2 = X_rand1[1]
            diff1 = X_rand2[0]
            diff2 = X_rand2[1]
            input1 = np.concatenate((same1,diff1), axis=0)
            input2 = np.concatenate((same2,diff2), axis=0)
            D_out = np.concatenate((Y_rand2,Y_rand1))     #false, true      
            ## update generator
            self.combined.train_on_batch([input0, input1, input2], [C_out, D_out])
            d_loss = (d_loss1+d_loss0)/2
            d_acc = (d_acc1+d_acc0)/2

            #print ('batch: %d, [Discriminator :: d_loss: %f, accuracy: %f], [ Classifier :: loss: %f, acc: %f]' % (batch, d_loss, d_acc, c_loss[0],c_loss[1]))
            if((batch+1) % 50 == 0):
                print ('batch: %d, [Discriminator :: d_loss: %f, accuracy: %f], [ Classifier :: loss: %f, acc: %f]' % (batch, d_loss, d_acc, c_loss[0],c_loss[1]))
                a = self.my_predict(X_test, Y_p_test)
                train_acc[i] = c_loss[1]
                train_acc[i] = c_loss[1]
                train_dis_loss[i] = d_loss
                train_dis_acc[i] = d_acc
                test_acc[i] = a[1]
                test_loss[i] = a[0]
                i = i+1      
            #if((batch+1) % 1000 == 0):
                #self.tsneshow(X_train, Y_p_train, X_test[0:1], Y_p_test[0:1])
                #scr[j] = self.testsvm(X_train,Y_p_train,X_test,Y_p_test)
                #scr_dense[j] = a[1]
                #j = j + 1
        #self.meanAndMae(X_train, Y_train, Y_p_train, X_test, Y_p_test, 0, action_start)
        #self.meanAndMae(X_train, Y_train, Y_p_train, X_test, Y_p_test, 1, action_start)
        #self.meanAndMae(X_train, Y_train, Y_p_train, X_test, Y_p_test, 2, action_start)
        #self.meanAndMae(X_train, Y_train, Y_p_train, X_test, Y_p_test, 3, action_start)
        #self.tsneshow(X_train, Y_p_train, X_test, Y_p_test)    
        # plot 
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(train_loss,label='train_loss')
        plt.plot(test_loss,label='test_loss')
        plt.plot(train_dis_loss, label='train_dis_loss')
        plt.legend()
        plt.xlabel('batch(50batchs)')
        plt.ylabel('loss')
        plt.grid(True)
        plt.subplot(1,2,2)
        plt.plot(train_acc,label='train_accuracy')
        plt.plot(test_acc,label='test_acuracy')
        plt.plot(train_dis_acc, label='train_dis_acc')
        plt.legend()
        plt.xlabel('batch(50batchs)')
        plt.ylabel('accuracy')
        plt.grid(True)
        plt.show()     
        #print(scr_dense)
        #print(scr)
        pre.send_mes_to_bot(a)

    def tsneshow(self, x_train, y_train, x_test, y_test):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], -1))
        feature_test = self.encoder.predict(x_test)
        x_train = x_train.reshape((x_train.shape[0],x_test.shape[1],-1))
        feature_train = self.encoder.predict(x_train)
        feature = np.concatenate((feature_train,feature_test),axis = 0)
        X_tsne = manifold.TSNE(n_components=2, init='random', random_state=15, verbose=1).fit_transform(feature)
        y_test = y_test + self.num_class
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
        
        
    def testsvm(self, X_train, Y_train, X_test, Y_test):      
        
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],-1))
        Feature_train = self.encoder.predict(X_train)
        
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],-1))
        Feature_test = self.encoder.predict(X_test)
        
        clf = svm.SVC(kernel='rbf')
        clf.fit(Feature_train, Y_train)
        
        prdict = clf.predict(Feature_test)
        
        return ((np.sum(prdict == Y_test))/Y_test.shape[0])

        
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
            mean_train[i] = np.mean(tmp,axis=0)
        
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
    """
    filename = 'D:/IndependentStudy/directly_behind_phone/'
    action_start = 0
    X_train = load(filename+"data.npy")
    Y_train = load(filename+"labels_action.npy")
    Y_p_train = load(filename+"labels_ID.npy")
    filename = 'gym_directily_behind_phone_012_5_'
    X_test = load(filename+"data_test.npy")
    Y_test = load(filename+"labels_test.npy")
    Y_p_test = load(filename+"labels_p_test.npy")
    """
    
    label_test = 0
    action_start = 2
    
    data = load("D:/IndependentStudy/directly_behind_phone/data.npy")
    labels_a = load("D:/IndependentStudy/directly_behind_phone/labels_action.npy")
    labels_p = load("D:/IndependentStudy/directly_behind_phone/labels_ID.npy")

    X_test = data[labels_a == label_test]
    Y_p_test = labels_p[labels_a == label_test]
    Y_test = labels_a[labels_a == label_test]

    filter_012 = (labels_a == 2)|(labels_a==3)|(labels_a==4)
    Y_p_train = labels_p[filter_012]
    Y_train = labels_a[filter_012]
    X_train = data[filter_012]
    
    
    tmp = pre.make_velocity(X_train)
    X_train = np.concatenate((X_train, tmp), axis=-1)
    
    tmp = pre.make_velocity(X_test)
    X_test = np.concatenate((X_test, tmp), axis=-1)
    #X_train = pre.make_velocity(X_train)
    #X_test = pre.make_velocity(X_test) 
    
    
    GAN0 = distanglingGAN(frame = 50, dim = 100, num_actions = 3, num_class = 4)
    GAN0.train(X_train, Y_train, Y_p_train, X_test, Y_test, Y_p_test, action_start, n_batch=8000,batch_size=64)
    
    

    
