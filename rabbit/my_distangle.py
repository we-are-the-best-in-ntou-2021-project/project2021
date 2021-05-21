from keras.models import Sequential,Model
from keras.layers import Conv1D,MaxPooling1D,BatchNormalization,Dense,Activation,Flatten,Dropout
from keras.layers import Concatenate,Input,Lambda
from keras.optimizers import Adam
import keras
import numpy as np
from numpy import load,save
# import tensorflow.compat.v1 as tf #使用1.0版本的方法
# tf.disable_v2_behavior() #禁用2.0版本的方法
import tensorflow as tf
# import keras.backend as K
# K.clear_session()

import GPU_setting as G
G.setting()

def get_data():
  person_name = ["bear","rabbit","haha","senior"]
  dirname = "jsondata"
  epochs = 50
  offset = "none"

  data = load(dirname + '/bear_data.npy')
  labels = load(dirname + '/bear_labels.npy')
  c = np.array(np.full(data.shape[0],0))
  labels = np.vstack((labels,c))
  for i in range(3):
    a = load(dirname + '/'+ person_name[i+1] +'_data.npy')
    b = load(dirname + '/'+ person_name[i+1] +'_labels.npy')
    c = np.array(np.full(a.shape[0],i+1))
    b = np.vstack((b,c))
    data = np.vstack((data,a))
    labels = np.hstack((labels,b))
  
  data_test = data[np.where(labels[0,:]>=5),:,:,:]
  data = data[np.where(labels[0,:]<=2),:,:,:]
  data_test = data_test.reshape(data_test.shape[1],data_test.shape[2],data_test.shape[3]*data_test.shape[4])
  data = data.reshape(data.shape[1],data.shape[2],data.shape[3]*data.shape[4])

  labels_test = labels[:,np.where(labels[0,:]>=5)]
  labels = labels[:,np.where(labels[0,:]<=2)]
  labels_test = labels_test.reshape(labels_test.shape[0],labels_test.shape[2])
  labels = labels.reshape(labels.shape[0],labels.shape[2])

  index=np.arange(data.shape[0])
  np.random.shuffle(index) 
  data = data[index]
  labels[0] = labels[0][index]
  labels[1] = labels[1][index]

  labels_v = labels[0]
  labels = keras.utils.to_categorical(labels[1])
  labels_test_v = labels_test[0]
  labels_test = keras.utils.to_categorical(labels_test[1])

  return data,data_test,labels,labels_test

class GAN():
  def __init__(self,frame=50,dim=50,num_class=4):

    self.latent_dim = 128
    self.num_class = num_class
    self.optimizer = Adam(lr=0.0002,beta_1=0.9)
    self.input_shape = (frame,dim)

    self.encoder = self.build_encoder()
    self.classifer = self.build_classifer()
    self._min = self.build_min(self.encoder)
    self.combined = self.build_combine(self.encoder,self.classifer,self._min)


  def build_encoder(self):
  
    model = Sequential()
    model.add(Conv1D(32,kernel_size=5,padding='same',activation='relu',input_shape=self.input_shape))
    model.add(Conv1D(32,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv1D(64,kernel_size=5,padding='same',activation='relu'))
    model.add(Conv1D(64,kernel_size=5,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(self.latent_dim))

    return model

  def build_classifer(self):

    model = Sequential()

    model.add(Dense(256,input_dim=self.latent_dim))
    model.add(Activation('relu'))
    
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(self.num_class,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=self.optimizer,loss_weights=[0.5],metrics=['accuracy'])

    return model
  
  def build_min(self,E):

    def abs_min(pair):
      f1,f2 = pair
      sub = keras.layers.Subtract()([feature_1,feature_2])
      return keras.backend.abs(sub)

    input1 = Input(shape=self.input_shape)
    input2 = Input(shape=self.input_shape)
    feature_1 = E(input1)
    feature_2 = E(input2)
    out = Lambda(abs_min)([feature_1,feature_2])

    model = Model([input1,input2],[out])

    return model

  def build_combine(self,E,C,_min):

    E.trainalbe = True
    C.trainalbe = False

    input0 = Input(shape=self.input_shape)
    input1 = Input(shape=self.input_shape)
    input2 = Input(shape=self.input_shape)
    
    feature_c = E(input0)
    
    C_out = C(feature_c)
    # print(labels.shape)

    min_out = _min([input1,input2])

    model = Model([input0,input1,input2],[C_out,min_out])
    
    # loss1 = keras.losses.categorical_crossentropy(labels,C_out)
    # loss2 = keras.losses.mse(feature_1,feature_2)
    # my_loss = 0.75*loss1 + 0.25*loss2

    # model.add_loss(my_loss)
    model.compile(loss=['categorical_crossentropy','mse'],optimizer=self.optimizer,loss_weights=[3,1],metrics=['accuracy'])
    return model

  # def my_loss():
    
  def random_select(self,data,labels,batch_size):
    
    rand_index = np.random.randint(low=0,high=data.shape[0],size=batch_size)
    rand_data0 = data[rand_index,:,:]
    rand_labels = labels[rand_index,:]

    rand_index = np.random.randint(low=0,high=data.shape[0],size=batch_size)
    rand_data1 = data[rand_index,:,:]

    rand_index = np.random.randint(low=0,high=data.shape[0],size=batch_size)
    rand_data2 = data[rand_index,:,:]

    # print(rand_data0)
    return rand_data0,rand_labels,rand_data1,rand_data2

  def my_predict(x_test,y_test):

    y_test = keras.utils.to_categorical(y_test)
    feature = self.encoder.predict(x_test)
    a = self.classifer.evaluate(feature,y_test)
    print(a)

  def train(self,data,labels,data_test,labels_test,batch_size,n_batch):

    zero = np.zeros((batch_size,self.latent_dim))
    for batch in range(n_batch):
      # rand_data,rand_labels,rand_data1,rand_data2 = self.random_select(data,labels,batch_size)

      # feature = self.encoder.predict(rand_data)
      # c_loss = self.classifer.train_on_batch(feature,rand_labels)

      rand_data,rand_labels,rand_data1,rand_data2 = self.random_select(data,labels,batch_size)
      # print(rand_labels)
      # rand_labels = keras.utils.to_categorical(rand_labels)
      # print(zero)
      loss = self.combined.train_on_batch([rand_data,rand_data1,rand_data2],[rand_labels,zero])

      if (batch+1)%100==0:
        print('batch: %d,loss: %f]'%(batch,loss))
        self.my_predict(data_test,labels_test)

if __name__ == '__main__':
  x,x_test,y,y_test = get_data()
  # print(a.shape,c.shape)

  distangle = GAN()
  distangle.train(x,y,x_test,y_test,batch_size=128,n_batch=2000)
