# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:50:58 2021

@author: User
"""

from numpy import load
import numpy as np

label_test = 5

bear_data = load("D:/IndependentStudy/json_gym_6kind/bear_data.npy")
bear_labels = load("D:/IndependentStudy/json_gym_6kind/bear_labels.npy")

bear_data_test = bear_data[bear_labels==label_test]
bear_labels_test = bear_labels[bear_labels==label_test]
bear_labels_p_test = np.full((bear_labels_test.shape[0]),1)

bear_data = bear_data[(bear_labels==1)|(bear_labels==2)]
bear_labels = bear_labels[(bear_labels==1)|(bear_labels==2)]
bear_labels_p = np.full((bear_labels.shape[0]),1)


rabbit_data = load("D:/IndependentStudy/json_gym_6kind/rabbit_data.npy")
rabbit_labels = load("D:/IndependentStudy/json_gym_6kind/rabbit_labels.npy")

rabbit_data_test = rabbit_data[rabbit_labels==label_test]
rabbit_labels_test = rabbit_labels[rabbit_labels==label_test]
rabbit_labels_p_test = np.full((rabbit_labels_test.shape[0]),0)

rabbit_data = rabbit_data[(rabbit_labels==1)|(rabbit_labels==2)]
rabbit_labels = rabbit_labels[(rabbit_labels==1)|(rabbit_labels==2)]
rabbit_labels_p = np.full((rabbit_labels.shape[0]),0)


senior_data = load("D:/IndependentStudy/json_gym_6kind/senior_data.npy")
senior_labels = load("D:/IndependentStudy/json_gym_6kind/senior_labels.npy")
senior_labels_test = senior_labels[senior_labels==label_test]
senior_data_test = senior_data[senior_labels==label_test]
senior_labels_p_test = np.full((senior_labels_test.shape[0]),3)

senior_data = senior_data[(senior_labels==1)|(senior_labels==2)]
senior_labels = senior_labels[(senior_labels==1)|(senior_labels==2)]
senior_labels_p = np.full((senior_labels.shape[0]),3)


haha_data = load("D:/IndependentStudy/json_gym_6kind/haha_data.npy")
haha_labels = load("D:/IndependentStudy/json_gym_6kind/haha_labels.npy")
haha_data_test = haha_data[haha_labels==label_test]
haha_labels_test = haha_labels[haha_labels==label_test]
haha_labels_p_test = np.full((haha_labels_test.shape[0]),2)

haha_data = haha_data[(haha_labels==1)|(haha_labels==2)]
haha_labels = haha_labels[(haha_labels==1)|(haha_labels==2)]
haha_labels_p = np.full((haha_labels.shape[0]),2)


data = np.concatenate((rabbit_data,bear_data,haha_data,senior_data), axis=0)
labels = np.concatenate((rabbit_labels,bear_labels,haha_labels,senior_labels),axis=0)
labels_p = np.concatenate((rabbit_labels_p,bear_labels_p, haha_labels_p, senior_labels_p), axis=0)

data_test = np.concatenate((rabbit_data_test,bear_data_test,haha_data_test,senior_data_test), axis=0)
labels_test = np.concatenate((rabbit_labels_test,bear_labels_test,haha_labels_test,senior_labels_test),axis=0)
labels_p_test = np.concatenate((rabbit_labels_p_test,bear_labels_p_test, haha_labels_p_test, senior_labels_p_test), axis=0)


#此為01234trian, 5test

filename = 'gym_6kind_12_5_'
np.save(filename+"data.npy",data)
np.save(filename+"labels.npy",labels)
np.save(filename+"labels_p.npy",labels_p)
np.save(filename+"data_test.npy",data_test)
np.save(filename+"labels_test.npy",labels_test)
np.save(filename+"labels_p_test.npy",labels_p_test)
"""
filename = 'gym_6kind_345_1_'

np.save(filename+"data.npy",data)
np.save(filename+"labels.npy",labels)
np.save(filename+"labels_p.npy",labels_p)
np.save(filename+"data_test.npy",data_test)
np.save(filename+"labels_test.npy",labels_test)
np.save(filename+"labels_p_test.npy",labels_p_test)

"""


