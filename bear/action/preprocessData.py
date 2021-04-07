# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 01:44:41 2021

@author: User
"""
import numpy as np


"""
data = load('data.npy')
data_test = load('data_test.npy')
labels = load('labels.npy')
labels_test = load('labels_test.npy')
"""
    
# 就是第i+1偵與第i之向量, 此shape = (data資料量, frame-1, 25,2)
def make_velocity(data):
    velocity = np.zeros((data.shape[0],1,data.shape[2],data.shape[3]))
    r = data.shape[1]-1
    for i in range(0,r):
        tmp = data[:,i+1]-data[:,i]
        tmp = tmp.reshape(tmp.shape[0],1,tmp.shape[1],-1)
        velocity = np.concatenate((velocity, tmp),axis=1)
    return velocity*30

def make_accel(data):
    velocity = make_velocity(data)
    accel = make_velocity(velocity)
    return accel
    
# 以第0偵當作基準點，再去算每一偵與基準點之差值(振福), shape = (data資料量, frame, 25, 2)
def make_offset(data):
    move = np.zeros((data.shape[0],1,25,2))
    for i in range(1,data.shape[1]):
        tmp = data[:,i]-data[:,0]
        tmp = tmp.reshape(tmp.shape[0],1,tmp.shape[1],-1)
        move = np.concatenate((move, tmp),axis=1)
    return move

    
# angle 234 567(left, right elbow) 91011 121314(left, rigth knee) 
# shape = (data量, frame, 4, 1)
def make_angle(data):
    a0 = data[:,:,2,:]- data[:,:,3,:]
    b0 = data[:,:,4,:]- data[:,:,3,:]
    c= np.sum(a0*b0,axis=2)
    nb = np.linalg.norm(b0,axis=2)
    na = np.linalg.norm(a0,axis=2)
    angle = np.arccos(c/na/nb).reshape(data.shape[0],-1,1)
    a1 = data[:,:,5,:]- data[:,:,6,:]
    b1 = data[:,:,7,:]- data[:,:,6,:]
    c= np.sum(a1*b1,axis=2)
    nb = np.linalg.norm(b1,axis=2)
    na = np.linalg.norm(a1,axis=2)
    tmp = np.arccos(c/na/nb).reshape(data.shape[0],-1,1)
    angle = np.concatenate((angle,tmp),axis = -1)
    a2 = data[:,:,9,:]- data[:,:,10,:]
    b2 = data[:,:,11,:]- data[:,:,10,:]
    c= np.sum(a2*b2,axis=2)
    nb = np.linalg.norm(b2,axis=2)
    na = np.linalg.norm(a2,axis=2)
    tmp = np.arccos(c/na/nb).reshape(data.shape[0],-1,1)
    angle = np.concatenate((angle,tmp),axis = -1)
    a3 = data[:,:,12,:]- data[:,:,13,:]
    b3 = data[:,:,14,:]- data[:,:,13,:]
    c= np.sum(a3*b3,axis=2)
    nb = np.linalg.norm(b3,axis=2)
    na = np.linalg.norm(a3,axis=2)
    tmp = np.arccos(c/na/nb).reshape(data.shape[0],-1,1)
    angle = np.concatenate((angle,tmp),axis = -1)
    angle = angle*100
    return angle


def send_mes_to_bot(mes):
    import requests

    r = requests.post(
        f"https://api.telegram.org/bot1706687838:AAGPTV4GnjsgB69zq6I1t7c97HUzRMzJTS8/sendMessage",
        json={
        "chat_id":"1378236226",
        "text":mes,
        },
    )
