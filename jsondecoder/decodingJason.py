"""
JasonDecoder which is a class is used to decode json files.
"""

import json
import numpy as np
from glob import glob

class JasonDecoder():
    
    def __init__(self, dataset_name, frame, nodes=25):
        self.dataset_name = dataset_name  # the list of actions (eg. [run, walk, upstair, downstair, ...])
        self.frame = frame  # how many frames to be one data
        self.nodes = nodes  # 25 node(joints)
        self.number_of_actions = len(dataset_name)
    
    # private function(don't call it)
    def __load_a_json(self, filename, datas, a, b):
        try:
            with open(filename, 'r') as fp:
                data = json.load(fp)
                data = data["people"][0]["pose_keypoints_2d"]
                X = np.array(data[0::3]).reshape((1,self.nodes))    #X
                Y = np.array(data[1::3]).reshape((1,self.nodes))    #Y
                datas[a][b] = np.concatenate((X,Y)).T
                return True
        except:
            # print('No')
            return False
    
    """
    # decoding json files to numpy array
    output: 
        dataset: 4dim np.array, shape=(number of data, frame, nodes,  2)
        labels: 1d list, shape=(numbr of data)
    """
    def decoding(self,):
        labels = list()
        label = 0
        for action in self.dataset_name:
            tmp_path = glob(".\%s\%s\*" % ("data", action))
            
            #path = glob('D:\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\output_jsons\*')
            n_person = 0
            
            for person in tmp_path:
                #print(person)
                path = glob(person+"\*")
                number_of_3d_array = len(path) // self.frame  #number_of_data_of_one_vedio
                datas = np.zeros((number_of_3d_array, self.frame, self.nodes, 2))
                
                a = 0
                b = 0
                for filename in path:
                    if a >= number_of_3d_array:
                        labels.append(label)
                        break
                    
                    if(self.__load_a_json(filename, datas, a, b) == True):
                        b = b + 1
                        if b >= self.frame:
                            b = 0
                            a = a + 1
                
                if(n_person == label == 0):
                    dataset = np.array(datas, copy = True) 
                else:
                    dataset = np.concatenate((dataset, datas))
                n_person = n_person + 1

            label = label + 1
        #print(dataset[1])
        return dataset, labels


#example
"""
actions = ['down', 'phone', 'raise', 'run']
jasondecoder = JasonDecoder(dataset_name=actions, frame=50)    
dataset, labels = jasondecoder.decoding()
"""
        
