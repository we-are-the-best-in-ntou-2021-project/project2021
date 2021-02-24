"""
JasonDecoder which is a class is used to decode json files.
"""

import json
import numpy as np
from glob import glob

class JasonDecoder():
    
    def __init__(self, dataset_name, frame=150, nodes=25):
        
        # dataset 的名稱，就是xxxx.avi的那個xxxx
        self.dataset_name = dataset_name
        # 一次讀幾禎
        self.frame = frame
        # 25個關節點
        self.nodes = nodes
        
    
    def _load_a_json(self, filename, X, Y, ACC, a, b):
        
        try:
            with open(filename, 'r') as fp:
                data = json.load(fp)
                data = data["people"][0]["pose_keypoints_2d"]
                X[a][b] = data[::3]
                Y[a][b] = data[1::3]
                ACC[a][b] = data[2::3]
            return True
        except:
            return False
    
    """
    objective:
        decoding json files to numpy array
    input: 
        filename: xxxxxx.avi
    output: 
    """
    def decoding(self, ):
        #path = glob('./%s/*' % (self.dataset_name))
        #path = glob('D:\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\output_jsons\*')
        
        # int number_of_2d_array
        number_of_2d_array = len(path) // self.frame
        data_X = np.zeros((number_of_2d_array, self.frame, self.nodes))
        data_Y = np.zeros((number_of_2d_array, self.frame, self.nodes))
        data_ACC = np.zeros((number_of_2d_array, self.frame, self.nodes))
        
        a = 0
        b = 0
        for filename in path:
            if a >= number_of_2d_array:
                return data_X, data_Y, data_ACC, number_of_2d_array

            if(self._load_a_json(filename, data_X, data_Y, data_ACC, a, b)):
                b = b + 1
                if b >= self.frame:
                    b = 0
                    a = a + 1

        # shape of array = (number_of_2d_array, frame, nodes=25) 
        return data_X, data_Y, data_ACC, number_of_2d_array
    
"""
if __name__ == '__main__':    
    a = JasonDecoder('sd')
    X, Y, ACC, number = a.decoding()
"""
        
