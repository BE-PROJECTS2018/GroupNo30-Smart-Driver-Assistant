# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:43:14 2018

@author: Aditya
"""
import pickle

target = "camera_matrix.pkl"
if os.path.getsize(target) > 0:
    cammat = pickle.load(open( "./camera_matrix.pkl", "rb" ))
    print(cammat)
else:
    print("File is empty")

cammat['imagesize'] = (640,360)

cammat['imagesize'] = (1280,720)

cammat['imagesize'] = (1200,720)


with open("./camera_matrix_1.pkl", "rb") as f:
    cammat = pickle.load(f)
    print(cammat)

with open("./camera_matrix_1.pkl", "wb") as fp:
    pickle.dump(cammat, fp)
    

