# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:26:01 2023

@author: Admin
"""

import numpy as np
import pickle

#Loading the model

load_model=pickle.load(open('C:/Users/Admin/Desktop/P265/Final Model/deployment model/trained_model.sav','rb'))

input_data=(4,33,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0)

#Changing the input data into numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we want predict one term
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=load_model.predict(input_data_reshaped)
# print(prediction)
print("The co2 emission is {}".format(prediction))