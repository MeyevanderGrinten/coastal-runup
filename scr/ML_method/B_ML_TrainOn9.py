# -*- coding: utf-8 -*-
"""
Script Name: B_ML_trainOn9.py
Author: Meye van der Grinten
Date: [2024-04-15]

Description:
    This script trains the ml model on 9 labeled timestack images to later test 
    it on new unseen data in C_ML_TestTrainedOn9.py.

Application:
    * The output model generated by this script is **not** ready for direct use
    on new beaches. It requires further refinement and training on 
    **larger, more diverse datasets** to improve its generalization 
    and performance across different environments.
    * Before applying this model to new locations, consider additional training
    and validation to ensure reliable results.
    * However, the results indicate that a relatively shallow, multi-channel 
    CNN network, informed by preprocessed images, shows promise in differentiating
    between water and sand and, ultimately, in extracting runup at a given beach 
    in Galveston, Texas.


Requirements:
    - numpy  
    - matplotlib  
    - os  
    - OpenCV (cv2)  
    - skimage  
    - scipy  
"""

import numpy as np
import os


# Import functions from A_functions.py file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from A_Functions import create_model, segment_image,\
    prepare, preprocess_channels



## Choose channels -------------------------------------------------------------
# Best combination based on highest mean acc
Channel_inputs = ['Grayscale', 'Grayscale_overSpace', 'S', 'Entropy', 'Entropy_overTime']


#%% TRAINING TIME
stacks_dict = {}
stacks_dict['SW_20231113_1346_GX010081.npy'] = [2300,7000]
stacks_dict['SW_20231113_1346_GX020081.npy'] = [1800,6500]
stacks_dict['SW_20231113_1346_GX030081.npy'] = [1500,6000]
stacks_dict['SW_20231114_0850_GX010084.npy'] = [4500,10000]
stacks_dict['SW_20231114_0850_GX020084.npy'] = [5000,10000]
stacks_dict['SW_20231114_0850_GX030084.npy'] = [4500,9500]
stacks_dict['SW_20231114_0850_GX040084.npy'] = [3500,8500]
stacks_dict['SW_20231114_0850_GX050084.npy'] = [2000,7500]
stacks_dict['SW_20231114_0850_GX060084.npy'] = [2500,7000]

# Make model for the correct number of input channels 
model = create_model(len(Channel_inputs))

for stack_name in stacks_dict:
    
    bounds = stacks_dict.get(stack_name)
    
    beach = stack_name[:2]

    # Load data
    timestack = np.load(os.path.join('Timestacks', stack_name))
    
    # Load labels
    shoreline = np.load(os.path.join('Runup_Manual', stack_name), allow_pickle=True)
    
    reduced_timestack, labels_ = prepare(timestack, shoreline, bounds)
    #%% This takes a while 
    data_train = preprocess_channels(Channel_inputs, reduced_timestack)
    X_patches_train = segment_image(data_train)
    Y_patches_train = segment_image(labels_)
    
    
    # Fit data to model
    model.fit(X_patches_train, Y_patches_train, batch_size=32, epochs=10, verbose=1)

model.save('Results/Model_trainedon9.h5')

