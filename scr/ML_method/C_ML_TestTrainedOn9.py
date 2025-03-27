# -*- coding: utf-8 -*-
"""
Script Name: C_ML_TestTrainedOn9.py
Author: Meye van der Grinten
Date: [2024-04-15]

Description:
    This script processes timestack images to calculate the runup position over time. 
    It utilizes a model trained on 9 timestack images for this purpose.

Application:
    * It shows how the model can extrapolate quite well to new data.
    * It also shows how the model struggles with some artifacts such as 
    low gradients in the backwash of singular waves or the nightfall. 
    
Usage:
    python Runup_EntropySaturation.py [input_file] [window_size]

Requirements:
    - numpy 
    - matplotlib 
    - os 
    - OpenCV (cv2) 
    - skimage
    - scipy 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model


# Import functions from A_functions.py file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from A_Functions import segment_image,\
    reconstruct_from_patches, preprocess_channels, prepareNOLABELS




stacks_dict = {}
stacks_dict['SW_20231116_0920_GX060085.npy'] = [3500,7500]
stacks_dict['SW_20231116_0920_GX010085.npy'] = [6000,12000]



Channel_inputs = ['Grayscale', 'Grayscale_overSpace', 'S', 'Entropy', 'Entropy_overTime']


model = load_model('Results/Model_trainedon9.h5')

for stack_name in stacks_dict:
    

    bounds = stacks_dict.get(stack_name)
    
    # Load and prepare timestack image
    timestack = np.load(os.path.join('Timestacks', stack_name))
    reduced_timestack = prepareNOLABELS(timestack, bounds)
    
    # Make a multi dimensional input array and segemnt into patches of 60x60xn
    data = preprocess_channels(Channel_inputs, reduced_timestack)
    X_patches = segment_image(data)
    
    
    # Binarise predictions 
    predictions = model.predict(X_patches)
    predicted_masks = predictions > 0.5
    reconstructed_prediction = reconstruct_from_patches(predicted_masks, reduced_timestack.shape[:-1])
    
    # Compute 1D runup array from 2D water/sand timestack prediction
    # The sum over each column is taken, counting all sand pixels.
    # Alternative approaches, such as smoothing, could yield a more refined runup signal.
    # However, these methods may obscure the true performance of the machine learning model.
    runup_ml = np.sum(reconstructed_prediction, axis=0)
    
    
    
        

    
    if stack_name == 'SW_20231116_0920_GX060085.npy':
        
        
        plt.figure(figsize=(18,4))
        plt.imshow(reduced_timestack)
        plt.plot(runup_ml, label='Runup by $ML_9$ method')
        plt.xlabel('Time [pixels] (1 pixel = 0.5 s)')
        plt.ylabel('Distance [pix] (1 pixel = 0.075 m)')
        plt.title(f'Video ID = {stack_name[-12:-4]} - Very similar timestack to the training data, leading to good performance.')
        plt.legend()
        plt.xlim(2000,4000)
        plt.savefig(f'Images/TestTrainedOn9/{stack_name[-12:-4]}_GoodPerformance')
        
        plt.figure(figsize=(18,4))
        plt.imshow(reduced_timestack)
        plt.plot(runup_ml, label='Runup by $ML_9$ method')
        plt.xlabel('Time [pixels] (1 pixel = 0.5 s)')
        plt.ylabel('Distance [pix] (1 pixel = 0.075 m)')
        plt.title(f'Video ID = {stack_name[-12:-4]} – model performance significantly deteriorates as night falls.')
        plt.legend()
        plt.xlim(7600,9600)
        plt.savefig(f'Images/TestTrainedOn9/{stack_name[-12:-4]}_NightFall')
        
    if stack_name == 'SW_20231116_0920_GX010085.npy':
        
        plt.figure(figsize=(18,4))
        plt.imshow(reduced_timestack)
        plt.plot(runup_ml, label='Runup by $ML_9$ method')
        plt.xlabel('Time [pixels] (1 pixel = 0.5 s)')
        plt.ylabel('Distance [pix] (1 pixel = 0.075 m)')
        plt.title(f'Video ID = {stack_name[-12:-4]} - Video taken a low tide hence little turbulence (white foam) in the swash ')
        plt.legend()
        plt.xlim(2500,6500)
        plt.savefig(f'Images/TestTrainedOn9/{stack_name[-12:-4]}_BadPerformance')
    
    
    

