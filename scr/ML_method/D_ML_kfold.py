# -*- coding: utf-8 -*-
"""
Script Name: D_ML_kfold.py
Author: Meye van der Grinten
Date: [2024-04-15]

Description:
    This script performs k-fold cross-validation using 9 timestack images. 
    Each of the 9 timestacks is used as a validation set once, while the remaining
    images are used for training.The runup line is predicted and plotted for each fold.
    The results of the k-fold (accuracy, loss) are stored in "Results/..."

Application:
    * When applying multiple channels or only the Grayscale channel the added 
    value of multiple channels becomes clear.


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
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.backend import clear_session


# Import functions from A_functions.py file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from A_Functions import segment_image, create_model,\
    reconstruct_from_patches, prepare, preprocess_channels



# Best based on highest mean acc
Channel_inputs = ['Grayscale', 'Grayscale_overSpace', 'S', 'Entropy', 'Entropy_overTime']
# Changing to a single Grayscale input shows that multiple channels enhance model performance
#Channel_inputs = ['Grayscale']

#%% TRAINING TIME
# Make dictionary of the timestack used for the k-fold (all with manually labeled data)
# Betwee brackets: the upper and lower pixel bounds of the timestacks,
# so that only the swash is considered.
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


n_input = len(Channel_inputs)


acc_per_fold = []
loss_per_fold = []
fold_no = 1


path = 'Results/' + 'kfold_'.join(Channel_inputs) + '.txt'

# Open a file to store the results
with open(path, 'w') as file:
    file.write("Fold,Validation Timestack,Accuracy,Loss\n")

    for stack_val in stacks_dict:
        # Prepare validation data ---------------------------------------------
        
        
        bounds_ = stacks_dict.get(stack_val)
        # Load data
        timestack = np.load(os.path.join('Timestacks', stack_val))
        
        # Load labels
        shoreline = np.load(os.path.join('Runup_Manual', stack_val), allow_pickle=True)
        
        reduced_timestack_, labels_ = prepare(timestack, shoreline, bounds_)
        data_val = preprocess_channels(Channel_inputs, reduced_timestack_)

        X_patches_val = segment_image(data_val)
        Y_patches_val = segment_image(labels_)
        
        reconstruct_dim = reduced_timestack_.shape[:-1]
        
        # Create a new dictionary excluding the current stack_name
        remaining_stacks = {k: v for k, v in stacks_dict.items() if k != stack_val}
        
        # Prepare training data
        X_train_list = []
        Y_train_list = []
        
        # Prepare training data -----------------------------------------------
        for stack_train in remaining_stacks:
            bounds = remaining_stacks.get(stack_train)
            
            # Load data
            timestack = np.load(os.path.join('Timestacks', stack_train))
            
            # Load labels
            shoreline = np.load(os.path.join('Runup_Manual', stack_train), allow_pickle=True)
            
            reduced_timestack, labels_ = prepare(timestack,shoreline, bounds)
            data_train = preprocess_channels(Channel_inputs, reduced_timestack)
            X_patches_train = segment_image(data_train)
            Y_patches_train = segment_image(labels_)
            X_train_list.append(X_patches_train)
            Y_train_list.append(Y_patches_train)
        
        X_train = np.concatenate(X_train_list, axis=0)
        Y_train = np.concatenate(Y_train_list, axis=0)
        
        # Create model
        model = create_model(n_input)
        
        # Fit data to model
        model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
        
        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_patches_val, Y_patches_val)
        
        # Add results of fold to list and write to .txt file
        acc_per_fold.append(val_accuracy * 100)
        loss_per_fold.append(val_loss)
        file.write(f"{fold_no},{stack_val},{val_accuracy * 100},{val_loss}\n")
        print(f"Fold {fold_no} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        
        # Predict on validation split
        predictions = model.predict(X_patches_val)
        predicted_masks = predictions > 0.5
        reconstructed_prediction = reconstruct_from_patches(predicted_masks, reconstruct_dim)
        
                
        # Compute runup estimate from prediction
        mask = (reconstructed_prediction == 0)
        lowest_indices = np.argmax(mask, axis=0)
        runup =  np.sum(reconstructed_prediction, axis=0)

        # Get start time and make xticks
        start_times = pd.read_csv('Time/Time_SW.txt', delimiter=',', names=['files', 'time', 't_end'])
        time = start_times[start_times['files'] == stack_val]['time'].iloc[0]
        start_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        timestamps = [start_time + timedelta(seconds=0.5*i) for i in range(reduced_timestack.shape[1])]
        x_ticks_indices = np.arange(0, len(timestamps), 1000)
        x_ticks = [timestamps[i].strftime('%H:%M:%S') for i in x_ticks_indices]
        
        # Plot resutls
        plt.figure(figsize=(18,4))
        plt.imshow(reduced_timestack_)
        plt.plot(runup, label='Runup by $ML_8$ method')
        plt.xticks(x_ticks_indices, x_ticks)
        plt.xlabel('Time [hh:mm:ss]')
        plt.ylabel('Distance [pix] (1 pixel = 0.075 m)')
        plt.title(f'Video ID = {stack_val[-12:-4]} - validation k-fold')
        plt.legend()
        plt.xlim(500,3000)
        plt.savefig(f'Images/k-fold/{stack_val[-12:-4]}_k_fold.png')
        
        fold_no += 1
        
        # Clear the session to free up resources
        clear_session()

# Print overall results
print("Average accuracy per fold: ", np.mean(acc_per_fold))
print("Average loss per fold: ", np.mean(loss_per_fold))