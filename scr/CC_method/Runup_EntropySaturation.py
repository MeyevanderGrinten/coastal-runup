# -*- coding: utf-8 -*-
"""
Script Name: Runup_entropySaturation.py
Author: Meye van der Grinten
Date: [2024-04-15]

Description:
    This script processes timestack images to compute the runup position over time.
    It contains two functions:
        1. The Shoreline_EntropySaturation() function which computes runup for a 
        given input timestack,
        2. The compute_runup_over_window() function which computes the runup over 
        time using a sliding window approach over time using the 
        Shoreline_EntropySaturation() function, and
        3. And example of how to use the function that requires the Timestack.npy 
        file and saves a plot of the timestack together the computed runup line.

Application:
    * Please be aware that this function is designed to compute runup for 
      dissipative conditions with a clear effluent line (and thus a seepage face) 
      and sufficient turbulence in the swash zone. If this is not the case, 
      please refer to the tip at line 140.
    * Note that the `window_size` parameter of the `compute_runup_over_window()` 
      function has been tailored to a specific scenario (dt = 0.5 s, sun exposure, etc.) 
      and may need adjustment to improve performance.
    * Other parameters, such as the disk size of the entropy function and the 
      kernel of the Gaussian blur (`gaussian_filter1d()`), may also benefit from 
      tuning if spatial or temporal dimensions differ significantly from the 
      presented case.

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
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d


def compute_runup_over_window(timestack, window_size):
    """
    Computes the runup position over time using a sliding window approach.
    Please notice the window_size parameter has been taylored to a specific 
    scenario (dt=0.5 s, sun exposure, etc...) and might need adjustment to 
    ameliorate performance.

    Parameters:
    -----------
    timestack : np.ndarray
            Timestack with shape (nt, nx, 3) with 
            nt the time dimension,  
            nx the spacial pixel dimension, and
            3 the RGB channels

    window_size : int
        The number of time steps in each window for runup computation with 
        Shoreline_EntropySaturation().

    Returns:
    --------
    runup : np.ndarray
        A 1D array of runup positions over time, with the a length of nt 
        (first dimension of input timestack array).
    """
    
    num_rows, total_time = timestack.shape[:2]
    
    # Initialize a 1D array to store the runup results
    runup = np.zeros(total_time)
    
    # Perform sliding window runup computation
    for start in range(0, total_time, window_size):
        end = min(start + window_size, total_time)  # Ensure it does not exceed total_time

        # Compute runup for the current window
        current_window = timestack[:, start:end]
        runup_window, _ = Shoreline_EntropySaturation(current_window)  # Assuming this function exists
        
        # Store the computed runup in the corresponding positions
        runup[start:end] = runup_window  # Assign values to the runup array

    return runup



def Shoreline_EntropySaturation(stack):
    '''
    Function computes the wet/dry boudnary and runup in pixel coordinates for 
    timestack images of dissipative beaches that have a clear effluent line and
    suffucuent turbulence in the swash zone.

    Parameters
    ----------
    stack : array
            Timestack with shape (nt, nx, 3) with 
            nt the time dimension,  
            nx the spacial pixel dimension, and
            3 the RGB channels

    Returns
    -------
    runup : array
        Instatanious waterline with dimension nt
        (first dimension of input stack array)
    WD : array
        Instantanious wet/dry boundary with dimension nt
        (first dimension of input stack array)
        
    '''
    
    # 1 -- ENTROPY: -----------------------------------------------------------
    # Grascale image 
    grayscale_image = cv.cvtColor(stack, cv.COLOR_RGB2GRAY)
    # Gradient over time
    gradient_columns = np.gradient(grayscale_image, axis=1).astype(np.uint8)
    #gradient_columns = normalize_columns(gradient_columns)
    # Entropy
    entropy_img = entropy(gradient_columns, disk(5))
    # Entropy threshold binarization --> need for clear bimodality
    thresh_entropy = threshold_otsu(entropy_img)
    binary_entropy = entropy_img >= thresh_entropy

    
    # 2 -- SATURATION:  -------------------------------------------------------
    I_RmB_mFoam = cv.cvtColor(stack, cv.COLOR_BGR2HSV)[:,:,1]
    # Blur image over time axis
    I_RmB_blur = cv.GaussianBlur(I_RmB_mFoam,(201,1),0)
    # Binarize wet from dry 
    thresh = threshold_otsu(I_RmB_blur)
    binary_wd = np.where((I_RmB_blur <= thresh), 0, 1)
    # Compute boundary
    WD = np.sum(binary_wd, axis=0)
    
    
    # 3 -- DELETE NOISE ABOVE EFFLUENT LINE -----------------------------------
    # Tip: If your beach does not have a well-defined seepage face, this step 
    # may do more harm than good. Consider modifying the function to retain only 
    # the results from the entropy analysis by skipping the next two lines and 
    # using `binary_entropy_mWD = binary_entropy`. 
    # This approach could yield better results for more reflective beach conditions!
    binary_entropy_mWD = np.copy(binary_entropy)
    binary_entropy_mWD[binary_wd == 1] = 0
    
    # Compute runup line
    runup = stack.shape[0] - np.sum(binary_entropy_mWD, axis=0)
    # Smooth runup
    runup = gaussian_filter1d(runup, 2)


    return runup, WD
    
    
#%%
if __name__ == "__main__":
    
    """
    This section provides an example of how to use the compute_runup_over_window()
    function. Please be aware you need the Timestack.npy located in your script 
    directory or adapt to your wishes. 
    """
    
    # Change working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    

    # Load timestack
    timestack = np.load('Timestack.npy')

    # Plot entire timestack to choose appropriate window
    plt.figure(figsize=(12,4))
    plt.imshow(timestack)

    # Compute runup
    runup = compute_runup_over_window(timestack, 500)
    
    # Plot resutls
    plt.figure(figsize=(12,4))
    plt.plot(runup, label='Runup by CC method')
    plt.imshow(timestack)
    plt.xlabel('Time [pixels] (1 pixel = 0.5 s)')
    plt.ylabel('Distance [pix] (1 pixel = 0.075 m)')
    plt.legend()
    plt.savefig('Timestack_Runup_CC.png')

    
 