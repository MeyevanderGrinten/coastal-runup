# -*- coding: utf-8 -*-
"""
Script Name: A_Functions.py
Author: Meye van der Grinten
Date: [2024-04-15]

Description:
    This script contains all base functions for B_ML_TrainOn9.py, C_ML_TestTrainedOn9.py,
    and D_ML_kfold.py.
    It contains six functions:
        1. create_model() to creates a convolutional autoencoder model using
        Keras Sequential API.
        2. segment_image() to segment an image into non-overlapping patches
        of a given size.
        3. reconstruct_from_patches() to reconstruct an image from the patches.
        4. prepare() to prepare a reduced timestack and corresponding binary 
        labels for segmentation for computational efficiency.
        5. prepareNOLABELS() to prepares a reduced timestack by cropping and
        downsampling the image for computational efficiency.
        6. preprocess_channels() to preprocesses the selected channels from the
        reduced timestack by applying various transformations.


Requirements:
    - numpy: For numerical computations and array manipulations.
    - os: For file and directory handling.
    - OpenCV (cv2): For image processing and transformations.
    - scikit-image (skimage): For additional image processing functions, such as entropy calculation.
    - scipy: For scientific computing and additional image processing utilities.
    - TensorFlow (tensorflow.keras): For building and training deep learning models.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk



#%%

def create_model(n_input=1):
    '''
    Creates and returns a convolutional autoencoder model using Keras Sequential API.
    
    Parameters:
        n_input (int): Number of input channels. Default is 1.
        
    Returns:
        model (tf.keras.Model): A compiled Keras model.
    '''
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=(60, 60, n_input)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2DTranspose(64, (3, 3), strides=(2, 2),
                        padding='same', activation='relu'),
        Conv2DTranspose(32, (3, 3), strides=(2, 2),
                        padding='same', activation='relu'),
        Conv2D(1, (1, 1), activation='sigmoid', padding='same'),
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


def segment_image(image, patch_size=(60, 60)):
    '''
    Segments an image into non-overlapping patches of a given size.

    Parameters:
        image (numpy.ndarray): The input image, either grayscale (2D) or multi-channel (3D).
        patch_size (tuple): The size of each patch (height, width). Default is (60, 60).

    Returns:
        numpy.ndarray: An array of image patches with shape (num_patches, patch_height, patch_width, num_channels).

    Functionality:
    - Determines if the image is single-channel or multi-channel based on its dimensions.
    - Extracts non-overlapping patches from the image, ensuring only full patches are included.
    - If the image is single channel (2D), patches are reshaped to include a channel dimension for consistency.

    Raises:
        ValueError: If the input image does not have 2 or 3 dimensions.
    '''
    
    # Determine if the image is grayscale or multi-channel by checking its dimension
    if image.ndim == 2:
        img_height, img_width = image.shape
        num_channels = 1  # It's a grayscale image
    elif image.ndim == 3:
        img_height, img_width, num_channels = image.shape
    else:
        raise ValueError("Image must be either 2 or 3 dimensions")

    patches = []

    # Loop through the image to extract patches
    for i in range(0, img_height, patch_size[0]):
        for j in range(0, img_width, patch_size[1]):
            # Check to ensure only full patches are taken
            if (i + patch_size[0] <= img_height) and (j + patch_size[1] <= img_width):
                if num_channels == 1:
                    patch = image[i:i + patch_size[0], j:j + patch_size[1]]
                    patches.append(patch.reshape(*patch_size, 1))  # Add channel dimension for consistency
                else:
                    patch = image[i:i + patch_size[0], j:j + patch_size[1], :]
                    patches.append(patch)
    
    return np.array(patches)


def reconstruct_from_patches(patches, original_dims, patch_size=(60, 60)):
    """
    Reconstructs an image from its patches.

    Parameters:
    - patches: The array of patches.
    - original_dims: The dimensions of the original image (height, width).
    - patch_size: The size of each patch (height, width).

    Returns:
    - A numpy array representing the reconstructed image.
    """
    reconstructed_image = np.zeros(original_dims)
    patch_idx = 0
    for i in range(0, original_dims[0], patch_size[0]):
        for j in range(0, original_dims[1], patch_size[1]):
            if i + patch_size[0] <= original_dims[0] and j + patch_size[1] <= original_dims[1]:
                reconstructed_image[i:i + patch_size[0], j:j +
                                    patch_size[1]] = patches[patch_idx].reshape(patch_size)
                patch_idx += 1
    return reconstructed_image


def prepare(timestack, shoreline, bounds):
    '''
    Prepares a reduced timestack and corresponding binary labels for segmentation.
    This because the extracted timestacks from the video footage had a way to high 
    resoltuion in space. Might certainly not me the case for your images...

    Parameters:
        timestack (numpy.ndarray): The input timestack image with shape (H, W, C).
        shoreline (numpy.ndarray): Array containing shoreline data. The fourth column (index 3) represents the runup.
        bounds (tuple): A tuple (lower_bound, upper_bound) defining the region of interest in the timestack.

    Returns:
        reduced_timestack (numpy.ndarray): A downsampled version of the selected timestack window.
        labels_ (numpy.ndarray): A binary label mask indicating water (0) or sand (1).

    '''
    
    # Take upper and lower bounds as defined in handpick
    timestack_window = timestack[bounds[0]:bounds[1], :]

    # Reshape to by averaging over n values in space for computational costs
    n = 10  # Reduction factor for the first axis
    height = timestack_window.shape[0] // n
    width = timestack_window.shape[1]
    reshaped = timestack_window.reshape(height, n, width, timestack_window.shape[2])
    reduced_timestack = np.mean(reshaped, axis=1).astype(np.uint8)
    
    # LOAD Labels:
    runup = shoreline[:, 3] - bounds[0]
    labels = np.ones((timestack_window.shape[0], timestack_window.shape[1]))
    # Create a mask where True indicates row indices in labels above the row indices in runup
    mask = np.arange(labels.shape[0])[:, np.newaxis] > runup
    # Set values in labels to 0 where the mask is True
    labels[mask] = 0
    labels_ = labels[::10]
    
    return reduced_timestack, labels_

def prepareNOLABELS(timestack, bounds):
    '''
    Prepares a reduced timestack by cropping and downsampling the image for computational efficiency.
    This because the extracted timestacks from the video footage had a way to high 
    resoltuion in space. Might certainly not me the case for your images...

    Parameters:
        timestack (numpy.ndarray): The input timestack image with shape (H, W, C).
        bounds (tuple): A tuple (lower_bound, upper_bound) defining the region of interest in the timestack.

    Returns:
        reduced_timestack (numpy.ndarray): A downsampled version of the selected timestack window.
    '''

    # Take upper and lower bounds as defined in handpick
    timestack_window = timestack[bounds[0]:bounds[1], :]

    # Reshape to by averaging over n values in space for computational costs
    n = 10  # Reduction factor for the first axis
    height = timestack_window.shape[0] // n
    width = timestack_window.shape[1]
    reshaped = timestack_window.reshape(height, n, width, timestack_window.shape[2])
    reduced_timestack = np.mean(reshaped, axis=1).astype(np.uint8)

    return reduced_timestack


def preprocess_channels(selected_channels, reduced_timestack):
    '''
    Preprocesses the selected channels from the reduced timestack by applying various transformations 
    such as grayscale conversion, gradient computation, entropy calculation, and color space conversions.

    Parameters:
        selected_channels (list of str): A list of channel names to process. Options include:
            - 'Grayscale': Converts the image to grayscale.
            - 'RGB': Uses the original RGB image.
            - 'Grayscale_overTime': Computes the temporal gradient of the grayscale image.
            - 'Grayscale_overSpace': Computes the spatial gradient of the grayscale image.
            - 'Entropy': Computes the entropy of the grayscale image.
            - 'Entropy_overTime': Computes the temporal gradient of the entropy image.
            - 'S': Extracts the saturation (S) channel from the HSV color space.
        reduced_timestack (numpy.ndarray): The input image array, expected to be a 3D RGB image.

    Returns:
        numpy.ndarray: A concatenated array of processed channels with shape (H, W, C), where C is 
        the number of selected channels.
    
    Notes:
    - If no valid channels are selected, returns an empty array with shape (H, W, 0).
    - Data normalization is applied to ensure values are scaled between 0 and 1.
    - Gaussian blur is used before computing gradients and entropy to reduce noise.
        If your are interested in why some are blured in time and/or space we 
        refer to the paper.
    '''
    processed_channels = []

    for channel in selected_channels:
        if channel == 'Grayscale':
            data = reduced_timestack
            data = cv.cvtColor(data, cv.COLOR_RGB2GRAY)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'RGB':
            data = reduced_timestack  # Already 3D, do not add new axis
            
        elif channel == 'Grayscale_overTime':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = np.gradient(cv.cvtColor(data, cv.COLOR_RGB2GRAY), axis=1)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'Grayscale_overSpace':
            data = cv.GaussianBlur(reduced_timestack, (3, 1), 0)
            data = np.gradient(cv.cvtColor(data, cv.COLOR_RGB2GRAY), axis=0)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'Entropy':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = entropy(cv.cvtColor(data, cv.COLOR_RGB2GRAY), disk(3))
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'Entropy_overTime':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            entropy_data = entropy(cv.cvtColor(data, cv.COLOR_RGB2GRAY), disk(3))
            data = np.gradient(entropy_data, axis=1)
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D
            
        elif channel == 'S':
            data = cv.GaussianBlur(reduced_timestack, (3, 3), 0)
            data = cv.cvtColor(data, cv.COLOR_BGR2HSV)[:, :, 1]
            data = data[:, :, np.newaxis]  # Add a new axis to make it 3D

        # Normalization of all channel to avoid scale dommination
        data_min = np.min(data)
        data_max = np.max(data)
        data = (data - data_min) / (data_max - data_min + 1e-6)
        

        processed_channels.append(data)
    
    if not processed_channels:
        return np.empty((reduced_timestack.shape[0], reduced_timestack.shape[1], 0))  # Return an empty array with the right shape but zero channels

    # Stack all processed channels along the last axis, handling RGB case appropriately
    return np.concatenate(processed_channels, axis=-1)

