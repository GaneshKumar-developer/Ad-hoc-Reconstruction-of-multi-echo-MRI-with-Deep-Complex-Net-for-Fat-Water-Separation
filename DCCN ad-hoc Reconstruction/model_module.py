import scipy
import scipy.signal
import scipy.io.wavfile
import librosa
from librosa.display import *

import os, ssl
import natsort
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.utils import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import*
from tensorflow.keras.activations import *
from tensorflow.keras.initializers import *
from tensorflow.python.client import device_lib

from complex_layers.layer import *
from complex_layers.activations import *
from complex_layers.normalization import *


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
tranposed_STFT  : transpoed Spectogram, ex) [None, 64, 512] --> [None, 512, 64, 1]
transpoed_ISTFT : transpoed and squeeze Spectogram (For Inverse Short Time Fourier Transform) [None 512 64 1] --> [None 64 512]
mask_processing : outputs of complex Unet would be multipled with complex ratio mask (modified)

complex_layers/
    activation.py
        Cleaky_ReLU
    networks.py
        complex_Conv2D
        complex_Conv2DTranspose
    normalization.py
        complex_NaiveBatchNormalization
        complex_BatchNormalization2d
    STFT.py
        STFT_layer
        ISTFT_layer
    
    networks.py, normalization.py, STFT.py All class module (Not activation.py)
    So, We create custom function module using class complex layers...
    But, Because inputs of complex_Batchnoramlization has to be combined, [real, imag] (concat) ==> inputs
    Make a seperate function module (complex BatchNomalization)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def tranposed_STFT (real, imag):
    real = tf.transpose(real, perm = (0, 2, 1))
    imag = tf.transpose(imag, perm = (0, 2, 1))

    real = tf.reshape(real, (-1, 512, 64, 1))
    imag = tf.reshape(imag, (-1, 512, 64, 1))
    
    return real, imag


def transpoed_ISTFT (real, imag):
    real = tf.transpose(real, (0, 2, 1, 3))
    imag = tf.transpose(imag, (0, 2, 1, 3))

    real = tf.squeeze(real, axis = 3)
    imag = tf.squeeze(imag, axis = 3)
    
    return real, imag
    

def mask_processing (real, imag, stft_real, stft_imag):
    magnitude = tf.tanh(tf.sqrt(tf.square(real) + tf.square(imag)))
    unit_real = tf.divide(real, tf.sqrt(tf.square(real) + tf.square(imag)))
    unit_imag = tf.divide(imag, tf.sqrt(tf.square(real) + tf.square(imag)))

    mask_real = tf.multiply(magnitude, unit_real)
    mask_imag = tf.multiply(magnitude, unit_imag)

    enhancement_real = stft_real * mask_real - stft_imag * mask_imag
    enhancement_imag = stft_real * mask_imag + stft_imag * mask_real
    
    return enhancement_real, enhancement_imag


def complex_BatchNormalization2d (real, imag, training = None):
    inputs  = tf.concat([real, imag], axis = -1)
    outputs = complex_BatchNorm2d()(inputs, training = training)

    input_dim = outputs.shape[-1] // 2
    real = outputs[ :, :, :, :input_dim]
    imag = outputs[ :, :, :, input_dim:]

    return real, imag


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'with Naive complex_BatchNormalization module'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def encoder_module (real, imag, filters, kernel_size, strides, training = True):
    conv_real, conv_imag = complex_Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    #out_real, out_imag   = CLeaky_ReLU(conv_real, conv_imag)
    out_real, out_imag   = complex_NaiveBatchNormalization()(conv_real, conv_imag, training = True)
    
    return out_real, out_imag, conv_real, conv_imag


def decoder_module (real, imag, concat_real, concat_imag, filters, kernel_size, strides, training = True):
    
    deconv_real, deconv_imag = complex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    
    if concat_real == None and concat_imag == None:
        pass
    else:
        deconv_real = concatenate([deconv_real, concat_real], axis = 3)
        deconv_imag = concatenate([deconv_imag, concat_imag], axis = 3)

    deconv_real, deconv_imag = complex_Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(deconv_real, deconv_imag)
    #deconv_real, deconv_imag = CLeaky_ReLU(deconv_real, deconv_imag)
    deconv_real, deconv_imag = complex_NaiveBatchNormalization()(deconv_real, deconv_imag, training = True)
    
    
        
    return deconv_real, deconv_imag


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'with Naive complex_BatchNormalization module'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def convariance_encoder_module (real, imag, filters, kernel_size, strides, training = True):
    conv_real, conv_imag = complex_Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    out_real, out_imag   = CLeaky_ReLU(conv_real, conv_imag)
    out_real, out_imag   = complex_BatchNormalization2d(conv_real, conv_imag, training = True)
    
    return out_real, out_imag, conv_real, conv_imag


def convariance_decoder_module (real, imag, concat_real, concat_imag, filters, kernel_size, strides, training = True):
    if concat_real == None and concat_imag == None:
        pass
    else:
        real = concatenate([real, concat_real], axis = 3)
        imag = concatenate([imag, concat_imag], axis = 3)
    deconv_real, deconv_imag = complex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    deconv_real, deconv_imag = CLeaky_ReLU(deconv_real, deconv_imag)
    deconv_real, deconv_imag = complex_BatchNormalization2d(deconv_real, deconv_imag, training = True)
    
    return deconv_real, deconv_imag