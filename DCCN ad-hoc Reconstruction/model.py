import os
import math
import string
import sys
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, Lambda, UpSampling2D
#from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalMaxPool2D
#from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import keras.backend as k
import keras.callbacks as cbks
from keras.callbacks import CSVLogger
from model_module import *
#from .stft import *
from complex_layers.layer import *
from complex_layers.activations import *
from complex_layers.normalization import *

def network(input_train_tm_tens,input_train_tp_tens, dfat_train_t_tens, te_train_t_tens):

 kernel_size = 2
 norm_trainig = False 

 real, imag, conv_real1, conv_imag1 = encoder_module(input_train_tm_tens, input_train_tp_tens, 12, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 real, imag                         = complex_MaxPooling()(real,imag)

 real, imag, conv_real2, conv_imag2 = encoder_module(real, imag, 24, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 real, imag                         = complex_MaxPooling()(real,imag)

 real, imag, conv_real3, conv_imag3 = encoder_module(real, imag, 48, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 real, imag                         = complex_MaxPooling()(real,imag)

 real, imag, conv_real4, conv_imag4 = encoder_module(real, imag, 96, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 real, imag                         = complex_MaxPooling()(real,imag)

 real, imag, conv_real5, conv_imag5 = encoder_module(real, imag, 192, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 real, imag                         = complex_MaxPooling()(real,imag)

 real, imag, conv_real6, conv_imag6 = encoder_module(real, imag, 384, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 real, imag                         = complex_UpSampling()(real,imag)
 
 #_____________DECODER STARTS__________________
 center_real1, center_imag1 = decoder_module(real, imag, conv_real5, conv_imag5, 192, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 center_real1, center_imag1 = complex_UpSampling()(center_real1,center_imag1)

 deconv_real1, deconv_imag1 = decoder_module(center_real1, center_imag1, conv_real4, conv_imag4, 96, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 deconv_real1, deconv_imag1 = complex_UpSampling()(deconv_real1,deconv_imag1)

 deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real3, conv_imag3, 48, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 deconv_real1, deconv_imag1 = complex_UpSampling()(deconv_real1,deconv_imag1)

 deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real2, conv_imag2, 24, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 deconv_real1, deconv_imag1 = complex_UpSampling()(deconv_real1,deconv_imag1)

 deconv_real1, deconv_imag1 = decoder_module(deconv_real1, deconv_imag1, conv_real1, conv_imag1, 12, (kernel_size, kernel_size), (1, 1), training = norm_trainig)
 #deconv_real1, deconv_imag1 = complex_UpSampling()(deconv_real1,deconv_imag1)

 deconv_real1, deconv_imag1 = complex_Conv2D_Linear(filters = 4, kernel_size = (1, 1), strides = (1,1))(deconv_real1, deconv_imag1)

 
 #output_pred = concatenate([deconv_real1,deconv_imag1])

 model = Model(inputs=[input_train_tm_tens,input_train_tp_tens, dfat_train_t_tens, te_train_t_tens], outputs=[deconv_real1, deconv_imag1])

 return model, deconv_real1, deconv_imag1 

