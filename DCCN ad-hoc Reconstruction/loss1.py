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
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalMaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import keras.backend as k
import keras.callbacks as cbks
from keras.callbacks import CSVLogger

def cart2pol(x, y):
    rho = tf.math.sqrt(tf.math.square(x) + tf.math.square(y))
    phi = tf.math.atan2(y, x, name=None)
    return(rho, phi)

def loss1(input_train_tm_tens,input_train_tp_tens,dfat_train_t_tens,te_train_t_tens,w_mean_r,w_std_r,f_mean_r,f_std_r,w_mean_i,w_std_i,f_mean_i,f_std_i,frq_mean,frq_std,r2_mean,r2_std,real,imag):

    pi = tf.constant(math.pi)

    num_echo = 3

    wat_r = real[:,:,:,0]
    watt_r = k.expand_dims(wat_r,3)
    watt_r = k.repeat_elements(watt_r,num_echo,3)
    
    fat_r = real[:,:,:,1]
    fatt_r = k.expand_dims(fat_r,3)
    fatt_r = k.repeat_elements(fatt_r,num_echo,3)

    frq_r = real[:,:,:,2]
    frqt_r = k.expand_dims(frq_r,3)
    frqt_r = k.repeat_elements(frqt_r,num_echo,3)

    r2_r = real[:,:,:,3]
    r2t_r = k.expand_dims(r2_r,3)
    r2t_r = k.repeat_elements(r2t_r,num_echo,3)

    wat_i = imag[:,:,:,0]
    watt_i = k.expand_dims(wat_i,3)
    watt_i = k.repeat_elements(watt_i,num_echo,3)

    fat_i = imag[:,:,:,1]
    fatt_i = k.expand_dims(fat_i,3)
    fatt_i = k.repeat_elements(fatt_i,num_echo,3)
 
    frq_i = imag[:,:,:,2]
    frqt_i = k.expand_dims(frq_i,3)
    frqt_i = k.repeat_elements(frqt_i,num_echo,3)
 
    r2_i = imag[:,:,:,3]
    r2t_i = k.expand_dims(r2_i,3)
    r2t_i = k.repeat_elements(r2t_i,num_echo,3)

    watt_r, watt_i = cart2pol(watt_r, watt_i)
    fatt_r, fatt_i = cart2pol(fatt_r, fatt_i)
    #frqt_r, frqt_i = cart2pol(frqt_r, frqt_i)
    #r2t_r, r2t_i = cart2pol(r2t_r, r2t_i)

    watt_c =  tf.cast(tf.complex(watt_r,0*watt_r), tf.complex64)
    fatt_c =  tf.cast(tf.complex(fatt_r,0*fatt_r), tf.complex64)
    
    watt_ci =  tf.cast(tf.complex(watt_i,0*watt_i), tf.complex64)
    fatt_ci =  tf.cast(tf.complex(fatt_i,0*fatt_i), tf.complex64)

    #r2t_c=  tf.cast(tf.complex(tf.abs(tf.complex(r2t_r,r2t_i)),0*r2t_i), tf.complex64)
    r2t_c=  tf.cast(tf.complex(r2t_r,r2t_i), tf.complex64)
    frqt_c = tf.cast(tf.complex(frqt_r,frqt_i),tf.complex64)
    
    dfat_train_t_c = tf.cast(tf.complex(dfat_train_t_tens,0*dfat_train_t_tens),tf.complex64)
    te_train_t_c = tf.cast(tf.complex(te_train_t_tens,0*te_train_t_tens),tf.complex64)

    pi_cmp =  tf.cast(tf.complex(pi,0*pi), tf.complex64)
  
  
    signal = tf.cast((( (watt_c)*tf.exp(1j*watt_ci) + tf.exp(1j*fatt_ci)*(fatt_c)*tf.exp(-1j*2*pi_cmp*dfat_train_t_c*te_train_t_c))*tf.exp(-1*r2t_c*te_train_t_c)*tf.exp(-1j*2*pi_cmp*frqt_c*te_train_t_c)),tf.complex64)


    input_train_t_mag = input_train_tm_tens[:,:,:,0:num_echo]
    input_train_t_phs = input_train_tp_tens[:,:,:,0:num_echo]

    gt_input_train2 = tf.complex(input_train_t_mag,input_train_t_phs)

    #fatt_img = tf.abs(tf.complex(fat_r,fat_i))
    #watt_img = tf.abs(tf.complex(wat_r, wat_i))
    #field_img = tf.abs(tf.complex(frq_r, frq_i))
    #R2_img = tf.abs(tf.complex(r2_r,r2_i))

    #TV1 = tf.reduce_sum(tf.image.total_variation(field_img))
    #TV2 = tf.reduce_sum(tf.image.total_variation(watt_img))
    #TV3 = tf.reduce_sum(tf.image.total_variation(fatt_img)) 
    #TV4 = tf.reduce_sum(tf.image.total_variation(R2_img)) 
    
    loss = tf.sqrt(tf.linalg.norm(tf.abs(gt_input_train2-signal)))  #+ ((0.000001)*(TV1))
    
    return loss


