'''
author: ramin jafari (rj259@cornell.edu)
https://doi.org/10.1002/mrm.28546
'''
#!/usr/bin/env python3
#!/usr/bin/env python2

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
#from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import keras.backend as k
import keras.callbacks as cbks
#from keras.backend.tensorflow_backend import set_session
#from keras.backend.tensorflow_backend import clear_session
#from keras.backend.tensorflow_backend import get_session
from keras.callbacks import CSVLogger


def loss2(input_train_tm_tens,input_train_tp_tens,output_train_t_tens,dfat_train_t_tens,te_train_t_tens,output_pred):

    loss = tf.sqrt(tf.linalg.norm(tf.abs(output_train_t_tens-output_pred)))

    return loss



