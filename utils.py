from glob import glob
import os, sys, json, math
import numpy as np
from numpy.random import permutation, random, choice
from shutil import copyfile
import bcolz

from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage import *
from scipy.misc import *

from matplotlib import pyplot as plt

import pandas as pd

from vgg16bn_ted import Vgg16BN

from keras.utils import get_file, to_categorical
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D 
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)
K.set_image_data_format('channels_first')

from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import categorical_crossentropy as crossentropy

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=64, 
                target_size=(224,224), class_mode='categorical'):
    return gen.flow_from_directory(path, target_size=target_size, 
                                   class_mode=class_mode,  batch_size=batch_size,
                                  shuffle=shuffle)

def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=1)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=1)
    
    return batches.classes, val_batches.classes, onehot(batches.classes), onehot(val_batches.classes), batches.filenames, val_batches.filenames, test_batches.filenames


def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, target_size=target_size, class_mode=None)
    return np.concatenate([batches.next() for i in range(batches.n)])


def vgg_ft_bn(out_dim):
    vgg = Vgg16BN()
    vgg.ft(out_dim)
    model = vgg.model
    return model


def split_at(model, layer_type):    
    idxs = [idx for idx, layer in enumerate(model.layers) if type(layer) is layer_type]
    last_idx = idxs[-1]
    return model.layers[:last_idx+1], model.layers[last_idx+1:]
    


def onehot(x):
    return to_categorical(x)


def ceil(x):
    return int(math.ceil(x))


def floor(x):
    return int(math.floor(x))


def save_array(fname, arr):
    # create an on-disk carray container
    c = bcolz.carray(arr, rootdir=fname)
    c.flush()
    
    
def load_array(fname):
    return bcolz.open(rootdir=fname)
    

    