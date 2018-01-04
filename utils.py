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

import PIL
from PIL import Image

from vgg16bn_ted import Vgg16BN

from keras.utils import get_file, to_categorical
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D 
from keras.layers import Flatten, Dropout, Input, Lambda, Add, Concatenate, Activation
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
    

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))
    

def onehot(x, num_classes=None):
    return to_categorical(x, num_classes=num_classes)


def ceil(x):
    return int(math.ceil(x))


def floor(x):
    return int(math.floor(x))


def save_array(fname, arr):
    # create an on-disk carray container
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
    
def load_array(fname):
    return bcolz.open(rootdir=fname)
    

class MixIterator(object):
    
    def __init__(self, iters):
        self.iters = iters
        self.n = int(np.sum([itr.n for itr in self.iters]))
        self.batch_size = int(np.sum([itr.batch_size for itr in self.iters]))
        self.steps_per_epoch = max([ceil(itr.n/itr.batch_size) for itr in self.iters])
    
    def reset(self):
        for itr in self.iters: itr.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = [next(itr) for itr in self.iters]
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)


class PseudoLabelGenerator(object):
    
    def __init__(self, iterator, model):
        self.iter = iterator
        self.n = self.iter.n
        self.batch_size = self.iter.batch_size
        self.steps_per_epoch = ceil(self.iter.n/self.iter.batch_size)
        self.model = model
        self.class_indices = self.iter.class_indices
        #self.classes = np.argmax(self.model.predict_generator(self.iter,steps=self.steps_per_epoch), axis=1)
    
    def reset(self):
        self.iter.reset()
    
    def __iter__(self):
        return self
    
    def __next__(self, *args, **kwargs):
        nexts = next(self.iter)
        results = self.model.predict(nexts[0], batch_size=self.batch_size)
        return (nexts[0], results)

