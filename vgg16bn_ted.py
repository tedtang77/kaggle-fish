import os, json, math
from glob import glob

import numpy as np

from keras.utils import get_file
from keras.models import Model, Sequential
from keras.layers import BatchNormalization, Input, Add, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D
from keras.regularizers import l2
from keras.optimizers import Adam



from keras import backend as K
K.set_image_data_format('channels_first')
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)



class Vgg16BN():
    """
        The VGG 16 Imagenet model with Batch Normalization for the Dense Layers
    """
    
    def __init__(self, size=(224,224), include_top=True):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.VGG_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
        self.VGG_DROPOUT = 0.5
        self._dropout = self.VGG_DROPOUT
        self._create(size, include_top)
        self._get_classes()
        
    
    def _vgg_preprocess(self, x):
        """
            Subtracts the mean RGB value, and transposes RGB to BGR.
            The mean RGB was computed on the image set used to train the VGG model
            (VGG-16 and VGG-19 were trained using Caffe, and Caffe uses OpenCV to load images which uses BGR by default, so both VGG models             are expecting BGR images.)
        
            Args:
                x: Image array (height x width x channels)
            Returns:
                Image array (height x width x transposed_channels)
        """
        x = x - self.VGG_MEAN
        return x[:,::-1] # reverse axis RGB into BGR
    
    
    def _get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it's not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        
        
    def conv_block(self, x, layers, filters):
        for i in range(layers):
            x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D((2, 2), strides=(2,2))(x)
        return x
        
        
    def fc_block(self, x):
        x = Dense(4096, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self._dropout)(x)
        return x
        
        
    def _create(self, size, include_top):
        input_shape = (3,)+size
        img_input = Input(shape=input_shape)
        
        x = Lambda(self._vgg_preprocess)(img_input)
        
        x = self.conv_block(x, 2, 64)
        x = self.conv_block(x, 2, 128)
        x = self.conv_block(x, 3, 256)
        x = self.conv_block(x, 3, 512)
        x = self.conv_block(x, 3, 512)
        
        if not include_top:
            fname = 'vgg16_bn_conv.h5'
        else: 
            x = Flatten()(x)
            x = self.fc_block(x)
            x = self.fc_block(x)
            x = Dense(1000, activation='softmax', name='softmax')(x)
            fname = 'vgg16_bn.h5'
        
        self.model = Model(inputs=img_input, outputs=x, name='vgg16')
        self.model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        
    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (Fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn 
            weights for the last layer in subsequent training.
            
            Args:
                num: Number of neurons of the last layer
        """
        
        model = self.model
        model.layers.pop()
        for layer in model.layers: layer.trainable=False
        sm = Dense(num, activation='softmax')(model.layers[-1].output)
        self.model = Model(model.input, sm)
        self.model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    def finetune(self, batches):
        """
            Modifies the original VGG16BN network architecture and update self.classes for new training data
            
            Args:
                batches : a keras.preprocessing.image.ImageDataGenerator object.
                          See definition of get_batches()
        """
        
        self.ft(batches.num_classes)
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}
        classes = list(iter(batches.class_indices))  # Get a list of all the class labels    
        # sort the class labels by index according to batches.class_indices and update model.classes
        for c in classes:
            classes[batches.class_indices[c]] = c
        self.classes = classes
        
    
    def fit(self, batches, val_batches, epochs=1, verbose=2):
        self.model.fit_generator(batches, epochs=epochs, verbose=verbose,
                                 steps_per_epoch=ceil(batches.n/batches.batch_size),
                                 validation_data=val_batches,
                                 validation_steps=ceil(val_batches.n/val_batches.batch_size))
        
    
    def test(self, path, batch_size=8):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch
            
            See Keras documentation: https://keras.io/models/model/#predict_generator
            
            Args:
                path (string) :  Path to the target directory. It should contain 
                                one subdirectory per class.
                batch_size (int) : The number of images to be considered in each batch.
            
            Returns:
                test_batches 
                numpy array(s) of predictions for the test batches.
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, steps=ceil(test_batches.n/test_batches.batch_size))
    
    
    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = np.array([all_preds[i][idxs[i]] for i in range(len(idxs))])
        classes = [self.classes[idx] for idx in idxs]
        return preds, idxs, classes
    
    