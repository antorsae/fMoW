"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


__author__ = 'jhuapl'
__version__ = 0.1

import json
from keras.applications import VGG16,imagenet_utils, InceptionResNetV2, Xception
from keras.layers import Dense,Input,Flatten,Dropout,LSTM, concatenate, Reshape
from keras.models import Sequential,Model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from keras_contrib.layers.normalization import InstanceNormalization, BatchRenormalization

import numpy as np

from data_ml_functions.dataFunctions import get_batch_inds, flip_axis
from glob import glob
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from collections import defaultdict
import copy
import random
import cv2
import scipy
import math

from data_ml_functions.iterm import show_image

from data_ml_functions.keras_squeeze_excite_network.se_inception_resnet_v2 import SEInceptionResNetV2

from data_ml_functions.DenseNet import densenet

def get_cnn_model(params):   
    """
    Load base CNN model and add metadata fusion layers if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :return model: CNN model with or without depending on params
    """
    
    input_tensor = Input(shape=(params.target_img_size, params.target_img_size, params.num_channels))

    classifier = globals()[params.classifier]
    if params.classifier == 'densenet':
        baseModel = densenet.DenseNetImageNet161(
            input_shape=(params.target_img_size, params.target_img_size, params.num_channels), include_top=False, input_tensor=input_tensor)
    else:
        baseModel = classifier(weights='imagenet', include_top=False, input_tensor=input_tensor, pooling='avg')
            
    trainable = False
    n_trainable = 0
    for i, layer in enumerate(baseModel.layers):
        if i >= params.freeze:
            trainable = True
            n_trainable += 1
        layer.trainable = trainable

    print("Base CNN model has " + str(n_trainable) + "/" + str(len(baseModel.layers)) + " trainable layers")

    modelStruct = baseModel.output

    if params.use_metadata:
        auxiliary_input = Input(shape=(params.metadata_length,), name='aux_input')
        auxiliary_input_norm = Reshape((1,1,-1))(auxiliary_input)
        auxiliary_input_norm = InstanceNormalization(axis=3, name='ins_norm_aux_input')(auxiliary_input_norm)
        auxiliary_input_norm = Flatten()(auxiliary_input_norm) if params.classifier != 'densenet' else auxiliary_input

        modelStruct = concatenate([modelStruct, auxiliary_input_norm])

        modelStruct = Dense(4096, activation='relu', name='fc1')(modelStruct)
        modelStruct = Dropout(0.2)(modelStruct)
        modelStruct = Dense(512, activation='relu', name='nfc2')(modelStruct)
        modelStruct = Dropout(0.1)(modelStruct)

    predictions = Dense(params.num_labels, activation='softmax', name='predictions')(modelStruct)

    if not params.use_metadata:
        model = Model(inputs=[baseModel.input], outputs=predictions)
    else:
        model = Model(inputs=[baseModel.input, auxiliary_input], outputs=predictions)

    return model

def img_metadata_generator(params, data, metadataStats):
    """
    Custom generator that yields images or (image,metadata) batches and their 
    category labels (categorical format). 
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to images and metadata features 
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (imgdata,labels) or (imgdata,metadata,labels): image data, metadata (if params set to use), and labels (categorical form) 
    """
    
    N = len(data)

    data_labels = [datum['category'] for datum in data]
    label_to_idx = defaultdict(list)
    for i, label in enumerate(data_labels):
        label_to_idx[label].append(i)

    executor = ThreadPoolExecutor(max_workers=params.num_workers)

    running_label_to_idx = copy.deepcopy(label_to_idx)

    while True:
        
        # class-aware supersampling
        idx = []
        num_labels = len(label_to_idx)
        for _ in range(N):
            random_label = np.random.randint(num_labels)
            if len(running_label_to_idx[random_label]) == 0:
                running_label_to_idx[random_label] = copy.copy(label_to_idx[random_label])
                random.shuffle(running_label_to_idx[random_label])
            idx.append(running_label_to_idx[random_label].pop())

        batchInds = get_batch_inds(params.batch_size, idx, N)

        for inds in batchInds:
            batchData = [data[ind] for ind in inds]
            imgdata,metadata,labels = load_cnn_batch(params, batchData, metadataStats, executor)
            if params.use_metadata:
                yield([imgdata,metadata],labels)
            else:
                yield(imgdata,labels)
        
def load_cnn_batch(params, batchData, metadataStats, executor):
    """
    Load batch of images and metadata and preprocess the data before returning.
    :param params: global parameters, used to find location of the dataset and json file
    :param batchData: list of objects in the current batch containing the category labels and paths to CNN codes and images 
    :param metadataStats: metadata stats used to normalize metadata features
    :return imgdata,metadata,labels: numpy arrays containing the image data, metadata, and labels (categorical form)
    """
    futures = []
    imgdata = np.zeros((params.batch_size,params.target_img_size,params.target_img_size,params.num_channels))
    metadata = np.zeros((params.batch_size,params.metadata_length))
    labels = np.zeros(params.batch_size)
    inputs = []
    results = []
    for i in range(0,len(batchData)):
        currInput = {}
        currInput['data'] = batchData[i]
        currInput['metadataStats'] = metadataStats
        currInput['target_img_size'] = params.target_img_size
        currInput['angle'] = params.angle
        currInput['flips'] = params.flips
        task = partial(_load_batch_helper, currInput)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i,result in enumerate(results):
        metadata[i,:] = result['metadata']
        imgdata[i, ...] = result['img']
        labels[i] = result['labels']
        
    #imgdata = imagenet_utils.preprocess_input(imgdata)
    #imgdata = imgdata / 255.0
    
    labels = to_categorical(labels, params.num_labels)

    return imgdata,metadata,labels

# counter-clockwise rotation
def rotate(a, angle, img_shape):
    center = np.array([img_shape[1], img_shape[0]]) / 2.
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta),  np.cos(theta)]])
    return np.dot(a - center, rotMatrix) + center

def transform_metadata(metadata, flip_h, flip_v, angle=0):
    metadata_angles = np.fmod(180. + np.array(metadata[19:27]) * 360., 360.) - 180.

    metadata_angles += angle

    if flip_h:
        metadata_angles = 180. - metadata_angles
    if flip_v:
        metadata_angles = 180. - metadata_angles

    metadata[19:27] = list(np.fmod(metadata_angles + 2*360., 360.) / 360.)

    assert all([i <= 1. and i >=0. for i in metadata[19:27]])
    return metadata

def _load_batch_helper(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    data = inputDict['data']
    metadataStats = inputDict['metadataStats']
    metadata = json.load(open(data['features_path']))

    img = scipy.misc.imread(data['img_path'])

    angle= (np.random.random() - 0.5 ) * inputDict['angle']

    target_img_size = inputDict['target_img_size']

    if angle != 0.:
        patch_size = img.shape[0]
        patch_center = patch_size / 2
        sq2 = 1.4142135624 

        src_points = np.float32([
            [ patch_center - patch_size / (2 * sq2) , patch_center - patch_size / (2 * sq2) ], 
            [ patch_center + patch_size / (2 * sq2) , patch_center - patch_size / (2 * sq2) ], 
            [ patch_center + patch_size / (2 * sq2) , patch_center + patch_size / (2 * sq2) ]])

        src_points = rotate(src_points, angle, img.shape).astype(np.float32)

        dst_points = np.float32([
            [ 0 , 0 ], 
            [ target_img_size - 1, 0 ], 
            [ target_img_size - 1, target_img_size - 1]]) 

        M   = cv2.getAffineTransform(src_points, dst_points)
        img = cv2.warpAffine(img, M, (target_img_size, target_img_size), borderMode = cv2.BORDER_REFLECT_101).astype(np.float32)

    else:
        crop_size = target_img_size
        x0 = int(img.shape[1]/2 - crop_size/2)
        x1 = x0 + crop_size
        y0 = int(img.shape[0]/2 - crop_size/2)
        y1 = y0 + crop_size

        img = img[y0:y1, x0:x1,...].astype(np.float32)

    flip_h = flip_v = False

    if inputDict['flips']:

        flip_h = (np.random.random() < 0.5)
        flip_v = (np.random.random() < 0.5)

        if flip_h:
            img = flip_axis(img, 1) # flips > into < 

        if flip_v:
            img = flip_axis(img, 0) # flips ^ into v 


    #show_image(img.astype(np.uint8))
    #raw_input("Press enter")
    metadata = transform_metadata(metadata, flip_h=flip_h, flip_v=flip_v, angle=angle)

    # TODO : FIX for multiple models
    #img = densenet.preprocess_input(img)
    img = imagenet_utils.preprocess_input(img) / 255.
    #img = imagenet_utils.preprocess_input(img, mode='tf')
#    img = imagenet_utils.preprocess_input(img) / 255. # this is for vgg, etc.

    labels = data['category']
    currOutput = {}
    currOutput['img'] = img
    metadata = np.divide(json.load(open(data['features_path'])) - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
    currOutput['metadata'] = metadata
    currOutput['labels'] = labels
    return currOutput

	
