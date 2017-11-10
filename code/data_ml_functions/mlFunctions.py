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
from keras.applications import VGG16,imagenet_utils, InceptionResNetV2
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

def get_cnn_model(params):   
    """
    Load base CNN model and add metadata fusion layers if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :return model: CNN model with or without depending on params
    """
    
    input_tensor = Input(shape=(params.target_img_size[0],params.target_img_size[1],params.num_channels))
    baseModel = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor, pooling='avg')
    trainable = True
    for layer in baseModel.layers:
        if layer.name == 'conv2d_158':
            trainable = True
        layer.trainable = trainable
    #baseModel.summary()
    modelStruct = baseModel.output
    #modelStruct = Flatten(input_shape=baseModel.output_shape[1:])(modelStruct)

    if params.use_metadata:
        auxiliary_input = Input(shape=(params.metadata_length,), name='aux_input')
        auxiliary_input_norm = Reshape((1,1,-1))(auxiliary_input)
        auxiliary_input_norm = InstanceNormalization(axis=3, name='ins_norm_aux_input')(auxiliary_input_norm)
        auxiliary_input_norm = Flatten()(auxiliary_input_norm)

        modelStruct = concatenate([modelStruct,auxiliary_input_norm])

    modelStruct = Dense(params.cnn_last_layer_length//4, activation='relu', name='fc1')(modelStruct)
    modelStruct = Dropout(0.2)(modelStruct)
    modelStruct = Dense(params.cnn_last_layer_length//8, activation='relu', name='fc2')(modelStruct)
    modelStruct = Dropout(0.1)(modelStruct)
    predictions = Dense(params.num_labels, activation='softmax')(modelStruct)

    if not params.use_metadata:
        model = Model(inputs=[baseModel.input], outputs=predictions)
    else:
        model = Model(inputs=[baseModel.input, auxiliary_input], outputs=predictions)

    #for i,layer in enumerate(model.layers):
    #    layer.trainable = True

    return model

def get_lstm_model(params, codesStats):
    """
    Load LSTM model and add metadata concatenation to input if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :param codesStats: dictionary containing CNN codes statistics, which are used to normalize the inputs
    :return model: LSTM model
    """

    model = Sequential()
    if params.use_metadata:
        layerLength = params.cnn_last_layer_length + params.metadata_length
    else:
        layerLength = params.cnn_last_layer_length
    model.add(LSTM(layerLength, return_sequences=True, input_shape=(codesStats['max_temporal'], layerLength), dropout=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params.num_labels, activation='softmax'))
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

        batchInds = get_batch_inds(params.batch_size_cnn, idx, N)

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
    imgdata = np.zeros((params.batch_size_cnn,params.target_img_size[0],params.target_img_size[1],params.num_channels))
    metadata = np.zeros((params.batch_size_cnn,params.metadata_length))
    labels = np.zeros(params.batch_size_cnn)
    inputs = []
    results = []
    for i in range(0,len(batchData)):
        currInput = {}
        currInput['data'] = batchData[i]
        currInput['metadataStats'] = metadataStats
        #results.append(_load_batch_helper(currInput))
        task = partial(_load_batch_helper, currInput)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i,result in enumerate(results):
        metadata[i,:] = result['metadata']
        imgdata[i,:,:,:] = result['img']
        labels[i] = result['labels']
        
    #imgdata = imagenet_utils.preprocess_input(imgdata)
    #imgdata = imgdata / 255.0
    
    labels = to_categorical(labels, params.num_labels)

    
    return imgdata,metadata,labels

def _load_batch_helper(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    data = inputDict['data']
    metadataStats = inputDict['metadataStats']
    #metadata = np.divide(json.load(open(data['features_path'])) - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
    metadata = json.load(open(data['features_path']))
    img = scipy.misc.imread(data['img_path'])

    if np.random.random() < 0.5:
        img = flip_axis(img, 1)

    if np.random.random() < 0.5:
        img = flip_axis(img, 0)

    def rect_coords(img_shape, sx, sy):
        x0 = (img_shape[1] - sx)/2
        x1 = x0 + sx
        y0 = (img_shape[0] - sy)/ 2
        y1 = y0 + sy
        return np.array([x0, x1, x1, x0]), np.array([y0 ,y0, y1,y1])

                                            
    def rotate(a, angle, img_shape):
        center = np.array([img_shape[1], img_shape[0]]) / 2.
        theta = (angle/180.) * np.pi
        rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                              [np.sin(theta),  np.cos(theta)]])
        return np.dot(a - center, rotMatrix) + center

    def enclosing_rect(edges):
        x0 = np.amin(edges[:,0])
        x1 = np.amax(edges[:,0])
        y0 = np.amin(edges[:,1])
        y1 = np.amax(edges[:,1])
        return int(x0),int(y0),int(math.ceil(x1)),int(math.ceil(y1)) # np.array(([x0,y0], [x1,y0], [x1,y1], [x0,y1]))

    angle=np.random.randint(360)

    sx,sy = metadata[15:17]
    x_side, y_side = sx/2, sy/2
    max_side = np.sqrt((x_side ** 2 ) + (y_side **2)) * 1.4142135624
    scaling = img.shape[0] / (max_side*2)

    edges = np.squeeze(np.dstack(rect_coords(img.shape, sx*scaling, sy*scaling)))
    rot_points = rotate(edges, angle, img.shape)

    img = scipy.ndimage.interpolation.rotate(img, angle=angle, reshape=False, mode='constant')
    x0,y0,x1,y1 = enclosing_rect(rot_points)
    img = img[y0:y1,x0:x1,...]
    img = cv2.resize(img, (299,299)).astype(np.float32)#params.target_img_size)
    img = imagenet_utils.preprocess_input(img, mode='tf')

#    img = imagenet_utils.preprocess_input(img) / 255.
    
    labels = data['category']
    currOutput = {}
    currOutput['img'] = img
    currOutput['metadata'] = metadata
    currOutput['labels'] = labels
    return currOutput

def codes_metadata_generator(params, data, metadataStats, codesStats):
    """
    Custom generator that yields a vector containign the 4096-d CNN codes output by VGG16 and metadata features (if params set to use).
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to CNN codes and images 
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (codesMetadata,labels): 4096-d CNN codes + metadata features (if set), and labels (categorical form) 
    """
    
    N = len(data)

    idx = np.random.permutation(N)

    batchInds = get_batch_inds(params.batch_size_lstm, idx, N)
    trainKeys = list(data.keys())
    
    while True:
        for inds in batchInds:
            batchKeys = [trainKeys[ind] for ind in inds]
            codesMetadata,labels = load_lstm_batch(params, data, batchKeys, metadataStats, codesStats)
            yield(codesMetadata,labels)
        
def load_lstm_batch(params, data, batchKeys, metadataStats, codesStats):
    """
    Load batch of CNN codes + metadata and preprocess the data before returning.
    :param params: global parameters, used to find location of the dataset and json file
    :param data: dictionary where the values are the paths to the files containing the CNN codes and metadata for a particular sequence
    :param batchKeys: list of keys for the current batch, where each key represents a temporal sequence of CNN codes and metadata
    :param metadataStats: metadata stats used to normalize metadata features
    :param codesStats: CNN codes stats used to normalize CNN codes and define the maximum number of temporal views
    :return codesMetadata,labels: 4096-d CNN codes + metadata (if set) and labels (categorical form)
    """
    
    if params.use_metadata:
        codesMetadata = np.zeros((params.batch_size_lstm, codesStats['max_temporal'], params.cnn_last_layer_length+params.metadata_length))
    else:
        codesMetadata = np.zeros((params.batch_size_lstm, codesStats['max_temporal'], params.cnn_last_layer_length))
        
    labels = np.zeros(params.batch_size_lstm)
    for i,key in enumerate(batchKeys):
        currData = data[key]
        labels[i] = currData['category']
        if params.use_metadata:
            inds = []
            for file in currData['metadata_paths']:
                underscores = [ind for ind,ltr in enumerate(file) if ltr == '_']
                inds.append(int(file[underscores[-3]+1:underscores[-2]]))
            inds = np.argsort(np.array(inds)).tolist()
        else:
            inds = range(len(currData['cnn_codes_paths']))
            
        for codesIndex in inds:
            cnnCodes = json.load(open(currData['cnn_codes_paths'][codesIndex])) - np.array(codesStats['codes_mean'])
            if params.use_metadata:
                metadata = np.divide(json.load(open(currData['metadata_paths'][codesIndex])) - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
                codesMetadata[i,codesIndex,0:params.metadata_length] = metadata
                codesMetadata[i,codesIndex,params.metadata_length:] = cnnCodes
            else:
                codesMetadata[i,codesIndex,:] = cnnCodes
    
    labels = to_categorical(labels, params.num_labels)

    
    return codesMetadata,labels

	
