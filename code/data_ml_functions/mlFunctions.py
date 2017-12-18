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
from keras.applications import VGG16, imagenet_utils, InceptionResNetV2, Xception, ResNet50, NASNetMobile, NASNetLarge
from keras.layers import Dense,Input,Flatten,Dropout,LSTM, GRU, concatenate, Reshape, Conv2D, MaxPooling2D, ConvLSTM2D, Activation
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

#from data_ml_functions.keras_squeeze_excite_network.se_inception_resnet_v2 import SEInceptionResNetV2
#from data_ml_functions.DenseNet import densenet

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

    modelStruct = baseModel.layers[-1].output

    if params.use_metadata:
        auxiliary_input = Input(shape=(params.metadata_length,), name='aux_input')
        auxiliary_input_norm = Reshape((1,1,-1))(auxiliary_input)
        auxiliary_input_norm = InstanceNormalization(axis=3, name='ins_norm_aux_input')(auxiliary_input_norm)
        auxiliary_input_norm = Flatten()(auxiliary_input_norm) #if params.classifier != 'densenet' else auxiliary_input

        modelStruct = concatenate([modelStruct, auxiliary_input_norm if params.norm_metadata else auxiliary_input])

        modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc1')(modelStruct)
        modelStruct = Dropout(0.2)(modelStruct)
        modelStruct = Dense(512, activation='relu', name='nfc2')(modelStruct)
        modelStruct = Dropout(0.1)(modelStruct)

    #modelStruct = Flatten()(modelStruct)
    predictions = Dense(params.num_labels, activation='softmax', name='predictions')(modelStruct)

    if not params.use_metadata:
        model = Model(inputs=[baseModel.input], outputs=predictions)
    else:
        model = Model(inputs=[baseModel.input, auxiliary_input], outputs=predictions)

    return model

def get_multi_model(params, codesStats):
    """
    Load LSTM model and add metadata concatenation to input if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :param codesStats: dictionary containing CNN codes statistics, which are used to normalize the inputs
    :return model: LSTM model
    """

    if params.use_metadata:
        layerLength = params.cnn_multi_layer_length + params.metadata_length
    else:
        layerLength = params.cnn_multi_layer_length
        layerShape  = params.cnn_multi_layer_shape

    print(codesStats['max_temporal'], layerLength)
    model = Sequential()
    arch = params.classifier

    if arch == 'lstm':
        model.add(InstanceNormalization(axis=2, input_shape=(codesStats['max_temporal'], layerLength)))
        model.add(LSTM(256, return_sequences=True, input_shape=(codesStats['max_temporal'], layerLength), dropout=0.5))
        model.add(LSTM(256, return_sequences=True, dropout=0.5))
        model.add(Flatten())
        model.add(Dense(params.num_labels, activation='softmax'))
    elif arch == 'lstm2':
        model.add(InstanceNormalization(axis=2, input_shape=(codesStats['max_temporal'], layerLength)))
        model.add(LSTM(256, return_sequences=True, input_shape=(codesStats['max_temporal'], layerLength)))
        model.add(LSTM(params.num_labels, return_sequences=False, dropout=0.5))
        model.add(Activation(activation='softmax'))
#        model.add(Dense(params.num_labels, activation='softmax'))
    elif arch == 'lstm3':
        model.add(InstanceNormalization(axis=2, input_shape=(codesStats['max_temporal'], layershape[0], layershape[1], layershape[2] )))
        #model.add(Reshape(target_shape=( codesStats['max_temporal'], 1, 1, layerLength)))
        model.add(ConvLSTM2D(128, 3, return_sequences=True,  dropout=0.5))
        model.add(ConvLSTM2D(params.num_labels,  3, return_sequences=False, dropout=0.5))
        model.add(Flatten())
        model.add(Activation(activation='softmax'))
        #model.add(Dense(params.num_labels, activation='softmax'))
    elif arch == 'gru':
        model.add(GRU(128, return_sequences=True, input_shape=(codesStats['max_temporal'], layerLength), dropout=0.5))
        model.add(GRU(128, return_sequences=True, input_shape=(codesStats['max_temporal'], layerLength), dropout=0.5))
        model.add(GRU(params.num_labels, activation='softmax', return_sequences=False))
    elif arch == 'pnet':
        model.add(Reshape(target_shape=(codesStats['max_temporal'], layerLength, 1), input_shape=(codesStats['max_temporal'], layerLength)))
        model.add(Conv2D(filters=1024,   kernel_size=(1, 1), activation='relu'))
        model.add(Conv2D(filters=2048,   kernel_size=(1, 1), activation='relu'))
        model.add(Conv2D(filters=4096,   kernel_size=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(codesStats['max_temporal'], 1), padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(params.num_labels, activation='softmax'))
    return model

def img_metadata_generator(params, data, metadataStats, class_aware_sampling = True, augmentation = True):
    """
    Custom generator that yields images or (image,metadata) batches and their 
    category labels (categorical format). 
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to images and metadata features 
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (imgdata,labels) or (imgdata,metadata,labels): image data, metadata (if params set to use), and labels (categorical form) 
    """
    
    N = len(data)

    if class_aware_sampling:
        data_labels = [datum['category'] for datum in data]
        label_to_idx = defaultdict(list)
        for i, label in enumerate(data_labels):
            label_to_idx[label].append(i)
        running_label_to_idx = copy.deepcopy(label_to_idx)

    executor = ThreadPoolExecutor(max_workers=params.num_workers)

    while True:
        
        if class_aware_sampling:
            # class-aware supersampling
            idx = []
            num_labels = len(label_to_idx)
            assert num_labels == params.num_labels
            for _ in range(N):
                random_label = np.random.randint(num_labels)
                if len(running_label_to_idx[random_label]) == 0:
                    running_label_to_idx[random_label] = copy.copy(label_to_idx[random_label])
                    random.shuffle(running_label_to_idx[random_label])
                idx.append(running_label_to_idx[random_label].pop())
        else:
            idx = np.random.permutation(N)

        batchInds = get_batch_inds(params.batch_size, idx, N)

        for inds in batchInds:
            batchData = [data[ind] for ind in inds]
            imgdata,metadata,labels = load_cnn_batch(params, batchData, metadataStats, executor, augmentation)
            if params.use_metadata:
                yield([imgdata,metadata],labels)
            else:
                yield(imgdata,labels)
        
def load_cnn_batch(params, batchData, metadataStats, executor, augmentation):
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
        currInput['flip_north_south'] = params.flip_north_south
        currInput['flip_east_west'] = params.flip_east_west
        currInput['mask_metadata'] = params.mask_metadata
        task = partial(_load_batch_helper, currInput, augmentation)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i,result in enumerate(results):
        metadata[i,:] = result['metadata']
        imgdata[i, ...] = result['img']
        labels[i] = result['labels']
    
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

    # b/c angles are clockwise we add the clockwise rotation angle
    metadata_angles += angle

    if flip_h: # > <
        metadata_angles =      - metadata_angles
    if flip_v: # v ^
        metadata_angles = 180. - metadata_angles

    metadata[19:27] = list(np.fmod(metadata_angles + 2*360., 360.) / 360.)

    assert all([i <= 1. and i >=0. for i in metadata[19:27]])
    return metadata

def mask_metadata(metadata):
    '''
        features[0] = float(jsonData['gsd'])
    x,y = utm_to_xy(jsonData['utm'])
    features[1] = x
    features[2] = y
    features[3] = float(jsonData['cloud_cover']) / 100.0
    date = dparser.parse(jsonData['timestamp'])
    features[4] = float(date.year)
    features[5] = float(date.month) / 12.0
    features[6] = float(date.day) / 31.0
    features[7] = float(date.hour) + float(date.minute)/60.0

    if jsonData['scan_direction'].lower() == 'forward':
        features[8] = 0.0
    else:
        features[8] = 1.0
    features[9] = float(jsonData['pan_resolution_dbl'])
    features[10] = float(jsonData['pan_resolution_start_dbl'])
    features[11] = float(jsonData['pan_resolution_end_dbl'])
    features[12] = float(jsonData['pan_resolution_min_dbl'])
    features[13] = float(jsonData['pan_resolution_max_dbl'])
    features[14] = float(jsonData['multi_resolution_dbl'])
    features[15] = float(jsonData['multi_resolution_min_dbl'])
    features[16] = float(jsonData['multi_resolution_max_dbl'])
    features[17] = float(jsonData['multi_resolution_start_dbl'])
    features[18] = float(jsonData['multi_resolution_end_dbl'])
    features[19] = float(jsonData['target_azimuth_dbl']) / 360.0
    features[20] = float(jsonData['target_azimuth_min_dbl']) / 360.0
    features[21] = float(jsonData['target_azimuth_max_dbl']) / 360.0
    features[22] = float(jsonData['target_azimuth_start_dbl']) / 360.0
    features[23] = float(jsonData['target_azimuth_end_dbl']) / 360.0
    features[24] = float(jsonData['sun_azimuth_dbl']) / 360.0
    features[25] = float(jsonData['sun_azimuth_min_dbl']) / 360.0
    features[26] = float(jsonData['sun_azimuth_max_dbl']) / 360.0
    features[27] = float(jsonData['sun_elevation_min_dbl']) / 90.0
    features[28] = float(jsonData['sun_elevation_dbl']) / 90.0
    features[29] = float(jsonData['sun_elevation_max_dbl']) / 90.0
    features[30] = float(jsonData['off_nadir_angle_dbl']) / 90.0
    features[31] = float(jsonData['off_nadir_angle_min_dbl']) / 90.0
    features[32] = float(jsonData['off_nadir_angle_max_dbl']) / 90.0
    features[33] = float(jsonData['off_nadir_angle_start_dbl']) / 90.0
    features[34] = float(jsonData['off_nadir_angle_end_dbl']) / 90.0
    features[35] = float(bb['box'][2])
    features[36] = float(bb['box'][3])
    features[37] = float(jsonData['img_width'])
    features[38] = float(jsonData['img_height'])
    features[39] = float(date.weekday())
    features[40] = min([features[35], features[36]]) / max([features[37], features[38]])
    features[41] = features[35] / features[37]
    features[42] = features[36] / features[38]
    features[43] = date.second
    if len(jsonData['bounding_boxes']) == 1:
        features[44] = 1.0
    else:
        features[44] = 0.0
    '''
    masked_attributes = [ \
        6,  # day
        7,  # hour/min
        35, # bbox loc
        36, # bbox loc
        37, # img_width
        38, # img_height
        39, # weekday
        43, # second
        44, # 1 bbox or more
        ]

    for to_mask in masked_attributes:
        metadata[to_mask] = 0.

    return metadata

def _load_batch_helper(inputDict, augmentation):
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

    if angle != 0. and augmentation:
        patch_size = img.shape[0]
        patch_center = patch_size / 2
        sq2 = 1.4142135624 

        src_points = np.float32([
            [ patch_center - patch_size / (2 * sq2) , patch_center - patch_size / (2 * sq2) ], 
            [ patch_center + patch_size / (2 * sq2) , patch_center - patch_size / (2 * sq2) ], 
            [ patch_center + patch_size / (2 * sq2) , patch_center + patch_size / (2 * sq2) ]])

        # src_points are rotated COUNTER-CLOCKWISE
        src_points = rotate(src_points, angle, img.shape).astype(np.float32)

        # dst_points are fixed
        dst_points = np.float32([
            [ 0 , 0 ], 
            [ target_img_size - 1, 0 ], 
            [ target_img_size - 1, target_img_size - 1]]) 

        # this is effectively a CLOCKWISE rotation
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

    if (inputDict['flip_north_south'] or inputDict['flip_east_west']) and augmentation:

        flip_v = (np.random.random() < 0.5)
        flip_h = (np.random.random() < 0.5)

        if flip_h and inputDict['flip_east_west']:
            img = flip_axis(img, 1) # flips > into < 

        if flip_v and inputDict['flip_north_south']:
            img = flip_axis(img, 0) # flips ^ into v 

    #show_image(img.astype(np.uint8))
    #raw_input("Press enter")
    if augmentation:
        metadata = transform_metadata(metadata, flip_h=flip_h, flip_v=flip_v, angle=angle)

    img = imagenet_utils.preprocess_input(img) / 255.

    labels = data['category']
    currOutput = {}
    currOutput['img'] = img
    metadata = np.divide(json.load(open(data['features_path'])) - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
    if inputDict['mask_metadata']:
        metadata = mask_metadata(metadata)   
    currOutput['metadata'] = metadata
    currOutput['labels'] = labels
    return currOutput

def codes_metadata_generator(params, data, metadataStats, codesStats, class_aware_sampling = True, temporal_dropout = True):
    """
    Custom generator that yields a vector containign the 4096-d CNN codes output by ResNet50 and metadata features (if params set to use).
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to CNN codes and images 
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (codesMetadata,labels): 4096-d CNN codes + metadata features (if set), and labels (categorical form) 
    """
    
    N = len(data)

    if class_aware_sampling:
        data_labels = [datum['category'] for datum in data.values()]
        label_to_idx = defaultdict(list)
        for i, label in enumerate(data_labels):
            label_to_idx[label].append(i)
        running_label_to_idx = copy.deepcopy(label_to_idx)

    trainKeys = list(data.keys())

    executor = ThreadPoolExecutor(max_workers=1)#params.num_workers)
    
    while True:
        if class_aware_sampling:
            #idx = np.random.permutation(N)
            # class-aware supersampling
            idx = []
            num_labels = len(label_to_idx)
            assert num_labels == params.num_labels
            for _ in range(N):
                random_label = np.random.randint(num_labels)
                if len(running_label_to_idx[random_label]) == 0:
                    running_label_to_idx[random_label] = copy.copy(label_to_idx[random_label])
                    random.shuffle(running_label_to_idx[random_label])
                idx.append(running_label_to_idx[random_label].pop())
        else:
            idx = np.random.permutation(N)

        batchInds = get_batch_inds(params.batch_size, idx, N)

        for inds in batchInds:
            batchKeys = [trainKeys[ind] for ind in inds]
            codesMetadata, labels = load_lstm_batch(params, data, batchKeys, metadataStats, codesStats, executor, temporal_dropout)
            yield(codesMetadata,labels)
        
def load_lstm_batch(params, data, batchKeys, metadataStats, codesStats, executor, temporal_dropout):
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
        codesMetadata = np.zeros((params.batch_size, codesStats['max_temporal'], params.cnn_multi_layer_length + params.metadata_length))
    else:
        codesMetadata = np.zeros((params.batch_size, codesStats['max_temporal'], params.cnn_multi_layer_shape[0], params.cnn_multi_layer_shape[1], params.cnn_multi_layer_shape[2]))

    labels = np.zeros(params.batch_size)

    futures = []
    for i,key in enumerate(batchKeys):
        currInput = {}
        currInput['currData'] = data[key]
        currInput['lastLayerLength'] = codesMetadata.shape[2]
        currInput['lastLayerShape'] = params.cnn_multi_layer_shape
        currInput['codesStats'] = codesStats
        currInput['use_metadata'] = params.use_metadata
        currInput['metadataStats'] = metadataStats
        currInput['mask_metadata'] = params.mask_metadata
        currInput['temporal_dropout'] = temporal_dropout
        labels[i] = data[key]['category']

        task = partial(_load_lstm_batch_helper, currInput)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i,result in enumerate(results):
        codesMetadata[i,:,:] = result['codesMetadata']

    labels = to_categorical(labels, params.num_labels)
    
    return codesMetadata,labels

def _load_lstm_batch_helper(inputDict):

    currData = inputDict['currData']
    codesStats = inputDict['codesStats']
    metadataStats = inputDict['metadataStats']
    currOutput = {}

    codesMetadata = np.zeros((codesStats['max_temporal'], inputDict['lastLayerLength']))
    timestamps = []

    temporal_dropout = inputDict['temporal_dropout']
    n_codes = len(currData['cnn_codes_paths'])
    n_codes_indexes = range(n_codes)
    if n_codes > 3 and temporal_dropout != 0:
        n_codes_to_train = int(math.ceil(n_codes * (1 - np.random.rand() * temporal_dropout)))
        n_codes_to_train = max(n_codes_to_train, 3)
        n_codes_indexes = random.sample(n_codes_indexes, n_codes_to_train)

    if len(n_codes_indexes) > codesStats['max_temporal']:
        n_codes_indexes = n_codes_indexes[:codesStats['max_temporal']]      

    for i, codesIndex in enumerate(n_codes_indexes):
        #cnnCodes = json.load(open(currData['cnn_codes_paths'][codesIndex]))
        cnnCodes = np.load(jcurrData['cnn_codes_paths'][codesIndex])
        metadata = json.load(open(currData['metadata_paths'][codesIndex]))
        # compute a timestamp for temporally sorting
        timestamp = (metadata[4]-1970)*525600 + metadata[5]*12*43800 + metadata[6]*31*1440 + metadata[7]*60
        timestamps.append(timestamp)

        cnnCodes = np.divide(cnnCodes - np.array(codesStats['codes_mean']), np.array(codesStats['codes_max']))
        metadata = np.divide(metadata - np.array(metadataStats['metadata_mean']), np.array(metadataStats['metadata_max']))

        if inputDict['use_metadata']:
            if inputDict['mask_metadata']:
                metadata = mask_metadata(metadata)
            codesMetadata[i,:] = np.concatenate((cnnCodes, metadata), axis=0)
        else:
            codesMetadata[i,...] = cnnCodes

    sortedInds = sorted(range(len(timestamps)), key=lambda k:timestamps[k])
    codesMetadata[range(len(sortedInds)),:] = codesMetadata[sortedInds,:]

    currOutput['codesMetadata'] = codesMetadata
    return currOutput

