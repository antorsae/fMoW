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
from keras.applications import VGG16, VGG19, MobileNet, imagenet_utils, InceptionResNetV2, InceptionV3, Xception, ResNet50
from keras.layers import Dense,Input,Flatten,Dropout,LSTM, GRU, concatenate, add, Reshape, Conv2D, Conv1D, \
                         MaxPooling2D, ConvLSTM2D, Activation, BatchNormalization, Permute, LocallyConnected1D, ConvLSTM2D, Conv3D
from keras.models import Sequential,Model
from keras.preprocessing.image import random_channel_shift
from keras.utils.np_utils import to_categorical

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
import datetime

#from data_ml_functions.keras_squeeze_excite_network.se_inception_resnet_v2 import SEInceptionResNetV2
#from data_ml_functions.DenseNet import densenet

def get_cnn_model(params):   
    """
    Load base CNN model and add metadata fusion layers if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :return model: CNN model with or without depending on params
    """
    
    if params.views == 0:
        input_tensor = Input(shape=(params.target_img_size, params.target_img_size, params.num_channels))
    else:
        input_tensors = []
        for _ in range(params.views):
            _i = Input(shape=(params.target_img_size, params.target_img_size, params.num_channels))
            input_tensors.append(_i)

    classifier = globals()[params.classifier]
    if params.classifier == 'densenet':
        baseModel = densenet.DenseNetImageNet161(
            input_shape=(params.target_img_size, params.target_img_size, params.num_channels), include_top=False)
    else:
        baseModel = classifier(weights='imagenet' if not params.no_imagenet else None, 
            include_top=False, 
            pooling=params.pooling if params.pooling != 'none' else None,
            input_shape=(params.target_img_size, params.target_img_size, params.num_channels))
            
    trainable = False
    n_trainable = 0
    for i, layer in enumerate(baseModel.layers):
        if i >= params.freeze:
            trainable = True
            n_trainable += 1
        layer.trainable = trainable

    print("Base CNN model has " + str(n_trainable) + "/" + str(len(baseModel.layers)) + " trainable layers")

    if params.views == 0:
        modelStruct = baseModel(input_tensor)
    else:
        modelStruct = None
        for _input_tensor in input_tensors:
            _modelStruct = baseModel(_input_tensor)
            if modelStruct == None:
                modelStruct = _modelStruct
            else:
                modelStruct = concatenate([modelStruct, _modelStruct])
                #modelStruct = add([modelStruct, _modelStruct])
        # new
        if params.pooling != 'none':
            mview_preffix = 'lstm_'
            modelStruct = Reshape((params.views, -1))(modelStruct)
            modelStruct = LSTM(1024, return_sequences=True, name=mview_preffix + '0_1024_' + str(params.views))(modelStruct)
            modelStruct = LSTM(512,  return_sequences=True, name=mview_preffix + '1_512_'  + str(params.views))(modelStruct)
            if True:
                #modelStruct = LSTM(params.num_labels, return_sequences=False, name=lstm_preffix + 'labels_' + str(params.views))(modelStruct)
                predictions = Activation('softmax')(modelStruct)
            else:
                #modelStruct = LSTM(params.num_labels, return_sequences=True, name='lstm_labels_' + str(params.views))(modelStruct)
                modelStruct = Flatten()(modelStruct)
                predictions = Dense(params.num_labels, activation='softmax', name='predictions')(modelStruct)
        else:
            last_shape = modelStruct.shape
            print(last_shape)
            assert last_shape[1] == last_shape[2]
            conv_features_grid_shape = int(last_shape[1])
            conv_features_filters_shape = int(last_shape[3])
            new_shape = (params.views, conv_features_grid_shape, conv_features_grid_shape, -1)
            print(new_shape)
            modelStruct = Reshape(new_shape)(modelStruct)
            if params.view_model == 'lstm2d':
                # make it adaptative rel to grid shape
                mview_preffix = 'lstm2d_'
                modelStruct = ConvLSTM2D(256, (1,1), activation='relu', return_sequences=True, name=mview_preffix + '0_256_' + str(params.views))(modelStruct)
                #modelStruct = ConvLSTM2D(256, (3,3), activation='relu', return_sequences=True, name=lstm_preffix + '1_256_' + str(params.views))(modelStruct)
                modelStruct = ConvLSTM2D(256, (3,3), activation='relu', return_sequences=True, name=mview_preffix + '2_256_' + str(params.views))(modelStruct)
                modelStruct = ConvLSTM2D(params.num_labels, (3,3), return_sequences=False, name=mview_preffix + 'labels_' + str(params.views))(modelStruct)
                modelStruct = Flatten()(modelStruct)
                modelStruct = Activation('softmax')(modelStruct)
            elif params.view_model == 'conv3d':
                mview_preffix = 'conv3d_'
                down_convs = max(conv_features_grid_shape, params.views)
                strides_views = np.diff(np.linspace(params.views,             1, down_convs, dtype=np.int32))
                strides_grids = np.diff(np.linspace(conv_features_grid_shape, 1, down_convs, dtype=np.int32))
                filters = 2** np.int32(np.log2(np.logspace(np.log2(conv_features_filters_shape)-1,6, down_convs, dtype=np.int32, base=2))) #todo change 6 to 2**(ceil(log2(params.num_labels))
                filters[-1] = params.num_labels
                print(down_convs, strides_views, strides_grids, filters) 

                for it, (stride_views, stride_grid, _filter) in enumerate(zip(strides_views, strides_grids, filters[1:])):
                    last = (it == down_convs - 2)
                    _stride_views = -stride_views + 1
                    _stride_grid  = -stride_grid  + 1
                    modelStruct = Conv3D( 
                        _filter, 
                        (_stride_views,_stride_grid,_stride_grid), 
                        activation='relu' if not last else 'softmax', 
                        name=mview_preffix + str(it) + '_s'+ str(_stride_views) + '_'  + str(_stride_grid) + '_f_' +  str(_filter) + '_' + str(params.views) + ('_softmax' if last else ''))(modelStruct)

                    if not last:
                        modelStruct = BatchNormalization(name=mview_preffix + str(it) + '_batchnorm_' + str(params.views))(modelStruct)

                # fixed
                #modelStruct = Conv3D(512, (1,1,1), activation='relu', name=mview_preffix + '0_512_' + str(params.views))(modelStruct)
                #modelStruct = BatchNormalization()(modelStruct)
                #modelStruct = Conv3D(256, (2,3,3), activation='relu', name=mview_preffix + '1_256_' + str(params.views))(modelStruct)
                #modelStruct = BatchNormalization()(modelStruct)
                #modelStruct = Conv3D(128, (2,3,3), activation='relu', name=mview_preffix + '2_128_' + str(params.views))(modelStruct)
                #modelStruct = BatchNormalization()(modelStruct)
                #modelStruct = Conv3D(params.num_labels, (1,1,1), activation='softmax', name=mview_preffix + '3_labels_' + str(params.views))(modelStruct)
                predictions = Flatten()(modelStruct)


        #model.add(Dense(params.num_labels, activation='softmax'))
        #modelStruct = Permute((2, 1))(modelStruct)
        #modelStruct = LocallyConnected1D(3, 1, activation='relu')(modelStruct)
        #modelStruct = LocallyConnected1D(2, 1, activation='relu')(modelStruct)
        #modelStruct = LocallyConnected1D(1, 1, activation='relu')(modelStruct)
        #modelStruct = Flatten()(modelStruct)
#        modelStruct = Conv1D(512, 1, activation='relu')(modelStruct)
        # new

    if params.use_metadata:

        auxiliary_input = Input(shape=(params.metadata_length,), name='aux_input')

        modelStruct = concatenate([modelStruct, auxiliary_input])

        modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc1')(modelStruct)
        modelStruct = Dropout(0.2)(modelStruct)
        #modelStruct = Dense(512, activation='relu', name='nfc2')(modelStruct)
        #modelStruct = Dropout(0.1)(modelStruct)

    #modelStruct = Dense(1024, activation='relu', name='nfc1')(modelStruct)
    #modelStruct = Dropout(0.3)(modelStruct)
    if params.views == 0:

        modelStruct = Dense(512, activation='relu', name='nfc2')(modelStruct)
        modelStruct = Dropout(0.5)(modelStruct)
        modelStruct = Dense(512, activation='relu', name='nfc3')(modelStruct)
        modelStruct = Dropout(0.5)(modelStruct)
        predictions = Dense(params.num_labels, activation='softmax', name='predictions')(modelStruct)

    if params.views == 0:
        inputs = [input_tensor]
    else:
        inputs = input_tensors

    if params.use_metadata:
        inputs.append(auxiliary_input)

    model = Model(inputs=inputs, outputs=predictions)

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
        if params.views == 0:
            data_labels = [datum['category'] for datum in data]
        else:
            data_labels = [datum[0]['category'] for datum in data]
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
            inputs = imgdata
            if params.views != 0:
                assert len(imgdata)  == params.views
                assert len(metadata) == params.views
                assert len(labels)   == params.views
                # all labels should be equal
                label = labels[0]
                for _label in labels:
                    assert np.argmax(_label) == np.argmax(label)
                labels = label
                metadata = np.mean(metadata, axis=0)

            if params.use_metadata:
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]
                inputs.append(metadata)

            yield(inputs,labels)
        
def load_cnn_batch(params, batchData, metadataStats, executor, augmentation):
    """
    Load batch of images and metadata and preprocess the data before returning.
    :param params: global parameters, used to find location of the dataset and json file
    :param batchData: list of objects in the current batch containing the category labels and paths to CNN codes and images 
    :param metadataStats: metadata stats used to normalize metadata features
    :return imgdata,metadata,labels: numpy arrays containing the image data, metadata, and labels (categorical form)
    """
    futures = []
    if params.views == 0:
        imgdata = np.zeros((params.batch_size,params.target_img_size,params.target_img_size,params.num_channels))
        metadata = np.zeros((params.batch_size,params.metadata_length))
        labels = np.zeros((params.batch_size, params.num_labels))
    else:
        imgdata  = []
        metadata = []
        labels   = []
        for _ in range(params.views):
            imgdata.append(np.zeros((params.batch_size,params.target_img_size,params.target_img_size,params.num_channels)))
            metadata.append(np.zeros((params.batch_size,params.metadata_length)))
            labels.append(np.zeros((params.batch_size, params.num_labels)))
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
        currInput['offset'] = params.offset
        currInput['zoom'] = params.zoom
        currInput['views'] = params.views
        currInput['num_labels'] = params.num_labels
        currInput['jitter_channel'] = params.jitter_channel
        currInput['jitter_metadata'] = params.jitter_metadata

        task = partial(_load_batch_helper, currInput, augmentation)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i,result in enumerate(results):
        if params.views == 0:
            imgdata[i, ...] = result['img']
            metadata[i,:]   = result['metadata'] 
            labels[i]       = result['labels']
        else:
            for view in range(params.views):
                imgdata[view][i]  = result[view]['img'] 
                metadata[view][i] = result[view]['metadata']
                labels[view][i]   = result[view]['labels']

    return imgdata,metadata,labels

# counter-clockwise rotation
def rotate(a, angle, img_shape):
    center = np.array([img_shape[1], img_shape[0]]) / 2.
    theta = (angle/180.) * np.pi
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta),  np.cos(theta)]])
    return np.dot(a - center, rotMatrix) + center

def zoom(a, scale, img_shape):
    center = np.array([img_shape[1], img_shape[0]]) / 2.
    return (a - center) * scale + center

def transform_metadata(metadata, flip_h, flip_v, angle=0, zoom = 1):
    metadata_angles = np.fmod(180. + np.array(metadata[19:27]) * 360., 360.) - 180.

    # b/c angles are clockwise we add the clockwise rotation angle
    metadata_angles += angle

    if flip_h: # > <
        metadata_angles =      - metadata_angles
    if flip_v: # v ^
        metadata_angles = 180. - metadata_angles

    metadata[19:27] = list(np.fmod(metadata_angles + 2*360., 360.) / 360.)

    # zoom > 1 is zoom OUT
    # zoom < 1 is zoom IN

    metadata[35] *= zoom
    metadata[36] *= zoom

    metadata[0] *= zoom # zoom > 1 zooms OUT so each pixel measures more

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

def get_timestamp(metadata):
    return (metadata[4]-1970)*525600 + metadata[5]*12*43800 + metadata[6]*31*1440 + metadata[7]*60

def jitter_metadata(metadata, scale, max_values):
    year   = int(metadata[4])
    month  = int(metadata[5]*12) # 1..12 / 12 => 1..12
    day    = int(metadata[6]*31) # 1..31 / 31 => 1..31
    hour   = int(metadata[7]) # 0..23 + 0..59/60 -> 0..23
    minute = int((metadata[7] - hour) * 60) # 0..59
    second = int(metadata[43]) # 0..59

    scan_direction = metadata[8]  # 0 or 1
    bounding_boxes = metadata[44] # 0 or 1

    _max_values = np.array(max_values)
    metadata = np.random.uniform(metadata - _max_values * scale / 2., metadata + _max_values * scale / 2.)

    timespan_1year = 365.25 # in days
    time_delta = datetime.timedelta(\
        days=int(np.random.uniform(-scale * 365/ 2, scale * 365 / 2)))

    metadata_time  = datetime.datetime(year, month, day, hour, minute) + time_delta

    metadata[4] = metadata_time.year
    metadata[5] = metadata_time.month/12.
    metadata[6] = metadata_time.day/31.
    metadata[7] = hour/12. # keep hour b/c lighting conditions
    metadata[39] = float(metadata_time.weekday())
    metadata[43] = metadata_time.second 
    
    metadata[8]   = scan_direction # keep scan direction
    metadata[44]  = bounding_boxes # keep bounding boxes (0 or 1)

    return metadata

def _load_batch_helper(inputDict, augmentation):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """
    datas = inputDict['data']
    metadataStats = inputDict['metadataStats']
    num_labels = inputDict['num_labels']

    # for 0-views make it a list so we can iterate later
    if not isinstance(datas, (list, tuple)):
        datas = [datas]

    currOutputs = [ ]
    target_img_size = inputDict['target_img_size']

    if augmentation:
        random_offset = (np.random.random((2,)) - 0.5 ) * (inputDict['offset'] * target_img_size)
        random_angle  = (np.random.random() - 0.5 ) * inputDict['angle']
        random_zoom   = np.random.uniform(1. - inputDict['zoom'] / 2., 1 + inputDict['zoom'] / 2.)
        flip_v = (np.random.random() < 0.5) and inputDict['flip_east_west']
        flip_h = (np.random.random() < 0.5) and inputDict['flip_north_south']
    else:
        random_offset = np.zeros(2,)
        random_zoom   = 1.
        random_angle  = 0.
        flip_v = flip_h = False

    if inputDict['views'] != 0:
        datas = random.sample(datas, inputDict['views'])

    timestamps = []
    for data in datas:

        metadata = json.load(open(data['features_path']))
        timestamps.append(get_timestamp(metadata))
        img = scipy.misc.imread(data['img_path'])

        if inputDict['jitter_channel'] != 0 and augmentation:
            img = random_channel_shift(img, inputDict['jitter_channel'] * 255., 2)

        if (random_angle != 0. or random_zoom != 1.) and augmentation:
            patch_size = img.shape[0]
            patch_center = patch_size / 2
            sq2 = 1.4142135624 

            src_points = np.float32([
                [ patch_center - target_img_size / 2 , patch_center - target_img_size / 2 ], 
                [ patch_center + target_img_size / 2 , patch_center - target_img_size / 2 ], 
                [ patch_center + target_img_size / 2 , patch_center + target_img_size / 2 ]])

            # src_points are rotated COUNTER-CLOCKWISE
            src_points = rotate(src_points, random_angle, img.shape)

            src_points = zoom(src_points, random_zoom, img.shape)

            src_points += random_offset 

            # dst_points are fixed
            dst_points = np.float32([
                [ 0 , 0 ], 
                [ target_img_size - 1, 0 ], 
                [ target_img_size - 1, target_img_size - 1]]) 

            # this is effectively a CLOCKWISE rotation
            M   = cv2.getAffineTransform(src_points.astype(np.float32), dst_points)
            img = cv2.warpAffine(img, M, (target_img_size, target_img_size), borderMode = cv2.BORDER_REFLECT_101).astype(np.float32)
        else:
            crop_size = target_img_size
            x0 = int(img.shape[1]/2 - crop_size/2 + random_offset[0])
            x1 = x0 + crop_size
            y0 = int(img.shape[0]/2 - crop_size/2 + random_offset[1])
            y1 = y0 + crop_size

            img = img[y0:y1, x0:x1,...].astype(np.float32)

        if flip_h:
            img = flip_axis(img, 1) # flips > into < 

        if flip_v:
            img = flip_axis(img, 0) # flips ^ into v 

        #show_image(img.astype(np.uint8))
        #raw_input("Press enter")

        if augmentation:
            metadata = transform_metadata(metadata, flip_h=flip_h, flip_v=flip_v, angle=random_angle, zoom=random_zoom)
            if inputDict['jitter_metadata'] != 0:
                metadata = jitter_metadata(metadata, inputDict['jitter_metadata'], metadataStats['metadata_max'])

        img = imagenet_utils.preprocess_input(img) / 255.

        labels = to_categorical(data['category'], num_labels)
        currOutput = {}
        currOutput['img'] = copy.deepcopy(img)
        metadata = np.divide(json.load(open(data['features_path'])) - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
        if inputDict['mask_metadata']:
            metadata = mask_metadata(metadata)   
        currOutput['metadata'] = metadata
        currOutput['labels'] = labels

        currOutputs.append(currOutput)

    if (len(currOutputs) == 1) and (inputDict['views'] == 0):
        currOutputs = currOutputs[0]
    else:
        # sort by timestamp
        sortedInds = sorted(range(len(timestamps)), key=lambda k:timestamps[k])
        currOutputs = [currOutputs[i] for i in sortedInds]

    return currOutputs

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
        timestamp = get_timestamp(metadata) 
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

