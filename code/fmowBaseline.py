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

from sklearn.model_selection import train_test_split
import json
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import VGG16,imagenet_utils
from data_ml_functions.mlFunctions import get_cnn_model, get_multi_model,img_metadata_generator, rotate, transform_metadata, mask_metadata, codes_metadata_generator
from data_ml_functions.dataFunctions import prepare_data,calculate_class_weights, flip_axis
import numpy as np
import os

from data_ml_functions.mlFunctions import load_cnn_batch
from data_ml_functions.dataFunctions import get_batch_inds
from multi_gpu_keras import multi_gpu_model
from keras import backend as K
from itertools import groupby

import time
from tqdm import tqdm
import keras.backend as K
import scipy.misc
import cv2
from data_ml_functions.iterm import show_image
import math
import random
import re
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import hickle
import keras.losses
from keras.losses import categorical_hinge
import glob
import tensorflow as tf

def softF1_loss(target, output):
    smooth = 0.001
    y_true_f = K.flatten(target)
    y_pred_f = K.flatten(output)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum((1. - y_true_f) * y_pred_f)
    fp = K.sum(y_true_f * (1. - y_pred_f))
    return 1 - (2. * tp + smooth) / ( 2 * tp + fn + fp + smooth) #(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def surrogateF1_loss(target, output): # WiP
    #https://arxiv.org/pdf/1608.04802.pdf
    y_true = K.flatten(target * 2 - 1)
    y_pred = K.flatten(output * 2 - 1)
    Yp = K.flatten(target)
    Yn = 1. - Yp

    tpl = K.sum(1. - categorical_hinge(y_true * Yp, y_pred * Yp))
    fpu = K.sum(     categorical_hinge(y_true * Yn, y_pred * Yn))

    # 2 tpl / |Y+| + tpl + fpu
    return 2 * tpl / (K.sum(Yp) + tpl + fpu)

def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)

category_weights = None

class FMOW_Callback(Callback):

    def __init__(self):
        super(Callback, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        np.random.seed(epoch)
        random.seed(epoch)
        tf.set_random_seed(epoch)
        return

    def on_epoch_end(self, epoch, logs={}):
#        print(logs)
#        y_pred_val = self.model.predict(self.x_val)
#        f1 = f1_score(self.y_val, y_pred_val, sample_weight=category_weights)
#        print(f1)        
        return
 

class FMOWBaseline:
    def __init__(self, params=None, argv=None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return: 
        """
        global category_weights

        self.params = params
        category_weights = self.get_class_weights(return_array=True)

        keras.losses.focal_loss       = focal_loss
        keras.losses.softF1_loss      = softF1_loss
        keras.losses.surrogateF1_loss = surrogateF1_loss

        np.random.seed(0)
        random.seed(0)
        tf.set_random_seed(0)

        if self.params.use_metadata:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_with_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_with_metadata.json')
            self.params.files['multi_training_struct'] = os.path.join(self.params.directories['working'], 'multi_training_struct_with_metadata.json')
            self.params.files['multi_test_struct'] = os.path.join(self.params.directories['working'], 'multi_test_struct_with_metadata.json')

        else:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_no_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_no_metadata.json')
            self.params.files['multi_training_struct'] = os.path.join(self.params.directories['working'], 'multi_training_struct_no_metadata.json')
            self.params.files['multi_test_struct'] = os.path.join(self.params.directories['working'], 'multi_test_struct_no_metadata.json')

    def get_class_weights(self, return_array=False):
        class_weights = {}

        low_impact = [ 'wind_farm', 'tunnel_opening', 'solar_farm', 'nuclear_powerplant', 'military_facility', 'crop_field', 
                        'airport', 'flooded_road', 'debris_or_rubble', 'single-unit_residential']
        high_impact = [ 'border_checkpoint', 'construction_site', 'educational_institution', 'factory_or_powerplant', 'fire_station',
                        'police_station', 'gas_station', 'smokestack', 'tower', 'road_bridge' ]

        for cat_id in range(self.params.num_labels):
            class_weights[cat_id] = 1.

        for category in low_impact:
            class_weights[self.params.category_names.index(category)] = 0.6
        for category in high_impact:
            class_weights[self.params.category_names.index(category)] = 1.4

        class_weights[self.params.category_names.index('false_detection')] = 0.

        if return_array:
            class_weights_array = [ ]
            for cat_id in range(self.params.num_labels):
                class_weights_array.append( class_weights[cat_id])
            return class_weights_array

        return class_weights

    def get_preffix(self):
        preffix_pairs = [ \
            'multi' if self.params.multi else 'cnn', \
            'views_' + str(self.params.views) if (self.params.views != 0 and not self.params.multi) else '', \
            'metadata' if self.params.use_metadata else 'no_metadata', \
            'd_' + self.params.directories_suffix, \
            'c_' + self.params.classifier + '_' + self.params.pooling, \
            'lr_' + str(self.params.learning_rate), \
            'lu' if self.params.leave_unbalanced else '', \
            'mm' if self.params.mask_metadata else '', \
            'nm' if self.params.norm_metadata else '', \
            'b_' + str(self.params.batch_size), \
            'td_' + str(self.params.temporal_dropout) if self.params.multi else '', \
            'a_' + str(self.params.angle) if not self.params.multi else '', \
            'o_' + str(self.params.offset) if not self.params.multi else '', \
            'z_' + str(self.params.zoom) if not self.params.multi else '', \
            'jc_' + str(self.params.jitter_channel) if not self.params.multi else '', \
            'jm_' + str(self.params.jitter_metadata) if not self.params.multi else '', \
            'freeze_' + str(self.params.freeze) if not self.params.multi else '', \
            'loss_' + self.params.loss if self.params.loss != 'categorical_crossentropy' else '', \
            'fns' if self.params.flip_north_south else '' if not self.params.multi else '', 
            'few' if self.params.flip_east_west else '' if not self.params.multi else '', 
            'w' if self.params.weigthed else '' \
            'nim' if self.params.no_imagenet else '' \
            ]

        preffix_pairs = [x for x in preffix_pairs if x != '']

        preffix = '-'.join(preffix_pairs)

        return preffix

    def get_initial_epoch(self, loaded_filename):
        initial_epoch = 0
        if loaded_filename:
            match = re.search(r'.*epoch_(\d+).*\.hdf5', loaded_filename)
            if match:
                initial_epoch = int(match.group(1))
        return initial_epoch

    def ensemble(self):
        prediction_maps = [ ]

        prediction_name_suffix = ''

        for predictions_map_hkl in self.params.ensemble:
            print("Loading " + predictions_map_hkl)
            prediction_maps.append(hickle.load(predictions_map_hkl))
            predictions_filename = glob.glob(predictions_map_hkl[:-4]+'*.txt')[0]
            match = re.search(r'.*-c_(\w+)-.*-epoch_(\d+).*-LB_(\d+)\.txt', predictions_filename)
            if match:
                classifier = match.group(1)
                epoch =      match.group(2)
                lb    =      match.group(3)
                prediction_name_suffix += '--' + classifier + "-" + epoch + "-" + lb

        n_maps = len(prediction_maps)
        assert n_maps > 1

        prediction_name_preffix = os.path.join(self.params.directories['predictions'], 
                'ensemble-%s-%s%s-%s' % (n_maps, self.params.ensemble_mean, prediction_name_suffix, time.strftime("%Y%m%d-%H%M%S")))
        fid = open(prediction_name_preffix + '.txt', 'w')
        epsilon = 1e-8
        for bbID in tqdm(prediction_maps[0]):
            prediction_shape = prediction_maps[0][bbID].shape
            predictions = np.zeros((n_maps, prediction_shape[1]))
            for i, prediction_map in enumerate(prediction_maps):

                pred = prediction_map[bbID]
                if self.params.ensemble_mean == 'geometric':
                    pred = np.log(pred + epsilon) # avoid numerical instability log(0)
                    predictions[i,...] = np.exp(np.mean(pred, axis=0))
                else:
                    predictions[i,...] = np.mean(pred, axis=0)

            if self.params.ensemble_mean == 'geometric':
                predictions = np.log(predictions + epsilon) # avoid numerical instability log(0)
            prediction = np.sum(predictions, axis=0)
            max_prediction = np.argmax(prediction)
            prediction_category = self.params.category_names[max_prediction]
            fid.write('%s,%s\n' % (bbID, prediction_category))
        fid.close()
        print(prediction_name_preffix + ".txt")

    def train(self):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        
        allTrainingData = json.load(open(self.params.files['training_struct']))

        # add 50% of the /val/ data (which has false_detection instances) to train dataset, leave rest for validation
        trainData    = [_t for _t in allTrainingData if _t['features_path'].find('/val/') == -1]
        allValidData = [_t for _t in allTrainingData if _t['features_path'].find('/val/') != -1]

        addToTrainData, validData = train_test_split(allValidData , test_size=0.5)

        trainData.extend(addToTrainData)

        assert len(allTrainingData) == len(trainData) + len(validData)

        if self.params.views != 0:
            allTrainingViews = []
            for k,g in groupby(sorted(allTrainingData), lambda x:x['features_path'].split('/')[-2]):
                group = list(g)
                if len(group) >= self.params.views:
                    allTrainingViews.append(group)
            trainViews    = [_t for _t in allTrainingViews if _t[0]['features_path'].find('/val/') == -1]
            allValidViews = [_t for _t in allTrainingViews if _t[0]['features_path'].find('/val/') != -1]
 
            addToTrainViews, validViews = train_test_split(allValidViews , test_size=0.5)
            trainViews.extend(addToTrainViews)

            assert len(allTrainingViews) == len(trainViews) + len(validViews)

            # for validation leave only exact number of views
            validViews = [_t for _t in validViews if len(_t) == self.params.views]

        metadataStats = json.load(open(self.params.files['dataset_stats']))

        loaded_filename = None
        if self.params.args.load_model:
            from keras.utils.generic_utils import CustomObjectScope
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                model = load_model(self.params.args.load_model)
            loaded_filename = os.path.basename(self.params.args.load_model)
        else:
            model = get_cnn_model(self.params)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True) # Keras 2.1.2, skip_mismatch=True)
            loaded_filename = os.path.basename(self.params.args.load_weights)

        initial_epoch = self.get_initial_epoch(loaded_filename)

        if self.params.print_model_summary:
            model.summary()

        model = multi_gpu_model(model, gpus=self.params.gpus)

        _loss = self.params.loss

        if  _loss== 'focal': 
            loss = focal_loss 
        elif _loss == 'softF1':
            loss = softF1_loss
        elif _loss == 'surrogateF1':
            loss = surrogateF1_loss
        else:
            loss = _loss

        model.compile(optimizer=Adam(lr=self.params.learning_rate),# amsgrad=self.params.amsgrad), 
            loss=loss, 
            metrics=['accuracy'])
        
        preffix = self.get_preffix()

        print("training single-image model: " + preffix)

        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], 
            preffix + '-epoch_' + '{epoch:02d}' + '-acc_' + '{acc:.4f}' + '-val_acc_' + '{val_acc:.4f}.hdf5')

        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=1, min_lr=1e-7, epsilon = 0.01, verbose=1)

        if self.params.views != 0:
            trainData = trainViews
            validData = validViews

        print("Train samples: %d, validation samples: %d" % ((len(trainData), len(validData))))

        model.fit_generator(
            generator=img_metadata_generator(self.params, trainData, metadataStats, class_aware_sampling = not self.params.leave_unbalanced),
            steps_per_epoch=int(math.ceil((len(trainData) / self.params.batch_size))),
            class_weight = self.get_class_weights() if self.params.weigthed else None,
            epochs=self.params.epochs, initial_epoch = initial_epoch,
            callbacks=[checkpoint, FMOW_Callback(), reduce_lr], 
            validation_data = img_metadata_generator(self.params, validData, metadataStats, class_aware_sampling = False, augmentation = False),
            validation_steps = int(math.ceil((len(validData) / self.params.batch_size))),
            shuffle=False,
            )

        model.save(self.params.files['cnn_model'])

    def train_multi(self):
        """
        Train LSTM pipeline using pre-generated CNN codes.
        :param: 
        :return: 
        """

        allCodesTrainingData = json.load(open(self.params.files['multi_training_struct']))

        # add 50% of the /val/ data (which has false_detection instances) to train dataset, leave rest for validation
        codesTrainData = { }
        codesValidData = { }

        for k,v in allCodesTrainingData.iteritems():
            if k.startswith('val/'):
                if np.random.randint(2):
                    codesTrainData[k] = v
                else:
                    codesValidData[k] = v
            else:
                codesTrainData[k] = v

        assert len(allCodesTrainingData) == len(codesTrainData) + len(codesValidData)

        codesStats = json.load(open(self.params.files['cnn_codes_stats']))
        if self.params.max_temporal != 0:
            codesStats['max_temporal'] = self.params.max_temporal

        metadataStats = json.load(open(self.params.files['dataset_stats']))
        
        loaded_filename = None
        if self.params.args.load_model:
            model = load_model(self.params.args.load_model)
            loaded_filename = os.path.basename(self.params.args.load_model)
        else:
            model = get_multi_model(self.params, codesStats)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True)
            loaded_filename = os.path.basename(self.params.args.load_weights)

        initial_epoch = self.get_initial_epoch(loaded_filename)

        if self.params.print_model_summary:
            model.summary()

        model = multi_gpu_model(model, gpus=self.params.gpus)

        model.compile(optimizer=RMSprop(lr=self.params.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        
        preffix = self.get_preffix()

        print("training multi-image model: " + preffix)

        filePath = os.path.join(self.params.directories['multi_checkpoint_weights'], 
            preffix + '-epoch_' + '{epoch:02d}' + '-acc_' + '{acc:.4f}' + '-val_acc_' + '{val_acc:.4f}.hdf5')
        
        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, 
            save_weights_only=False, mode='auto', period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=1, min_lr=1e-7, epsilon = 0.01, verbose=1)

        model.fit_generator(generator=codes_metadata_generator(self.params, \
                                                                codesTrainData, metadataStats, codesStats, \
                                                                class_aware_sampling = not self.params.leave_unbalanced, \
                                                                temporal_dropout = self.params.temporal_dropout),
                            steps_per_epoch=int(math.ceil((len(codesTrainData) / self.params.batch_size))),
                            epochs=self.params.epochs, 
                            callbacks=[checkpoint, FMOW_Callback(), reduce_lr],
                            initial_epoch = initial_epoch,
                            validation_data = codes_metadata_generator(self.params, \
                                                                        codesValidData, metadataStats, codesStats, \
                                                                        class_aware_sampling = False,\
                                                                        temporal_dropout = 0.),
                            validation_steps = int(math.ceil((len(codesValidData) / self.params.batch_size))),
                            )

        model.save(self.params.files['multi_model'])
        
    def test(self):

        if self.params.multi:
            codesTestData = json.load(open(self.params.files['multi_test_struct']))
            codesStats = json.load(open(self.params.files['cnn_codes_stats']))
            if self.params.max_temporal != 0:
                codesStats['max_temporal'] = self.params.max_temporal
            
        metadataStats = json.load(open(self.params.files['dataset_stats']))
    
        metadataMean = np.array(metadataStats['metadata_mean'])
        metadataMax = np.array(metadataStats['metadata_max'])

        loaded_filename = None
        if self.params.args.load_model:
            from keras.utils.generic_utils import CustomObjectScope
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                model = load_model(self.params.args.load_model)
            loaded_filename = os.path.basename(self.params.args.load_model)
        else:
            model = get_cnn_model(self.params) if not self.params.multi else get_multi_model(self.params, codesStats)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True)
            loaded_filename = os.path.basename(self.params.args.load_weights)

        model = multi_gpu_model(model, gpus=self.params.gpus)

        index = 0
        
        timestr = time.strftime("%Y%m%d-%H%M%S")

        prediction_name_preffix = os.path.join(self.params.directories['predictions'], 
                'predictions-%s-%s' % (loaded_filename[:-5], timestr))
        fid = open(prediction_name_preffix + '.txt', 'w')

        predictions_map = { }

        def walkdir(folder):
            for root, dirs, files in os.walk(folder):
                if len(files) > 0:
                    yield (root, dirs, files)
        
        num_sequences = 0
        for _ in walkdir(self.params.directories['test_data']):
            num_sequences += 1
        
        for root, dirs, files in tqdm(walkdir(self.params.directories['test_data']), total=num_sequences):
            if len(files) > 0:
                imgPaths = []
                metadataPaths = []
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']
                bbID = int(root[slashes[-1]+1:])
                
            for file in files:
                if file.endswith(self.params.image_format_processed):
                    imgPaths.append(os.path.join(root,file))
                    metadataPaths.append(os.path.join(root, file[:-4]+'_features.json'))
                    
            if len(files) > 0:
                inds = []
                for metadataPath in metadataPaths:
                    underscores = [ind for ind,ltr in enumerate(metadataPath) if ltr == '_']
                    inds.append(int(metadataPath[underscores[-3]+1:underscores[-2]]))
                inds = np.argsort(np.array(inds)).tolist()

                if not self.params.multi:
                    # single-image
                
                    tta_flip_v = self.params.flip_north_south
                    tta_flip_h = self.params.flip_east_west

                    currBatchSize = len(inds) * (2 if tta_flip_v else 1) * (2 if tta_flip_h else 1)
                    imgdata = np.zeros((currBatchSize, self.params.target_img_size, self.params.target_img_size, self.params.num_channels))
                    metadataFeatures = np.zeros((currBatchSize, self.params.metadata_length))
                        
                    for ind in inds:
                        features = np.array(json.load(open(metadataPaths[ind])))

                        img = scipy.misc.imread(imgPaths[ind]) #image.load_img(imgPaths[ind])
                        crop_size = self.params.target_img_size
                        x0 = int(img.shape[1]/2 - crop_size/2)
                        x1 = x0 + crop_size
                        y0 = int(img.shape[0]/2 - crop_size/2)
                        y1 = y0 + crop_size

                        img = img[y0:y1, x0:x1, ...].astype(np.float32)

                        #show_image(img)
                        #raw_input("press enter")

                        metadataFeatures[ind,:] = features

                        img = imagenet_utils.preprocess_input(img) / 255.
                        imgdata[ind, ...] = img

                        tta_idx = len(inds) + ind
                        if tta_flip_v:
                            imgdata[tta_idx,...] = flip_axis(img, 0)
                            metadataFeatures[tta_idx,:] = transform_metadata(features, flip_h = False, flip_v=True)
                            tta_idx += len(inds)

                        if tta_flip_h:
                            imgdata[tta_idx,...] = flip_axis(img, 1)
                            metadataFeatures[tta_idx,:] = transform_metadata(features, flip_h = True, flip_v=False)
                            tta_idx += len(inds)

                            if tta_flip_v:
                                imgdata[tta_idx,...] = flip_axis(flip_axis(img, 1), 0)
                                metadataFeatures[tta_idx,:] = transform_metadata(features, flip_h = True, flip_v=True)
                                tta_idx += len(inds)
                    
                    if self.params.use_metadata:
                        metadataFeatures = np.divide(metadataFeatures - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
                        if self.params.mask_metadata:
                            for ind in inds:
                                metadataFeatures[ind] = mask_metadata(metadataFeatures[ind])

                        _predictions = model.predict([imgdata, metadataFeatures], batch_size=currBatchSize)
                    else:
                        _predictions = model.predict(imgdata, batch_size=currBatchSize)

                    predictions_map[str(bbID)] = _predictions
                    predictions  = np.sum(_predictions, axis=0) 

                else:
                    # multi-image

                    currBatchSize = len(inds)
                    metadataFeatures = np.zeros((currBatchSize, self.params.metadata_length))

                    codesIndex = 0
                    code_index = '/'.join(root.split('/')[-3:])
                    codesPaths = codesTestData[code_index]
                    codesFeatures = []
                    for ind in inds:

                        features = np.array(json.load(open(metadataPaths[ind])))
                        metadataFeatures[ind,:] = features
                        
                        codesFeatures.append(json.load(open(codesPaths['cnn_codes_paths'][codesIndex])))
                        codesIndex += 1

                    if self.params.use_metadata:
                        codesMetadata = np.zeros((1, codesStats['max_temporal'], self.params.cnn_multi_layer_length + self.params.metadata_length))
                    else:
                        codesMetadata = np.zeros((1, codesStats['max_temporal'], self.params.cnn_multi_layer_length))

                    timestamps = []
                    for codesIndex in range(currBatchSize):
                        cnnCodes = codesFeatures[codesIndex]
                        metadata = metadataFeatures[codesIndex]
                        #print(metadata)
                        timestamp = (metadata[4]-1970)*525600 + metadata[5]*12*43800 + metadata[6]*31*1440 + metadata[7]*60
                        timestamps.append(timestamp)
                        cnnCodes = np.divide(cnnCodes - np.array(codesStats['codes_mean']), np.array(codesStats['codes_max']))
                        metadata = np.divide(metadata - metadataMean, metadataMax)
                        #print(metadata)

                        if self.params.use_metadata:
                            if self.params.mask_metadata:
                                metadata = mask_metadata(metadata)
                            codesMetadata[0,codesIndex,:] = np.concatenate((cnnCodes, metadata), axis=0)
                        else:
                            codesMetadata[0,codesIndex,:] = cnnCodes
                    
                    sortedInds = sorted(range(len(timestamps)), key=lambda k:timestamps[k])
                    codesMetadata[0,range(len(sortedInds)),:] = codesMetadata[0,sortedInds,:]
                    predictions = model.predict(codesMetadata, batch_size=1)
                                
            if len(files) > 0:
                prediction = np.argmax(predictions)
                prediction_category = self.params.category_names[prediction]
                fid.write('%d,%s\n' % (bbID,prediction_category))
                index += 1

        hickle.dump(predictions_map, prediction_name_preffix + ".hkl")
        fid.close()

    def generate_cnn_codes(self):
        """
        Use trained CNN to generate CNN codes/features for each image or (image, metadata) pair
        which can be used to train an LSTM.
        :param: 
        :return: 
        """
        
        metadataStats = json.load(open(self.params.files['dataset_stats']))
        trainData = json.load(open(self.params.files['training_struct']))
        testData = json.load(open(self.params.files['test_struct']))

        if self.params.args.load_model:
            model = load_model(self.params.args.load_model)
        else:
            model = get_cnn_model(self.params)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True)

        features_layer_index = -3

        featuresModel = Model(model.inputs, model.layers[features_layer_index].output)

        #assert model.layers[features_layer_index].output.shape[1] == self.params.cnn_multi_layer_length

        if self.params.print_model_summary:
            featuresModel.summary()

        featuresModel = multi_gpu_model(featuresModel, gpus=self.params.gpus)
        
        allTrainCodes = []
        
        featureDirs = ['train', 'test']

        for featureDir in featureDirs:
            
            codesData = {}
            
            isTrain = (featureDir == 'train')
            index = 0

            if isTrain:
                data = trainData
            else:
                data = testData
            
            outDir = os.path.join(self.params.directories['cnn_codes'], featureDir)
            if not os.path.isdir(outDir):
                os.mkdir(outDir)

            N = len(data)
            initBatch = True
            for i,currData in enumerate(tqdm(data)):
                if initBatch:
                    if N-i < self.params.batch_size:
                        batchSize = 1
                    else:
                        batchSize = self.params.batch_size
                    imgdata = np.zeros((batchSize, self.params.target_img_size, self.params.target_img_size, self.params.num_channels))
                    metadataFeatures = np.zeros((batchSize, self.params.metadata_length))
                    batchIndex = 0
                    tmpBasePaths = []
                    tmpFeaturePaths = []
                    tmpCategories = []
                    initBatch = False

                path,_  = os.path.split(currData['img_path'])
                if isTrain:
                    basePath = path[len(self.params.directories['train_data'])+1:]
                else:
                    basePath = path[len(self.params.directories['test_data'])+1:]
                    
                tmpBasePaths.append(basePath)
                if isTrain:
                    tmpCategories.append(currData['category'])
                
                origFeatures = np.array(json.load(open(currData['features_path'])))
                tmpFeaturePaths.append(currData['features_path'])

                metadataFeatures[batchIndex, :] = np.divide(origFeatures - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
                
                img = scipy.misc.imread(currData['img_path']) #image.load_img(imgPaths[ind])
                crop_size = self.params.target_img_size
                x0 = int(img.shape[1]/2 - crop_size/2)
                x1 = x0 + crop_size
                y0 = int(img.shape[0]/2 - crop_size/2)
                y1 = y0 + crop_size

                img = img[y0:y1, x0:x1, ...].astype(np.float32)

                imgdata[batchIndex,...] = img 

                batchIndex += 1

                if batchIndex == batchSize:
                    imgdata = imagenet_utils.preprocess_input(imgdata) / 255.

                    if self.params.use_metadata:
                        cnnCodes = featuresModel.predict([imgdata,metadataFeatures], batch_size=batchSize)
                    else:
                        cnnCodes = featuresModel.predict(imgdata, batch_size=batchSize)

                    for codeIndex,currCodes in enumerate(cnnCodes):
                        currBasePath = tmpBasePaths[codeIndex]
                        #outFile = os.path.join(outDir, '%07d.npy' % index)
                        outFile = os.path.join(outDir, '%07d' % index)
                        index += 1
                        np.save(outFile, currCodes)
                        #json.dump(currCodes.tolist(), open(outFile, 'w'))
                        if currBasePath not in codesData.keys():
                            codesData[currBasePath] = {}
                            codesData[currBasePath]['cnn_codes_paths'] = []
                            codesData[currBasePath]['metadata_paths'] = []
                            if isTrain:
                                codesData[currBasePath]['category'] = tmpCategories[codeIndex]
                        codesData[currBasePath]['cnn_codes_paths'].append(outFile)
                        codesData[currBasePath]['metadata_paths'].append(tmpFeaturePaths[codeIndex])
                        if isTrain:
                            allTrainCodes.append(currCodes)
                        initBatch = True
        
            if isTrain:
                codesTrainData = codesData
            else:
                codesTestData = codesData

        N = len(allTrainCodes[0])
        sumCodes = np.zeros(N)
        for currCodes in allTrainCodes:
            sumCodes += currCodes
        avgCodes = sumCodes / len(allTrainCodes)
        maxCodes = np.zeros(N)
        for currCodes in allTrainCodes:
            maxCodes = np.maximum(maxCodes, np.abs(currCodes-avgCodes))
        maxCodes[maxCodes == 0] = 1
            
        maxTemporal = 0
        for key in codesTrainData.keys():
            currTemporal = len(codesTrainData[key]['cnn_codes_paths'])
            if currTemporal > maxTemporal:
                maxTemporal = currTemporal

        codesStats = {}
        codesStats['codes_mean'] = avgCodes.tolist()
        codesStats['codes_max'] = maxCodes.tolist()
        codesStats['max_temporal'] = maxTemporal

        json.dump(codesTrainData, open(self.params.files['multi_training_struct'], 'w'))
        json.dump(codesStats, open(self.params.files['cnn_codes_stats'], 'w'))
        json.dump(codesTestData, open(self.params.files['multi_test_struct'], 'w'))
