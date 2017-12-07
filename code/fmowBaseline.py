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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications import VGG16,imagenet_utils
from data_ml_functions.mlFunctions import get_cnn_model, get_multi_model,img_metadata_generator, rotate, transform_metadata, codes_metadata_generator
from data_ml_functions.dataFunctions import prepare_data,calculate_class_weights, flip_axis
import numpy as np
import os

from data_ml_functions.mlFunctions import load_cnn_batch
from data_ml_functions.dataFunctions import get_batch_inds
from multi_gpu_keras import multi_gpu_model

import time
from tqdm import tqdm
import keras.backend as K
import scipy.misc
import cv2
from data_ml_functions.iterm import show_image
import math
import random
import re

def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)

class FMOWBaseline:
    def __init__(self, params=None, argv=None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return: 
        """
        self.params = params

        np.random.seed(0)
        random.seed(0)
                
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

    def get_class_weights(self):
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

        return class_weights

    def train(self):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        
        trainData = json.load(open(self.params.files['training_struct']))
        metadataStats = json.load(open(self.params.files['dataset_stats']))

        loaded_filename = None
        if self.params.args.load_model:
            model = load_model(self.params.args.load_model)
            loaded_filename = os.path.basename(self.params.args.load_model)
        else:
            model = get_cnn_model(self.params)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True)
            loaded_filename = os.path.basename(self.params.args.load_weights)

        initial_epoch = 0
        if loaded_filename:
            match = re.search(r'.*epoch_(\d+).*\.hdf5', loaded_filename)
            if match:
                initial_epoch = int(match.group(1)) 

        if self.params.print_model_summary:
            model.summary()

        model = multi_gpu_model(model, gpus=self.params.gpus)

        import keras.losses
        keras.losses.focal_loss = focal_loss

        if self.params.loss == 'focal':
            loss = focal_loss
        else:
            loss = self.params.loss

        model.compile(optimizer=Adam(lr=self.params.learning_rate), loss=loss, metrics=['accuracy'])

        train_datagen = img_metadata_generator(self.params, trainData, metadataStats)
        
        preffix_pairs = [ \
            'd_' + self.params.directories_suffix, \
            'c_' + self.params.classifier , 'lr_' + str(self.params.learning_rate), \
            'b_' + str(self.params.batch_size), 'a_' + str(self.params.angle), 
            'freeze_' + str(self.params.freeze), \
            'w_' if self.params.weigthed else '', \
            'loss_' + self.params.loss if self.params.loss != 'categorical_crossentropy' else '', \
            'f_' if self.params.flips else '', 'w_' if self.params.weigthed else '' \
            ]

        preffix_pairs = [x for x in preffix_pairs if x != '']
        
        preffix = '-'.join(preffix_pairs)

        print("training single-image model: " + preffix)

        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], 
            preffix + '-epoch_' + '{epoch:02d}' + '-' + '-acc_' + '{acc:.4f}.hdf5')

        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]

        model.fit_generator(train_datagen,
            steps_per_epoch=int(math.ceil((len(trainData) / self.params.batch_size))),
            class_weight = self.get_class_weights() if self.params.weigthed else None,
            epochs=self.params.epochs, callbacks=callbacks_list, initial_epoch = initial_epoch)

        model.save(self.params.files['cnn_model'])

    def train_multi(self):
        """
        Train LSTM pipeline using pre-generated CNN codes.
        :param: 
        :return: 
        """

        codesTrainData = json.load(open(self.params.files['multi_training_struct']))
        codesStats = json.load(open(self.params.files['cnn_codes_stats']))
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

        initial_epoch = 0
        if loaded_filename:
            match = re.search(r'\w+\.(\d+)\.hdf5', loaded_filename)
            if match:
                initial_epoch = int(match.group(1)) 

        if self.params.print_model_summary:
            model.summary()

        model = multi_gpu_model(model, gpus=self.params.gpus)

        model.compile(optimizer=Adam(lr=self.params.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        train_datagen = codes_metadata_generator(self.params, codesTrainData, metadataStats, codesStats)
        
        print("training multi-image model: ")
        filePath = os.path.join(self.params.directories['multi_checkpoint_weights'], 'weights.{epoch:02d}.hdf5')

        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, 
            save_weights_only=False, mode='auto', period=1)
        
        callbacks_list = [checkpoint]

        model.fit_generator(train_datagen,
                            steps_per_epoch=int(math.ceil((len(codesTrainData) / self.params.batch_size))),
                            epochs=self.params.epochs, callbacks=callbacks_list,
                            max_queue_size=20)

        model.save(self.params.files['multi_model'])
        
    def test(self):
        metadataStats = json.load(open(self.params.files['dataset_stats']))
    
        metadataMean = np.array(metadataStats['metadata_mean'])
        metadataMax = np.array(metadataStats['metadata_max'])

        if self.params.args.load_model:
            model = load_model(self.params.args.load_model)
        else:
            model = get_cnn_model(self.params)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True)

        model = multi_gpu_model(model, gpus=self.params.gpus)

        index = 0
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        fidCNN = open(os.path.join(self.params.directories['predictions'], 'predictions-cnn-%s.txt' % timestr), 'w')
        
        for root, dirs, files in tqdm(os.walk(self.params.directories['test_data'])):
            if len(files) > 0:
                imgPaths = []
                metadataPaths = []
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']
                bbID = int(root[slashes[-1]+1:])
                
            for file in files:
                if file.endswith('.jpg'):
                    imgPaths.append(os.path.join(root,file))
                    metadataPaths.append(os.path.join(root, file[:-4]+'_features.json'))
                    
            if len(files) > 0:
                inds = []
                for metadataPath in metadataPaths:
                    underscores = [ind for ind,ltr in enumerate(metadataPath) if ltr == '_']
                    inds.append(int(metadataPath[underscores[-3]+1:underscores[-2]]))
                inds = np.argsort(np.array(inds)).tolist()
                
                tta_flip_v = tta_flip_h = self.params.flips

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
   
#                imgdata = imagenet_utils.preprocess_input(imgdata, mode='tf')
                #imgdata = imgdata / 255.0
                
                if self.params.use_metadata:
                    metadataFeatures = np.divide(metadataFeatures - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
                    predictionsCNN = np.sum(model.predict([imgdata, metadataFeatures], batch_size=currBatchSize), axis=0)
                else:
                    predictionsCNN = np.sum(model.predict(imgdata, batch_size=currBatchSize), axis=0)
                                
            if len(files) > 0:
                predCNN = np.argmax(predictionsCNN)
                oursCNNStr = self.params.category_names[predCNN]
                fidCNN.write('%d,%s\n' % (bbID,oursCNNStr))
                index += 1

        fidCNN.close()

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

        featuresModel = Model(model.inputs, model.layers[-6].output)

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

                imgdata[batchIndex,...] = img #image.img_to_array(image.load_img(currData['img_path']))

                # imgdata[batchIndex,:,:,:] = image.img_to_array(image.load_img(currData['img_path']))

                batchIndex += 1

                if batchIndex == batchSize:
                    imgdata = imagenet_utils.preprocess_input(imgdata)
                    imgdata = imgdata / 255.0

                    if self.params.use_metadata:
                        cnnCodes = featuresModel.predict([imgdata,metadataFeatures], batch_size=batchSize)
                    else:
                        cnnCodes = featuresModel.predict(imgdata, batch_size=batchSize)

                    for codeIndex,currCodes in enumerate(cnnCodes):
                        currBasePath = tmpBasePaths[codeIndex]
                        outFile = os.path.join(outDir, '%07d.json' % index)
                        index += 1
                        json.dump(currCodes.tolist(), open(outFile, 'w'))
                        if currBasePath not in codesData.keys():
                            codesData[currBasePath] = {}
                            codesData[currBasePath]['cnn_codes_paths'] = []
                            if self.params.use_metadata:
                                codesData[currBasePath]['metadata_paths'] = []
                            if isTrain:
                                codesData[currBasePath]['category'] = tmpCategories[codeIndex]
                        codesData[currBasePath]['cnn_codes_paths'].append(outFile)
                        if self.params.use_metadata:
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