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
from data_ml_functions.mlFunctions import get_cnn_model,img_metadata_generator,get_lstm_model,codes_metadata_generator
from data_ml_functions.dataFunctions import prepare_data,calculate_class_weights, flip_axis
import numpy as np
import os

from data_ml_functions.mlFunctions import load_cnn_batch
from data_ml_functions.dataFunctions import get_batch_inds
#from data_ml_functions.multi_gpu import make_parallel
#from keras.utils.training_utils import multi_gpu_model
from multi_gpu_keras import multi_gpu_model

import time
from tqdm import tqdm

import keras.backend as K

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
        
        for arg in argv:
            if arg == '-prepare':
                prepare_data(params)
                calculate_class_weights(params)
            performAll = (arg == '-all')
            if performAll or arg == '-cnn':
                self.params.train_cnn = True
            if performAll or arg == '-codes':
                self.params.generate_cnn_codes = True
            if performAll or arg == '-lstm':
                self.params.train_lstm = True
            if performAll or arg == '-test':
                self.params.test_cnn = True
                self.params.test_lstm = True
            if performAll or arg == '-test_cnn':
                self.params.test_cnn = True
            if performAll or arg == '-test_lstm':
                self.params.test_lstm = True
            
            if arg == '-nm':
                self.params.use_metadata = False
                
        if self.params.use_metadata:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_with_metadata.model')
            self.params.files['lstm_model'] = os.path.join(self.params.directories['lstm_models'], 'lstm_model_with_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_with_metadata.json')
            self.params.files['lstm_training_struct'] = os.path.join(self.params.directories['working'], 'lstm_training_struct_with_metadata.json')
        else:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_no_metadata.model')
            self.params.files['lstm_model'] = os.path.join(self.params.directories['lstm_models'], 'lstm_model_no_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_no_metadata.json')
            self.params.files['lstm_training_struct'] = os.path.join(self.params.directories['working'], 'lstm_training_struct_no_metadata.json')
    
    def train_cnn(self):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        
        trainData = json.load(open(self.params.files['training_struct']))

        metadataStats = json.load(open(self.params.files['dataset_stats']))

        model = get_cnn_model(self.params)
        #model = load_model("test.hdf5")
        #model.summary()
        model.load_weights('../data/working-fixed/cnn_checkpoint_weights/weights.01.hdf5', by_name=True)
        model = multi_gpu_model(model, gpus=2)

        import keras.losses
        keras.losses.focal_loss = focal_loss

        model.compile(optimizer=Adam(lr=self.params.cnn_adam_learning_rate), loss=focal_loss, metrics=['accuracy'])
        #model.save("test.hdf5")
        
#        classWeights = np.array(json.load(open(self.params.files['class_weight'])))

        train_datagen = img_metadata_generator(self.params, trainData, metadataStats)
        
        print("training")
        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], 'weights.{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]

        #import copy
        #_config =  model.get_config()
        #print(_config)
        #_config_copy = copy.deepcopy(_config)

        model.fit_generator(train_datagen,
            steps_per_epoch=(len(trainData) / self.params.batch_size_cnn + 1),
            epochs=self.params.cnn_epochs, callbacks=callbacks_list)
#            epochs=self.params.cnn_epochs, class_weight=classWeights, callbacks=callbacks_list)

        model.save(self.params.files['cnn_model'])
        
    def train_lstm(self):
        """
        Train LSTM pipeline using pre-generated CNN codes.
        :param: 
        :return: 
        """
        codesTrainData = json.load(open(self.params.files['lstm_training_struct']))
        codesStats = json.load(open(self.params.files['cnn_codes_stats']))
        metadataStats = json.load(open(self.params.files['dataset_stats']))
        
        model = get_lstm_model(self.params, codesStats)
        #model = make_parallel(model, 4)
        model.compile(optimizer=Adam(lr=self.params.lstm_adam_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

#        classWeights = np.array(json.load(open(self.params.files['class_weight'])))

        train_datagen = codes_metadata_generator(self.params, codesTrainData, metadataStats, codesStats)
        
        print("training")
        filePath = os.path.join(self.params.directories['lstm_checkpoint_weights'], 'weights.{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        callbacks_list = [checkpoint]

        model.fit_generator(train_datagen,
            steps_per_epoch=(len(codesTrainData) / self.params.batch_size_lstm + 1),
            epochs=self.params.lstm_epochs, callbacks=callbacks_list)
#            epochs=self.params.lstm_epochs, class_weight=classWeights, callbacks=callbacks_list)

        model.save(self.params.files['lstm_model'])
    
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
        model = load_model(self.params.files['cnn_model'])
#        model = get_cnn_model(self.params)
#        model.load_weights('../data/working/cnn_checkpoint_weights/weights.14.hdf5')
        
        featuresModel = Model(model.input, model.layers[-3].output)

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
            for i,currData in enumerate(data):
                if initBatch:
                    if N-i < self.params.batch_size_eval:
                        batchSize = 1
                    else:
                        batchSize = self.params.batch_size_eval
                    imgdata = np.zeros((batchSize, self.params.target_img_size[0], self.params.target_img_size[1], self.params.num_channels))
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
                imgdata[batchIndex,:,:,:] = image.img_to_array(image.load_img(currData['img_path']))

                batchIndex += 1

                print(i)

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
            maxCodes = np.maximum(maxCodes, currCodes-avgCodes)

        maxTemporal = 0
        for key in codesTrainData.keys():
            currTemporal = len(codesTrainData[key]['cnn_codes_paths'])
            if currTemporal > maxTemporal:
                maxTemporal = currTemporal
                
        codesStats = {}
        codesStats['codes_mean'] = avgCodes.tolist()
        codesStats['codes_max'] = maxCodes.tolist()
        codesStats['max_temporal'] = maxTemporal

        json.dump(codesTrainData, open(self.params.files['lstm_training_struct'], 'w'))
        json.dump(codesStats, open(self.params.files['cnn_codes_stats'], 'w'))
        
        
    def test_models(self):
        metadataStats = json.load(open(self.params.files['dataset_stats']))
    
        metadataMean = np.array(metadataStats['metadata_mean'])
        metadataMax = np.array(metadataStats['metadata_max'])

        #cnnModel = load_model(self.params.files['cnn_model'])
        cnnModel = get_cnn_model(self.params)
        #cnnModel = multi_gpu_model(cnnModel, gpus=2)

        #cnnModel = make_parallel(cnnModel, 4)
        cnnModel.load_weights('../data/working-fixed/cnn_checkpoint_weights/weights.08.hdf5')
        #cnnModel = multi_gpu_model(cnnModel, gpus=2)
        #cnnModel = cnnModel.layers[-2]
        
        if self.params.test_lstm:
            codesStats = json.load(open(self.params.files['cnn_codes_stats']))
            featuresModel = Model(cnnModel.input, cnnModel.layers[-3].output)
            lstmModel = load_model(self.params.files['lstm_model'])
#            lstmModel = get_lstm_model(self.params, codesStats)
#            lstmModel.load_weights('../data/working/lstm_checkpoint_weights/weights.14.hdf5')
            

        index = 0
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        
        if self.params.test_cnn:
            fidCNN = open(os.path.join(self.params.directories['predictions'], 'predictions-cnn-%s.txt' % timestr), 'w')
        if self.params.test_lstm:
            fidLSTM = open(os.path.join(self.params.directories['predictions'], 'predictions-lstm-%s.txt' % timestr), 'w')
        
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
                
                tta_flip_v = tta_flip_h = True

                currBatchSize = len(inds) * (2 if tta_flip_v else 1) * (2 if tta_flip_h else 1)
                imgdata = np.zeros((currBatchSize, self.params.target_img_size[0], self.params.target_img_size[1], self.params.num_channels))
                metadataFeatures = np.zeros((currBatchSize, self.params.metadata_length))
                    
                for ind in inds:
                    img = image.load_img(imgPaths[ind])
                    img = image.img_to_array(img)
                    img.setflags(write=True)
                    imgdata[ind,:,:,:] = img

                    features = np.array(json.load(open(metadataPaths[ind])))
                    features = np.divide(features - metadataMean, metadataMax)
                    metadataFeatures[ind,:] = features

                    tta_idx = len(inds) + ind
                    if tta_flip_v:
                        imgdata[tta_idx,:,:,:] = flip_axis(img, 0)
                        metadataFeatures[tta_idx,:] = features
                        tta_idx += len(inds)

                    if tta_flip_h:
                        imgdata[tta_idx,:,:,:] = flip_axis(img, 1)
                        metadataFeatures[tta_idx,:] = features
                        tta_idx += len(inds)

                        if tta_flip_v:
                            imgdata[tta_idx,:,:,:] = flip_axis(flip_axis(img, 1), 0)
                            metadataFeatures[tta_idx,:] = features
                            tta_idx += len(inds)
   
                imgdata = imagenet_utils.preprocess_input(imgdata, mode='tf')
                #imgdata = imgdata / 255.0
                
                if self.params.test_cnn:
                    if self.params.use_metadata:
                        predictionsCNN = np.sum(cnnModel.predict([imgdata, metadataFeatures], batch_size=currBatchSize), axis=0)
                    else:
                        predictionsCNN = np.sum(cnnModel.predict(imgdata, batch_size=currBatchSize), axis=0)
                
                if self.params.test_lstm:
                    
                    if self.params.use_metadata:
                        codesMetadata = np.zeros((1, codesStats['max_temporal'], self.params.cnn_last_layer_length+self.params.metadata_length))
                        currFeatures = featuresModel.predict([imgdata, metadataFeatures], batch_size=currBatchSize)
                    else:
                        codesMetadata = np.zeros((1, codesStats['max_temporal'], self.params.cnn_last_layer_length))
                        currFeatures = featuresModel.predict(imgdata, batch_size=currBatchSize)
                    
                    for codesIndex in range(currBatchSize):
                        metadata = metadataFeatures[codesIndex,:]
                        cnnCodes = currFeatures[codesIndex,:]
                        if self.params.use_metadata:
                            codesMetadata[0,codesIndex,0:self.params.metadata_length] = metadata
                            codesMetadata[0,codesIndex,self.params.metadata_length:] = cnnCodes
                        else:
                            codesMetadata[0,codesIndex,:] = cnnCodes
                    predictionsLSTM = lstmModel.predict(codesMetadata, batch_size=1)
                
            if len(files) > 0:
                if self.params.test_cnn:
                    predCNN = np.argmax(predictionsCNN)
                    oursCNNStr = self.params.category_names[predCNN]
                    fidCNN.write('%d,%s\n' % (bbID,oursCNNStr))
                if self.params.test_lstm:
                    predLSTM = np.argmax(predictionsLSTM)
                    oursLSTMStr = self.params.category_names[predLSTM]
                    fidLSTM.write('%d,%s\n' % (bbID,oursLSTMStr))
                index += 1
                #print(index)
                
        if self.params.test_cnn:            
            fidCNN.close()
        if self.params.test_lstm:
            fidLSTM.close()
        
                    
    
    
