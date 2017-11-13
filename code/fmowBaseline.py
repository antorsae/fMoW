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
from data_ml_functions.mlFunctions import get_cnn_model,img_metadata_generator, rect_coords, rotate, enclosing_rect
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
                
        if self.params.use_metadata:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_with_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_with_metadata.json')
        else:
            self.params.files['cnn_model'] = os.path.join(self.params.directories['cnn_models'], 'cnn_model_no_metadata.model')
            self.params.files['cnn_codes_stats'] = os.path.join(self.params.directories['working'], 'cnn_codes_stats_no_metadata.json')
    
    def train(self):
        """
        Train CNN with or without metadata depending on setting of 'use_metadata' in params.py.
        :param: 
        :return: 
        """
        
        trainData = json.load(open(self.params.files['training_struct']))
        metadataStats = json.load(open(self.params.files['dataset_stats']))

        if self.params.args.load_model:
            model = load_model(params.args.load_model)
        else:
            model = get_cnn_model(self.params)

        if self.params.args.load_weights:
            model.load_weights(self.params.args.load_weights, by_name=True)

        model = multi_gpu_model(model, gpus=self.params.gpus)

        #import keras.losses
        #keras.losses.focal_loss = focal_loss

        model.compile(optimizer=Adam(lr=self.params.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        train_datagen = img_metadata_generator(self.params, trainData, metadataStats)
        
        print("training")
        filePath = os.path.join(self.params.directories['cnn_checkpoint_weights'], 'weights.{epoch:02d}.hdf5')

        checkpoint = ModelCheckpoint(filepath=filePath, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        callbacks_list = [checkpoint]

        model.fit_generator(train_datagen,
            steps_per_epoch=int(math.ceil((len(trainData) / self.params.batch_size))),
            epochs=self.params.epochs, callbacks=callbacks_list)

        model.save(self.params.files['cnn_model'])
        
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
                
                tta_flip_v = tta_flip_h = True

                currBatchSize = len(inds) * (2 if tta_flip_v else 1) * (2 if tta_flip_h else 1)
                imgdata = np.zeros((currBatchSize, self.params.target_img_size, self.params.target_img_size, self.params.num_channels))
                metadataFeatures = np.zeros((currBatchSize, self.params.metadata_length))
                    
                for ind in inds:
                    img = scipy.misc.imread(imgPaths[ind]) #image.load_img(imgPaths[ind])

                    features = np.array(json.load(open(metadataPaths[ind])))

                    crop_size = self.params.target_img_size
                    x0 = int(img.shape[1]/2 - crop_size/2)
                    x1 = x0 + crop_size
                    y0 = int(img.shape[0]/2 - crop_size/2)
                    y1 = y0 + crop_size

                    img = img[y0:y1,x0:y1,...]

                    #show_image(img)
                    #raw_input("press enter")

                    metadataFeatures[ind,:] = features

                    tta_idx = len(inds) + ind
                    if tta_flip_v:
                        imgdata[tta_idx,...] = flip_axis(img, 0)
                        metadataFeatures[tta_idx,:] = features
                        tta_idx += len(inds)

                    if tta_flip_h:
                        imgdata[tta_idx,...] = flip_axis(img, 1)
                        metadataFeatures[tta_idx,:] = features
                        tta_idx += len(inds)

                        if tta_flip_v:
                            imgdata[tta_idx,...] = flip_axis(flip_axis(img, 1), 0)
                            metadataFeatures[tta_idx,:] = features
                            tta_idx += len(inds)
   
                imgdata = imagenet_utils.preprocess_input(imgdata, mode='tf')
                #imgdata = imgdata / 255.0
                
                if self.params.use_metadata:
                    predictionsCNN = np.sum(model.predict([imgdata, metadataFeatures], batch_size=currBatchSize), axis=0)
                else:
                    predictionsCNN = np.sum(model.predict(imgdata, batch_size=currBatchSize), axis=0)
                                
            if len(files) > 0:
                predCNN = np.argmax(predictionsCNN)
                oursCNNStr = self.params.category_names[predCNN]
                fidCNN.write('%d,%s\n' % (bbID,oursCNNStr))
                index += 1
                
        fidCNN.close()        
                    
    
    
