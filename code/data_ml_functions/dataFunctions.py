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
import os
import errno
import numpy as np
import string
import dateutil.parser as dparser
from PIL import Image
from sklearn.utils import class_weight
from keras.preprocessing import image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import warnings

import code
from iterm import show_image
import math
import scipy.misc
import cv2

def prepare_data(params):
    """
    Saves sub images, converts metadata to feature vectors and saves in JSON files, 
    calculates dataset statistics, and keeps track of saved files so they can be loaded as batches
    while training the CNN.
    :param params: global parameters, used to find location of the dataset and json file
    :return: 
    """

    # suppress decompression bomb warnings for Pillow
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    walkDirs = ['train', 'val', 'test']

    executor = ThreadPoolExecutor(max_workers=params.num_workers)
    futures = []
    paramsDict = vars(params)
    keysToKeep = ['image_format', 'target_img_size', 'metadata_length', 'category_names', 'context_factor']
    paramsDict = {keepKey: paramsDict[keepKey] for keepKey in keysToKeep}
    
    results = []
    for currDir in walkDirs:
        isTrain = (currDir == 'train') or (currDir == 'val')
        if isTrain:
            outDir = params.directories['train_data']
        else:
            outDir = params.directories['test_data']

        print('Looping through sequences in: ' + currDir)
        for it, (root, dirs, files) in enumerate(tqdm(os.walk(os.path.join(params.directories['dataset'], currDir)))):
            if len(files) > 0:
                slashes = [i for i,ltr in enumerate(root) if ltr == '/']
                        
            for file in files:
                if file.endswith('_rgb.json'): #skip _msrgb images
                    task = partial(_process_file, file, slashes, root, isTrain, outDir, params)
                    futures.append(executor.submit(task))

    print('Preprocessing all files...')
    results = []
    [results.extend(future.result()) for future in tqdm(futures)]
    allTrainFeatures = [np.array(r[0]) for r in results if r[0] is not None]
    
    metadataTrainSum = np.zeros(params.metadata_length)
    for features in allTrainFeatures:
        metadataTrainSum += features

    trainingData = [r[1] for r in results if r[1] is not None]
    trainCount = len(trainingData)
    testData = [r[2] for r in results if r[2] is not None]

    metadataMean = metadataTrainSum / trainCount
    metadataMax = np.zeros(params.metadata_length)
    for currFeat in allTrainFeatures:
        currFeat = currFeat - metadataMean
        for i in range(params.metadata_length):
            if abs(currFeat[i]) > metadataMax[i]:
                metadataMax[i] = abs(currFeat[i])
    for i in range(params.metadata_length):
        if metadataMax[i] == 0:
            metadataMax[i] = 1.0
    metadataStats = {}
    metadataStats['metadata_mean'] = metadataMean.tolist()
    metadataStats['metadata_max'] = metadataMax.tolist()
    json.dump(testData, open(params.files['test_struct'], 'w'))
    json.dump(trainingData, open(params.files['training_struct'], 'w'))
    json.dump(metadataStats, open(params.files['dataset_stats'], 'w'))

def _process_file(file, slashes, root, isTrain, outDir, params):
    """
    Helper for prepare_data that actually loads and resizes each image and computes
    feature vectors. This function is designed to be called in parallel for each file
    :param file: file to process
    :param slashes: location of slashes from root walk path
    :param root: root walk path
    :param isTrain: flag on whether or not the current file is from the train set
    :param outDir: output directory for processed data
    :param params: dict of the global parameters with only the necessary fields
    :return (allFeatures, allTrainResults, allTestResults)
    """
    noResult = [(None, None, None)]
    baseName = file[:-5]

    imgFile = baseName + '.' + params.image_format
        
    if not os.path.isfile(os.path.join(root, imgFile)):
        print(os.path.join(root, imgFile))
        return noResult

    jsonData = json.load(open(os.path.join(root, file)))
    if not isinstance(jsonData['bounding_boxes'], list):
        jsonData['bounding_boxes'] = [jsonData['bounding_boxes']]

    allResults = []
    img = None
    for bb in jsonData['bounding_boxes']:
        if isTrain:
            category = bb['category']
        box = bb['box']

        outBaseName = '%d' % bb['ID']
        if isTrain:
            outBaseName = ('%s_' % category) + outBaseName

        if isTrain:
            currOut = os.path.join(outDir, root[slashes[-3] + 1:], outBaseName)
        else:
            currOut = os.path.join(outDir, root[slashes[-2] + 1:], outBaseName)

        if not os.path.isdir(currOut):
            try:
                os.makedirs(currOut)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass

        featuresPath = os.path.join(currOut, baseName + '_features.json')
        imgPath = os.path.join(currOut, imgFile)

        if True:#not os.path.isfile(imgPath):

            if img is None:
                try:
                    img = scipy.misc.imread(os.path.join(root, imgFile))
                except:
                    print(os.path.join(root, imgFile))
                    return noResult

            x_size, y_size = box[2], box[3]
            x0, y0 = box[0], box[1]
            x1, y1 = min(x0 + x_size, img.shape[1]-1), min(y0 + y_size, img.shape[0]-1)

            x_side, y_side = x_size /2 , y_size /2

            # don't train on tiny boxes
            if x_size <= 2 or y_size <= 2:
                print("Tiny box @ " + file)
                continue

            x_center = x0 + x_side
            y_center = y0 + y_side

            _x0 = np.clip(x_center - x_side * params.context_factor, 0, img.shape[1]-1)
            _x1 = np.clip(x_center + x_side * params.context_factor, 0, img.shape[1]-1)
            _y0 = np.clip(y_center - y_side * params.context_factor, 0, img.shape[0]-1)
            _y1 = np.clip(y_center + y_side * params.context_factor, 0, img.shape[0]-1)

            src_points = np.float32([[_x0,_y0], [_x1, _y0], [_x1, _y1]])
            sq2 = 1.4142135624 
            patch_size   = int(math.ceil(params.target_img_size * sq2))
            patch_center = patch_size / 2
            dst_points = np.float32((
                [ patch_center - patch_size / (2 * sq2) , patch_center - patch_size / (2 * sq2) ], 
                [ patch_center + patch_size / (2 * sq2) , patch_center - patch_size / (2 * sq2) ], 
                [ patch_center + patch_size / (2 * sq2) , patch_center + patch_size / (2 * sq2) ])) 

            M   = cv2.getAffineTransform(src_points, dst_points)
            _img = cv2.warpAffine(img,M,(patch_size, patch_size), borderMode = cv2.BORDER_REFLECT_101).astype(np.float32)

            if False:
                show_image(_img)
                print(category)
                raw_input("Press it now")

            scipy.misc.imsave(imgPath, _img)

        features = json_to_feature_vector(params.metadata_length, jsonData)
        features = features.tolist()

        json.dump(features, open(featuresPath, 'w'))

        if isTrain:
            allResults.append((features, {"features_path": featuresPath, "img_path": imgPath, "category": params.category_names.index(category)}, None))
        else:
            allResults.append((None, None, {"features_path": featuresPath, "img_path": imgPath}))

    return allResults

def json_to_feature_vector(metadata_length, jsonData):
    """
    Generates feature vector for CNN fusion from metadata
    :param metadata_length: total number of metadata parameters being used
    :param jsonData: metadata from a JSON file
    :return features: numpy feature vector representation of the metadata
    """
    features = np.zeros(metadata_length, dtype=float)
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
    features[10] = float(jsonData['multi_resolution_dbl'])
    features[11] = float(jsonData['target_azimuth_dbl']) / 360.0
    features[12] = float(jsonData['sun_azimuth_dbl']) / 360.0
    features[13] = float(jsonData['sun_elevation_dbl']) / 90.0
    features[14] = float(jsonData['off_nadir_angle_dbl']) / 90.0
    features[15] = float(jsonData['bounding_boxes'][0]['box'][2])
    features[16] = float(jsonData['bounding_boxes'][0]['box'][3])
    features[17] = float(jsonData['img_width'])
    features[18] = float(jsonData['img_height'])
    features[19] = float(len(jsonData['approximate_wavelengths']))
    features[20] = float(date.weekday())
    
    return features

def flip_axis(x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
                  
def utm_to_xy(zone):
    """
    Converts UTM zone to x,y values between 0 and 1.
    :param zone: UTM zone (string)
    :return (x,y): values between 0 and 1
    """
    nums = range(1,61);
    letters = string.ascii_lowercase[2:-2]
    if len(zone) == 2:
        num = int(zone[0:1])
    else:
        num = int(zone[0:2])
    letter = zone[-1].lower()
    numIndex = nums.index(num)
    letterIndex = letters.index(letter)
    x = float(numIndex) / float(len(nums)-1)
    y = float(letterIndex) / float(len(letters)-1)
    return (x,y)

def get_batch_inds(batch_size, idx, N):
    """
    Generates an array of indices of length N
    :param batch_size: the size of training batches
    :param idx: data to split into batches
    :param N: Maximum size
    :return batchInds: list of arrays of data of length batch_size
    """
    batchInds = []
    idx0 = 0

    toProcess = True
    while toProcess:
        idx1 = idx0 + batch_size
        if idx1 > N:
            idx1 = N
            idx0 = idx1 - batch_size
            toProcess = False
        batchInds.append(idx[idx0:idx1])
        idx0 = idx1

    return batchInds

def calculate_class_weights(params):
    """
    Computes the class weights for the training data and writes out to a json file 
    :param params: global parameters, used to find location of the dataset and json file
    :return: 
    """
    
    counts = {}
    for i in range(0,params.num_labels):
        counts[i] = 0

    trainingData = json.load(open(params.files['training_struct']))

    ytrain = []
    for i,currData in enumerate(trainingData):
        ytrain.append(currData['category'])
        counts[currData['category']] += 1

    classWeights = class_weight.compute_class_weight('balanced', np.unique(ytrain), np.array(ytrain))

    with open(params.files['class_weight'], 'w') as json_file:
        json.dump(classWeights.tolist(), json_file)
    
    
