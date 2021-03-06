# alternative for params.py reading from command line
# note the duplicate stuff to make rest of of fmow code compatible as it was w/ params.py

import os
from multiprocessing import cpu_count
import argparse

parser = argparse.ArgumentParser()

# train 
parser.add_argument('-lw', '--load-weights', type=str, help='load model weights (and continue training)')
parser.add_argument('-lm', '--load-model',   type=str, help='load model (and continue training)')

#num_workers = cpu_count() 
parser.add_argument('--num-workers', type=int, default=cpu_count() , help='Workers for multi-thread generators')

#use_metadata = True
parser.add_argument('--no-metadata', action='store_true', help='Use metadata')

#batch_size_cnn = 48
#batch_size_multi = 64
#batch_size_eval = 128
parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')

#metadata_length = 21
parser.add_argument('--metadata_length', type=int, default=45, help='Metadata length')
#num_channels = 3
parser.add_argument('--num-channels', type=int, default=3, help='Number of channels (bands)')


#target_img_size = (299,299)
parser.add_argument('-i', '--image-size', type=int, default=224, help='Image size (side) in pixels (e.g. -i 299)')

#cnn_adam_learning_rate = 1e-4
parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='Initial learning rate, e.g. -l 1e-4')

#cnn_epochs = 15
parser.add_argument('--max-epoch', type=int, default=40, help='Epoch to run')

## 
parser.add_argument('-d', '--dir-suffix', type=str, default='_r224', help='Suffix for directory names')
parser.add_argument('--prepare', action='store_true', help='Prepare data')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('-gm', '--gpu-memory-frac', default=1., type=float, action='store', help='Use fraction of GPU memory (tensorflow only)')
parser.add_argument('-s', '--seed', default=0, type=int, action='store', help='Initial seed')

# ensembling
parser.add_argument('-e', '--ensemble', type=str, nargs='*', default=None, help='Generate predictions based on ensemble of multiple hkl files, e.g. -e *.hkl')
parser.add_argument('-em', '--ensemble-mean', type=str, default='arithmetic', help='Type of mean for averageing -em arithmetic|geometric')

parser.add_argument('--test', action='store_true', help='Evaluate model and generate predictions')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--loss', type=str, default='categorical_crossentropy', help='Loss function to use, i.e. categorical_crossentropy or focal')
parser.add_argument('-w', '--weigthed', action='store_true', help='Use weights for more important classes as per Scoring here: https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16996&pm=14684')
parser.add_argument('-pms', '--print-model-summary', action='store_true', help='Ditto')
parser.add_argument('--img-suffix', type=str, default='rgb', help='When doing --prepare use _rgb or _msrgb images, e.g. --img-suffix msrgb')
parser.add_argument('-lu', '--leave-unbalanced', action='store_true', help='Do not use class-aware sampling (i.e. leave dataset unbalanced as-is)')

parser.add_argument('-mm', '--mask-metadata', action='store_true', help='Mask some of the metadata attributes (e.g. minutes, bbox coords, etc.)')
parser.add_argument('-c', '--classifier', type=str, default=None, help='Base classifier to use -m InceptionResNetV2|SEInceptionResNetV2|Xception|densenet or lstm|...')
parser.add_argument('-ifp', '--image-format-processed', type=str, default='jpg', help='Image format for output --prepare (models will be trained/evaluated with that output) -ig jpg|png')
parser.add_argument('-nm', '--norm-metadata', action='store_true', help='Normalize (std 0, var 1) metadata')

# single specific
parser.add_argument('--generate-cnn-codes', action='store_true', help='Generate CNN codes')
parser.add_argument('-jc', '--jitter-channel',  type=float, default=0., help='Percentage to jitter image channels, e.g. -jc 0.1')
parser.add_argument('-jm', '--jitter-metadata', type=float, default=0., help='Percentage to jitter metadata, e.g. -jm 0.1')
parser.add_argument('-a', '--angle', type=int, default=0, help='Angle range for rotation augmentation, e.g. -a 360')
parser.add_argument('-o', '--offset', type=float, default=0., help='Offset percentage to take a RANDOM crop relative to image size, e.g. -o 0.2 means 20 per cent of image size, e.g. 112 * 0.2 = 22.4 pixels')
parser.add_argument('-z', '--zoom', type=float, default=0., help='Zoom percentage to take a RANDOM crop relative to image size, e.g. -z 0.2 means 20 per cent of image size, e.g. 112 * 0.2 = 22.4 pixels')
parser.add_argument('-cf', '--context-factor', type=float, default=1.5, help='Context around bound box selection, e.g. -cf 1 (no context, just bb), -cf 2 (effectively doubles size of bb)')
parser.add_argument('-f', '--flips', action='store_true', help='Use horizontal/vertical flips augmentation (same as -fns -few)')
parser.add_argument('-fns', '--flip-north-south', action='store_true', help='Use north/south flip augmentation')
parser.add_argument('-few', '--flip-east-west', action='store_true', help='Use east-west flip augmentation')
parser.add_argument('--freeze', type=int, default=0, help='Freeze first n CNN layers, e.g. --freeze 10')
parser.add_argument('--amsgrad', action='store_true', help='Use amsgrad with Adam optimizer (requires Keras 2.1.3)')
parser.add_argument('--no-imagenet', action='store_true', help='Do NOT use imagenet-trained weights to init model')
parser.add_argument('--pooling', default='avg', help='Pooling to use for feature extraction, e.g. --pooling avg|max')

parser.add_argument('-v', '--views', default=0, type=int, help='Number of views to use in multi-view model (defaults to single-view if not specified)')
parser.add_argument('-vm', '--view-model', default='lstm2d', help='Submodel for multi-view: -vm lstm2d|conv3d')

# multi specific
parser.add_argument('-m', '--multi', action='store_true', help='Use multi model')
parser.add_argument('-td', '--temporal-dropout', type=float, default=0.3, help='Percentage of images to drop while training multi-image model')
parser.add_argument('-mt', '--max-temporal', type=int, default=0, help='If specified caps max number of frames for multi-image model')

args = parser.parse_args()

cnn_last_layer_length  = 4096
cnn_multi_layer_length = 1536 # 2208

if args.classifier == None:
	args.classifier = 'InceptionResNetV2' if not args.multi else 'lstm'
	
# PARSED PARAMS
gpus = args.gpus
num_workers = args.num_workers
use_metadata = not args.no_metadata
batch_size = args.batch_size * gpus
metadata_length = args.metadata_length
num_channels = args.num_channels
target_img_size = args.image_size
learning_rate = args.learning_rate
epochs = args.max_epoch
directories_suffix = args.dir_suffix
loss = args.loss
angle = args.angle
context_factor = args.context_factor
weigthed = args.weigthed
classifier = args.classifier
flip_north_south = args.flip_north_south or args.flips
flip_east_west = args.flip_east_west or args.flips
freeze = args.freeze
print_model_summary = args.print_model_summary
multi = args.multi
leave_unbalanced = args.leave_unbalanced
mask_metadata = args.mask_metadata
temporal_dropout = args.temporal_dropout
max_temporal = args.max_temporal
image_format_processed = args.image_format_processed
image_format_dataset = 'jpg'
norm_metadata = args.norm_metadata
amsgrad = args.amsgrad
no_imagenet = args.no_imagenet
pooling = args.pooling
views = args.views
view_model = args.view_model
offset = args.offset
zoom = args.zoom
ensemble = args.ensemble
ensemble_mean = args.ensemble_mean
jitter_channel = args.jitter_channel
jitter_metadata = args.jitter_metadata
gpu_memory_frac = args.gpu_memory_frac
seed = args.seed

#DIRECTORIES AND FILES
directories = {}
directories['dataset'] = '../../fmow_dataset'
directories['input'] = os.path.join('..', 'data', 'input' + directories_suffix)
directories['output'] = os.path.join('..', 'data', 'output' + directories_suffix)
directories['working'] = os.path.join('..', 'data', 'working' + directories_suffix)
directories['train_data'] = os.path.join(directories['input'], 'train_data')
directories['test_data'] = os.path.join(directories['input'], 'test_data')
directories['cnn_models'] = os.path.join(directories['working'], 'cnn_models')
directories['predictions'] = os.path.join(directories['output'], 'predictions')
directories['cnn_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_checkpoint_weights')
directories['multi_checkpoint_weights'] = os.path.join(directories['working'], 'multi_checkpoint_weights')

directories['cnn_codes'] = os.path.join(directories['working'], 'cnn_codes')

files = {}
files['training_struct'] = os.path.join(directories['working'], 'training_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')
files['dataset_stats'] = os.path.join(directories['working'], 'dataset_stats.json')
files['class_weight'] = os.path.join(directories['working'], 'class_weights.json')
#files['cnn_model'] = os.path.join(directories['cnn_checkpoint_weights'], 'weights.02.hdf5')

#
category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

num_labels = len(category_names)

for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)