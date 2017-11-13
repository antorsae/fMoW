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
#batch_size_lstm = 64
#batch_size_eval = 128
parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')

#metadata_length = 21
parser.add_argument('--metadata_length', type=int, default=21, help='Metadata length')
#num_channels = 3
parser.add_argument('--num-channels', type=int, default=3, help='Number of channels (bands)')


#target_img_size = (299,299)
parser.add_argument('-i', '--image-size', type=int, default=299, help='Image size (side) in pixels (e.g. -i 299)')

#cnn_adam_learning_rate = 1e-4
parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='Initial learning rate, e.g. -l 1e-4')

#cnn_epochs = 15
parser.add_argument('--max-epoch', type=int, default=20, help='Epoch to run')

## 
parser.add_argument('--dir-suffix', type=str, default='-rotready2', help='Suffix for directory names')
parser.add_argument('--prepare', action='store_true', help='Prepare data')
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('--test', action='store_true', help='Evaluate model and generate predictions')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')

args = parser.parse_args()

image_format = 'jpg'
	
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

directories['cnn_codes'] = os.path.join(directories['working'], 'cnn_codes')

files = {}
files['training_struct'] = os.path.join(directories['working'], 'training_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')
files['dataset_stats'] = os.path.join(directories['working'], 'dataset_stats.json')
files['class_weight'] = os.path.join(directories['working'], 'class_weights.json')
files['cnn_model'] = os.path.join(directories['cnn_checkpoint_weights'], 'weights.02.hdf5')

#
category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

num_labels = len(category_names)

for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)