"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory LLC
and Andres Torrubia
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

__author__ = 'jhuapl, antor'
__version__ = 0.1

import sys

from fmowBaseline import FMOWBaseline
import params_args as params
from data_ml_functions.dataFunctions import prepare_data, calculate_class_weights

def main(argv):
    baseline = FMOWBaseline(params, argv)

    if params.args.prepare:
        prepare_data(params)
        calculate_class_weights(params)

    if params.args.train:
        if params.args.multi:
            baseline.train_multi()
        else:
            baseline.train()

    if params.args.generate_cnn_codes:
        baseline.generate_cnn_codes()

    if params.args.test:
        baseline.test()

    if params.args.ensemble:
        baseline.ensemble()
    
if __name__ == "__main__":
    main(sys.argv[1:])
