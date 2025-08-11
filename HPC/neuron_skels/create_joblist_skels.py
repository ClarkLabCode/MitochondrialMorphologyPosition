#!/usr/bin/env python

import os
import importlib
from os.path import isfile
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

home_dir = config.home_dir
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()

unique_neuron_types, counts = np.unique( neuron_quality_np[:,1], return_counts = True )
unique_neuron_types = unique_neuron_types[ counts >= 10 ]
counts = counts[ counts >= 10 ] # only use LC neurons with at least 10 neurons
analyze_neurons = list( np.append( unique_neuron_types[-3:], unique_neuron_types[:-3] ) )

i_neurons = np.append(np.where( np.isin(all_bodyIds[:,1], analyze_neurons) )[0], 
                      np.where( np.array([neuron_type.startswith('KC') for neuron_type in all_bodyIds[:,1]]) )[0])

is_analyzed = np.ones( len(i_neurons), dtype=bool )
for ii, i_neuron in enumerate(i_neurons):
    bodyId, neuron_type = all_bodyIds[i_neuron, [0,1]]
    is_analyzed[ii] = isfile(f'{home_dir}/saved_neuron_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv')

to_be_analyzed = i_neurons[~is_analyzed] # np.where( ~is_analyzed )[0]

# Save info to a joblist file
log_file = 'joblist_skels.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for i_neuron in to_be_analyzed:
        f.write(f'module load miniconda; source activate flyem-stuff; python3 skels_main.py --i_neuron={i_neuron}\n')
