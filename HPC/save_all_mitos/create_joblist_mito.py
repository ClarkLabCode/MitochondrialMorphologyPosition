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
all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()

good_neurons = []
for i_neuron in np.where( all_bodyIds[:,-1] == False )[0]:
    bodyId, neuron_type = all_bodyIds[i_neuron, [0,1]]
    if not isfile( home_dir + f'/saved_data/saved_mito_df_all/{neuron_type}_{bodyId}_mito_df.csv' ):
        good_neurons.append( i_neuron )
good_neurons = np.array(good_neurons)

# Save info to a joblist file
for i_joblist in range( int( np.ceil(len(good_neurons) / 5000) ) ):
    init = i_joblist * 5000
    final = np.min([ len(good_neurons), (i_joblist+1)*5000 ])
    log_file = f'joblist_mito_synapse_{i_joblist}.txt'
    with open(log_file, 'w') as f:
        f.truncate()

        # go through all uncropped neurons
        for i_neuron in good_neurons[ init:final ]:
            f.write(f'module load miniconda; source activate flyem-stuff; python3 mito_main.py --i_neuron={i_neuron}\n')
