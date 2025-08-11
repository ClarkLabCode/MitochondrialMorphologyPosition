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

unique_neuron_types, counts = np.unique( neuron_quality_np[:,1], return_counts = True )
unique_neuron_types = unique_neuron_types[ counts >= 10 ]
counts = counts[ counts >= 10 ] # only use LC neurons with at least 10 neurons
analyze_neurons = list( np.append( unique_neuron_types[-3:], unique_neuron_types[:-3] ) )

is_analyzed = np.ones( len(neuron_quality), dtype=bool )
for i_neuron in np.where( np.isin(neuron_quality_np[:,1], analyze_neurons) )[0]:
    bodyId, neuron_type = neuron_quality_np[i_neuron, [0,1]]

    skel_file = f'{home_dir}/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'
    mito_file = f'{home_dir}/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv'
    synapse_file = f'{home_dir}/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv'
    if isfile(skel_file):
        is_analyzed[i_neuron] = isfile(mito_file) and isfile(synapse_file)
to_be_analyzed = np.where( ~is_analyzed )[0]

# Save info to a joblist file
log_file = 'joblist_mito_synapse.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for i_neuron in to_be_analyzed:
        f.write(f'module load miniconda; source activate flyem-stuff; python3 mito_synapse_main.py --i_neuron={i_neuron}\n')
