#!/usr/bin/env python

import os
import importlib
from os.path import isfile
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
analyze_neurons = config.vis_neurons
bool_neurons = np.all([np.isin(neuron_quality_np[:,1], analyze_neurons), np.any(neuron_quality_np[:,2:],axis=1)], axis=0)

i_neurons = []
for i_neuron in np.where( bool_neurons )[0]:
    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'
    mito_file = home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv'
    mesh_files = [ ~isfile(home_dir + f'/saved_data/neuron_meshes/{bodyId}_{neuron_type}_{arbor}.pkl') for arbor in ['axon', 'dendrite', 'connecting cable', 'cell body fiber', 'soma'] ]
    if isfile(skel_file) and isfile(mito_file) and np.any(mesh_files):
        i_neurons.append( i_neuron )

# Save info to a joblist file
log_file = 'joblist_neuron_mesh.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for i_neuron in i_neurons:
        f.write(f'module load miniconda; source activate flyem-stuff; python3 neuron_mesh_main.py --i_neuron={i_neuron}\n')
