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
jitter_strengths = np.logspace( np.log10(0.1), np.log10(10), 15) * 1000/8
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

def run_neuron(i_neuron):
    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'
    synapse_file = home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv'
    mito_file = home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv'
    has_arbors = np.any([ neuron_quality[f'has_{arbor}'].iloc[i_neuron] for arbor in ['axon', 'dendrite'] ])
    if has_arbors and isfile(skel_file) and isfile(synapse_file) and isfile(mito_file):
        for jitter_strength in config.jitter_strengths:
            if not isfile( home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_X_df_Jitter_{int(jitter_strength)}_{i_neuron}.csv' ):
                return True
    return False


# Save info to a joblist file
log_file = 'joblist.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for i_neuron in np.where( np.isin(neuron_quality_np[:,1], config.analyze_neurons) )[0]:
        if run_neuron(i_neuron):
            f.write(f'module load miniconda; source activate flyem-stuff; python3 main.py --i_neuron={i_neuron}\n')
