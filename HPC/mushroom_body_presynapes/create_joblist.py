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
analyze_neurons = config.analyze_neurons

all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()
analyze_types = []
for neuron_type in np.unique( all_bodyIds[:,1] ):
    if neuron_type.startswith('KC'):
        analyze_types.append( neuron_type )
rows = np.where( np.isin( all_bodyIds[:,1], analyze_types ) )[0]

# Save info to a joblist file
log_file = 'joblist.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for row in rows:
        pre_bodyId, pre_type = all_bodyIds[row,[0,1]]
        if not isfile(home_dir + f'/saved_data/MB_presynapses/{pre_bodyId}_{pre_type}.csv'):
            f.write(f'module load miniconda; source activate flyem-stuff; python3 main.py --row={row}\n')
