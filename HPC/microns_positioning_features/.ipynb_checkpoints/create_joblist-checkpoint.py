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
pandas_names = ['rowId', 'compartment', 'x', 'y', 'z', 'radius', 'link']
compartments_analyze = ['Basal', 'Apical', 'Axonal']
compartment_dict = {0: 'Somatic', 1: 'Axonal', 2: 'Basal', 3: 'Apical', 4: 'undefined', 5: 'custom'}

cell_df = pd.read_csv(home_dir + '/saved_data/mouse_microns/cell_type_classification_filtered.csv')
i_cells = []
for i_cell in range(len(cell_df)):
    bodyId = cell_df.iloc[i_cell]['cell_id']
    cell_type = cell_df.iloc[i_cell]['cell_type']
    skel_file = home_dir + f'/saved_data/mouse_microns/cell_morphologies/{bodyId}.swc'
    mito_nodes_file = home_dir + f'/saved_data/mouse_microns/associated_nodes/{bodyId}_assoc_nodes.json'
    mito_file = home_dir + f'/saved_data/mouse_microns/microns_cells/{cell_type}_{bodyId}.csv'
    if isfile(mito_file) and isfile(mito_nodes_file) and isfile(skel_file):
        skel_compartment = pd.read_csv(skel_file, header = None, names = pandas_names, delimiter = ' ')['compartment'].to_numpy()
        bool_nodes = np.zeros(len(skel_compartment), dtype = bool)
        for section in compartments_analyze:
            i_section = np.where( section == np.array(list(compartment_dict.values())) )[0][0]
    
            bool_nodes[skel_compartment == i_section] = True
        if np.any(bool_nodes):
            i_cells.append(i_cell)

# Save info to a joblist file
log_file = 'joblist.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for i_cell in i_cells:
        f.write(f'module load miniconda; source activate flyem-stuff; python3 main.py --i_neuron={i_cell}\n')
