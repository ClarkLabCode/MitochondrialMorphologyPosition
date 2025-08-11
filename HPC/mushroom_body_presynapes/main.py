#!/usr/bin/env python
from absl import app
from absl import flags

# % matplotlib inline
from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_mitochondria
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
import importlib
import random
from os.path import isfile
import pickle
import os
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore") # ignore all warnings

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

_FLAGS = flags.FLAGS
flags.DEFINE_integer('row', None, 'row in all_bodyIds', lower_bound=0)
flags.mark_flags_as_required(['row'])

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict
all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()

# import con_utils file
spec = importlib.util.spec_from_file_location('con_utils', home_dir+'/util_files/connectivity_utils.py')
con_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(con_utils)

# import voxel_utils file
spec = importlib.util.spec_from_file_location('voxel_utils', home_dir+'/util_files/voxel_utils.py')
voxel_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voxel_utils)

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

cols = ['type', 'confidence', 'x', 'y', 'z', 'roi_compartment', 'connecting_type', 'connecting_bodyId', 'is_on']

pos_val_compartments = ["a'3(R)", "a2(R)", "a'1(R)", "g1(R)", "g2(R)", "g3(R)"]
neg_val_compartments = ["g4(R)", "g5(R)", "b'2(R)"]
#compartments = np.append(pos_val_compartments, neg_val_compartments)
compartments = np.array(["a1(R)", "a2(R)", "a3(R)", "a'1(R)", "a'2(R)", "a'3(R)", "b'1(R)", "b'2(R)",
                         "b1(R)", "b2(R)","g1(R)", "g2(R)", "g3(R)", "g4(R)", "g5(R)"])

def main(_):
    row=_FLAGS.row
    warnings.filterwarnings("ignore") # ignore all warnings
    
    bodyId, neuron_type = all_bodyIds[row,[0,1]]
    coords_roi = fetch_synapses(NC(bodyId=bodyId), SC(primary_only=False, type='pre'))[['x','y','z','roi']]
    coords_roi = coords_roi.iloc[ np.where( np.isin(coords_roi['roi'].to_numpy(), compartments) )[0] ]

    synapse_df = utils.get_synapse_df( bodyId, 'pre')
    dist_matrix = cdist(synapse_df[['x','y','z']].to_numpy(), coords_roi[['x','y','z']].to_numpy())
    i_closest_syns = np.argmin(dist_matrix,axis=1)
    keep_syns = np.array([ dist_matrix[row,col] == 0 for row,col in enumerate(i_closest_syns)])
    i_closest_syns = i_closest_syns[keep_syns]
    synapse_df = synapse_df.iloc[ np.where(keep_syns)[0] ]
    synapse_df['roi_compartment'] = coords_roi['roi'].to_numpy()[i_closest_syns]

    syn_coords = synapse_df[['x','y','z']].to_numpy()
    feat_vec = []
    for i_synapse in range(len(synapse_df)):
        mito_CS = voxel_utils.get_CrossSection_info(bodyId, syn_coords[i_synapse], 50, return_mito = True)[1]
        if mito_CS is not None:
            for i_post in range(len(synapse_df.iloc[i_synapse]['connecting_type'])):
                post_type = synapse_df.iloc[i_synapse]['connecting_type'][i_post]
                post_bodyId = synapse_df.iloc[i_synapse]['connecting_bodyId'][i_post]
                feat_vec.append( np.append(synapse_df.iloc[i_synapse][ ['type', 'confidence', 'x', 'y', 'z', 'roi_compartment'] ].to_numpy(), 
                                            np.array([post_type, post_bodyId, np.any(mito_CS)], dtype=object)) )
    final_synapse_df = pd.DataFrame(np.array(feat_vec), columns = cols)
    final_synapse_df.to_csv(home_dir + f'/saved_data/MB_presynapses/{bodyId}_{neuron_type}.csv', index=False)
    
if __name__ == '__main__':
    app.run(main)
