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

def main(_):
    row=_FLAGS.row
    warnings.filterwarnings("ignore") # ignore all warnings
    pre_bodyId, pre_type = all_bodyIds[row,[0,1]]
    post_types = con_utils.get_unique_post_types([pre_type], count_thresh = 0, put_LC_first = False, use_broad_names = False)[0]
    synapse_df = utils.get_synapse_df(pre_bodyId, 'pre', group_synapses = True)
    connecting_types = synapse_df['connecting_type'].to_numpy()
    num_connections, num_connections_on = np.zeros( (2,len(post_types)) , dtype=int)
    for i_synapse in range(len(synapse_df)):
        if np.any([post_type in post_types for post_type in connecting_types[i_synapse]]):
            mito_CS = voxel_utils.get_CrossSection_info(pre_bodyId, synapse_df.iloc[i_synapse][['x','y','z']].to_numpy(), 100, return_mito = True)[1]
            if mito_CS is not None:
                pre_on = np.any(mito_CS)
                for post_type in connecting_types[i_synapse]:
                    num_connections[ post_type == post_types ] += 1
                    num_connections_on[ post_type == post_types ] += int(pre_on)
    df = pd.DataFrame( data = np.array([num_connections, num_connections_on]), columns = post_types )
    df.to_csv(home_dir + f'/saved_data/all_mitoconnectome/{pre_bodyId}_{pre_type}.csv', index=False)

if __name__ == '__main__':
    app.run(main)
