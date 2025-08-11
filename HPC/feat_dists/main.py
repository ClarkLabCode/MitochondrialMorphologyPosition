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
import time
from os import listdir
import importlib
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from os.path import isfile
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.cluster.hierarchy as sch
import os

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

import warnings
warnings.filterwarnings("ignore")

_FLAGS = flags.FLAGS
flags.DEFINE_integer('i_type', None, 'ith neuron type in analyze_neurons', lower_bound=0)
flags.mark_flags_as_required(['i_type'])

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
analyze_neurons = config.analyze_neurons

# import GLM_utils file
spec = importlib.util.spec_from_file_location('GLM_utils', home_dir+'/util_files/GLM_utils.py')
GLM_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GLM_utils)

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def append_dists(dist_matrix, cur_list, dist_thresh):
    for dists in dist_matrix:
        cur_list.append( dists[ dists < dist_thresh ] )

def main(_):
    i_type=_FLAGS.i_type
    warnings.filterwarnings("ignore") # ignore all warnings
    neuron_type = analyze_neurons[i_type]

    bodyId_type_arbor_measured = []
    presynapse_dists = []
    postsynapse_dists = []
    branch_dists = []
    mito_dists = []
    dist_thresh = 10
    for i_arbor, arbor in enumerate(['dendrite', 'axon']):
        this_dx = []
        for i_neuron in np.where(neuron_quality_np[:,1] == neuron_type)[0]:
            bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
            synapse_file = home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv'
            if neuron_quality[f'has_{arbor}'].iloc[i_neuron] and isfile(synapse_file):# and arbor == 'axon':
                # this neuron contains arbor
                s_pandas = pd.read_csv( home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv' )
                node_classes = s_pandas['node_classes'].to_numpy()
                s_np = s_pandas.to_numpy().astype(float)
                branch_nodes = utils.find_leaves_and_branches(s_np)[1]
                synapse_df = pd.read_csv(synapse_file)
                arbor_synapse_df = synapse_df.iloc[ np.where(node_classes[utils.find_closest_idxs(s_np, synapse_df)] == node_class_dict[arbor])[0] ]
    
                mito_df = pd.read_csv(home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv')
                arbor_mito_df = mito_df.iloc[ np.where(node_classes[utils.find_closest_idxs(s_np, mito_df)] == node_class_dict[arbor])[0] ]
    
                branch_idxs = np.where(np.isin(s_np[:,0], branch_nodes))[0]
    
                if len(arbor_mito_df) > 1 and len(arbor_synapse_df) > 1:
                    shuffled_mito_df = utils.get_shuffled_mito_df(bodyId, arbor)
                    all_mito_df = pd.concat([shuffled_mito_df, arbor_mito_df])
                    
                    mito_presynapse_dists  = utils.get_mito_object_dists(all_mito_df, arbor_synapse_df[ arbor_synapse_df['type'] == 'pre' ], s_np) * 8/1000
                    append_dists(mito_presynapse_dists, presynapse_dists, dist_thresh)
                    
                    mito_postsynapse_dists = utils.get_mito_object_dists(all_mito_df, arbor_synapse_df[ arbor_synapse_df['type'] == 'post'], s_np) * 8/1000
                    append_dists(mito_postsynapse_dists, postsynapse_dists, dist_thresh)
                    
                    if len(branch_idxs) > 0:
                        branch_coord_df = pd.DataFrame(data = s_np[branch_idxs,:][:,[1,2,3]], columns = ['x','y','z'])
                        mito_branch_dists = utils.get_mito_object_dists(all_mito_df, branch_coord_df, s_np) * 8/1000
                    else:
                        mito_branch_dists = np.ones( (len(all_mito_df),1) ) * np.inf
                    append_dists(mito_branch_dists, branch_dists, dist_thresh)
    
                    for i_mito_df, this_mito_df in enumerate([shuffled_mito_df, arbor_mito_df]):
                        mito_mito_dists = utils.get_mito_object_dists(this_mito_df, this_mito_df, s_np) * 8/1000
                        append_dists(mito_mito_dists, mito_dists, dist_thresh)
    
                    for Y in np.append( np.zeros(len(shuffled_mito_df), dtype=bool), np.ones(len(arbor_mito_df), dtype=bool) ):
                        bodyId_type_arbor_measured.append( [bodyId, neuron_type, arbor, Y] )
    
                    print(len(bodyId_type_arbor_measured), len(mito_dists), len(branch_dists), len(postsynapse_dists), len(presynapse_dists))
    with open(home_dir + f'/saved_data/saved_feat_dists/{neuron_type}_bodyId_type_arbor_measured.pkl', 'wb') as f:
        pickle.dump(bodyId_type_arbor_measured, f)
    
    with open(home_dir + f'/saved_data/saved_feat_dists/{neuron_type}_presynapse_dists.pkl', 'wb') as f:
        pickle.dump(presynapse_dists, f)
    
    with open(home_dir + f'/saved_data/saved_feat_dists/{neuron_type}_postsynapse_dists.pkl', 'wb') as f:
        pickle.dump(postsynapse_dists, f)
    
    with open(home_dir + f'/saved_data/saved_feat_dists/{neuron_type}_branch_dists.pkl', 'wb') as f:
        pickle.dump(branch_dists, f)
    
    with open(home_dir + f'/saved_data/saved_feat_dists/{neuron_type}_mito_dists.pkl', 'wb') as f:
        pickle.dump(mito_dists, f)

if __name__ == '__main__':
    app.run(main)
















