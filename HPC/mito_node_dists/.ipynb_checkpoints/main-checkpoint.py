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
import pickle

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

import warnings
warnings.filterwarnings("ignore")

_FLAGS = flags.FLAGS
flags.DEFINE_integer('i_neuron', None, 'ith bodyId in analyze_neurons', lower_bound=0)
flags.mark_flags_as_required(['i_neuron'])

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
analyze_neurons = config.analyze_neurons

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

sections = ['axon', 'dendrite', 'connecting cable']
sections_num = [node_class_dict[ section ] for section in sections]

def main(_):
    i_neuron = _FLAGS.i_neuron
    warnings.filterwarnings("ignore") # ignore all warnings
    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    mito_file =  home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv'
    skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'
    if isfile(skel_file) and isfile(mito_file):
        s_pandas = pd.read_csv(skel_file)
        s_np = s_pandas.to_numpy()
        mito_df = pd.read_csv(mito_file)
        mito_idxs = utils.find_closest_idxs(s_np, mito_df)
        
        node_dists = []
        for section in ['dendrite', 'connecting cable', 'axon']:
            bool_mitos = node_class_dict[section] == mito_df['class'].to_numpy()
            if np.sum(bool_mitos) > 1:
                arbor_idxs = np.where( s_pandas['node_classes'].to_numpy() == node_class_dict[section] )[0]
                arbor_mito_idxs = mito_idxs[mito_df['class'].to_numpy() == node_class_dict[section]]

                node_dists.append( utils.get_pairwise_dists( s_pandas.iloc[ arbor_mito_idxs ], s_np, df_2 = s_pandas.iloc[arbor_idxs] ) * 8 / 1000 )
            else:
                node_dists.append( None )
                
        with open(home_dir + f'/saved_data/saved_mito_node_dists/{neuron_type}_{bodyId}_{section}_mito_node_dists.pkl', 'wb') as f:
            pickle.dump(node_dists, f)

if __name__ == '__main__':
    app.run(main)
















