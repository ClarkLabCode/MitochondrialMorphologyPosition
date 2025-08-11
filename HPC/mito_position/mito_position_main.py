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

def main(_):
    i_type=_FLAGS.i_type
    warnings.filterwarnings("ignore") # ignore all warnings
    neuron_type = analyze_neurons[i_type]

    dist_bins, titles, scalar_features = GLM_utils.get_mito_pos_features()
    num_hist_bins = [ len(dist_bins[i_hist])-1 for i_hist in range(len(dist_bins)) ]
    all_titles = []
    for i_hist in range(len(titles)):
        for i_bin in range(len(dist_bins[i_hist])-1):
            all_titles.append( titles[i_hist] + f'_{i_bin}' )
    for scalar_feat in scalar_features:
        all_titles.append(scalar_feat)

    arbors = ['axon', 'dendrite', 'connecting cable']

    this_X, Y, bodyId_type_arbor = GLM_utils.get_mito_position_space([neuron_type], arbors, offset = -np.inf)
    X = np.where( np.any([this_X == np.inf, this_X == -np.inf],axis=0), np.nan, this_X )
    X_df = pd.DataFrame(data=X, columns = all_titles)
    Y_df = pd.DataFrame(data=Y, columns = ['Is Existing Mitochondrion'])
    bodyId_type_arbor_df = pd.DataFrame(data=bodyId_type_arbor.T, columns = ['bodyId', 'neuron_type', 'arbor'])

    X_df.to_csv(home_dir + f'/saved_data/position_feats/{neuron_type}_X_df.csv', index=False)
    Y_df.to_csv(home_dir + f'/saved_data/position_feats/{neuron_type}_Y_df.csv', index=False)
    bodyId_type_arbor_df.to_csv(home_dir + f'/saved_data/position_feats/{neuron_type}_bodyId_type_arbor.csv', index=False)

if __name__ == '__main__':
    app.run(main)
