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
flags.DEFINE_integer('i_neuron', None, 'ith neuron in neuron_quality', lower_bound=0)
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

# import GLM_utils file
spec = importlib.util.spec_from_file_location('GLM_utils', home_dir+'/util_files/GLM_utils.py')
GLM_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GLM_utils)

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def main(_):
    i_neuron=_FLAGS.i_neuron
    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    warnings.filterwarnings("ignore") # ignore all warnings
    offset = -np.inf

    arbors = ['axon', 'dendrite']
    dist_bins, titles, scalar_features = GLM_utils.get_mito_pos_features()
    num_hist_bins = [ len(dist_bins[i_hist])-1 for i_hist in range(len(dist_bins)) ]
    final_idx = np.cumsum(num_hist_bins)
    init_idx = np.append(np.array([0]),final_idx[:-1])
    num_features = len(scalar_features) + np.sum(num_hist_bins)

    all_titles = []
    for i_hist in range(len(titles)):
        for i_bin in range(len(dist_bins[i_hist])-1):
            all_titles.append( titles[i_hist] + f'_{i_bin}' )
    for scalar_feat in scalar_features:
        all_titles.append(scalar_feat)

    for jitter_strength in config.jitter_strengths:
        if not isfile( home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_X_df_Jitter_{int(jitter_strength)}_{i_neuron}.csv' ):
            bodyId_type_arbor = []
            all_X = np.array( [ [] for _ in range(num_features) ] ).T
            all_Y = np.array( [ [] for _ in range(1) ] ).T
            mean_dx = np.zeros( 2 )
            rms = np.zeros( 2 )
            for i_arbor, arbor in enumerate(arbors):
                this_dx = np.array([])
                synapse_file = home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv'
                if neuron_quality[f'has_{arbor}'].iloc[i_neuron]:
                    # this neuron contains arbor
                    s_pandas = pd.read_csv( home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv' )
                    node_classes = s_pandas['node_classes'].to_numpy()
                    s_np = s_pandas.to_numpy().astype(float)
                    branch_nodes = utils.find_leaves_and_branches(s_np)[1]
                    synapse_df = pd.read_csv(synapse_file)
                    is_pre = (synapse_df['type'].to_numpy() == 'pre').astype(float)
                    arbor_synapse_df = synapse_df.iloc[ np.where(node_classes[utils.find_closest_idxs(s_np, synapse_df)] == node_class_dict[arbor])[0] ]

                    mito_df = pd.read_csv(home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv')
                    all_mito_idxs = utils.find_closest_idxs(s_np, mito_df)
                    bool_mito = node_classes[ all_mito_idxs ] == node_class_dict[arbor]

                    branch_idxs = np.where(np.isin(s_np[:,0], branch_nodes))[0]

                    if np.sum( bool_mito ) > 1 and len(arbor_synapse_df) > 1:
                        shuffled_mito_df = utils.get_shuffled_mito_df(bodyId, arbor)
                        if jitter_strength > 0:
                            mito_df, true_dxs, dxs = utils.get_shuffled_mito_df(bodyId, arbor, method = 'jitter', jitter_strength = jitter_strength)
                            this_dx = np.append(this_dx, true_dxs)
                        else:
                            mito_df = mito_df.iloc[ np.where(bool_mito)[0] ]
                        all_mito_df = pd.concat([shuffled_mito_df, mito_df])
                        has_mito = s_pandas['mito_CA'].to_numpy() > 0

                        #is_original = np.append( np.zeros(shuffled_mito_df.shape[0]), np.ones(mito_df.shape[0]) )[:,np.newaxis]
                        #Y = np.concatenate( [is_original, all_mito_df[['cristae SA', 'mito SA']].to_numpy()], axis=1 )
                        Y = np.append( np.zeros(shuffled_mito_df.shape[0]), np.ones(mito_df.shape[0]) ).astype(int)[:,np.newaxis]

                        X = np.zeros( (len(Y),num_features) ) + offset

                        mito_synapse_dists = utils.get_mito_object_dists(all_mito_df, arbor_synapse_df, s_np) * 8/1000
                        for i_hist, synapse_type in enumerate(['pre','post']):
                            if np.any( arbor_synapse_df['type'].to_numpy() == synapse_type ):
                                this_bins = dist_bins[i_hist]
                                for i_d in range(len(this_bins)-1):
                                    bool_1 = mito_synapse_dists[:,arbor_synapse_df['type'].to_numpy() == synapse_type] > this_bins[i_d]
                                    bool_2 = mito_synapse_dists[:,arbor_synapse_df['type'].to_numpy() == synapse_type] <= this_bins[i_d+1]
                                    X[:,init_idx[i_hist] + i_d] = np.sum( np.all([bool_1,bool_2],axis=0) , axis=1)
                                assert init_idx[i_hist]+i_d+1 == final_idx[i_hist]

                        # histogram features for branches
                        i_hist += 1
                        if len(branch_idxs) > 0:
                            branch_coord_df = pd.DataFrame(data = s_np[branch_idxs,:][:,[1,2,3]], columns = ['x','y','z'])
                            mito_branch_dists = utils.get_mito_object_dists(all_mito_df, branch_coord_df, s_np) * 8/1000
                            this_bins = dist_bins[i_hist]
                            for i_d in range(len(this_bins)-1):
                                bool_1 = mito_branch_dists > this_bins[i_d]
                                bool_2 = mito_branch_dists <= this_bins[i_d+1]
                                X[:,init_idx[i_hist] + i_d] = np.sum( np.all([bool_1,bool_2],axis=0) , axis=1)
                            assert init_idx[i_hist]+i_d+1 == final_idx[i_hist]
                            
                        # compute intramitochondria distances
                        i_hist += 1
                        this_bins = dist_bins[i_hist]
                        for i_mito_df, this_mito_df in enumerate([shuffled_mito_df, mito_df]):
                            mito_mito_dists = utils.get_mito_object_dists(this_mito_df, this_mito_df, s_np) * 8/1000
                            init_mito_idx = len(shuffled_mito_df) if i_mito_df == 1 else 0
                            for i_d in range(len(this_bins)-1):
                                bool_1 = mito_mito_dists > this_bins[i_d]
                                bool_2 = mito_mito_dists <= this_bins[i_d+1]
                                assert np.all(X[init_mito_idx + np.arange(len(this_mito_df)),:][:,init_idx[i_hist] + i_d] == offset)
                                X[init_mito_idx + np.arange(len(this_mito_df)),init_idx[i_hist] + i_d] = np.sum( np.all([bool_1,bool_2],axis=0) , axis=1)

                        # compute scalar features
                        all_mito_idxs = utils.find_closest_idxs(s_np, all_mito_df)
                        branch_orders = utils.get_branch_order( s_np[all_mito_idxs,0], s_np )
                        arbor_base_idx = utils.find_arbor_base_idx(s_np, node_classes == node_class_dict[arbor])
                        X[:,final_idx[-1]] = branch_orders - utils.get_branch_order( s_np[[arbor_base_idx],0], s_np )[0]
                        X[:,final_idx[-1]+1] = utils.get_leaf_number(s_np[all_mito_idxs,0], s_np)

                        for i_mito, this_edge_nodes in enumerate(all_mito_df['edge_nodes'].to_numpy()):
                            if type(this_edge_nodes) == str:
                                this_edge_nodes = eval(this_edge_nodes)
                            most_down_node = this_edge_nodes[-1][1]
                            assert this_edge_nodes[-1][3] == 'down'
                            most_up_node = this_edge_nodes[0][1]
                            if np.sum(most_down_node == s_np[:,5]) > 1 and most_up_node != most_down_node:
                                # most down node is a branch_node
                                path_idxs = utils.get_down_idxs(s_np, most_up_node, s_np[:,0] == most_down_node, count_start = True)
                                assert s_np[path_idxs[-1],0] == most_down_node
                                most_down_node = s_np[path_idxs[-2],0]
                            
                            d1_idxs, d2_idxs, m_idxs = GLM_utils.get_d1_d2_m_seg_idxs(most_down_node, s_np, branch_nodes)
                            if d2_idxs is not None:
                                X[i_mito,final_idx[-1]+2] = np.mean(s_np[d1_idxs,10]) > np.mean(s_np[d2_idxs,10]) # is_on_thicker_daughter
                            X[i_mito,final_idx[-1]+3] = np.log10(np.mean(s_np[d1_idxs,10]) / np.mean(s_np[m_idxs,10])) # segment circumference

                            this_COM = all_mito_df.iloc[i_mito][['x','y','z']].to_numpy()
                            closest_idx = d1_idxs[ np.argmin( np.sum( (this_COM[np.newaxis,:] - s_np[d1_idxs,:][:,[1,2,3]])**2, axis=1) ) ]
                            theta, phi = s_np[closest_idx,[7,8]]
                            xyz = np.array( utils.spherical_2_cart(1, theta, phi) )
                            rem_down_idxs = utils.get_down_idxs(s_np, s_np[closest_idx,0], np.isin(s_np[:,0], branch_nodes), count_start=True)
                            down_seg_dist = np.sum(s_np[rem_down_idxs[:-1],6]) - (np.sum(xyz*this_COM) - np.sum(xyz*s_np[closest_idx,[1,2,3]]))
                            X[i_mito,final_idx[-1]+4] = 1 - (down_seg_dist / np.sum( s_np[d1_idxs[d1_idxs!= rem_down_idxs[-1]],6] ))

                            i_synapses = all_mito_df.iloc[i_mito]['i_synapses_on']
                            if type(i_synapses) == str:
                                i_synapses = eval(i_synapses)
                            X[i_mito,final_idx[-1]+5] = np.sum(1==is_pre[np.array(i_synapses).astype(int)] ) # number of presynapses on mitochondrion
                            X[i_mito,final_idx[-1]+6] = np.sum(0==is_pre[np.array(i_synapses).astype(int)] ) # number of postsynapses on mitochondrion

                        branch_nodes_in = all_mito_df.iloc[i_mito]['branch_nodes_in']
                        if type(branch_nodes_in) == str:
                            branch_nodes_in = eval(branch_nodes_in)
                        X[:,final_idx[-1]+7] = len(branch_nodes_in)
                        assert final_idx[-1]+7 == X.shape[1]-1, 'check number of expected features'

                        all_X = np.append( all_X, X, axis=0 )
                        all_Y = np.append( all_Y, Y, axis=0 )
                        for i in range(len(Y)):
                            bodyId_type_arbor.append( [bodyId, neuron_type, arbor] )
                if len(this_dx) > 0:
                    mean_dx[i_arbor] = np.mean(this_dx)
                    rms[i_arbor] = np.sqrt(np.mean(this_dx**2))

            if all_X.shape[0] > 0:
                # save results
                X = np.where( np.any([all_X == np.inf, all_X == -np.inf],axis=0), np.nan, all_X )
                X_df = pd.DataFrame(data=X, columns = all_titles)
                Y_df = pd.DataFrame(data=all_Y, columns = ['Is Jittered Mitochondrion'])
                bodyId_type_arbor_df = pd.DataFrame(data=np.array(bodyId_type_arbor), columns = ['bodyId', 'neuron_type', 'arbor'])
                mean_dx_df = pd.DataFrame(data = np.array(mean_dx)[np.newaxis,:], columns = arbors)
                rms_df = pd.DataFrame(data = np.array(rms)[np.newaxis,:], columns = arbors)

                X_df.to_csv(home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_X_df_Jitter_{int(jitter_strength)}_{i_neuron}.csv', index=False)
                Y_df.to_csv(home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_Y_df_Jitter_{int(jitter_strength)}_{i_neuron}.csv', index=False)
                bodyId_type_arbor_df.to_csv(home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_bodyId_type_arbor_Jitter_{int(jitter_strength)}_{i_neuron}.csv', index=False)
                mean_dx_df.to_csv(home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_mean_dx_Jitter_{int(jitter_strength)}_{i_neuron}.csv', index=False)
                rms_df.to_csv(home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_rms_Jitter_{int(jitter_strength)}_{i_neuron}.csv', index=False)

if __name__ == '__main__':
    app.run(main)
