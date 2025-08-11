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
from os.path import isfile
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import iqr as IQR
import os
import pickle
import json

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

import warnings
warnings.filterwarnings("ignore")

_FLAGS = flags.FLAGS
flags.DEFINE_integer('i_neuron', None, 'ith cell in microns dataset', lower_bound=0)
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

# import skel_clean_utils file
spec = importlib.util.spec_from_file_location('skel_clean_utils', home_dir+'/util_files/skel_clean_utils.py')
skel_clean_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skel_clean_utils)

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def append_theta_phi(s_pandas):
    s_np = s_pandas.to_numpy()
    theta_phi = []
    for idx in range(len(s_pandas)):
        if s_np[idx,5] == -1:
            theta_phi.append(np.zeros(2))
        else:
            down_idx = np.where( s_np[:,0] == s_np[idx,5] )[0][0]
            r = s_np[down_idx,[1,2,3]] - s_np[idx,[1,2,3]]
            theta_phi.append( utils.cart_2_spherical(r[0], r[1], r[2])[1:] )
    theta_phi = np.array(theta_phi)
    s_pandas['theta'] = theta_phi[:,0]
    s_pandas['phi'] = theta_phi[:,1]
    return s_pandas

def adjust_node_labels(s_pandas):
    if s_pandas['rowId'].min() == 0:
        s_pandas['rowId'] = s_pandas['rowId'].to_numpy() + 1
        s_pandas['link'] = np.where(s_pandas['link'] != -1, s_pandas['link'].to_numpy() + 1, s_pandas['link'].to_numpy())
    return s_pandas

def get_shuffled_mito_nodes(s_np, bool_nodes, L):
    init_idx = np.random.choice( np.where(bool_nodes)[0] )
    idx = init_idx + 0
    nodes = [s_np[idx,0]]
    dist = 0
    while dist < L/2:
        dist += s_np[idx,6]
        if s_np[idx,5] == -1:
            return None, None
        idx = np.where( s_np[:,0] == s_np[idx,5] )[0][0]
        if ~bool_nodes[idx]:
            return None, None
        nodes.append( s_np[idx,0] )
        
    idx = init_idx + 0
    while dist < L/2:
        if s_np[idx,0] not in s_np[:,5]:
            return None, None
        idx = np.random.choice(s_np[ s_np[:,5] == s_np[idx,0], 0])
        if ~bool_nodes[idx]:
            return None, None
        dist += s_np[idx,6]
        nodes.append( s_np[idx,0] )
    return np.array(nodes), init_idx

def main(_):
    dist_bins, titles, scalar_features = GLM_utils.get_mito_microns_pos_features()
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

    cell_df = pd.read_csv(home_dir + '/saved_data/mouse_microns/cell_type_classification_filtered.csv')
    i_cell=_FLAGS.i_neuron

    all_synapse_df = pd.read_csv(home_dir + '/saved_data/mouse_microns/synapse_filtered.csv')
    pandas_names = ['rowId', 'compartment', 'x', 'y', 'z', 'radius', 'link']
    ordered_pandas_names = ['rowId', 'x', 'y', 'z', 'radius', 'link', 'distance', 'compartment']
    
    compartments_analyze = ['Axonal'] #['Basal', 'Apical', 'Axonal']
    compartment_dict = {0: 'Somatic', 1: 'Axonal', 2: 'Basal', 3: 'Apical', 4: 'undefined', 5: 'custom'}
    colors =           [   'k',             'r',       'g',       'purple',            'cyan',               'yellow' ]
    
    offset = -np.inf
    bodyId_type_subtype_arbor = []
    all_X = np.array( [ [] for _ in range(num_features) ] ).T
    all_Y = np.array( [ [] for _ in range(1) ] ).T
    
    bodyId = cell_df.iloc[i_cell]['cell_id']
    cell_type = cell_df.iloc[i_cell]['cell_type']

    synapse_df = all_synapse_df[ all_synapse_df['cell_id'] == bodyId ]

    s_pandas = pd.read_csv(home_dir + f'/saved_data/mouse_microns/cell_morphologies/{bodyId}.swc', header = None, names = pandas_names, 
                           delimiter = ' ')[['rowId', 'x', 'y', 'z', 'radius', 'link', 'compartment']]
    skel_compartment = s_pandas['compartment'].to_numpy()
    s_pandas = skel_clean_utils.append_distance(s_pandas)[['rowId', 'x', 'y', 'z', 'radius', 'link', 'distance']]
    s_pandas = adjust_node_labels(s_pandas)
    
    skel_coords = s_pandas[['x','y','z']].to_numpy()
    mito_df = pd.read_csv(home_dir + f'/saved_data/mouse_microns/microns_cells/{cell_type}_{bodyId}.csv')
    mito_df = mito_df[ mito_df['compartment'] != 'Somatic' ]
    mito_nodes_file = home_dir + f'/saved_data/mouse_microns/associated_nodes/{bodyId}_assoc_nodes.json'

    with open(mito_nodes_file, 'r') as f:
        # Load the JSON data into a Python dictionary
        mito_nodes = json.load(f)
    #mito_nodes = mito_nodes['nodes']
    for key in mito_nodes.keys():
        mito_nodes[key] = np.array(mito_nodes[key]) + 1
    #mito_nodes = {k: v for k, v in mito_nodes.items() if len(v) > 0}
    #mito_df = mito_df.iloc[ np.where( np.isin(mito_df['mito_id'].to_numpy(), np.array(list(mito_nodes.keys()), dtype=int)) )[0] ]
    assert np.all( mito_df['mito_id'].to_numpy() == np.array(list(mito_nodes.keys()), dtype=int) )

    for section in compartments_analyze:
        i_section = np.where( section == np.array(list(compartment_dict.values())) )[0][0]

        bool_nodes = skel_compartment == i_section
        if np.any(bool_nodes):
            root_idx = np.where(~bool_nodes)[0][ np.argmax( np.min(cdist(skel_coords[bool_nodes], skel_coords[~bool_nodes]),axis=0) ) ]
            skeleton.reorient_skeleton( s_pandas, rowId = s_pandas['rowId'].iloc[root_idx] )
            s_pandas = skel_clean_utils.append_distance(skeleton.heal_skeleton(s_pandas))
            s_pandas = append_theta_phi(s_pandas)
            section_idxs = np.where( [ compartment_dict[ this_comp ] == section for this_comp in skel_compartment ] )[0]
            s_np = s_pandas.to_numpy()
            
            branch_nodes = utils.find_leaves_and_branches(s_np)[1]
            branch_idxs = np.where(np.isin(s_np[:,0], branch_nodes))[0]

            section_mito_df = mito_df[ mito_df['compartment'] == section ]
            synapse_idxs = utils.find_closest_idxs(s_np, synapse_df)
            bool_synapses = np.isin( synapse_idxs, section_idxs )
            section_synapse_df = synapse_df.iloc[np.where(bool_synapses)[0]]
                
            if len(section_synapse_df) > 1 and len(section_mito_df) > 0:
                this_mito_df_nodes = [ mito_nodes[ str(mito_id) ] for mito_id in section_mito_df['mito_id'] ]
                shuffle_mito_list = []
                for mito_id in section_mito_df['mito_id']:
                    L = np.sum( s_np[ np.isin(s_np[:,0], mito_nodes[ str(mito_id) ]), 6 ] )
                    this_nodes = None
                    while this_nodes is None:
                        this_nodes, idx = get_shuffled_mito_nodes(s_np, bool_nodes, L)
                        if this_nodes is not None:
                            this_mito_df_nodes.append( this_nodes )
                            shuffle_mito_list.append(s_np[idx,[1,2,3]])
                shuffle_mito_np = np.array(shuffle_mito_list)
                
                unique_mito_nodes = np.unique(np.concatenate(this_mito_df_nodes, axis=0))
                unique_mito_nodes_df = pd.DataFrame( [ s_np[ np.where(mito_node == s_np[:,0])[0][0], [1,2,3]] for mito_node in unique_mito_nodes ], columns = ['x','y','z'])
    
                Y = np.append( np.ones(len(section_mito_df)), np.zeros(len(shuffle_mito_np)) )[:,np.newaxis].astype(int)
                mito_cols = ['x','y','z']
                this_mito_df = pd.DataFrame(np.append( section_mito_df[mito_cols], shuffle_mito_np , axis=0), columns = mito_cols)
    
                X = np.zeros( (len(Y),num_features) ) + offset
                
                mito_nodes_synapse_dists = utils.get_pairwise_dists( unique_mito_nodes_df, s_np, df_2 = section_synapse_df ) / 1000
                mito_synapse_dists = np.zeros( (len(this_mito_df), len(section_synapse_df)) )
                for i_mito in range(len(this_mito_df)):
                    bool_mito_nodes = np.isin( unique_mito_nodes, this_mito_df_nodes[i_mito] )
                    mito_synapse_dists[i_mito] = np.min(mito_nodes_synapse_dists[bool_mito_nodes],axis=0)
                for i_hist, synapse_type in enumerate(['pre','post']):
                    if np.any( section_synapse_df['synapse_type'].to_numpy() == synapse_type ):
                        this_bins = dist_bins[i_hist]
                        for i_d in range(len(this_bins)-1):
                            bool_1 = mito_synapse_dists[:,section_synapse_df['synapse_type'].to_numpy() == synapse_type] > this_bins[i_d]
                            bool_2 = mito_synapse_dists[:,section_synapse_df['synapse_type'].to_numpy() == synapse_type] <= this_bins[i_d+1]
                            X[:,init_idx[i_hist] + i_d] = np.sum( np.all([bool_1,bool_2],axis=0) , axis=1)
                        assert init_idx[i_hist]+i_d+1 == final_idx[i_hist]
                
                # histogram features for branches
                i_hist += 1
                if len(branch_idxs) > 0:
                    branch_coord_df = pd.DataFrame(data = s_np[branch_idxs,:][:,[1,2,3]], columns = ['x','y','z'])
                    mito_nodes_branch_dists = utils.get_pairwise_dists( unique_mito_nodes_df, s_np, df_2 = branch_coord_df ) / 1000
                    mito_branch_dists = np.zeros( (len(this_mito_df), len(branch_coord_df)) )
                    for i_mito in range(len(this_mito_df)):
                        bool_mito_nodes = np.isin( unique_mito_nodes, this_mito_df_nodes[i_mito] )
                        mito_branch_dists[i_mito] = np.min(mito_nodes_branch_dists[bool_mito_nodes],axis=0)
                        
                    this_bins = dist_bins[i_hist]
                    for i_d in range(len(this_bins)-1):
                        bool_1 = mito_branch_dists > this_bins[i_d]
                        bool_2 = mito_branch_dists <= this_bins[i_d+1]
                        X[:,init_idx[i_hist] + i_d] = np.sum( np.all([bool_1,bool_2],axis=0) , axis=1)
                    assert init_idx[i_hist]+i_d+1 == final_idx[i_hist]
                    
                # compute intramitochondria distances
                i_hist += 1
                this_bins = dist_bins[i_hist]
                for i_mito_df, i_mitos in enumerate( [ np.arange(len(section_mito_df)), np.arange(len(section_mito_df), len(this_mito_df)) ] ):
                    small_this_mito_df_nodes = [ this_mito_df_nodes[ii] for ii in i_mitos ]
                    if i_mito_df == 0:
                        # measured mitos
                        small_this_mito_df_nodes = [ this_mito_df_nodes[ii] for ii in range(len(section_mito_df)) ]
                    else:
                        # shuffled mitos
                        small_this_mito_df_nodes = [ this_mito_df_nodes[ii] for ii in range(len(section_mito_df), len(this_mito_df)) ]
                        
                    small_unique_mito_nodes = np.unique(np.concatenate(small_this_mito_df_nodes, axis=0))
                    small_unique_mito_nodes_df = pd.DataFrame( [ s_np[ np.where(mito_node == s_np[:,0])[0][0], [1,2,3]] for mito_node in small_unique_mito_nodes ], columns = ['x','y','z'])
                    mito_nodes_mito_nodes_dists = utils.get_pairwise_dists( small_unique_mito_nodes_df, s_np ) / 1000
                    mito_mito_dists = np.zeros( (len(i_mitos), len(i_mitos)) )

                    for row, row_mito in enumerate(i_mitos):
                        row_bool_mito_nodes = np.isin( small_unique_mito_nodes, this_mito_df_nodes[row_mito] )
                        for col, col_mito in enumerate(i_mitos):
                            col_bool_mito_nodes = np.isin( small_unique_mito_nodes, this_mito_df_nodes[col_mito] )
                            
                            mito_mito_dists[row,col] = np.min( mito_nodes_mito_nodes_dists[row_bool_mito_nodes,:][:,col_bool_mito_nodes] )
                    
                    init_mito_idx = len(section_mito_df) if i_mito_df == 1 else 0
                    for i_d in range(len(this_bins)-1):
                        bool_1 = mito_mito_dists > this_bins[i_d]
                        bool_2 = mito_mito_dists <= this_bins[i_d+1]
                        assert np.all(X[init_mito_idx + np.arange(len(i_mitos)),:][:,init_idx[i_hist] + i_d] == offset)
                        X[init_mito_idx + np.arange(len(i_mitos)),init_idx[i_hist] + i_d] = np.sum( np.all([bool_1,bool_2],axis=0) , axis=1)
    
                # compute scalar features
                all_mito_idxs = utils.find_closest_idxs(s_np, this_mito_df)
                X[:,final_idx[-1]+1] = utils.get_leaf_number(s_np[all_mito_idxs,0], s_np)
    
                for i_mito, mito_idx in enumerate(all_mito_idxs):
                    X[i_mito,final_idx[-1]] = np.sum( np.isin(utils.get_down_idxs(s_np, s_np[mito_idx,0], ~bool_nodes, count_start = True), branch_idxs) )
    
                    d1_idxs, d2_idxs, m_idxs = GLM_utils.get_d1_d2_m_seg_idxs( s_np[mito_idx,0], s_np, branch_nodes)
                    this_COM = this_mito_df.iloc[i_mito][['x','y','z']].to_numpy()
                    closest_idx = d1_idxs[ np.argmin( np.sum( (this_COM[np.newaxis,:] - s_np[d1_idxs,:][:,[1,2,3]])**2, axis=1) ) ]
                    theta, phi = s_np[closest_idx,[7,8]]
                    xyz = np.array( utils.spherical_2_cart(1, theta, phi) )
                    rem_down_idxs = utils.get_down_idxs(s_np, s_np[closest_idx,0], np.isin(s_np[:,0], branch_nodes), count_start=True)
                    down_seg_dist = np.sum(s_np[rem_down_idxs[:-1],6]) - (np.sum(xyz*this_COM) - np.sum(xyz*s_np[closest_idx,[1,2,3]]))
                    X[i_mito,final_idx[-1]+2] = 1 - (down_seg_dist / np.sum( s_np[d1_idxs[d1_idxs!= rem_down_idxs[-1]],6] ))

                presynapse_nodes = s_np[utils.find_closest_idxs(s_np, section_synapse_df[ section_synapse_df['synapse_type'] == 'pre' ]),0]
                num_pre_in = np.array([ np.sum(np.isin(nodes, presynapse_nodes)) for nodes in this_mito_df_nodes ])
                X[:,final_idx[-1]+3] = num_pre_in # number of presynapses on mitos

                postsynapse_nodes = s_np[utils.find_closest_idxs(s_np, section_synapse_df[ section_synapse_df['synapse_type'] == 'post' ]),0]
                num_post_in = np.array([ np.sum(np.isin(nodes, postsynapse_nodes)) for nodes in this_mito_df_nodes ])
                X[:,final_idx[-1]+4] = num_post_in # number of postsynapses on mitos

                num_branches_in = np.zeros( len(this_mito_df) ) if len(branch_idxs) == 0 else np.array([ np.sum(np.isin(nodes, branch_nodes)) for nodes in this_mito_df_nodes ])
                X[:,final_idx[-1]+5] = num_branches_in # number of branches in mitos
                
                assert final_idx[-1]+5 == X.shape[1]-1, 'check number of expected features'
    
                all_X = np.append( all_X, X, axis=0 )
                all_Y = np.append( all_Y, Y, axis=0 )
                for i in range(len(Y)):
                    bodyId_type_subtype_arbor.append( [bodyId, cell_type, cell_df.iloc[i_cell]['cell_subtype'], section] )
    X_df = pd.DataFrame(data = np.where( np.any([all_X == np.inf, all_X == -np.inf],axis=0), np.nan, all_X ), columns = all_titles)
    Y_df = pd.DataFrame(data = all_Y, columns = ['Is Existing Mitochondrion'])
    bodyId_type_subtype_arbor_df = pd.DataFrame(data=bodyId_type_subtype_arbor, columns = ['bodyId', 'neuron_type', 'neuron_subtype', 'arbor'])
    if len(all_Y) > 0:
        X_df.to_csv(home_dir + f'/saved_data/position_feats/mouse/{i_cell}_X_df.csv', index=False)
        Y_df.to_csv(home_dir + f'/saved_data/position_feats/mouse/{i_cell}_Y_df.csv', index=False)
        bodyId_type_subtype_arbor_df.to_csv(home_dir + f'/saved_data/position_feats/mouse/{i_cell}_bodyId_type_subtype_arbor.csv', index=False)

if __name__ == '__main__':
    app.run(main)
