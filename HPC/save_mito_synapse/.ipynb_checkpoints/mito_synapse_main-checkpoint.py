#!/usr/bin/env python
from absl import app
from absl import flags

# % matplotlib inline
from neuprint import Client, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_mitochondria
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import random
from os.path import isfile
import time
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing, binary_erosion, measurements, convolve
from skimage import measure
import os

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

import warnings
warnings.filterwarnings("ignore")

_FLAGS = flags.FLAGS
flags.DEFINE_integer('i_neuron', None, 'ith neuron in all_bodyIds', lower_bound=0)
flags.mark_flags_as_required(['i_neuron'])

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

os.environ['TENSORSTORE_CA_BUNDLE'] = config.tensorstore_ca_bundle
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.google_application_credentials

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# import voxel_utils file
spec = importlib.util.spec_from_file_location('voxel_utils', home_dir+'/util_files/voxel_utils.py')
voxel_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voxel_utils)

# import cristae_segmentation file
spec = importlib.util.spec_from_file_location('cris_seg', home_dir+'/util_files/cristae_segmentation.py')
cris_seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cris_seg)

# import skel_clean_utils file
spec = importlib.util.spec_from_file_location('skel_clean_utils', home_dir+'/util_files/skel_clean_utils.py')
skel_clean_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skel_clean_utils)

def main(_):
    i_neuron=_FLAGS.i_neuron
    warnings.filterwarnings("ignore") # ignore all warnings

    new_feats = []
    new_feats.append( 'mito SA' )
    new_feats.append( 'mito CA' )
    new_feats.append( 'skel CA' )
    new_feats.append( 'skel Circum' )
    new_feats.append( 'convex hull compactness' )
    new_feats.append( 'PC1 Length' )
    new_feats.append( 'PC1 inertia moment' )
    new_feats.append( 'PC1 symmetry' )
    new_feats.append( 'PC1 CA' )
    new_feats.append( 'PC1 Circum' )
    
    new_feats.append( 'PC2 Length' )
    new_feats.append( 'PC2 inertia moment' )
    new_feats.append( 'PC2 symmetry' )
    new_feats.append( 'PC2 CA' )
    new_feats.append( 'PC2 Circum' )
    
    new_feats.append( 'PC3 Length' )
    new_feats.append( 'PC3 inertia moment' )
    new_feats.append( 'PC3 symmetry' )
    new_feats.append( 'PC3 CA' )
    new_feats.append( 'PC3 Circum' )
    
    new_feats.append( 'mito diameter' )
    new_feats.append( 'length along skeleton' )

    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'

    s_pandas = pd.read_csv( skel_file )
    node_classes = s_pandas['node_classes'].to_numpy()
    s_np = s_pandas.to_numpy()

    branch_nodes = utils.find_leaves_and_branches(s_np)[1]
    branch_idxs = np.array([ np.where(branch_node == s_np[:,0])[0][0] for branch_node in branch_nodes ])
    branch_classes = node_classes[ branch_idxs ]

    uncleaned_synapse_sites = pd.concat( [utils.get_synapse_df( bodyId, 'pre'), utils.get_synapse_df( bodyId, 'post')], axis=0 )
    synapse_sites = skel_clean_utils.clean_synapses(s_np, uncleaned_synapse_sites )

    synapse_classes = node_classes[ utils.find_closest_idxs(s_np, synapse_sites) ]
    synapse_sites = synapse_sites.iloc[ np.where( np.isin( synapse_classes, [node_class_dict[section] for section in ['axon', 'dendrite', 'connecting cable']] ) )[0] ]
    # see if synapse in on mitochondrion
    syn_coords = synapse_sites[['x','y','z']].to_numpy()
    pre_on = np.zeros( synapse_sites.shape[0] )
    for i_synapse in range(len(synapse_sites)):
        _, mito_CS = voxel_utils.get_CrossSection_info(bodyId, syn_coords[i_synapse], 50, return_mito = True)
        pre_on[i_synapse] = np.any(mito_CS)
    synapse_graph = utils.build_object_graph(s_pandas, synapse_sites, pad_root = True)

    mito_df = fetch_mitochondria(NC(bodyId=bodyId))
    mito_classes = node_classes[ utils.find_closest_idxs(s_np, mito_df) ]
    mito_df = mito_df.iloc[ np.where( np.isin( mito_classes, [node_class_dict[section] for section in ['axon', 'dendrite', 'connecting cable']] ) )[0] ]
    mito_idxs = utils.find_closest_idxs(s_np, mito_df)
    mito_branch_nodes = [ [] for _ in range(len(mito_df)) ]
    edge_nodes = [ [] for _ in range(len(mito_df)) ]

    new_feat_space = np.zeros( (len(mito_df), len(new_feats)) )
    syn_mito_dists = np.ones( (len(synapse_sites), len(mito_df)) ) + np.inf

    for i_mito in range(len(mito_df)):
        feat_vec = []
        coords = voxel_utils.get_full_mito_coords(mito_df.iloc[i_mito])
        box_zyx = [ np.flip( np.min(coords,axis=0) ) - 2, np.flip( np.max(coords,axis=0) ) + 2 ]
        center_idx = np.flip( mito_df.iloc[i_mito][['x','y','z']].to_numpy(dtype=int) ) - box_zyx[0]
        mito_subvol = voxel_utils.get_center_object( voxel_utils.get_subvol_any_size(box_zyx, 'mito-objects') == bodyId , center = center_idx)
        skel_subvol = voxel_utils.get_center_object( voxel_utils.get_subvol_any_size(box_zyx, 'segmentation') == bodyId )
        grayscale_subvol = voxel_utils.get_subvol_any_size(box_zyx, 'grayscale')

        syn_mito_dists[:,i_mito] = np.min( cdist( syn_coords, coords ), axis=1)

        # get cross-sectional area of each mitochondrion at its centroid
        skel_CS, mito_CS = voxel_utils.get_CrossSection_info(bodyId, mito_df.iloc[i_mito][['x','y','z']].to_numpy(), s_np[mito_idxs[i_mito],4]*2, return_mito = True)
        this_skel_CA, this_skel_Circum = voxel_utils.get_foreground_stats(skel_CS, box_zyx, return_COM = False)
        this_mito_CA, _ = voxel_utils.get_foreground_stats(mito_CS, box_zyx, return_COM = False)
        this_mito_SA =  voxel_utils.get_foreground_stats(mito_subvol, box_zyx, return_COM = False)[1]
        feat_vec.append( this_mito_SA )
        feat_vec.append( this_mito_CA )
        feat_vec.append( this_skel_CA )
        feat_vec.append( this_skel_Circum )

        hull = ConvexHull(coords)
        feat_vec.append( hull.area )

        pca = PCA()
        pca.fit(coords[random.sample(range(len(coords)), int(len(coords)/100) )])
        PC_coords = pca.transform(coords)
        for i_PC in range(3):
            proj_coords = PC_coords[:,i_PC] #np.matmul( coords, pca.components_[i_PC][:,np.newaxis] )[:,0]
            if i_PC == 0:
                PC1_length = np.max(proj_coords) - np.min(proj_coords)
            feat_vec.append( np.max(proj_coords) - np.min(proj_coords) ) # PC length
            feat_vec.append( np.sum( np.sum(PC_coords[:,np.arange(3)!=i_PC]**2,axis=1) ) ) # PC internia moment

            PC_coords_flipped = np.where(np.repeat( np.arange(3)[np.newaxis,:], len(PC_coords), axis=0) == i_PC, -PC_coords, PC_coords)
            symmetry = 2 - (np.unique(np.append(PC_coords_flipped, PC_coords, axis=0).astype(int), axis=0).shape[0] / len(np.unique(PC_coords.astype(int),axis=0)))
            feat_vec.append( symmetry ) # symmetry about PC
            
            theta_PC, phi_PC = utils.cart_2_spherical(*pca.components_[i_PC])[1:]
            mito_CS_PC = voxel_utils.find_cross_section( theta_PC, phi_PC, mito_subvol)
            
            mito_CS_PC_CA, mito_CS_PC_Circum = voxel_utils.get_foreground_stats(mito_CS_PC, box_zyx, return_COM = False)
            
            feat_vec.append(mito_CS_PC_CA) # PC Cross-sectional Area
            feat_vec.append(mito_CS_PC_Circum) # PC Circumference
        feat_vec.append( np.max(pdist(coords[random.sample(range(len(coords)), int(len(coords)/100) )])) )

        edge_nodes[i_mito] = utils.overlay_mito_on_df(s_pandas, s_pandas, s_np, coords, i_mito, PC1_length, np.sqrt(this_mito_CA/np.pi))
        # get length along skeleton
        orig_up_node = edge_nodes[i_mito][0][1]
        orig_down_node = edge_nodes[i_mito][-1][1]
        assert edge_nodes[i_mito][0][3] == 'up'
        assert edge_nodes[i_mito][-1][3] == 'down'
        if orig_up_node == edge_nodes[i_mito][-1][0]:
            # this mito is between two nodes
            skel_length = PC1_length + 0
        else:
            path_idxs = utils.get_down_idxs(s_np, orig_up_node, s_np[:,0] == orig_down_node, count_start = True)
            assert s_np[path_idxs[-1],0] == orig_down_node
            skel_length = edge_nodes[i_mito][0][2] + np.sum(s_np[path_idxs[:-1],6]) + edge_nodes[i_mito][-1][2]
            if skel_length < 0:
                # something went wrong, so use PC1 length
                skel_length = PC1_length + 0
        feat_vec.append(skel_length)
        new_feat_space[i_mito] = feat_vec

        for branch_idx in branch_idxs:
            if np.min( np.sqrt( np.sum( (s_np[branch_idx,[1,2,3]][np.newaxis,:] - coords)**2, axis=1) ) ) < s_np[branch_idx,4]:
                # this mitochondrion is in this branch
                mito_branch_nodes[i_mito].append( s_np[branch_idx,0] )

    mito_df['branch_nodes_in'] = mito_branch_nodes
    mito_df['edge_nodes'] = edge_nodes
    for i_feat, this_feat_name in enumerate(new_feats):
        mito_df[this_feat_name] = new_feat_space[:,i_feat]
    mito_graph_df = utils.build_object_graph(s_pandas, mito_df, pad_root = True)

    # REMOVE FIRST ROW OF MITO_GRAPH AND SYNAPSE_GRAPH
    mito_graph_np = mito_graph_df.to_numpy()
    mito_graph_np[:,5] = np.where( mito_graph_np[:,5] == 1, -1, mito_graph_np[:,5])
    mito_graph_df = pd.DataFrame(data= mito_graph_np[1:], columns = mito_graph_df.columns)
    mito_graph_np = mito_graph_df.to_numpy()

    for i_mito in range(len(mito_graph_df)):

        if mito_graph_np[i_mito,5] == -1:
            mito_graph_np[i_mito,6] = np.inf
        else:
            i_df = None
            i_overhang = None
            for this_edges in edge_nodes[i_mito]:
                if this_edges[3] == 'down':
                    if this_edges[2] < 0:
                        i_df = mito_df.iloc[[i_mito]]
                    else:
                        i_df = s_pandas.iloc[np.where(s_np[:,0] == this_edges[1])[0]]
                    i_overhang = this_edges[2]
            assert i_df is not None and i_overhang is not None

            j_mito = np.where(mito_graph_np[i_mito,5] == mito_graph_np[:,0])[0][0]

            j_idxs = []; j_overhangs = []
            for this_edges in edge_nodes[j_mito]:
                if this_edges[3] == 'up':
                    j_idxs.append( np.where(s_np[:,0] == this_edges[1])[0][0] )
                    j_overhangs.append( this_edges[2] )
            assert len(j_idxs) > 0
            j_df = s_pandas.iloc[j_idxs]

            dists = utils.get_pairwise_dists(i_df, s_np, df_2 = j_df)[0]
            mito_graph_np[i_mito,6] = np.min(dists) - i_overhang - j_overhangs[ np.argmin(dists) ]
    mito_graph_df = pd.DataFrame(data= mito_graph_np, columns = mito_graph_df.columns)

    all_synapses_on = [ [] for _ in range(len(mito_df)) ]
    for i_synapse in np.where( pre_on==1 )[0]:
        i_mito = np.argmin( syn_mito_dists[i_synapse] )
        all_synapses_on[i_mito].append(i_synapse)
    mito_graph_df['i_synapses_on'] = all_synapses_on
    mito_graph_df.to_csv(home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv', index = False)

    all_mitos_on = [ [] for _ in range(len(synapse_sites)) ]
    for i_synapse in np.where( pre_on==1 )[0]:
        i_mito = np.argmin( syn_mito_dists[i_synapse] )
        all_mitos_on[i_synapse].append(i_mito)

    synapse_sites['i_mitos_on'] = all_mitos_on
    synapse_graph = utils.build_object_graph(s_pandas, synapse_sites, pad_root = True)
    synapse_graph_np = synapse_graph.to_numpy()
    synapse_graph_np[:,5] = np.where( synapse_graph_np[:,5] == 1, -1, synapse_graph_np[:,5])
    synapse_graph = pd.DataFrame(data= synapse_graph_np[1:], columns = synapse_graph.columns)
    synapse_graph.to_csv(home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv', index = False)

if __name__ == '__main__':
    app.run(main)
