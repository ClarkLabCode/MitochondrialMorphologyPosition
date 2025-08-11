"""
Utility functions for cleaning and processing neuron skeletons in the Drosophila hemibrain dataset.

This module provides tools for:
- Skeleton node classification and labeling
- Skeleton healing and resampling
- Leaf and branch analysis
- Synapse cleaning and validation
- Feature extraction for skeleton analysis

Key dependencies:
- numpy: Numerical computations
- pandas: Data manipulation
- neuprint: FlyEM database access
- skimage: Image processing
- scipy: Scientific computing
- sklearn: Machine learning utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
import importlib
import time
import scipy
from skimage import measure
from scipy.ndimage import measurements, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
import os
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from os.path import isfile

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
spec = importlib.util.spec_from_file_location('config', os.path.dirname(__file__) + '/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
# uuid of the hemibrain-flattened repository
node_class_dict = config.node_class_dict

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

# import utils file
os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def get_node_labels(idx, i_coord, num_coords, next_node, old_nodes, available_label):
    """
    Assigns labels to nodes in a resampled skeleton.
    
    This function determines the appropriate rowId and link values for nodes
    in a resampled skeleton based on their position and relationship to the
    original skeleton nodes.
    
    Args:
        idx (int): Index in the old skeleton
        i_coord (int): Position in the coordinate sequence
        num_coords (int): Total number of coordinates
        next_node (int): Next node in the sequence
        old_nodes (numpy.ndarray): Array of nodes from the old skeleton
        available_label (int): Next available label to assign
        
    Returns:
        tuple: (rowId, link, available_label) where:
            rowId: Assigned row ID for the node
            link: Link to the next node
            available_label: Updated next available label
    """

    if num_coords == 1:
        # this means the rowId and link are both nodes in the old skeleton
        rowId = old_nodes[idx]
        link = next_node
        # don't update available_node
    elif i_coord == 0:
        # this is a node stored in the old skeleton
        # link should be available node
        rowId = old_nodes[idx]
        link = available_label
        available_label += 1
    elif i_coord == num_coords-1:
        # the link node is from the old skeleton
        rowId = available_label-1
        link = next_node
    else:
        # rowId should be the link from the
        rowId, link = available_label + np.array([-1,0])
        available_label += 1

    return rowId, link, available_label

def classify_nodes(orig_s_pandas, synapse_sites, this_neuron_quality):
    """
    Classifies nodes in a neuron skeleton into different anatomical regions.
    
    This function identifies and labels nodes as belonging to different
    anatomical regions (soma, axon, dendrite, etc.) based on synapse
    locations and neuron quality information.
    
    Args:
        orig_s_pandas (pandas.DataFrame): Original skeleton data
        synapse_sites (pandas.DataFrame): Synapse location data
        this_neuron_quality (pandas.Series): Quality metrics for the neuron
        
    Returns:
        tuple: (node_classes, important_nodes) where:
            node_classes: Array of class labels for each node
            important_nodes: Dictionary of key node IDs
    """

    s_pandas = orig_s_pandas.copy()
    assert np.sum(s_pandas['link'].to_numpy() == -1) == 1, 'please input a healed skeleton'
    node_classes = np.ones( (len(s_pandas),), dtype=int ) * node_class_dict['other'] # initially assume all nodes are other

    if ~this_neuron_quality['has_axon'] and ~this_neuron_quality['has_dendrite']:
        return None, None # this neuron is useless
    s_np = s_pandas.to_numpy()
    synapse_coords = synapse_sites[['x','y','z']].to_numpy()
    synapse_idxs = utils.find_closest_idxs(s_np, synapse_sites)
    is_LO = np.any([synapse_sites['roi'].to_numpy() == 'LO(R)', synapse_sites['roi'].to_numpy() == 'LOP(R)' ],axis=0)
    is_None = np.array([this_roi is None for this_roi in synapse_sites['roi'].to_numpy()])
    is_CB = np.all([~is_LO, ~is_None],axis=0)

    if this_neuron_quality['has_soma']:
        root_idx = s_pandas['radius'].argmax()
        root_node = s_pandas['rowId'].iloc[root_idx]
        node_classes[root_idx] = node_class_dict['soma']
    elif this_neuron_quality['has_axon'] and this_neuron_quality['has_dendrite']:
        # has axon and dendrite but no soma
        # find separating node using linear SVM on synapse coords
        root_node = find_separating_node( s_pandas, synapse_sites, synapse_idxs )
        if root_node is None: return None, None
    else:
        # find skeleton node farthest from median of axon or dendrite arbor
        synapse_bool = is_LO if this_neuron_quality['has_dendrite'] else is_CB
        median_coord = np.median( synapse_coords[synapse_bool], axis = 0 ).reshape((1,3))
        s_np = s_pandas.to_numpy()
        node_dists = np.sum((s_np[:,[1,2,3]] - median_coord)**2,axis=1)
        root_idx = np.argmax(node_dists)
        root_node = s_np[root_idx,0]
    skeleton.reorient_skeleton( s_pandas, rowId = root_node )
    important_nodes = {'root node': root_node}
    s_np = s_pandas.to_numpy()
    root_idx = np.where(s_np[:,0] == root_node)[0][0]
    start_node = root_node
    if this_neuron_quality['has_soma'] and this_neuron_quality['has_axon'] and this_neuron_quality['has_dendrite']:
        # find main bifurcation node
        main_bifurcation_node, CBF_idxs = find_main_bifurcation_node(s_np, synapse_sites)
        important_nodes['main bifurcation node'] = main_bifurcation_node
        node_classes[ CBF_idxs ] = node_class_dict['cell body fiber']
        start_node = main_bifurcation_node

        for up_idx in np.where( root_node == s_np[:,5] )[0]:
            if up_idx not in CBF_idxs:
                node_classes[ utils.find_up_idxs(s_np, s_np[up_idx,0]) ] = node_class_dict['soma']
        radii_labels = measure.label( s_np[:,4] >= np.median( s_np[CBF_idxs,4] ) )
        if radii_labels[root_idx] > 0:
            node_classes[ radii_labels == radii_labels[root_idx] ] = node_class_dict['soma']

    if this_neuron_quality['has_axon']:
        axon_base_node = find_arbor_base( s_np, synapse_idxs[is_CB] )
        if axon_base_node is None: return None, None
        important_nodes['axon base node'] = axon_base_node
        node_classes[ utils.find_up_idxs(s_np, axon_base_node) ] = node_class_dict['axon']
    if this_neuron_quality['has_dendrite']:
        dendrite_base_node = find_arbor_base( s_np, synapse_idxs[is_LO] )
        if dendrite_base_node is None: return None, None
        important_nodes['dendrite base node'] = dendrite_base_node
        node_classes[ utils.find_up_idxs(s_np, dendrite_base_node) ] = node_class_dict['dendrite']

    if this_neuron_quality['has_axon'] and this_neuron_quality['has_dendrite']:
        # get nodes connecting axon base and dendrite base
        base_idxs = [utils.get_down_idxs(s_np, s_np[np.where(base_node==s_np[:,0])[0][0],5], s_np[:,5] == -1) for base_node in [axon_base_node, dendrite_base_node]]
        node_classes[ base_idxs[0][ ~np.isin(base_idxs[0], base_idxs[1]) ] ] = node_class_dict['connecting cable']
        node_classes[ base_idxs[1][ ~np.isin(base_idxs[1], base_idxs[0]) ] ] = node_class_dict['connecting cable']

    return node_classes, important_nodes

def find_arbor_base( s_np, synapse_idxs ):
    """
    Finds the base node of an arbor (axon or dendrite) in the skeleton.
    
    The base node is defined as the point where the arbor branches off from
    the main neuron structure, determined by analyzing synapse distribution.
    
    Args:
        s_np (numpy.ndarray): Skeleton data as numpy array
        synapse_idxs (numpy.ndarray): Indices of synapses in the arbor
        
    Returns:
        int: Node ID of the arbor base or None if not found
    """

    median_synapse_coord = np.median( s_np[synapse_idxs,:][:,[1,2,3]], axis=0 )
    root_idx = np.where( s_np[:,5] == -1 )[0][0]
    idx = np.argmin( np.sum( (median_synapse_coord[np.newaxis,:] - s_np[:,[1,2,3]])**2, axis=1) )
    while idx != root_idx:
        num_up_synapses = np.sum( np.isin(synapse_idxs, utils.find_up_idxs(s_np, s_np[idx,0])) )
        if num_up_synapses >= int(len(synapse_idxs)*.98):
            return s_np[idx,0]
        idx = np.where( s_np[idx,5] == s_np[:,0] )[0][0]
    return None

def find_main_bifurcation_node(s_np, synapse_sites):
    """
    Identifies the main bifurcation node where the cell body fiber splits.
    
    This function locates the point where the cell body fiber branches into
    the axon and dendrite arbors, using synapse distribution analysis.
    
    Args:
        s_np (numpy.ndarray): Skeleton data as numpy array
        synapse_sites (pandas.DataFrame): Synapse location data
        
    Returns:
        tuple: (main_bifurcation_node, CBF_idxs) where:
            main_bifurcation_node: Node ID of the bifurcation point
            CBF_idxs: Indices of the cell body fiber nodes
    """

    synapse_coords = synapse_sites[['x','y','z']].to_numpy()
    synapse_idxs = utils.find_closest_idxs(s_np, synapse_sites)
    is_LO = np.any([synapse_sites['roi'].to_numpy() == 'LO(R)', synapse_sites['roi'].to_numpy() == 'LOP(R)' ],axis=0)
    arbor_idxs = []
    for bool_synapses in [is_LO, ~is_LO]:
        median_arbor_coord = np.median( synapse_coords[bool_synapses], axis = 0 ).reshape((1,3))
        idxs = utils.get_down_idxs(s_np, s_np[np.argmin( np.sum((s_np[:,[1,2,3]] - median_arbor_coord)**2,axis=1) ), 0], s_np[:,5] == -1)
        arbor_idxs.append(idxs)
    CBF_idxs = arbor_idxs[0][ np.isin( arbor_idxs[0], arbor_idxs[1] ) ]
    main_bifurcation_node = s_np[ CBF_idxs[0], 0]
    CBF_idxs = np.flip(CBF_idxs[1:]) # flip CBF_idxs such that the root is first
    return main_bifurcation_node, CBF_idxs

def find_separating_node( orig_s_pandas, orig_synapse_sites, synapse_idxs ):
    """
    Finds the node that best separates lobula and non-lobula synapses.
    
    Uses a linear SVM to find the optimal node that divides the neuron
    into regions containing predominantly lobula or non-lobula synapses.
    
    Args:
        orig_s_pandas (pandas.DataFrame): Original skeleton data
        orig_synapse_sites (pandas.DataFrame): Original synapse data
        synapse_idxs (numpy.ndarray): Indices of synapses in the skeleton
        
    Returns:
        int: Node ID of the separating node or None if not found
    """

    is_not_None = np.array([this_roi is not None for this_roi in orig_synapse_sites['roi'].to_numpy()])
    synapse_sites = orig_synapse_sites.iloc[ np.where(is_not_None)[0] ] # exclude synapses in unknown ROI
    synapse_idxs = synapse_idxs[ is_not_None ]

    s_pandas = orig_s_pandas.copy()
    s_np = s_pandas.to_numpy()
    is_LO = np.any([synapse_sites['roi'].to_numpy() == 'LO(R)', synapse_sites['roi'].to_numpy() == 'LOP(R)' ],axis=0)
    is_CB = ~is_LO
    LO_idxs = synapse_idxs

    synapse_coords = synapse_sites[ ['x','y','z'] ].to_numpy()
    num_LO = np.sum(is_LO); num_CB = np.sum(is_CB) # number of lobula and central brain synapses
    model = LinearSVC(dual=False, max_iter=10000)
    model.fit(synapse_coords, is_LO)
    M = model.coef_[0].reshape((1,3)) # normal vector of hyperplane
    B = model.intercept_[0] # intercept of hyperplane

    node_dists = ( np.sum(M * s_np[:,[1,2,3]],axis=1) + B)**2 # square distances of nodes to hyperplane
    sorted_idxs = np.argsort(node_dists) # ascending sorted indices
    for node in s_np[ sorted_idxs,0 ]:
        skeleton.reorient_skeleton( s_pandas, rowId = node ) #use_max_radius=True)
        s_np = s_pandas.to_numpy()

        is_up_LO = [ is_LO[np.isin(synapse_idxs, utils.find_up_idxs(s_np, node))] for node in s_np[s_np[:,5] == node, 0] ]
        frac_up_LO = [ np.sum(this_up_LO)/np.sum(is_LO) for this_up_LO in is_up_LO ] # fraction of lobula synapses upstream from each node
        frac_up_CB = [ np.sum(~this_up_LO)/np.sum(~is_LO) for this_up_LO in is_up_LO ] # fraction of lobula synapses upstream from each node

        if np.argmax(frac_up_LO) != np.argmax(frac_up_CB) and np.max(frac_up_LO) > 0.95 and np.max(frac_up_CB) > 0.95:
            return node
    return None

def append_distance(s_pandas):
    """
    Adds Euclidean distance information to the skeleton dataframe.
    
    Calculates and appends the distance between each node and its parent
    node in the skeleton structure.
    
    Args:
        s_pandas (pandas.DataFrame): Skeleton data
        
    Returns:
        pandas.DataFrame: Skeleton data with distance field added
    """
    s_np = s_pandas.to_numpy()
    distances = np.zeros( (len(s_np),) )
    for cur_idx in range(s_pandas.shape[0]):
        next_node = s_np[cur_idx,5]
        if next_node >= 0:
            next_idx = np.where(s_np[:,0] == next_node)[0][0]
            cur_pos = s_np[cur_idx,[1,2,3]]; next_pos = s_np[next_idx,[1,2,3]]
            distances[cur_idx] = np.sqrt( np.sum((cur_pos - next_pos) ** 2) ) # sub-segment length
        else:
            # we're at the root, so set distance to infinity
            distances[cur_idx] = np.Inf
    s_pandas['distance'] = distances # add the distances to the pandas dataframe
    return s_pandas

def get_OldNew_mother_leaf_features( leaf_node, s_np ):
    """
    Extracts features of a leaf node and its mother branch.
    
    Calculates various metrics including radii and lengths for both
    the leaf segment and its parent branch.
    
    Args:
        leaf_node (int): ID of the leaf node
        s_np (numpy.ndarray): Skeleton data as numpy array
        
    Returns:
        tuple: (leaf_radius, branch_radius, mother_radius, leaf_length)
    """

    branch_nodes = utils.find_leaves_and_branches( s_np )[1]
    assert np.all( np.isin(branch_nodes,s_np[:,0]) ), 'not all branches were found in skeleton'
    is_branch = np.isin(s_np[:,0],branch_nodes)
    leaf_idxs = utils.get_down_idxs(s_np, leaf_node, is_branch)
    branch_idx = leaf_idxs[-1]
    assert s_np[branch_idx,0] in branch_nodes or s_np[branch_idx,5] == -1

    leaf_idxs_bool = np.sqrt( np.sum( (s_np[leaf_idxs,:][:,[1,2,3]] - s_np[branch_idx,[1,2,3]][np.newaxis,:])**2, axis=1) ) > s_np[branch_idx,4]
    leaf_radius = np.mean( s_np[ leaf_idxs[leaf_idxs_bool], 4 ] ) if np.any( leaf_idxs_bool ) else s_np[leaf_idxs[0],4]
    root_node = s_np[ np.where(s_np[:,5]==-1)[0][0], 0]

    if s_np[branch_idx,0] == root_node:
        mother_radius = 0
    else:
        mother_idxs = utils.get_down_idxs(s_np, s_np[branch_idx,0], is_branch)
        mother_idxs_bool = np.all([np.sqrt( np.sum( (s_np[mother_idxs,:][:,[1,2,3]] - s_np[branch_idx,[1,2,3]][np.newaxis,:] )**2, axis=1) ) > s_np[branch_idx,4],
                                    np.sqrt( np.sum( (s_np[mother_idxs,:][:,[1,2,3]] - s_np[mother_idxs[-1],[1,2,3]][np.newaxis,:] )**2, axis=1) ) > s_np[mother_idxs[-1],4]],axis=0)
        mother_radius = np.mean( s_np[ mother_idxs[ mother_idxs_bool ], 4 ] ) if np.any( mother_idxs_bool ) else 0

    leaf_length = np.sum(s_np[leaf_idxs[ s_np[leaf_idxs,5] != -1 ],6]) if np.any(s_np[leaf_idxs,5] != -1) else 0
    return leaf_radius, s_np[branch_idx,4], mother_radius, leaf_length

def heal_resampled_skel(resampled_s_pandas, bodyId):
    """
    Heals a resampled skeleton by connecting nodes according to the original skeleton.
    
    Ensures that the resampled skeleton maintains the same connectivity
    structure as the original skeleton from the database.
    
    Args:
        resampled_s_pandas (pandas.DataFrame): Resampled skeleton data
        bodyId (int): ID of the neuron
        
    Returns:
        pandas.DataFrame: Healed skeleton data
    """
    resampled_s_np = resampled_s_pandas.to_numpy()
    assert (resampled_s_np[0,5] == -1) and (resampled_s_np[0,0] == 1), 'Please do not reorient the resampled skeleton before healing'

    if np.sum(resampled_s_np[:,5] == -1) == 1: return resampled_s_pandas

    unhealed_s_np = c.fetch_skeleton( bodyId, format='pandas', heal=False, with_distances=False).to_numpy()
    healed_s_np = c.fetch_skeleton( bodyId, format='pandas', heal=True, with_distances=False).to_numpy()
    assert np.all( healed_s_np[:,0] == unhealed_s_np[:,0] )
    old_nodes = np.concatenate( (healed_s_np[:,0],[-1]) )

    for i in np.arange(healed_s_np.shape[0]):
        node, down_node = healed_s_np[i,[0,5]]
        idx = np.where(resampled_s_np[:,0] == node)[0][0]
        while not (resampled_s_np[idx,5] in old_nodes):
            idx = np.where(resampled_s_np[:,0] == resampled_s_np[idx,5])[0][0]
        resampled_s_np[idx,5] = down_node
    assert np.sum(resampled_s_np[:,5] == -1) == 1, 'There should be no remaining unhealead sections of the skeleton'

    return pd.DataFrame( data=resampled_s_np, columns = resampled_s_pandas.columns )

def get_is_trivial_leaf_space(bodyId, leaf_nodes, num_features, new_s_pandas):
    """
    Computes feature space for determining if leaf nodes are trivial.
    
    Extracts various features from both old and new skeletons to create
    a feature space for classifying leaf nodes as trivial or non-trivial.
    
    Args:
        bodyId (int): ID of the neuron
        leaf_nodes (numpy.ndarray): Array of leaf node IDs
        num_features (int): Number of features to compute
        new_s_pandas (pandas.DataFrame): New skeleton data
        
    Returns:
        tuple: (X, leaf_nodes) where:
            X: Feature matrix for leaf nodes
            leaf_nodes: Array of leaf node IDs
    """

    i_neuron = np.where( neuron_quality_np[:,0] == bodyId )[0][0]
    neuron_type = neuron_quality_np[i_neuron,1]
    if (neuron_quality_np[i_neuron,3] or neuron_quality_np[i_neuron,4]):
        old_s_pandas = c.fetch_skeleton( bodyId, format='pandas', heal=True, with_distances=False) # I will heal the skeleton later
        node_classes, important_nodes = classify_nodes(old_s_pandas, fetch_synapses(NC(bodyId=bodyId)), neuron_quality.iloc[i_neuron])
        if node_classes is None: return None, None
        #if important_nodes['root node'] in leaf_nodes:
            # get rid of the root in leaf_nodes
        #    leaf_nodes = leaf_nodes[ leaf_nodes != important_nodes['root node'] ]
        skeleton.reorient_skeleton( old_s_pandas, rowId = important_nodes['root node'] )
        old_s_pandas = append_distance( old_s_pandas )
        old_s_np = old_s_pandas.to_numpy()

        # download recomputed skeleton
        new_cols = np.concatenate( [np.array(new_s_pandas.columns)[:6], ['distance'], np.array(new_s_pandas.columns)[6:] ] )
        new_s_pandas = append_distance( new_s_pandas ) # create a distance field in the dataframe
        new_s_pandas = new_s_pandas.reindex(columns=new_cols)
        new_s_np = new_s_pandas.to_numpy()
        branch_nodes = utils.find_leaves_and_branches( new_s_np )[1]

        X = np.zeros( (len(leaf_nodes), num_features) )
        for i_node, leaf_node in enumerate(leaf_nodes):
            assert leaf_node not in new_s_np[:,5]
            old_leaf_radius, old_branch_radius, old_mother_radius, old_leaf_length = get_OldNew_mother_leaf_features( leaf_node, old_s_np )
            new_leaf_radius, new_branch_radius, new_mother_radius, new_leaf_length = get_OldNew_mother_leaf_features( leaf_node, new_s_np )
            branch_idx = utils.get_down_idxs(new_s_np, leaf_node, np.isin(new_s_np[:,0],branch_nodes) )[-1]

            X[i_node,0] = old_branch_radius; X[i_node,1] = old_leaf_radius; X[i_node,2] = old_mother_radius
            X[i_node,3] = new_branch_radius; X[i_node,4] = new_leaf_radius; X[i_node,5] = new_mother_radius
            X[i_node,6] = np.log10( new_branch_radius / old_leaf_radius ) if (old_leaf_radius > 0) and (new_branch_radius > 0) else 0
            X[i_node,7] = np.log10( new_branch_radius / new_leaf_radius ) if (new_leaf_radius > 0) and (new_branch_radius > 0) else 0
            X[i_node,8] = np.log10( old_leaf_radius / new_leaf_radius ) if (new_leaf_radius > 0) and (old_leaf_radius > 0) else 0
            X[i_node,9] = old_leaf_length
            X[i_node,10] = np.sqrt( np.sum( (old_s_np[ np.where(old_s_np[:,0]==leaf_node)[0][0], [1,2,3] ] - new_s_np[ np.where(new_s_np[:,0]==leaf_node)[0][0], [1,2,3] ])**2 ) )
            X[i_node,11] = np.log10( (old_leaf_length - new_branch_radius) / new_leaf_radius ) if (new_leaf_radius > 0) and ((old_leaf_length - new_branch_radius) > 0) else 0
            X[i_node,12] = new_branch_radius / old_leaf_radius if (old_leaf_radius > 0) and (new_branch_radius > 0) else 0
            X[i_node,13] = new_branch_radius / new_leaf_radius if (new_leaf_radius > 0) and (new_branch_radius > 0) else 0
            X[i_node,14] = old_leaf_radius / new_leaf_radius if (new_leaf_radius > 0) and (old_leaf_radius > 0) else 0
            X[i_node,15] = np.log10( (old_leaf_length - new_branch_radius) / new_branch_radius ) if (old_leaf_length - new_branch_radius) > 0 else 0
            X[i_node,16] = np.sum( new_s_np[:,5] == new_s_np[branch_idx,0] ) > 2
        assert np.all( X != np.inf )
        return X, leaf_nodes
    return None, None

def clean_synapses(s_np, synapse_sites):
    """
    Filters synapses to remove those not within the neuron's radius.
    
    Removes synapses that are too far from the nearest skeleton point,
    ensuring only valid synapses are retained.
    
    Args:
        s_np (numpy.ndarray): Skeleton data as numpy array
        synapse_sites (pandas.DataFrame): Synapse location data
        
    Returns:
        pandas.DataFrame: Filtered synapse data
    """
    heights = []
    synapse_coords = synapse_sites[ ['x', 'y', 'z'] ].to_numpy(dtype=float)
    good_synapses = np.zeros( len(synapse_sites), dtype=bool )
    for i, idx in enumerate( utils.find_closest_idxs(s_np, synapse_sites) ):
        xyz = np.array(utils.spherical_2_cart(1, s_np[idx,7], s_np[idx,8]))
        base = np.sum((synapse_coords[i] - s_np[idx,[1,2,3]]) * xyz) # base of right triangle
        hyp = np.sqrt( np.sum( (synapse_coords[i] - s_np[idx,[1,2,3]])**2 ) ) # hypotenuse of triangle
        height = np.sqrt( hyp**2 - base**2 ) # pythagorean theorem
        heights.append( height - s_np[idx,4] )
        good_synapses[i] = height < ( s_np[idx,4] + 500/8)
    return synapse_sites.iloc[ np.where(good_synapses)[0] ]
