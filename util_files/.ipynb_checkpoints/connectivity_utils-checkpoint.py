"""
Utility functions for analyzing neuronal connectivity in the Drosophila brain.
This module contains functions for:
- Neuron type classification and naming
- Connectivity analysis between neuron types
- Lobula surface modeling and PCA analysis
"""

# list of computing utilities for dvid code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
import importlib
import os
from sklearn.decomposition import PCA
from os.path import isfile

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
spec = importlib.util.spec_from_file_location('config', os.path.dirname(__file__) + '/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Load configuration parameters
token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict

# Load neuron quality data
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

# Import utils module
os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def get_this_broad_name(neuron_type):
    """
    Extract the broad name from a neuron type by removing numeric suffixes.
    
    Args:
        neuron_type (str): The full neuron type name
        
    Returns:
        str: The broad name without numeric suffixes
    """
    isdigits = np.array([i.isdigit() for i in neuron_type])
    if np.all(isdigits):
        return neuron_type
    if isdigits[0]:
        isdigits[ : np.where(~isdigits)[0][0] ] = False
        if np.any(isdigits):
            neuron_type = neuron_type[ :np.where( isdigits )[0][0] ]
    elif np.any(isdigits):
        neuron_type = neuron_type[ :np.where( isdigits )[0][0] ]
    return neuron_type

def get_unique_post_types(neuron_types, count_thresh = 1, put_LC_first = True, use_broad_names = False):
    """
    Get list of postsynaptic neuron types and their connection frequencies.
    
    Args:
        neuron_types (list): List of presynaptic neuron types
        count_thresh (float): Threshold for minimum number of connections relative to number of neurons
        put_LC_first (bool): If True, put local cell (LC) types first in the output
        use_broad_names (bool): If True, use broad neuron type names instead of specific ones
        
    Returns:
        tuple: (sorted_post_types, is_above_thresh) where:
            sorted_post_types (numpy.ndarray): Array of postsynaptic neuron types
            is_above_thresh (numpy.ndarray): Boolean array indicating which connections are above threshold
    """
    # get list of postsynaptic connections
    all_post_types = [] # list of pst types above threshold for each neuron in neuron_types
    all_post_types_count = []
    keep_post_types = []
    num_in_type = []
    for pre_type in neuron_types:
        this_post_types = []
        q = f"""\
            MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron)
            WHERE a.type = "{pre_type}" and b.type is not null
            RETURN a.bodyId as pre_bodyId, w.weight as weight, b.bodyId as post_bodyId, b.type as post_type
        """
        results = c.fetch_custom(q)

        unique_post_types = results['post_type'].unique()
        num_neurons = np.sum( neuron_quality_np[:,1] == pre_type )
        num_in_type.append(num_neurons)
        results = results.to_numpy()
        unique_post_types_count = np.array([ np.sum(results[ results[:,3] == posttype, 1 ]) for posttype in unique_post_types ])
        all_post_types_count.append( unique_post_types_count )

        if use_broad_names:
            bool_keep = np.ones( len(unique_post_types_count), dtype = bool )
        else:
            bool_keep = unique_post_types_count >= num_neurons * count_thresh
        all_post_types.append( unique_post_types[bool_keep] )
        for post_type in unique_post_types[bool_keep]:
            if post_type not in keep_post_types:
                keep_post_types.append( post_type )
    if put_LC_first:
        bool_nonLC_posttypes = [ posttype not in neuron_types for posttype in np.sort(keep_post_types) ]
        sorted_post_types = np.append( np.array(neuron_types), np.sort(keep_post_types)[bool_nonLC_posttypes] )
    else:
        sorted_post_types = np.sort(keep_post_types)

    if use_broad_names:
        broad_names = []
        for posttype in sorted_post_types:
            broad_names.append( get_this_broad_name(posttype) )
        broad_names = np.unique(broad_names)

        broad_names_count = np.zeros( (len(neuron_types), len(broad_names)) )
        for i_type in range(len(neuron_types)):
            for i_post_type, post_type in enumerate(all_post_types[i_type]):
                if post_type is not None:
                    bool_post_types = np.array([post_type.startswith(this_broad_name) for this_broad_name in broad_names])
                    if np.any(bool_post_types):
                        broad_names_count[i_type,np.where(bool_post_types)[0][0]] += all_post_types_count[i_type][i_post_type]
        is_above_thresh = broad_names_count >= (np.array(num_in_type)[:,np.newaxis] * count_thresh)

        good_broad_names = np.any( is_above_thresh, axis=0 )
        is_above_thresh = is_above_thresh[:,good_broad_names]
        sorted_post_types = [ broad_names[i] for i in np.where(good_broad_names)[0] ]
    else:
        is_above_thresh = np.zeros( (len(neuron_types), len(sorted_post_types)) , dtype=bool)
        for i_type in range(len(neuron_types)):
            is_above_thresh[i_type] = np.isin( sorted_post_types, all_post_types[i_type] )
            if put_LC_first:
                assert is_above_thresh[i_type, i_type] # homotypic connections should count
    return sorted_post_types, is_above_thresh

def loadlobulamodel(**kwargs):
    """
    Load or create a model of the lobula surface using PCA and quadratic fitting.
    Based on Ryosuke's paper, this function:
    1. Queries postsynaptic synapses in the lobula region
    2. Performs PCA on the synapse locations
    3. Fits a quadratic surface to the PCA-transformed data
    
    Args:
        **kwargs: Additional keyword arguments (currently unused)
        
    Returns:
        tuple: (pca, modelcoeff) where:
            pca (PCA): Fitted PCA model
            modelcoeff (numpy.ndarray): Coefficients of the quadratic surface model
    """
    # From Ryosuke's paper
    neuron_type = 'LT1'
    #neuron_type = 'LC25'
    q = f"""\
        MATCH (a:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse)
        WHERE a.type='{neuron_type}' AND s.type='post' AND s.`LO(R)`
        RETURN DISTINCT s.location.x as x, s.location.y as y, s.location.z as z
        """

    # run the query
    landmark = c.fetch_custom(q)

    # Now run PCA on x/y/z in the "landmark"
    # Also convert 8 nm px to microns unit
    X = landmark["x"].to_numpy().flatten()*8/1000
    Y = landmark["y"].to_numpy().flatten()*8/1000
    Z = landmark["z"].to_numpy().flatten()*8/1000
    XYZ = np.array([X,Y,Z]).T
    pca = PCA(n_components=3)
    PCs = pca.fit_transform(XYZ)


    # Do fitting of quadric model
    # This model can be improved
    # fit quadric model
    PC1 = PCs[:,0].flatten()
    PC2 = PCs[:,1].flatten()
    PC3 = PCs[:,2].flatten()

    # predictors
    A = np.array([PC1*0+1, PC1, PC2, PC1**2, PC2**2, PC1*PC2]).T

    # fitting
    modelcoeff, r, rank, s = np.linalg.lstsq(A,PC3,rcond=None)

    # show goodness of fit
    r2 = 1 - r / np.sum(PC3**2)
    print('R2 of the lobula model was: ',r2)

    return pca, modelcoeff
