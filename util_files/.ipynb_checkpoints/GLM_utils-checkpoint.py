"""
Utility functions for generalized linear modeling (GLM) analysis of neuron morphology and connectivity data.
This module contains functions for:
- Statistical analysis (correlation, confidence intervals)
- GLM model training and evaluation
- Cross-validation and model selection
- Feature space construction for mitochondria positioning analysis
"""

from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_mitochondria
import numpy as np
import pandas as pd
import importlib
import random
from os.path import isfile
from scipy.optimize import minimize
from ast import literal_eval
#from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from scipy import stats

import warnings
warnings.filterwarnings("error")

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
import os
spec = importlib.util.spec_from_file_location('config', os.path.dirname(__file__) + '/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

token_id = config.token_id
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = config.home_dir
c = config.c
server = config.server
node_class_dict = config.node_class_dict

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

def pearsonr_ci(x,y,alpha=0.05):
    """
    Calculate Pearson correlation along with the confidence interval using scipy and numpy.
    
    Args:
        x, y (numpy.ndarray): Input arrays for correlation calculation
        alpha (float): Significance level. 0.05 by default
        
    Returns:
        tuple: (r, p, lo, hi) where:
            r (float): Pearson's correlation coefficient
            p (float): The corresponding p value
            lo (float): Lower bound of confidence interval
            hi (float): Upper bound of confidence interval
    """

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def train_GLM(X, Y, alpha = None, model = 'Logit'):
    """
    Train a generalized linear model (GLM) with optional regularization.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        Y (pandas.Series): Target variable
        alpha (float, optional): Regularization strength. If None, no regularization is applied
        model (str): Model type, either 'Logit' for logistic regression or 'OLS' for linear regression
        
    Returns:
        tuple: (betas, conf_int) where:
            betas (numpy.ndarray): Model coefficients
            conf_int (numpy.ndarray): Confidence intervals for coefficients
    """
    assert X.shape[0] == Y.shape[0]
    #assert np.all( ~np.all(np.isnan(X.to_numpy()), axis=0) ), f'{~np.all(np.isnan(X.to_numpy()), axis=0)}'
    if alpha is None:
        try:
            if model == 'Logit':
                log_reg = sm.Logit(Y, X, missing = 'drop').fit(disp=0, maxiter = 10000)
            elif model == 'OLS':
                log_reg = sm.OLS(Y, X, missing = 'drop').fit()
        except: return None, None
        betas = np.array(log_reg.params)
        conf_int = np.array(log_reg.conf_int())
    else:
        if model == 'OLS':
            reg = sm.OLS(Y, X, missing = 'drop').fit_regularized(alpha = alpha, L1_wt=1.0, refit=True)
            return reg.params, reg.conf_int()
        try:
            log_reg = sm.Logit(Y, X, missing = 'drop').fit_regularized(disp=0, maxiter = 10000, alpha = alpha, method = 'l1')
        except: return None, None
        nonzero_betas = np.abs(np.array(log_reg.params)) > 0
        betas = np.zeros( len(nonzero_betas) )
        conf_int = np.zeros( (len(nonzero_betas),2) )
        if np.all(nonzero_betas):
            # all coefficients are nonzero
            try:
                #log_reg = sm.GLM(Y, X,family=sm.families.Binomial(),missing='drop').fit_regularized(alpha=0.01, L1_wt = 0.99, method='elastic_net')
                log_reg = sm.Logit(Y, X, missing = 'drop').fit(disp=0, maxiter = 10000)
            except: return None, None
            betas = np.array(log_reg.params)
            conf_int = np.array(log_reg.conf_int())
        elif np.all(~nonzero_betas):
            # all coefficients are zero, so no reason to refit
            pass
        else:
            drop_titles = [ X.columns[idx] for idx in np.where(~nonzero_betas)[0] ]
            small_X = X.drop(columns = drop_titles)
            try:
                log_reg = sm.Logit(Y, small_X, missing = 'drop').fit(disp=0, maxiter = 10000)
            except: return None, None
            betas[ nonzero_betas ] = np.array(log_reg.params)
            conf_int[ nonzero_betas ] = np.array(log_reg.conf_int())
    assert len(betas) == len(conf_int), f'{len(betas)}, {len(conf_int)}'
    assert len(betas) == X.shape[1]
    return betas, conf_int

def train_random_forest(X, Y):
    """
    Train a random forest classifier.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        Y (pandas.Series): Target variable
        
    Returns:
        RandomForestClassifier: Trained random forest model
    """
    rnd_clf = RandomForestClassifier(n_estimators=100, max_depth=2)
    rnd_clf.fit(X, Y.to_numpy()[:,0])
    return rnd

def get_AUC(X, Y, group_ids, k_groups, alpha, model = 'GLM', return_r2 = False):
    """
    Calculate area under the ROC curve (AUC) using k-fold cross-validation.
    
    Args:
        X (pandas.DataFrame): Feature matrix
        Y (pandas.Series): Target variable
        group_ids (numpy.ndarray): Group identifiers for cross-validation
        k_groups (list): List of groups for cross-validation
        alpha (float): Regularization strength
        model (str): Model type ('GLM', 'OLS', or 'random forest')
        return_r2 (bool): If True, return R-squared instead of AUC
        
    Returns:
        tuple: (test_AUC, train_AUC) where:
            test_AUC (float): AUC on test set
            train_AUC (float): AUC on training set
    """
    num_points = 1000
    train_AUC = 0
    X_array = np.array(X)
    pred_test = np.array([]); Y_test = np.array([])
    for k_group in k_groups:
        neuron_bool = ~np.isin(group_ids, k_group) # true if in training group
        Y_train = np.array(Y.iloc[np.where(neuron_bool)[0]])[:,0]
        Y_test = np.append( Y_test, np.array(Y.iloc[np.where(~neuron_bool)[0]])[:,0] )
        if model == 'random forest':
            rnd_clf = train_random_forest(X.iloc[np.where(neuron_bool)[0]], Y.iloc[np.where(neuron_bool)[0]].to_numpy()[:,0])
            pred_train = rnd_clf.predict_proba( X.iloc[np.where(neuron_bool)[0]] )[:,1]
            pred_test = np.append( pred_test, rnd_clf.predict_proba( X.iloc[np.where(~neuron_bool)[0]] )[:,1] )
        elif model == 'GLM':
            betas, conf_int = train_GLM(X.iloc[np.where(neuron_bool)[0]], Y.iloc[np.where(neuron_bool)[0]], alpha = alpha)
            if betas is None:
                return None, None

            q_vals = np.nansum( X_array[~neuron_bool] * betas[np.newaxis,:], axis=1)
            q_vals = np.where( q_vals > 100, 100, q_vals)
            q_vals = np.where( q_vals < -100, -100, q_vals)
            pred_test = np.append( pred_test, 1 / (1 + np.exp(-q_vals)) )

            q_vals = np.nansum( X_array[neuron_bool] * betas[np.newaxis,:], axis=1)
            q_vals = np.where( q_vals > 100, 100, q_vals)
            q_vals = np.where( q_vals < -100, -100, q_vals)

            pred_train = 1 / (1 + np.exp(-q_vals))
        elif model == 'OLS':
            betas, conf_int = train_GLM(X.iloc[np.where(neuron_bool)[0]], Y.iloc[np.where(neuron_bool)[0]], alpha = alpha, model = 'OLS')
            pred_test = np.append( pred_test, np.nansum( X_array[~neuron_bool] * betas[np.newaxis,:], axis=1) )
            pred_train= np.nansum( X_array[ neuron_bool] * betas[np.newaxis,:], axis=1)

        if return_r2:
            train_AUC += utils.spearman_ci( pred_train, Y_train )[0]**2 / len(k_groups)
        else:
            TP = np.zeros((num_points,)); FP = np.zeros((num_points,))
            for i, thresh in enumerate(np.linspace(0,1,num_points)):
                Y_predict = pred_train >= thresh
                TP[i] = np.mean( Y_predict[ Y_train==1 ] )
                FP[i] = np.mean( Y_predict[ Y_train==0 ] )
            train_AUC += np.trapz(np.flip(TP),np.flip(FP)) / len(k_groups)

    assert len(Y_test) == X.shape[0]
    if return_r2:
        #print( pearsonr_ci(pred_test,Y_test,alpha=0.05) )
        test_AUC = utils.spearman_ci(pred_test, Y_test)[0]**2
    else:
        TP = np.zeros((num_points,)); FP = np.zeros((num_points,))
        for i, thresh in enumerate(np.linspace(0,1,num_points)):
            Y_predict = pred_test > thresh
            TP[i] = np.mean( Y_predict[ Y_test==1 ] )
            FP[i] = np.mean( Y_predict[ Y_test==0 ] )
        test_AUC = np.trapz(np.flip(TP),np.flip(FP))
    return test_AUC, train_AUC

def get_cross_val_groups(k, ids):
    """
    Divide a set of IDs into k groups for cross-validation.
    
    Args:
        k (int): Number of groups
        ids (numpy.ndarray): Array of IDs to divide
        
    Returns:
        list: List of k groups, each containing a subset of IDs
    """

    unique_ids = np.unique( ids )
    if len(unique_ids) >= k:
        np.random.shuffle(unique_ids)
        k_groups = [ [] for _ in range(k) ]
        group_size = int(len(unique_ids) / k)
        for i_k in range(k):
            for i in range(group_size):
                k_groups[i_k].append( unique_ids[i + i_k*group_size] )
        for i in range( len(unique_ids) % k ):
            k_groups[i].append( unique_ids[ -(i+1) ] )
    else:
        k_groups = None
    return k_groups

# get feature space for mito position GLM
def get_mito_pos_features():
    """
    Define feature space for mitochondria positioning GLM analysis.
    
    Returns:
        tuple: (dist_bins, titles, scalar_features) where:
            dist_bins (list): List of distance bin boundaries for different features
            titles (list): Names of the histogram features
            scalar_features (list): Names of scalar features
    """

    # create feaures to use for GLM
    presynapse_dist_bins = np.array( [0,0.3,0.6,1.2,2.5] )
    postsynapse_dist_bins = np.array( [0,0.3,0.6,1.2,2.5] )
    branch_dist_bins = np.array( [0,0.3,0.6,1.2,2.5] )
    intra_neuron_mito_dist_bins = np.array( [0, 0.5, 1.2, 2.5, 5, 10] )

    dist_bins = [presynapse_dist_bins, postsynapse_dist_bins, branch_dist_bins, intra_neuron_mito_dist_bins]
    titles = ['Presynapse Histogram', 'Postsynapse Histogram', 'Branch Histogram', 'Intramitochondria Histogram']

    scalar_features = []
    scalar_features.append( 'branch order' )
    scalar_features.append( 'leaf number' )
    scalar_features.append( 'is on thicker daughter' )
    scalar_features.append( 'Daughter Circumference / Mother Circumference' )
    scalar_features.append( 'distance up segment' )
    scalar_features.append( 'num presynapses on' )
    scalar_features.append( 'num postsynapses on' )
    scalar_features.append( 'number branches in' )
    return dist_bins, titles, scalar_features

def get_mito_microns_pos_features():
    """
    Define feature space for mitochondria positioning GLM analysis in microns.
    
    Returns:
        tuple: (dist_bins, titles, scalar_features) where:
            dist_bins (list): List of distance bin boundaries for different features
            titles (list): Names of the histogram features
            scalar_features (list): Names of scalar features
    """

    # create feaures to use for GLM
    presynapse_dist_bins = np.array( [0,0.3,0.6,1.2,2.5] )
    postsynapse_dist_bins = np.array( [0,0.3,0.6,1.2,2.5] )
    branch_dist_bins = np.array( [0,0.3,0.6,1.2,2.5] )
    intra_neuron_mito_dist_bins = np.array( [0, 0.5, 1.2, 2.5, 5, 10] )

    dist_bins = [presynapse_dist_bins, postsynapse_dist_bins, branch_dist_bins, intra_neuron_mito_dist_bins]
    titles = ['Presynapse Histogram', 'Postsynapse Histogram', 'Branch Histogram', 'Intramitochondria Histogram']

    scalar_features = []
    scalar_features.append( 'branch order' )
    scalar_features.append( 'leaf number' )
    scalar_features.append( 'distance up segment' )
    scalar_features.append( 'num presynapses on' )
    scalar_features.append( 'num postsynapses on' )
    scalar_features.append( 'number branches in' )
    return dist_bins, titles, scalar_features

def get_d1_d2_m_seg_idxs(node, s_np, branch_nodes):
    """
    Get indices for daughter segments and mother segment at a branch point.
    
    Args:
        node (int): Node ID at the branch point
        s_np (numpy.ndarray): Skeleton data
        branch_nodes (numpy.ndarray): Array of branch node IDs
        
    Returns:
        tuple: (d1_idxs, d2_idxs, m_idxs) where:
            d1_idxs (numpy.ndarray): Indices of first daughter segment
            d2_idxs (numpy.ndarray): Indices of second daughter segment
            m_idxs (numpy.ndarray): Indices of mother segment
    """

    '''
    d1 is the daughter segment with "node" in it
    d2 is the other daughter segment coming out of the branch downstream of node
    '''
    d1_idxs = utils.get_down_idxs(s_np, node, np.isin(s_np[:,0], branch_nodes), count_start=True)

    branch_idx = d1_idxs[-1]
    if s_np[branch_idx,5] == -1:
        m_idxs = np.array([branch_idx])
    else:
        m_idxs = utils.get_down_idxs(s_np, s_np[branch_idx,5], np.isin(s_np[:,0], branch_nodes), count_start=False)

    while np.sum( node == s_np[:,5] ) == 1:
        d1_idxs = np.append( d1_idxs, np.where(node == s_np[:,0])[0][0] )
        node = s_np[ np.where(node == s_np[:,5])[0][0], 0]

    branch_up_idxs = np.where( s_np[:,5] == s_np[branch_idx,0] )[0]
    if len(branch_up_idxs) == 1:
        assert s_np[branch_idx,5] == -1
        return np.unique(d1_idxs), None, m_idxs
    d2_idxs = branch_up_idxs[ ~np.isin(branch_up_idxs, d1_idxs) ]
    assert len(d2_idxs) > 0
    node = s_np[d2_idxs[0],0]
    while np.sum( node == s_np[:,5] ) == 1:
        d2_idxs = np.append( d2_idxs, np.where(node == s_np[:,0])[0][0] )
        node = s_np[ np.where(node == s_np[:,5])[0][0], 0]
    return np.unique(d1_idxs), np.unique(d2_idxs), m_idxs

def get_mito_position_space(neuron_types, arbors, offset = 0, jitter_strength = 0):
    """
    Construct feature space for mitochondria positioning analysis.
    
    Args:
        neuron_types (numpy.ndarray): Array of neuron types to analyze
        arbors (list): List of arbor types to analyze
        offset (float): Value to assign to features that cannot be determined
        jitter_strength (float): Distance to jitter mitochondria positions
        
    Returns:
        tuple: (all_X, all_Y, bodyId_type_arbor) where:
            all_X (numpy.ndarray): Feature matrix
            all_Y (numpy.ndarray): Target variable
            bodyId_type_arbor (numpy.ndarray): Array of neuron IDs, types, and arbor types
    """

    dist_bins, titles, scalar_features = get_mito_pos_features()
    num_hist_bins = [ len(dist_bins[i_hist])-1 for i_hist in range(len(dist_bins)) ]
    final_idx = np.cumsum(num_hist_bins)
    init_idx = np.append(np.array([0]),final_idx[:-1])
    num_features = len(scalar_features) + np.sum(num_hist_bins)

    bodyId_type_arbor = []
    all_X = np.array( [ [] for _ in range(num_features) ] ).T
    all_Y = np.array( [ [] for _ in range(1) ] ).T
    mean_dx = np.zeros( 2 )
    for i_arbor, arbor in enumerate(arbors):
        this_dx = []
        for i_neuron in np.where(np.isin(neuron_quality_np[:,1], neuron_types))[0]:
            bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
            synapse_file = home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv'
            if arbor == 'connecting cable':
                has_section = np.all([neuron_quality[f'has_{this_arbor}'].iloc[i_neuron] for this_arbor in ['axon', 'dendrite']]) 
            else: 
                has_section = neuron_quality[f'has_{arbor}'].iloc[i_neuron]
            if has_section and isfile(synapse_file):# and arbor == 'axon':
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
                    #Y = is_original.copy()
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
                        #
                        d1_idxs, d2_idxs, m_idxs = get_d1_d2_m_seg_idxs(most_down_node, s_np, branch_nodes)
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
        if jitter_strength > 0:
            mean_dx[i_arbor] = np.mean(this_dx)
    bodyId_type_arbor = np.array(bodyId_type_arbor).T
    if jitter_strength > 0:
        return all_X, all_Y, bodyId_type_arbor, mean_dx
    else:
        return all_X, all_Y, bodyId_type_arbor
