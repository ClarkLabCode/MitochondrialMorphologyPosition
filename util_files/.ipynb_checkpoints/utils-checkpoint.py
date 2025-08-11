import numpy as np
import pandas as pd
from neuprint import Client, skeleton
import matplotlib.pyplot as plt
import copy
import networkx as nx
import importlib
from scipy import stats
from ast import literal_eval
import os
from os.path import isfile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.spatial.distance import pdist
from sklearn.neighbors import KernelDensity

import warnings
warnings.filterwarnings("ignore") # ignore all warnings

# import config file
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

#os.environ['TENSORSTORE_CA_BUNDLE'] = config.tensorstore_ca_bundle
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.google_application_credentials

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

def get_hist_bin_size(vals):
    '''
    Freedman–Diaconis rule for histogram bin sizes
    '''
    return 2 * stats.iqr(vals) /  ( len(vals)**(1/3) )

def get_kde_bin_size(label_coords):
    '''
    Silverman's rule of thumb for kde bandwidth
    
    Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis. London: Chapman & Hall/CRC. p. 45. ISBN 978-0-412-24620-3.
    '''
    X = pdist(label_coords)
    h = 0.9 * np.min( [np.std(X), stats.iqr(X) / 1.34] ) * (len(label_coords)**(-1/5))
    
    return h

def get_kde(vals, x_vals, h = None):
    '''
    Silverman's rule of thumb for kde bandwidth
    
    Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis. London: Chapman & Hall/CRC. p. 45. ISBN 978-0-412-24620-3.
    
    
    Inputs:
            vals : measured values you want when fitting kde model
            x_vals : values you want evaluated by kde model
            
    Outputs:
            probs : estimated probability density
    '''
    
    coords = vals[:,np.newaxis]
    if h is None:
        h = get_kde_bin_size(coords)
    kde = KernelDensity(kernel='gaussian', bandwidth = h).fit(coords)
    x_vals = x_vals[:,np.newaxis]
    probs = np.exp(kde.score_samples(x_vals))
    probs /= np.sum(probs) * h
    
    return probs

def get_stars(number, new_line = 4):

    stars = ''
    if number > 0.05:
        return stars

    num_stars = 0
    while not (number < 10**-(num_stars) and number > 10**-(num_stars+1)):
        num_stars += 1
        assert num_stars < 100


    for i_star in range(num_stars):
        if i_star % new_line == 0:
            stars += '\n'
        stars += '*'
    return stars

def spearman_ci(x,y,alpha=0.05):
    ''' calculate Spearman correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Spearman's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.spearmanr(x, y)

    #r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def pearson_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

def get_bonferroni_holm_thresh(pvals, N):
    
    pvals = np.sort(pvals)
    ns = np.flip(np.arange(N))
    thresh = None
    i = 0
    while pvals[i] < (0.05/ns[i]):
        i += 1
    if i == 0:
        return 0
    else:
        return 0.05/ns[i]

def spherical_2_cart(r,theta,phi):

    # given r, theta, and phi, compute the cartesian coordinates
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return x, y, z

def cart_2_spherical(x,y,z):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos( z / r )
    if x==0:
        # if y=0, phi is undefined but we'll call it zero
        phi = 0 if y==0 else np.sign(y) * np.pi/2
    elif y==0:
        # x is non-zero but y is 0
        phi = 0 if x>0 else np.pi
    else:
        # x and y are non-zero, so figure out which quadrant we're in
        if (x>0) and (y>0): phi = np.arctan(y/x)
        elif (x<0) and (y>0): phi = np.arctan(np.abs(x)/y) + np.pi/2
        elif (x<0) and (y<0): phi = np.arctan( np.abs(y) / np.abs(x) ) + np.pi
        elif (x>0) and (y<0):phi = np.arctan( x / np.abs(y) ) + 3*np.pi/2
    return r, theta, phi

def calc_orthonormal_basis( vec ):
    # vec should be the theta,phi components of the initial vector
    assert len(vec)==2

    theta = np.zeros((3,),dtype='float'); phi = np.zeros((3,),dtype='float')
    theta[0] = vec[0]; phi[0] = vec[1]
    phi[1] = phi[0] - np.pi/4 # ensures cos( phi_1 - phi_2 ) is not zero

    if theta[0] == 0:
        # the above equation will have a divide by zero error
        # this is because the input vector is entirely along z-axis
        x,y,z = spherical_2_cart(1, vec[0], vec[1])
        _, theta[1], phi[1] = cart_2_spherical(1,0,0)
    else:
        theta[1] = np.arctan( -np.cos(theta[0]) / (np.sin(theta[0])*np.cos(phi[0]-phi[1])) ) # equation from mathematica

    # find third vector by cross product of first two vectors in cartesian coordinates
    x = np.zeros((3,),dtype=float); y = np.zeros((3,),dtype=float); z = np.zeros((3,),dtype=float)
    for i in [0,1]:
        x[i], y[i], z[i] = spherical_2_cart(1, theta[i], phi[i])
    x[2], y[2], z[2] = np.cross([x[0],y[0],z[0]], [x[1],y[1],z[1]])

    # normalize vectors (although, they should aready by normalized
    norm_factor = np.sqrt( x**2 + y**2 + z**2 )
    x /= norm_factor; y /= norm_factor; z /= norm_factor

    # make sure all vector pairs are orthogonal
    for pair in [[0,1],[0,2],[1,2]]:
        assert np.abs(x[pair[0]]*x[pair[1]] + y[pair[0]]*y[pair[1]] + z[pair[0]]*z[pair[1]]) < 10**-3
    return x,y,z

def find_closest_idxs(s_np, obj_df):
    '''
        Inputs
            - s_pandas: neuron skeleton
            - obj_df : dataframe of objects to attach to skeleton
        Outputs
            - closest_idxs : array of closest skeleton indices to each object in df
    '''

    coords = obj_df[ ['x','y','z'] ].to_numpy() # nx3 array of object coordinates
    closest_idxs = [ np.argmin(np.sum((s_np[:,[1,2,3]] - coord.reshape((1,3)))**2,axis=1)) for coord in coords ]
    return np.array(closest_idxs, dtype=int)
def find_up_idxs(s_np, node):
    '''

    '''

    all_idxs = np.array([], dtype=int)
    idxs = np.where( s_np[:,0] == node )[0]
    while len(idxs) > 0:
        all_idxs = np.append( all_idxs, idxs )
        idxs = np.where( np.isin(s_np[:,5], s_np[idxs,0]) )[0]
    return all_idxs

def get_down_idxs(s_np, node, bool_stop, count_start = False):
    '''
    Starting from node, find all skeleton indices that connect the node to the root or a branch

    Inputs:
        s_np : numpy array of neuron skeleton
        node : node to start search
        bool_stop : boolean array of which node(s) to stop when going down. It always stops at the skeleton root though
        count_start : whether or not to stop if "node" is included as a stop node
    '''
    idxs = np.where( node == s_np[:,0] )[0]
    if not count_start:
        bool_stop[idxs[0]] = False
    while s_np[idxs[-1],5] != -1 and not bool_stop[idxs[-1]]:
        idxs = np.append(idxs, np.where( s_np[idxs[-1],5] == s_np[:,0] )[0][0] )
    return idxs

def find_leaves_and_branches(s_np):
    '''
    Find the leaf and branch nodes in s_np

    Inputs:
        s_np : numpy array of neuron skeleton
    Ouputs:
        leaf_nodes : nodes in s_np that are leafs
        branch_nodes : nodes in s_np that are branches
    '''

    leaf_nodes = s_np[ ~np.isin(s_np[:,0],s_np[:,5]) , 0]

    nodes, count = np.unique(s_np[:,5], return_counts = True)
    branch_nodes = nodes[ np.all([nodes != -1, count > 1],axis=0) ]
    return leaf_nodes, branch_nodes

def get_branch_order(nodes, s_np):
    branch_nodes = find_leaves_and_branches(s_np)[1]
    return np.array([np.sum(np.isin(s_np[get_down_idxs(s_np, node, s_np[:,5]==-1),0], branch_nodes)) for node in nodes ])

def get_leaf_number(nodes, s_np):
    leaf_nodes = find_leaves_and_branches(s_np)[0]
    return np.array([np.sum(np.isin(s_np[find_up_idxs(s_np, node),0], leaf_nodes)) for node in nodes ])

def find_arbor_base_idx(s_np, node_bool):
    node_bool = np.all([node_bool, s_np[:,5]!=-1],axis=0)
    down_idxs = np.array([ np.where(node == s_np[:,0])[0][0] for node in s_np[ node_bool, 5]])
    base_idx = down_idxs[ ~node_bool[down_idxs] ]
    return base_idx[0]

def build_object_graph( orig_s_pandas, df, dist_matrix = None , pad_root = False):
    '''
        Inputs
            - s_pandas: neuron skeleton
            - df : dataframe of objects to attach to skeleton
            - dist_matrix : len(df) x len(df) of pairwise geodesic distances
            - pad_root : include skeleton root as root of object_graph to ensure
                         the directionality of object_graph points toward the skeleton root
        Outputs
            - s_pandas style graph, except the nodes are the objects in df
    '''
    s_pandas = orig_s_pandas.copy()
    s_np = s_pandas.to_numpy()[:,:9].astype(float)
    assert 'distance' in s_pandas.columns

    if pad_root:
        root_idx = np.where( s_np[:,5] == -1 )[0][0]
        root_vec = []
        for column in df.columns:
            if column == 'x': root_vec.append( s_np[root_idx,1] )
            elif column == 'y': root_vec.append( s_np[root_idx,2] )
            elif column == 'z': root_vec.append( s_np[root_idx,3] )
            else: root_vec.append( None )
        new_row = pd.DataFrame(data = np.array(root_vec)[np.newaxis,:], columns = df.columns)
        df = pd.concat([new_row,df[:]]).reset_index(drop = True)
    if dist_matrix is None:
        dist_matrix = get_pairwise_dists( df, s_np )

    labels = ['rowId', 'x', 'y', 'z', 'radius', 'link', 'distance']
    object_np = np.zeros( (len(df),len(labels)) , dtype=float)
    object_np[:,0] = np.arange(len(df))
    object_np[:,[1,2,3]] = df[['x','y','z']].to_numpy()

    if 'size' in df:
        # this df is about mitochondria
        object_np[:,4] = (df['size'].to_numpy(dtype=float) * 3 / (4*np.pi))**(1/3)

    closest_idxs = find_closest_idxs(s_np, df)
    skeleton.reorient_skeleton( s_pandas, rowId = s_np[closest_idxs[0],0] )
    s_np = s_pandas.to_numpy()
    object_np[0,5] = -1; object_np[0,6] = np.inf # treat 1st object as root
    for i_object, idx in enumerate(closest_idxs):
        down_i_object = None
        if np.sum( idx == closest_idxs ) > 1:
            down_i_objects = np.where( idx == closest_idxs )[0]
            xyz = np.array( spherical_2_cart(1, s_np[idx,7], s_np[idx,8]) )[np.newaxis,:]
            down_i_objects = down_i_objects[np.argsort( np.sum( xyz * object_np[down_i_objects,:][:,[1,2,3]], axis=1 ) )] # ascending order of distances
            if down_i_objects[-1] != i_object:
                down_i_object = down_i_objects[ np.where( i_object == down_i_objects )[0][0] + 1 ]
        if down_i_object is None:
            # go down skeleton until you hit a node with an object
            down_idxs = get_down_idxs(s_np, s_np[idx,0], np.isin(np.arange(len(s_np)), closest_idxs))
            if np.sum( down_idxs[-1] == closest_idxs ) > 1:
                # connect i_object to most oupstream object at last down_idx
                down_i_objects = np.where( down_idxs[-1] == closest_idxs )[0]
                xyz = np.array( spherical_2_cart(1, s_np[down_idxs[-1],7], s_np[down_idxs[-1],8]) )[np.newaxis,:]
                down_i_object = down_i_objects[np.argmin( np.sum( xyz * object_np[down_i_objects,:][:,[1,2,3]], axis=1 ) )]
            else:
                down_i_object = np.where( down_idxs[-1] == closest_idxs )[0][0]
        object_np[i_object,5] = down_i_object
        object_np[i_object,6] = dist_matrix[i_object,down_i_object]
        assert i_object != down_i_object or i_object == 0
    # add one to the node labels so that I don't have a node 0
    object_np[:,0] += 1; object_np[:,5] += 1
    object_np[0,5] = -1
    object_graph_df = pd.DataFrame( data=object_np, columns=labels )
    object_graph_df['class'] = s_pandas['node_classes'].to_numpy()[ closest_idxs ]
    for label in np.array(df.columns):
        if label not in labels:
            object_graph_df[label] = df[label]
    return object_graph_df

def find_shared_element(i_path,k_path):
    '''
        Given 2 lists, find first shared element

        Output:
            i_idx : index of first element in i_path that appears in k_path
            k_idx : index of first element in k_path that appears in i_path
    '''

    i_in = np.isin( i_path, k_path, assume_unique = True )
    i_idx = np.argmax( i_in ) # note, argmax stops at first true value
    shared_node = i_path[i_idx]
    k_idx = np.where( k_path==shared_node )[0][0]
    return i_idx, k_idx

def get_up_idx_dist(s_np, coord):
    '''
        Inputs
            - s_np : neuron skeleton
            - coord : coordinate of object under investigation
        Outputs
            - up_idx : index of s_np that is upstream of the object
            - up_dist : distance to the upstream node in skeleton
    '''

    found_up = False
    for s_idx in np.argsort( np.sum( (s_np[:,[1,2,3]] - coord.reshape((1,3)))**2, axis=1 ) - s_np[:,4] )[:20]:
        xyz = np.array( spherical_2_cart(1, s_np[s_idx,7], s_np[s_idx,8]) )
        up_dist = np.sum(xyz * (coord - s_np[s_idx,[1,2,3]]) )
        if up_dist >= 0:
            if up_dist < s_np[s_idx,6]:
                return s_idx, up_dist
            if not found_up:
                found_up = True
                best_s_idx = int(s_idx); best_up_dist = up_dist + 0
    if not found_up:
        # this object is a leaf, so it doesn't have an upstream skeleton node
        # make up_dist negative
        s_idx = np.argmin( np.sum( (s_np[:,[1,2,3]] - coord.reshape((1,3)))**2, axis=1 ) - s_np[:,4] )
        xyz = np.array( spherical_2_cart(1, s_np[s_idx,7], s_np[s_idx,8]) )
        up_dist = np.sum(xyz * (coord - s_np[s_idx,[1,2,3]]) )
        return s_idx, -np.abs(up_dist)

    return best_s_idx, best_up_dist

def path_dist_to_root(coords, s_np, distance_type):
    '''
        Inputs:
            coords : n x 3 array of coordinates
            s_np : skeleton of neuron
        Outputs:
            paths : list of node paths to root
            dists : list of geodesic distances from node to all nodes in paths
    '''
    all_paths = []; all_dists = []
    for i in np.arange(coords.shape[0]):
        up_idx, up_dist = get_up_idx_dist(s_np, coords[i])
        root_idxs = get_down_idxs(s_np, s_np[up_idx,0], s_np[:,5] == -1)
        offset_dist = up_dist if up_dist < s_np[root_idxs[0],6] else s_np[root_idxs[0],6]

        all_paths.append( root_idxs )
        if distance_type == 'geodesic':
            all_dists.append( np.append(np.abs(up_dist), np.cumsum(s_np[root_idxs[:-1],6]) - offset_dist) )
        elif distance_type == 'electronic':
            all_dists.append( np.append( np.array([np.abs(up_dist), s_np[root_idxs[0],6] - offset_dist]), s_np[root_idxs[:-1],6] ) )
    return all_paths, all_dists

def get_pairwise_dists( df_1, s_np, df_2 = None, distance_type = 'geodesic'):
    '''
        Inputs:
            df_1 : pandas dataframe that must contain fields x,y,z
            s_np : numpy array of neuron skeleton
            df_2 : optional 2nd pandas dataframe that must contain fields x,y,z
            distance_type : whether to use 'geodesic' distance or 'electronic' length
        Outputs:
            dist_matrix : len(df_1) x len(df_2) matrix of pairwise distances
    '''

    length_constant = lambda CA, SA : np.sqrt( config.rhoM_rhoI * (CA / SA) )

    # get paths to root for all points in first dataframe
    df_coords = df_1[['x','y','z']].to_numpy()
    i_paths, i_dists = path_dist_to_root(df_coords, s_np, distance_type)

    if df_2 is None:
        # assume we want to compare distance within df_1
        k_paths = i_paths.copy(); k_dists = i_dists.copy()
        #print(k_paths)
        k_df_coords = df_coords.copy()
    else:
        # user passed a 2nd dataframe
        k_df_coords = df_2[['x','y','z']].to_numpy()
        k_paths, k_dists = path_dist_to_root(k_df_coords, s_np, distance_type)


    dist_matrix = np.zeros( (len(i_paths),len(k_paths)) )
    i_all = np.arange( len(i_paths) - 1 ) if (df_2 is None) else np.arange( len(i_paths) )
    for i in i_all:
        i_path = i_paths[i]; i_dist = i_dists[i]
        k_all = np.arange(i+1, len(k_paths)) if (df_2 is None) else np.arange( len(k_paths) )
        for k in k_all:
            k_path = k_paths[k]; k_dist = k_dists[k]
            i_idx, k_idx = find_shared_element(i_path,k_path)
            if (i_idx <= 1) and (k_idx <= 1):
                # project coordinates onto the skeleton and get euclidean distances
                skel_coords = s_np[i_path[0], [1,2,3]]
                xyz = np.array(spherical_2_cart(1, s_np[i_path[0],7], s_np[i_path[0],8]))
                this_dist = np.abs( np.sum((  df_coords[i]-skel_coords) * xyz) -
                                    np.sum((k_df_coords[k]-skel_coords) * xyz) )
                if distance_type == 'electronic':
                    this_dist = np.exp( - (this_dist* 8/1000) / length_constant(s_np[i_path[0],9] * (8/1000)**2, s_np[i_path[0],10] * 8/1000) )
            else:
                if distance_type == 'electronic':
                    total_path = np.append( np.append(i_path[0],i_path[:i_idx]), np.append(k_path[0],k_path[:k_idx]) )
                    lengths = np.append(i_dist[:i_idx+1], k_dist[:k_idx+1]) * 8/1000
                    this_dist = np.exp( - np.sum(lengths / length_constant(s_np[total_path,9] * (8/1000)**2, s_np[total_path,10] * 8/1000) ) )
                else:
                    this_dist = i_dist[i_idx] + k_dist[k_idx]
            dist_matrix[i,k] = this_dist
            if df_2 is None:
                dist_matrix[k,i] = this_dist
    assert np.all(dist_matrix >= 0), 'Distance matrix should be nonnegative'
    return dist_matrix


def overlay_mito_on_df(df, df_graph, s_np, mito_coords, orig_i_mito, length_along_skeleton, mito_radius):
    '''
    Inputs:
        df : either synapse_sites or s_pandas
        df_graph : directed graph of objects in df. If df_type is 'skeleton', this is just s_pandas.
                   Otherwise, this is synapse_graph
        s_np : numpy array of neuron skeleton
        mito_coords : n x 3 array of coordinates belonging to the mitochondria I am overlaying
        orig_i_mito : ith mitochondrion in mito_df
        length_along_skeleton : length of mitochondrion when projected along orientation of skeleton at mito's centroid
        mito_radius : radius with equivalent cross-sectional area of mitochondrion at its centroid
    '''

    def get_overhang(on_idx, mito_coords, s_np, off_idx):
        if off_idx is None:
            # I am computing the up overhang on a leaf node, so the off_idx is not defined
            xyz = - np.array(spherical_2_cart(1, s_np[on_idx,7], s_np[on_idx,8] ))
            overhang = np.max( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[on_idx,[1,2,3]] * xyz) )
        else:
            xyz = s_np[off_idx,[1,2,3]] - s_np[on_idx,[1,2,3]]
            xyz = xyz / np.sqrt(np.sum(xyz**2))
            overhang = np.max( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[on_idx,[1,2,3]] * xyz) )
        return overhang
    def is_overlapping(skel_idx, this_coords):
        # are mito_coords overlapping on this_coords given the orientation specified by skel_idx
        xyz = np.array(spherical_2_cart(1, s_np[skel_idx,7], s_np[skel_idx,8]))
        proj_mito_dists = np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(this_coords * xyz)
        if np.max(proj_mito_dists) >= 0 and np.min(proj_mito_dists) <= 0:
            if np.min( np.sqrt(np.sum( (this_coords[np.newaxis,:] - mito_coords)**2, axis=1)) ) < s_np[skel_idx,4]*2:
                return True
        return False

    edge_nodes = []

    df_coords = df_graph[['x','y','z']].to_numpy(dtype=float)
    this_is_on = np.zeros( len(df_graph), dtype=bool)

    obj_nodes = df_graph['rowId'].to_numpy()
    down_obj_nodes = df_graph['link'].to_numpy()
    dists = np.sqrt( np.sum( (df_graph[['x','y','z']].to_numpy() - np.mean(mito_coords,axis=0)[np.newaxis,:])**2 , axis=1) )
    for i_object in np.where( dists < 2 * length_along_skeleton )[0]:
        skel_idx = np.argmin( np.sum((df_coords[i_object][np.newaxis,:] - s_np[:,[1,2,3]])**2, axis=1) )
        this_is_on[i_object] = is_overlapping(skel_idx, df_coords[i_object])

    if np.any(this_is_on):
        if np.sum(this_is_on) > 1:
            num_nodes = np.zeros( (np.sum(this_is_on), np.sum(this_is_on)) )
            for ii, node in enumerate(obj_nodes[this_is_on]):
                for jj, target_node in enumerate(obj_nodes[this_is_on]):
                    if node != target_node:
                        obj_idx = np.where(obj_nodes == node)[0][0]
                        num_nodes[ii,jj] += 1
                        while down_obj_nodes[obj_idx] > 0 and down_obj_nodes[obj_idx] != target_node:
                            obj_idx = np.where(down_obj_nodes[obj_idx] == obj_nodes)[0][0]
                            num_nodes[ii,jj] += 1
                        if down_obj_nodes[obj_idx] == -1:
                            num_nodes[ii,jj] = 0
            # keep longest length that doesn't touch the root
            if np.max(num_nodes) == 0:
                # the touching nodes are on different segments of the arbor
                # take the closest synapse to the mitochondria as the true synapse on the mitochondrion
                this_dists = [ np.min( np.sqrt(np.sum( (this_coord[np.newaxis,:] - mito_coords)**2, axis=1)) ) for this_coord in df_coords[this_is_on] ]
                closest_i_object = np.where(this_is_on)[0][np.argmin(this_dists)]
                most_up_node, most_down_node = obj_nodes[closest_i_object], obj_nodes[closest_i_object]
            else:
                row, col = np.where( num_nodes == np.max(num_nodes) )
                most_up_node, most_down_node = obj_nodes[np.where(this_is_on)[0][row[0]]], obj_nodes[np.where(this_is_on)[0][col[0]]]
        else:
            most_up_node, most_down_node = obj_nodes[np.where(this_is_on)[0][0]], obj_nodes[np.where(this_is_on)[0][0]]

        path_idxs = get_down_idxs(s_np, most_up_node, s_np[:,0] == most_down_node, count_start = True)
        assert s_np[path_idxs[-1],0] == most_down_node and s_np[path_idxs[0],0] == most_up_node
        if s_np[path_idxs[0],0] not in s_np[:,5]: # most_up_node is a leaf
            edge_nodes.append([None, s_np[path_idxs[0],0], get_overhang(path_idxs[0], mito_coords, s_np, None), 'up'])
        for i_path, idx in enumerate(path_idxs):
            for off_idx in np.where( s_np[:,5] == s_np[idx,0] )[0]:
                if off_idx not in path_idxs:
                    # keep going up the off_idx until mito does not overlap with it
                    if i_path == 0:
                        # this is the most upstream node, so use projected distances
                        while is_overlapping(off_idx, s_np[off_idx,[1,2,3]]) and (s_np[off_idx,0] in s_np[:,5]):
                            off_idx = np.where( s_np[off_idx,0] == s_np[:,5] )[0][0]
                        on_idx = np.where( s_np[off_idx,5] == s_np[:,0] )[0][0]
                        edge_nodes.append([s_np[off_idx,0], s_np[on_idx,0], get_overhang(on_idx, mito_coords, s_np, off_idx), 'up'])
                    else:
                        is_on = np.sqrt(np.sum((s_np[off_idx,[1,2,3]] - s_np[idx,[1,2,3]])**2)) < mito_radius
                        while is_on and (s_np[off_idx,0] in s_np[:,5]):
                            off_idx = np.where( s_np[off_idx,0] == s_np[:,5] )[0][0]
                            is_on = np.sqrt(np.sum((s_np[off_idx,[1,2,3]] - s_np[idx,[1,2,3]])**2)) < mito_radius
                        on_idx = np.where( s_np[off_idx,5] == s_np[:,0] )[0][0]
                        overhang = mito_radius - np.sqrt(np.sum((s_np[on_idx,[1,2,3]] - s_np[idx,[1,2,3]])**2))
                        edge_nodes.append([s_np[off_idx,0], s_np[on_idx,0], overhang, 'up'])
        off_idx = np.where( s_np[path_idxs[-1],5] == s_np[:,0] )[0][0]
        edge_nodes.append([s_np[off_idx,0], s_np[path_idxs[-1],0], get_overhang(path_idxs[-1], mito_coords, s_np, off_idx), 'down'])
    else:
        up_idx = get_up_idx_dist(s_np, np.mean(mito_coords,axis=0))[0]
        down_idx = np.where( s_np[up_idx,5] == s_np[:,0] )[0][0]

        xyz = np.array(spherical_2_cart(1, s_np[up_idx,7], s_np[up_idx,8] ))
        overhang = np.min( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[up_idx,[1,2,3]] * xyz) )
        edge_nodes.append([s_np[down_idx,0], s_np[up_idx,0], -overhang, 'up'])

        xyz = -np.array(spherical_2_cart(1, s_np[down_idx,7], s_np[down_idx,8] ))
        overhang = np.min( np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(s_np[down_idx,[1,2,3]] * xyz) )
        edge_nodes.append([s_np[up_idx,0], s_np[down_idx,0], -overhang, 'down'])

    return edge_nodes

def get_synapse_df( bodyId, synapse_type, group_synapses = True):
    '''
    Get the dataframe describing synapses

    '''

    keep_cols = ['type', 'confidence', 'x', 'y', 'z', 'connecting_bodyId', 'connecting_type', 'connecting_x', 'connecting_y', 'connecting_z']

    neuron_ids = ['a','b','d','c'] if synapse_type == 'pre' else ['d', 'c', 'a', 'b']
    rois = c.all_rois

    q = f"""\
        MATCH (a:Neuron)-[:Contains]->(:`SynapseSet`)-[:Contains]->(b:Synapse)-[:SynapsesTo]->(c:Synapse)<-[:Contains]-(:`SynapseSet`)<-[:Contains]-(d:Neuron)
        WHERE {neuron_ids[0]}.bodyId = {bodyId}
        RETURN {neuron_ids[0]}.bodyId as bodyId,
               {neuron_ids[0]}.type as neuron_type,
               {neuron_ids[1]}.confidence as confidence,
               {neuron_ids[1]}.location.x as x,
               {neuron_ids[1]}.location.y as y,
               {neuron_ids[1]}.location.z as z,
               {neuron_ids[2]}.bodyId as connecting_bodyId,
               {neuron_ids[2]}.type as connecting_type,
               {neuron_ids[3]}.location.x as connecting_x,
               {neuron_ids[3]}.location.y as connecting_y,
               {neuron_ids[3]}.location.z as connecting_z
    """


    synapse_sites = c.fetch_custom(q).drop_duplicates()
    synapse_sites['type'] = synapse_type
    if len(synapse_sites) == 0:
        # this skeleton has no presynaptic sites
        return None

    neuron_type = synapse_sites.iloc[0]['neuron_type']

    coords = synapse_sites[['x','y','z']].to_numpy()
    connecting_coords = synapse_sites[['connecting_x','connecting_y','connecting_z']].to_numpy()

    synapse_ids = np.unique(coords,axis=0, return_inverse=True)[1]
    if group_synapses:
        unique_synapse_sites = synapse_sites.drop_duplicates( ['x', 'y', 'z'] ).copy()
        all_connecting_bodyIds = synapse_sites['connecting_bodyId'].to_numpy()
        all_connecting_types = synapse_sites['connecting_type'].to_numpy()
        connecting_bodyIds = []; connecting_types = []; mean_coords = []

        for synapse_id in np.unique(synapse_ids):
            synapse_bool = synapse_id == synapse_ids
            connecting_bodyIds.append( all_connecting_bodyIds[ synapse_bool ] )
            connecting_types.append( all_connecting_types[ synapse_bool ] )
            mean_coords.append( np.mean(connecting_coords[synapse_bool],axis=0) )
        unique_synapse_sites[ 'connecting_bodyId' ] = connecting_bodyIds
        unique_synapse_sites[ 'connecting_type' ] = connecting_types
        for i_dim, dim in enumerate(['x','y','z']):
            unique_synapse_sites[ f'connecting_{dim}' ] = np.array(mean_coords)[:,i_dim]
            synapse_sites = unique_synapse_sites.copy()
    return synapse_sites[keep_cols]

def generate_mito_coords(s_np, base_idx, other_idx, length):

    if other_idx is None:
        r = - np.array(spherical_2_cart(1, s_np[base_idx,7], s_np[base_idx,8] ))
    else:
        r = s_np[other_idx,[1,2,3]] - s_np[base_idx,[1,2,3]]
        r = r / np.sqrt(np.sum(r**2))
    rs = r[np.newaxis,:] * np.linspace(0,np.abs(length),int(np.abs(length)+1))[:,np.newaxis] * np.sign(length)
    return s_np[base_idx,[1,2,3]][np.newaxis,:] + rs

def generate_path_idxs(s_np, init_idx, length):
    path_idxs = np.array([init_idx])
    while np.sum(s_np[path_idxs,6]) < length:
        path_idxs = np.append( path_idxs, np.where( s_np[path_idxs[-1],5] == s_np[:,0] )[0][0] )
    return path_idxs

def get_shuffled_synapse_df(bodyId):

    i_neuron = np.where( neuron_quality_np[:,0] == bodyId )[0][0]
    neuron_type = neuron_quality_np[i_neuron,1]

    synapse_file = home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv'
    skel_file = home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'
    if not isfile(synapse_file):
        # there is not synapse file, so return None
        return None

    synapse_df = pd.read_csv(synapse_file)
    s_pandas = pd.read_csv(skel_file)
    s_np = s_pandas.to_numpy()
    node_classes = s_pandas['node_classes'].to_numpy()

    syn_idxs = find_closest_idxs(s_np, synapse_df)
    syn_classes = node_classes[syn_idxs]
    syn_coords = np.zeros( (len(synapse_df), 3) )
    for this_class in np.unique(syn_classes):
        for i_synapse in np.where( syn_classes == this_class )[0]:
            idx = np.random.choice( np.where(this_class == node_classes)[0] )
            next_idx = np.where( s_np[:,0] == s_np[idx,5] )[0][0]
            syn_coords[i_synapse] = s_np[idx,[1,2,3]] + np.random.rand() * (s_np[next_idx,[1,2,3]] - s_np[idx,[1,2,3]])
    return pd.DataFrame( data = syn_coords, columns = ['x','y','z'])


def get_shuffled_mito_df(bodyId, arbor, method = 'shuffle', jitter_strength = 0, return_mitos_on = False, shuffle_synapses = False):
    '''
    This function randomly places mitochondria in previously unoccupied regions of the arbor.

    Inputs:
        s_pandas : pandas dataframe of neuron skeleton
        arbor_mito_df : dataframe of mitochondria inside arbor you want to analyze
        arbor : which arbor you want to analyze
        method : 'shuffle' or 'jitter'
        jitter_strength : when method == 'jitter', this is the standard deviation of the gaussian RNG
    Output:
        dataframe of mitochondria after shuffling
    '''
    assert method == 'shuffle' or method == 'jitter'

    def get_this_jitter_edge(edge_nodes, s_np, jitter_strength, bool_stop, length):
        factor = np.random.choice([-1,1]) # 1 means go down skeleton; -1 means go up skeleton
        cur_dist = factor * edge_nodes[2]
        if cur_dist > jitter_strength:
            return [ edge_nodes[0], edge_nodes[1], edge_nodes[2] - factor * jitter_strength ]

        d_dist = 0
        next_idx = np.where( s_np[:,0] == edge_nodes[1] )[0][0] # on_node of the most up node
        num_flips = 0
        while cur_dist + d_dist < jitter_strength:
            cur_dist += d_dist
            cur_idx = int(next_idx)
            if factor == -1:
                # move up the skeleton
                if s_np[cur_idx,0] not in s_np[:,5]:
                    # change direction, because you're at a leaf node
                    d_dist = 0
                    factor *= -1
                    num_flips += 1
                else:
                    # go up the skeleton
                    next_idx = np.random.choice(np.where( s_np[cur_idx,0] == s_np[:,5] )[0])
                    d_dist = s_np[next_idx,6]
                    num_flips = 0
            elif factor == 1:
                # go down the skeleton
                if bool_stop[np.where( s_np[cur_idx,5] == s_np[:,0] )[0][0]] or bool_stop[cur_idx]:
                    # the next node is outside of the region boundaries, so move in other direction
                    factor *= -1
                    d_dist = 0
                    num_flips += 1
                elif np.sum(s_np[cur_idx,0] == s_np[:,5]) > 1:
                    # this is a branch node
                    up_idxs = np.append( np.where( s_np[cur_idx,5] == s_np[:,0] )[0][0], np.where(s_np[cur_idx,0] == s_np[:,5])[0] )
                    next_idx = np.random.choice(up_idxs)
                    if np.where(next_idx == up_idxs)[0][0] == 0:
                        # I'm moving down
                        d_dist = s_np[cur_idx,6]
                        num_flips = 0
                    else:
                        # start moving up
                        d_dist = s_np[next_idx,6]
                        num_flips = 0
                        factor *= -1
                else:
                    # move down
                    next_idx = np.where( s_np[cur_idx,5] == s_np[:,0] )[0][0]
                    d_dist = s_np[cur_idx,6]
                    num_flips = 0
            assert num_flips < 10, 'Infinite While Loop Detected'
        rem_dist = jitter_strength - cur_dist
        if factor == -1:
            # next_idx is up the skeleton
            assert s_np[next_idx,5] == s_np[cur_idx,0]
            return [s_np[next_idx,0], s_np[cur_idx,0], rem_dist, 'up']
        elif factor == 1:
            # next_idx is down the skeleton
            assert s_np[next_idx,0] == s_np[cur_idx,5]
            return [s_np[cur_idx,0], s_np[next_idx,0], s_np[cur_idx,6] - rem_dist, 'up']
    def is_overlapping_func(s_np, skel_idx, this_coords, mito_coords):
        # are mito_coords overlapping on this_coords given the orientation specified by skel_idx
        xyz = np.array(spherical_2_cart(1, s_np[skel_idx,7], s_np[skel_idx,8]))
        proj_mito_dists = np.matmul(mito_coords, xyz[:,np.newaxis]) - np.sum(this_coords * xyz)
        if np.max(proj_mito_dists) >= 0 and np.min(proj_mito_dists) <= 0:
            if np.min( np.sqrt(np.sum( (this_coords[np.newaxis,:] - mito_coords)**2, axis=1)) ) < s_np[skel_idx,4]*2:
                return True
        return False

    neuron_type = neuron_quality_np[np.where( neuron_quality_np[:,0] == bodyId )[0][0],1]
    s_pandas = pd.read_csv( home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv' )
    s_np = s_pandas.to_numpy()
    node_classes = s_pandas['node_classes'].to_numpy()
    if shuffle_synapses:
        synapse_df = get_shuffled_synapse_df(bodyId)
    else:
        synapse_df = pd.read_csv(home_dir + f'/saved_synapse_df/{neuron_type}_{bodyId}_synapse_df.csv')
    syn_coords = synapse_df[['x','y','z']].to_numpy()
    syn_closest_idxs = find_closest_idxs(s_np, synapse_df)
    mito_df = pd.read_csv(home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv')
    mito_idxs = find_closest_idxs(s_np, mito_df)
    mito_classes = node_classes[mito_idxs]
    arbor_mito_df = mito_df.iloc[ np.where(mito_classes == node_class_dict[arbor])[0] ]

    keep_cols = ['class', 'bodyId', 'mitoType', 'roi', 'size', 'r0', 'r1', 'r2', 'mito SA', 'mito CA', 'PC1 Length', 'PC2 Length', 'PC3 Length',
                 'relaxed mito size', 'relaxed mito SA', 'mean matrix intensity', 'mean cristae intensity', 'median matrix intensity', 'median cristae intensity',
                 'std matrix intensity', 'std cristae intensity', 'cristae volume', 'cristae SA', 'number of cristae']
    unshuffled_edge_nodes = [eval(edge_node) for edge_node in arbor_mito_df['edge_nodes'].to_numpy()]
    shuffled_mito_df = arbor_mito_df[keep_cols].copy()
    new_COMs = np.zeros((len(arbor_mito_df),3))

    all_mitos_on = [ [] for _ in range(len(synapse_df)) ]
    all_synapses_on = [ [] for _ in range(len(shuffled_mito_df)) ]
    all_edge_nodes = [ [] for _ in range(len(shuffled_mito_df)) ]
    mito_branch_nodes = [ [] for _ in range(len(shuffled_mito_df)) ]

    if jitter_strength != 0:
        rng = np.random.default_rng()
        jitter_strengths = np.abs( rng.normal(loc = 0, scale = jitter_strength, size = len(arbor_mito_df)) )
        assert method == 'jitter'
    valid_mito = np.ones(len(arbor_mito_df), dtype=bool)
    for i_mito in range(len(arbor_mito_df)):
        new_class = np.inf # ensures we enter while loop
        num_tries = 0
        while new_class != node_class_dict[arbor]:
            num_tries += 1
            try:
                this_edge_nodes = []
                mito_radius = np.sqrt(shuffled_mito_df.iloc[i_mito]['mito CA'] / np.pi)
                assert unshuffled_edge_nodes[i_mito][0][3] == 'up'
                assert unshuffled_edge_nodes[i_mito][-1][3] == 'down'
                orig_up_node = unshuffled_edge_nodes[i_mito][0][1]
                orig_down_node = unshuffled_edge_nodes[i_mito][-1][1]
                shuffled_mito_coords = np.array([ [] for _ in range(3) ]).T
                if method == 'shuffle':
                    on_up_idx = np.random.choice( np.where( node_classes == node_class_dict[arbor])[0] ) # get random node in arbor
                if orig_up_node == unshuffled_edge_nodes[i_mito][-1][0]:
                    if method == 'shuffle':
                        # this mito was between two nodes, so center mitochondrion on on_up_idx
                        overhang = shuffled_mito_df.iloc[i_mito]['PC1 Length'] / 2
                        if s_np[on_up_idx,0] not in s_np[:,5]:
                            # this is a leaf node
                            this_edge_nodes.append( [None, s_np[on_up_idx,0], overhang, 'up'] )
                            shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, None, overhang), axis=0)
                        else:
                            for off_up_idx in np.where( s_np[on_up_idx,0] == s_np[:,5] )[0]:
                                this_edge_nodes.append( [s_np[off_up_idx,0], s_np[on_up_idx,0], overhang, 'up'] )
                                shuffled_mito_coords = np.append(shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, off_up_idx, overhang), axis=0)

                        off_idx = np.where( s_np[on_up_idx,5] == s_np[:,0] )[0][0]
                        shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, off_idx, overhang), axis=0)
                        this_edge_nodes.append( [s_np[off_idx,0], s_np[on_up_idx,0], overhang, 'down'] )
                    else:
                        skel_length = shuffled_mito_df.iloc[i_mito]['PC1 Length']
                        jittered_most_up_edge = get_this_jitter_edge(unshuffled_edge_nodes[i_mito][0], s_np, jitter_strengths[i_mito], node_classes != node_class_dict[arbor], skel_length)
                        on_up_idx = np.where( jittered_most_up_edge[1] == s_np[:,0] )[0][0]
                        up_overhang = jittered_most_up_edge[2]

                        shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, None, up_overhang), axis=0)
                        if s_np[on_up_idx,0] not in s_np[:,5]:
                            # this is a leaf node
                            this_edge_nodes.append( [None, s_np[on_up_idx,0], up_overhang, 'up'] )
                            shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, None, up_overhang), axis=0)
                else:
                    path_idxs = get_down_idxs(s_np, orig_up_node, s_np[:,0] == orig_down_node, count_start = True)
                    assert s_np[path_idxs[-1],0] == orig_down_node
                    skel_length = unshuffled_edge_nodes[i_mito][0][2] + np.sum(s_np[path_idxs[:-1],6]) + unshuffled_edge_nodes[i_mito][-1][2]
                    if skel_length < 0:
                        # something went wrong, so use PC1 length
                        skel_length = shuffled_mito_df.iloc[i_mito]['PC1 Length']

                    if method == 'jitter':
                        #print(unshuffled_edge_nodes[i_mito])
                        jittered_most_up_edge = get_this_jitter_edge(unshuffled_edge_nodes[i_mito][0], s_np, jitter_strengths[i_mito], node_classes != node_class_dict[arbor], skel_length)
                        on_up_idx = np.where( jittered_most_up_edge[1] == s_np[:,0] )[0][0]
                        up_overhang = jittered_most_up_edge[2]
                    else:
                        up_overhang = np.random.rand() * 200/8

                    if s_np[on_up_idx,0] not in s_np[:,5]:
                        # this is a leaf node
                        this_edge_nodes.append( [None, s_np[on_up_idx,0], up_overhang, 'up'] )
                        shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, None, up_overhang), axis=0)
                    else:
                        for off_up_idx in np.where( s_np[on_up_idx,0] == s_np[:,5] )[0]:
                            #this_edge_nodes.append( [s_np[off_up_idx,0], s_np[on_up_idx,0], up_overhang, 'up'] )
                            shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_up_idx, off_up_idx, up_overhang), axis=0)
                if orig_up_node != unshuffled_edge_nodes[i_mito][-1][0] or method == 'jitter':
                    # get new path_idxs
                    path_idxs = np.array([on_up_idx])
                    while np.sum(s_np[path_idxs,6]) < skel_length - up_overhang:
                        path_idxs = np.append( path_idxs, np.where( s_np[path_idxs[-1],5] == s_np[:,0] )[0][0] )
                    for i_path_idx in range(len(path_idxs)):
                        on_idx = path_idxs[i_path_idx]
                        if np.sum(s_np[:,5] == s_np[on_idx,0]) > 1:
                            mito_branch_nodes[i_mito].append( s_np[on_idx,0] )
                        if i_path_idx < len(path_idxs) - 1:
                            shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_idx, path_idxs[i_path_idx+1], s_np[on_idx,6]), axis=0)
                        for off_idx in np.where( s_np[:,5] == s_np[on_idx,0] )[0]:
                            if off_idx not in path_idxs:
                                # keep going up the off_idx until mito does not overlap with it
                                if i_path_idx == 0:
                                    this_edge_nodes.append([s_np[off_idx,0], s_np[on_idx,0], up_overhang, 'up'])
                                else:
                                    is_overlapping = np.sqrt(np.sum((s_np[off_idx,[1,2,3]] - s_np[on_idx,[1,2,3]])**2)) < mito_radius
                                    while is_overlapping and (s_np[off_idx,0] in s_np[:,5]):
                                        off_idx = np.where( s_np[off_idx,0] == s_np[:,5] )[0][0]
                                        is_overlapping = np.sqrt(np.sum((s_np[off_idx,[1,2,3]] - s_np[on_idx,[1,2,3]])**2)) < mito_radius
                                    new_on_idx = np.where( s_np[off_idx,5] == s_np[:,0] )[0][0]
                                    overhang = mito_radius - np.sqrt(np.sum((s_np[new_on_idx,[1,2,3]] - s_np[on_idx,[1,2,3]])**2))
                                    this_edge_nodes.append([s_np[off_idx,0], s_np[new_on_idx,0], overhang, 'up'])
                    on_idx = path_idxs[-1]
                    off_idx = np.where( s_np[on_idx,5] == s_np[:,0] )[0][0]
                    overhang = skel_length - (up_overhang + np.sum(s_np[path_idxs[:-1],6]))
                    shuffled_mito_coords = np.append( shuffled_mito_coords, generate_mito_coords(s_np, on_idx, off_idx, overhang), axis=0)
                    this_edge_nodes.append( [s_np[off_idx,0], s_np[on_idx,0], overhang, 'down'] )
                new_COMs[i_mito] = np.mean(shuffled_mito_coords,axis=0)
                new_class = node_classes[ np.argmin( np.sum( (s_np[:,[1,2,3]] - new_COMs[i_mito][np.newaxis,:])**2, axis=1) ) ]
            except:
                if method == 'jitter':
                    if num_tries == 5:
                        this_edge_nodes = []
                        valid_mito[i_mito] = False
                        new_class = node_class_dict[arbor]
                    else:
                        num_tries += 1
        # find synapses on this mitochondrion
        all_edge_nodes[i_mito] = this_edge_nodes

        # loop through synapses possibly on this mito
        dists = np.sqrt( np.sum( (syn_coords - new_COMs[i_mito][np.newaxis,:])**2 , axis=1) )
        for i_syn in np.where( dists < (2 * shuffled_mito_df.iloc[i_mito]['PC1 Length']) )[0]:
            skel_idx = syn_closest_idxs[i_syn]
            if is_overlapping_func(s_np, skel_idx, syn_coords[i_syn], shuffled_mito_coords):
                all_mitos_on[i_syn].append( i_mito )
                all_synapses_on[i_mito].append( i_syn )
    for this_COM in new_COMs:
        assert node_classes[ np.argmin( np.sum( (s_np[:,[1,2,3]] - this_COM[np.newaxis,:])**2, axis=1) ) ] == node_class_dict[arbor]
    shuffled_mito_df['x'] = new_COMs[:,0]
    shuffled_mito_df['y'] = new_COMs[:,1]
    shuffled_mito_df['z'] = new_COMs[:,2]
    assert np.all( shuffled_mito_df[['x','y','z']].to_numpy() == new_COMs )

    shuffled_mito_df['i_synapses_on'] = all_synapses_on
    shuffled_mito_df['branch_nodes_in'] = mito_branch_nodes
    shuffled_mito_df['edge_nodes'] = all_edge_nodes
    shuffled_mito_graph_df = build_object_graph(s_pandas, shuffled_mito_df.iloc[np.where(valid_mito)[0]], pad_root = True)

    edge_nodes = [ all_edge_nodes[i_mito] for i_mito in np.where(valid_mito)[0] ]

    # REMOVE FIRST ROW OF MITO_GRAPH AND SYNAPSE_GRAPH
    shuffled_mito_graph_np = shuffled_mito_graph_df.to_numpy()
    shuffled_mito_graph_np[:,5] = np.where( shuffled_mito_graph_np[:,5] == 1, -1, shuffled_mito_graph_np[:,5])
    shuffled_mito_graph_df = pd.DataFrame(data= shuffled_mito_graph_np[1:], columns = shuffled_mito_graph_df.columns)
    shuffled_mito_graph_np = shuffled_mito_graph_df.to_numpy()

    for i_mito in range(len(shuffled_mito_graph_df)):
        if shuffled_mito_graph_np[i_mito,5] == -1:
            shuffled_mito_graph_np[i_mito,6] = np.inf
        else:
            i_df = None
            i_overhang = None
            for this_edges in edge_nodes[i_mito]:
                if this_edges[3] == 'down':
                    if this_edges[2] < 0:
                        i_df = shuffled_mito_df.iloc[[i_mito]]
                    else:
                        i_df = s_pandas.iloc[np.where(s_np[:,0] == this_edges[1])[0]]
                    i_overhang = this_edges[2]
            assert i_df is not None and i_overhang is not None

            j_mito = np.where(shuffled_mito_graph_np[i_mito,5] == shuffled_mito_graph_np[:,0])[0][0]

            j_idxs = []; j_overhangs = []
            for this_edges in edge_nodes[j_mito]:
                if this_edges[3] == 'up':
                    j_idxs.append( np.where(s_np[:,0] == this_edges[1])[0][0] )
                    j_overhangs.append( this_edges[2] )
            assert len(j_idxs) > 0
            j_df = s_pandas.iloc[j_idxs]

            dists = get_pairwise_dists(i_df, s_np, df_2 = j_df)[0]
            shuffled_mito_graph_np[i_mito,6] = np.min(dists) - i_overhang - j_overhangs[ np.argmin(dists) ]
    shuffled_mito_graph_df = pd.DataFrame(data= shuffled_mito_graph_np, columns = shuffled_mito_graph_df.columns)
    if method == 'shuffle':
        if return_mitos_on:
            return shuffled_mito_df, all_mitos_on
        return shuffled_mito_graph_df

    # get displacement from original
    true_jittered_dists = np.zeros(len(shuffled_mito_graph_df))
    for i_mito in range(len(shuffled_mito_graph_df)):
        comp_coords = []
        for df in [shuffled_mito_graph_df, arbor_mito_df]:
            this_edge = df['edge_nodes'].to_numpy()[i_mito]
            if type(this_edge) == str:
                this_edge = eval(this_edge)
            base_idx = np.where( s_np[:,0] == this_edge[0][1] )[0][0]
            if this_edge[0][0] is None:
                other_idx = None
            else:
                other_idx = np.where( s_np[:,0] == this_edge[0][0] )[0][0]
            comp_coords.append( generate_mito_coords(s_np, base_idx, other_idx, this_edge[0][2])[-1] )

        coord_df = pd.DataFrame(data=comp_coords, columns = ['x','y','z'])
        true_jittered_dists[i_mito] = get_pairwise_dists(coord_df, s_np )[0,1]
    if return_mitos_on:
        return shuffled_mito_df, true_jittered_dists, jitter_strengths, all_mitos_on
    return shuffled_mito_graph_df, true_jittered_dists, jitter_strengths

def get_mito_object_dists(mito_df, object_df, s_np):
    skel_coords_df = pd.DataFrame( data=s_np[:,[1,2,3]], columns = ['x', 'y', 'z'] )
    if 'edge_nodes' in object_df.columns:
        # the other object is also a dataframe of mitochondria
        other_edge_nodes = []
        for i_mito, this_edge_nodes in enumerate(mito_df['edge_nodes'].to_numpy()):
            if type(this_edge_nodes) == str:
                this_edge_nodes = eval(this_edge_nodes)
            other_edge_nodes.append(this_edge_nodes)
        col_info = []
        for i_mito in range(len(other_edge_nodes)):
            for i_node in range(len(other_edge_nodes[i_mito])):
                on_idx = np.where( s_np[:,0] == other_edge_nodes[i_mito][i_node][1] )[0][0]
                col_info.append( [i_mito, on_idx, other_edge_nodes[i_mito][i_node][2], other_edge_nodes[i_mito][i_node][3]] )
        col_info = np.array( col_info, dtype=object )
        col_ids = col_info[:,0]
        object_idxs = col_info[:,1]
        object_overhangs = col_info[:,2]
        is_object_down = (col_info[:,3]=='down').astype(float) * 2 - 1
        col_df = skel_coords_df.iloc[ object_idxs ]
    else:
        col_df = object_df.copy()
        object_idxs = find_closest_idxs(s_np, object_df)
        object_overhangs = np.zeros(len(object_idxs))
        col_ids = np.arange(len(object_idxs))


    # get all the edge nodes in mito_df and what mitochondrion they belong to
    edge_nodes = []
    for i_mito, this_edge_nodes in enumerate(mito_df['edge_nodes'].to_numpy()):
        if type(this_edge_nodes) == str:
            this_edge_nodes = eval(this_edge_nodes)
        edge_nodes.append(this_edge_nodes)
    row_info = []
    all_object_overhangs = []
    for i_mito in range(len(edge_nodes)):
        for i_node in range(len(edge_nodes[i_mito])):
            on_idx = np.where( s_np[:,0] == edge_nodes[i_mito][i_node][1] )[0][0]
            off_idx = np.where( s_np[:,0] == edge_nodes[i_mito][i_node][0] )[0][0] if edge_nodes[i_mito][i_node][0] is not None else None
            row_info.append( [i_mito, off_idx, on_idx, edge_nodes[i_mito][i_node][2], edge_nodes[i_mito][i_node][3]] )

            if 'edge_nodes' in object_df.columns:
                sign = int(row_info[-1][4] == 'up') * 2 - 1 # whether to flip the sign of is_object_down
                all_object_overhangs.append( object_overhangs * is_object_down * sign )
            else:
                all_object_overhangs.append( object_overhangs )
    all_object_overhangs = np.array(all_object_overhangs)
    row_info = np.array( row_info, dtype=object ) # i_mito, off idx, on idx, overhang distance, direction
    count_other_node = np.zeros( (len(row_info), len(object_idxs)), dtype=bool )

    for i_row in range(len(row_info)):
        if row_info[i_row,4] == 'up':
            if row_info[i_row,1] is not None:
                this_idxs = np.append( row_info[i_row,2], find_up_idxs(s_np, s_np[row_info[i_row,1],0]) )
            else:
                this_idxs = np.array( row_info[i_row,2] )
        elif row_info[i_row,4] == 'down':
            all_idxs = np.arange(len(s_np))
            up_idxs = find_up_idxs(s_np, s_np[row_info[i_row,2],0])
            this_idxs = np.append( row_info[i_row,2], all_idxs[ ~np.isin(all_idxs,up_idxs) ] )
        count_other_node[i_row] = np.isin( object_idxs, this_idxs )

    row_df = skel_coords_df.iloc[ row_info[:,2] ]

    all_dist_matrix = get_pairwise_dists(row_df, s_np, df_2 = col_df )
    all_dist_matrix = (all_dist_matrix - row_info[:,3][:,np.newaxis]) - all_object_overhangs

    assert len(np.unique(row_info[:,0])) == len(mito_df)
    final_dist_matrix = np.zeros( (len(np.unique(row_info[:,0])), len(np.unique(col_ids))) ) + np.inf
    for row, i_mito in enumerate(np.unique(row_info[:,0])):
        for all_row in np.where(i_mito==row_info[:,0])[0]:
            # loop through each object in column dataframe
            for col_id in np.unique(col_ids[ count_other_node[all_row] ]):
                bool_cols = np.all([count_other_node[all_row], col_ids==col_id],axis=0)
                final_dist_matrix[i_mito,col_id] = np.min([final_dist_matrix[i_mito,col_id], np.min( all_dist_matrix[all_row, bool_cols] ) ])
    return final_dist_matrix

def is_isolated_synapse(synapse_df):

    synapse_nx = skeleton.skeleton_df_to_nx(synapse_df.query(f'distance < {500/8}'), directed = False)
    synapse_nodes = synapse_df['rowId'].to_numpy()
    is_isolated = np.ones( len(synapse_df), dtype = bool )
    for cc in nx.connected_components(synapse_nx):
        if len(cc) >= 4:
            is_isolated[ np.isin(synapse_nodes, list(cc)) ] = False
    return is_isolated

def fit_power_laws(neuron_type, arbor, yname, xnames, yfactor = 1, xfactors = None, zscore_thresh = np.inf, fit_separate = True):
    '''
    Fit cristae surface area power law with...
    (1) cristae volume
    (2) outer membrane surface area
    (3) mitochondrion volume
    (4) mitochondrion length
    
    Outputs:
        reg : regularizer object from linear regression of log values
        CoDs : coefficient of determination for each fit
    '''
    
    
    #feat_names = ['cristae SA', 'cristae volume', 'relaxed mito SA', 'relaxed mito size', 'PC1 Length']
    if xfactors is None:
        xfactors = np.ones(len(xnames))
    
    Y = []
    X = np.array( [ [] for _ in range(len(xnames)) ]).T
    for i_neuron in np.where( neuron_quality_np[:,1] == neuron_type )[0]:
        bodyId = neuron_quality_np[i_neuron,0]
        mito_file = home_dir + f'/saved_mito_df/{neuron_type}_{bodyId}_mito_df.csv'
        if isfile(mito_file):
            mito_df = pd.read_csv( mito_file)
            if np.any( mito_df['class'].to_numpy() == node_class_dict[arbor] ):
                mito_df = mito_df.iloc[ np.where(mito_df['class'].to_numpy() == node_class_dict[arbor])[0] ]

                y = mito_df[yname].to_numpy()
                Y = np.append( Y, y[y>0] )
                X = np.append( X, mito_df[xnames].to_numpy()[y>0], axis=0)
    if len(Y) == 0:
        return None, None, None, None
    Y = Y * yfactor
    Y_zscores = np.where( Y > 0, (np.log10(Y) - np.nanmean(np.log10(Y[Y>0]))) / np.nanstd(np.log10(Y[Y>0])), 0)
    X = X * xfactors[np.newaxis,:]
    
    if fit_separate:
        regs = []
        CoDs = np.zeros(len(xnames))
        for i in range(len(xnames)):
            vals = X[:,i]
            zscores = np.where( vals > 0, (np.log10(vals) - np.nanmean(np.log10(vals[vals>0]))) / np.nanstd(np.log10(vals[vals>0])), 0)
            fit_bool = np.all([vals > 0, Y>0, np.abs(zscores) <= zscore_thresh, np.abs(Y_zscores) <= zscore_thresh],axis=0)

            x = sm.add_constant(np.log10(vals[fit_bool]))
            y = np.log10(Y[fit_bool])
            reg = sm.OLS(y, x).fit()
            #reg = LinearRegression().fit(np.log10(vals[fit_bool])[:,np.newaxis], np.log10(Y[fit_bool]))
            regs.append(reg)
            #CoDs[i] = r2_score(np.log10(Y[fit_bool]), reg.predict(np.log10(vals[fit_bool])[:,np.newaxis]))
            CoDs[i] = r2_score(y, reg.predict(x))
        return regs, CoDs, Y, X
    else:
        fit_bool = np.all([np.all(X > 0, axis=1), Y > 0], axis=0)
        x = sm.add_constant(np.log10(X[fit_bool]))
        y = np.log10(Y[fit_bool])
        reg = sm.OLS(y, x).fit()
        CoD = r2_score(y, reg.predict(x))
        
        return reg, CoD, Y, X
