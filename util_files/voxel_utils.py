"""
Utility functions for working with voxel data in the Drosophila hemibrain dataset.

This module provides tools for:
- Voxel data manipulation and analysis
- Cross-section generation and analysis
- Mitochondria segmentation and tracking
- 3D volume processing
- Coordinate transformations
- Statistical analysis of 3D structures

Key dependencies:
- numpy: Numerical computations
- pandas: Data manipulation
- neuprint: FlyEM database access
- skimage: Image processing
- scipy: Scientific computing
- tensorstore: Large-scale data storage
"""

import numpy as np
import pandas as pd
from neuprint import Client, skeleton
import matplotlib.pyplot as plt
from skimage import measure
import importlib
from ast import literal_eval
from scipy.optimize import minimize
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing, binary_erosion, measurements, convolve
import os
import tensorstore as ts

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

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

def adjust_center(center_init, dr_small, bodyId, data_instance = 'segmentation'):
    """
    Adjusts the initial center point to be centered within the neuron body or mitochondria.
    
    This function iteratively searches around the initial center point to find a valid
    center that lies within the specified structure (neuron body or mitochondria).
    
    Args:
        center_init: Initial center coordinates [x, y, z]
        dr_small: Radius for searching around the center
        bodyId: ID of the neuron body to analyze
        data_instance: Type of data to analyze ('segmentation' or 'mito-objects')
    
    Returns:
        Adjusted center coordinates or None if no valid center found
    """
    def sphere_mask(radius):
        """
        Creates a boolean spherical mask of given radius.
        
        Args:
            radius: Radius of the sphere in voxels
        
        Returns:
            3D boolean array representing spherical mask
        """
        n = radius * 2
        x, y, z = np.meshgrid( np.linspace(-1,1,n), np.linspace(-1,1,n), np.linspace(-1,1,n) )
        return x**2 + y**2 + z**2 <= 1

    subvol = [False]
    num_increment = 1#0
    while not np.any(subvol):
        num_increment = int( num_increment * 2 )
        if dr_small*num_increment > 200: #5/8*1000:
            # center_init is too far from any tissue, so return None
            return None
        circ_bool = sphere_mask(dr_small*num_increment)
        min_zyx = np.flip(center_init).astype(int) - dr_small*num_increment
        max_zyx = np.flip(center_init).astype(int) + dr_small*num_increment
        box_zyx = np.array([ min_zyx, max_zyx])
        subvol = (get_subvol_any_size(box_zyx, data_instance) == bodyId) * circ_bool
        if data_instance == 'mito-objects':
            # multiply subvol by segmentation bool to get voxels that contain the mito and neuron
            subvol = (get_subvol_any_size(box_zyx, 'segmentation') == bodyId) * subvol
        if np.any(subvol):
            subvol = get_center_object(subvol)
    if dr_small == 1:
        # just return coordinates of true value closest to scenter
        body_idxs = np.array( np.where(subvol) )
        voxel_idx = np.argmin(np.sum( (body_idxs - num_increment)**2 ,axis=0))
        center = np.flip( body_idxs[:,voxel_idx] + min_zyx) # x,y,z coordinates
        return center.astype(int)

    if num_increment > 1:
        circ_bool = sphere_mask(dr_small)

    COM_idx = np.array(measurements.center_of_mass(subvol)).astype(int)
    past_subvol = np.copy(subvol); past_min_zyx = np.copy(min_zyx)
    dist_thresh = 2; count = 0
    past_COM_idx = COM_idx + 2*dist_thresh
    #old_min_zyx = min_zyx
    while np.sqrt(np.sum((COM_idx-past_COM_idx)**2)) >= dist_thresh:
        # center of mass has not converged yet
        count += 1
        past_COM_idx = np.copy(COM_idx)
        min_zyx = COM_idx + min_zyx - dr_small
        box_zyx = np.array([ min_zyx, min_zyx + 2*dr_small])
        subvol = (get_subvol_any_size(box_zyx, data_instance) == bodyId) * circ_bool
        if data_instance == 'mito-objects':
            # multiply subvol by segmentation bool to get voxels that contain the mito and neuron
            subvol = (get_subvol_any_size(box_zyx, 'segmentation') == bodyId) * subvol

        if np.any(subvol):
            subvol = get_center_object(subvol)
            past_subvol = np.copy(subvol)
            past_min_zyx = np.copy(min_zyx)
            COM_idx = np.array(measurements.center_of_mass(subvol)).astype(int)
        else:
            min_zyx = np.copy(past_min_zyx)
            subvol = np.copy(past_subvol)
        assert count < 10, 'infinite while loop detected'

    COM_idx = get_closest_true_idx(subvol, COM_idx)

    if subvol[COM_idx[0],COM_idx[1],COM_idx[2]]:
        # use COM as the center
        center = np.flip(COM_idx + min_zyx)
    else:
        # find coordinate closest to COM belonging to bodyId
        body_idxs = np.array( np.where(subvol) )
        voxel_idx = np.argmin(np.sum( (body_idxs - dr_small)**2 ,axis=0))
        center = np.flip( body_idxs[:,voxel_idx] + min_zyx) # x,y,z coordinates

    return center.astype(int)

def get_closest_true_idx(vals, idx):
    """
    Finds the closest True value in a boolean array to a given index.
    
    This function is used to find the nearest valid point in a binary mask
    to a reference point, which is useful for ensuring operations are performed
    on valid data points.
    
    Args:
        vals: Boolean array
        idx: Reference index to find closest True value to
    
    Returns:
        Index of closest True value
    """
    vals = vals==1
    if len(vals.shape) == 3:
        if vals[idx[0],idx[1],idx[2]]:
            # use idx as the center
            return idx
    elif len(vals.shape) == 2:
        if vals[idx[0],idx[1]]:
            # use idx as the center
            return idx
    # find index closest to idx that is true
    all_idxs = np.array( np.where(vals) )
    closest_true_idx = np.argmin(np.sum( (all_idxs - idx[:,np.newaxis])**2 ,axis=0))
    return all_idxs[:,closest_true_idx]

def get_subvol_any_size(init_box_zyx, datatype):
    """
    Retrieves a subvolume of any size from the specified data instance.
    
    This function handles the retrieval of subvolumes from large datasets,
    automatically managing chunking and memory constraints. It supports
    various data types including grayscale, segmentation, and mitochondria.
    
    Args:
        init_box_zyx: Bounding box coordinates in zyx order [min_zyx, max_zyx]
        datatype: Type of data to retrieve ('grayscale', 'segmentation', etc.)
    
    Returns:
        Subvolume data as a numpy array
    """
    if datatype == 'grayscale':
        kvstore = 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg'
    if datatype == 'grayscale_clahe':
        kvstore = 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg'
    elif datatype == 'segmentation':
        kvstore = 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/'
    elif datatype == 'mito-objects':
        kvstore = 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped'
    elif datatype == 'rois':
        kvstore = 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/rois'

    dataset_future = ts.open({'driver': 'neuroglancer_precomputed',
                              'kvstore': kvstore,
                              'context': {'cache_pool': {'total_bytes_limit': 100_000_000}},
                              'recheck_cached_data': 'open'})
    dataset_3d = dataset_future.result()[ts.d['channel'][0]]

    init_box_zyx = np.array(init_box_zyx).astype(int)
    min_point_zyx = np.flip(np.array([0,0,0], dtype=int))
    max_point_zyx = np.flip(np.array([34367,37888,41344], dtype=int))
    min_box_zyx, max_box_zyx = init_box_zyx
    if np.any( np.array(init_box_zyx[0]) <= min_point_zyx ):
        # I'm trying to grab voxels that don't exist in the dataset
        prepend_zyx = min_point_zyx - init_box_zyx[0] + 1
        prepend_zyx = np.max( [prepend_zyx, np.zeros(3, dtype=int)], axis=0 )
        min_box_zyx = min_box_zyx + prepend_zyx
    else:
        prepend_zyx = np.zeros(3, dtype=int)
    if np.any( np.array(init_box_zyx[1]) >= max_point_zyx ):
        # I'm trying to grab voxels that don't exist in the dataset
        append_zyx = init_box_zyx[1] - max_point_zyx + 1
        append_zyx = np.max([append_zyx, np.zeros(3, dtype=int)],axis=0)
        max_box_zyx = max_box_zyx - append_zyx
    else:
        append_zyx = np.zeros(3, dtype=int)
    assert np.all( max_box_zyx > min_box_zyx ), f'{min_box_zyx} {max_box_zyx}'
    min_box_zyx = np.array(min_box_zyx).astype(int)
    max_box_zyx = np.array(max_box_zyx).astype(int)
    box_zyx = np.array([min_box_zyx, max_box_zyx]).astype(int)

    subvol_xyz = np.array( dataset_3d[min_box_zyx[2] : max_box_zyx[2], min_box_zyx[1] : max_box_zyx[1], min_box_zyx[0] : max_box_zyx[0]] ).astype(float)
    subvol = np.transpose(subvol_xyz, axes = [2,1,0]) # subvol_zyx

    for i_axis in range(3):
        if prepend_zyx[i_axis] > 0:
            box_zyx[0][i_axis] = box_zyx[0][i_axis] - prepend_zyx[i_axis]

            prepend_shape = np.array(subvol.shape)
            prepend_shape[i_axis] = prepend_zyx[i_axis]
            subvol = np.append( np.zeros(prepend_shape), subvol, axis=i_axis )
        if append_zyx[i_axis] > 0:
            box_zyx[1][i_axis] = box_zyx[1][i_axis] + append_zyx[i_axis]

            append_shape = np.array(subvol.shape)
            append_shape[i_axis] = append_zyx[i_axis]
            subvol = np.append( subvol, np.zeros(append_shape), axis=i_axis )
    for i in range(2):
        assert np.all( np.array(init_box_zyx[i]) == np.array(box_zyx[i]) ), f'init_box_zyx = {init_box_zyx}   |   box_zyx = {box_zyx}'
    assert np.all( np.array(subvol.shape) == (box_zyx[1] - box_zyx[0]) ), f'subvol shape = {subvol.shape}  |  box_shape = {box_zyx[1] - box_zyx[0]}'
    return subvol

def get_center_object(array, center = None):
    """
    Extracts the central connected component from a binary volume.
    
    This function identifies the largest connected component in the volume
    and returns a binary mask containing only that component. This is useful
    for isolating individual objects or structures in the data.
    
    Args:
        array: Binary volume to process
        center: Optional center coordinates to use as reference
    
    Returns:
        Binary mask containing only the central connected component
    """
    assert np.any(array==1)
    # consider using measure.regionprops for this part
    plane_labels = measure.label(array.astype(int), background=0)
    unique_labels = np.unique(plane_labels)
    unique_labels = unique_labels[ unique_labels>0 ] # get rid of label for background

    if len(array.shape) == 2:
        # this is a cross section
        row, col = np.ceil( np.array(plane_labels.shape) / 2).astype(int) if center is None else center
        if plane_labels[ row, col ] == 0:
            # find cloest row,col element with labels greater than 0
            foreground_idxs = np.array( np.where(plane_labels > 0) ) # 2xn matrix
            label_idx = np.argmin( np.sum( ( foreground_idxs - np.array([[row],[col]]) )**2, axis=0 ) )
            row, col = foreground_idxs[:,label_idx]
        center_label = plane_labels[row,col]
    else:
        # this is a 3D Volume
        row, col, page = np.ceil( np.array(plane_labels.shape) / 2).astype(int) if center is None else center
        if plane_labels[ row, col, page ] == 0:
            # find cloest row,col element with labels greater than 0
            foreground_idxs = np.array( np.where(plane_labels > 0) ) # 3xn matrix
            label_idx = np.argmin( np.sum( ( foreground_idxs - np.array([[row],[col],[page]]) )**2, axis=0 ) )
            row, col, page = foreground_idxs[:,label_idx]
        center_label = plane_labels[row,col,page]
    assert center_label > 0, "center label should not be the background"
    return center_label == plane_labels

def calc_plane_coords(theta, phi, box_zyx, height=None, width=None):
    """
    Calculates coordinates for a plane in 3D space.
    
    This function generates the coordinates for a plane defined by spherical
    angles theta and phi, within a given bounding box. The plane can be
    specified with custom height and width, or will default to the maximum
    dimensions that fit within the bounding box.
    
    Args:
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        box_zyx: Bounding box coordinates in zyx order [min_zyx, max_zyx]
        height: Optional height of the plane
        width: Optional width of the plane
    
    Returns:
        Array of plane coordinates
    """
    center = np.flip(np.array(box_zyx[1] + box_zyx[0])/2).astype(int)
    x, y, z = utils.calc_orthonormal_basis( [theta,phi] )
    if height is None:
        max_dr = np.max( box_zyx[1] - box_zyx[0] )
        height, width = int(max_dr), int(max_dr)

    height_vec = np.array( [x[1],y[1],z[1]] ).reshape((1,1,3)) * height
    width_vec = np.array( [x[2],y[2],z[2]] ).reshape((1,1,3)) * width
    base_point = np.ceil(center - (height_vec + width_vec)*(1/2.0)).astype(int)
    hor_grid, ver_grid = np.meshgrid( np.linspace(-0.5,0.5,num=width),np.linspace(-0.5,0.5,num=height) )
    assert hor_grid.shape[0]==height; assert hor_grid.shape[1]==width
    hor_grid = np.repeat( hor_grid[:,:,np.newaxis], 3, axis=2 )
    ver_grid = np.repeat( ver_grid[:,:,np.newaxis], 3, axis=2 )

    plane_coords = center.reshape((1,1,3)) + (hor_grid * width_vec) + (ver_grid * height_vec)
    return plane_coords.astype(int)

def find_cross_section(theta, phi, subvol, height=None, width=None, skel_CS=None):
    """
    Finds the cross-section of a 3D volume along a specified plane.
    
    This function calculates the intersection of a plane (defined by theta and phi)
    with a 3D volume. It can optionally use a skeleton cross-section as a reference.
    
    Args:
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        subvol: 3D volume data
        height: Optional height of the cross-section
        width: Optional width of the cross-section
        skel_CS: Optional skeleton cross-section for reference
    
    Returns:
        Array containing the cross-section values
    """
    box_zyx = np.array([ [0,0,0], subvol.shape ]).astype(int) # I don't need to real coordinates just to get the CS values
    plane_coords = calc_plane_coords(theta, phi, box_zyx, height = height, width = width)
    flat_plane_vals = np.zeros( (plane_coords.shape[0] * plane_coords.shape[1], ), dtype=type(subvol[0,0,0]) )
    min_coords = np.flip(box_zyx[0]) # x,y,z coordinates
    plane_idxs = (plane_coords - min_coords.reshape((1,1,3))).astype(int)
    flat_plane_idxs = plane_idxs.reshape( ( plane_idxs.shape[0]*plane_idxs.shape[1], 3 ) )
    flat_valid_idxs = ~np.any( [np.any(flat_plane_idxs<0,axis=1), np.any(flat_plane_idxs>=np.flip(box_zyx[1] - box_zyx[0])[np.newaxis,:],axis=1)], axis=0 )
    flat_plane_vals[ flat_valid_idxs ] = subvol[ flat_plane_idxs[flat_valid_idxs,2], flat_plane_idxs[flat_valid_idxs,1], flat_plane_idxs[flat_valid_idxs,0] ]
    plane_vals = flat_plane_vals.reshape( (plane_coords.shape[0], plane_coords.shape[1]) )

    if np.all( np.isin( plane_vals.astype(int), [0,1] ) ) and np.all( np.isin( [0,1], plane_vals.astype(int) ) ):
        # this is a boolean array, so only keep the forground group nearest to the center
        if skel_CS is None:
            plane_vals = get_center_object( binary_fill_holes(plane_vals) )
        else:
            plane_labels = measure.label(plane_vals.astype(int), background=0)
            unique_labels = np.unique( plane_labels[skel_CS==1] )

            if np.all(unique_labels == 0):
                return np.zeros(plane_vals.shape)
            unique_labels = unique_labels[ unique_labels>0 ] # get rid of label for background
            plane_vals = binary_fill_holes(np.isin(plane_labels, unique_labels))
    elif skel_CS is not None:
        plane_vals = np.where( binary_fill_holes(skel_CS==1), plane_vals, 0 )

    return plane_vals

def find_best_cross_section(subvol):
    """
    Finds the optimal cross-section orientation by minimizing the cross-sectional area.
    
    This function uses optimization to find the plane orientation that produces
    the most compact cross-section of the volume, with a penalty for cropped sections.
    
    Args:
        subvol: 3D volume to analyze
    
    Returns:
        Minimizer object containing the optimal theta and phi angles
    """
    def CA_obj_fun(plane_norm, subvol):
        theta, phi = plane_norm
        CS = find_cross_section( theta, phi, subvol )
        CA = np.sum( CS > 0 )
        if is_cropped(CS, box_shape = subvol.shape, theta=theta, phi=phi):
            # multiply loss funtion to penalize for cropping
            CA *= 1000
        return CA

    # get best initial guess of theta, phi
    init_thetas = np.linspace(0, np.pi, 7)
    init_phis = np.linspace(0, np.pi, 7)
    min_area = np.inf
    for theta, phi in zip(init_thetas, init_phis):
        this_area = CA_obj_fun((theta,phi), subvol)
        if this_area < min_area:
            # found a plane normal with smaller cross-sectional area than area_0
            init_guess = (theta, phi)
            min_area = this_area + 0
    param_bounds = ( (init_guess[0]-np.pi/2,init_guess[0]+np.pi/2), (init_guess[1]-np.pi/2,init_guess[1]+np.pi/2) )
    res = minimize(CA_obj_fun, init_guess, args=(subvol),method='Nelder-Mead',bounds=param_bounds)
    assert res.fun <= min_area
    return res

def get_border_mask(bool_array, border_type):
    """
    Creates a mask of the border/edge of a boolean array.
    
    This function can identify either the inner border (True pixels touching False)
    or outer border (False pixels touching True) of a binary mask.
    
    Args:
        bool_array: Boolean array to find border of
        border_type: Type of border to extract ('inner' or 'outer')
    
    Returns:
        Boolean array marking the border region
    """
    if border_type == 'off':
        border_mask = np.all([ binary_dilation(bool_array == 1) == 1, bool_array == 0], axis=0)
    elif border_type == 'on':
        border_mask = np.all([ binary_erosion(bool_array == 1) == 0, bool_array == 1], axis=0)
    return border_mask

def is_cropped(vals, box_shape=None, theta=None, phi=None):
    """
    Determines if a volume or cross-section is cropped at the edges.
    
    This function checks whether the data extends to the boundaries of the
    volume or cross-section, indicating that it might be cropped.
    
    Args:
        vals: Array to check for cropping
        box_shape: Optional shape of bounding box
        theta: Optional azimuthal angle for cross-sections
        phi: Optional polar angle for cross-sections
    
    Returns:
        Boolean indicating if the array is cropped
    """
    if theta is None:
        # vals should be 3D
        assert len(vals.shape) == 3
        cropped = False
        for side in [0,-1]:
            cropped = cropped or np.any(vals[side,:,:] == 1)
            cropped = cropped or np.any(vals[:,side,:] == 1)
            cropped = cropped or np.any(vals[:,:,side] == 1)
    else:
        # vals should be 2D
        assert len(vals.shape) == 2
        is_included = find_cross_section( theta, phi, np.ones( box_shape ))
        cropped = np.any( np.all([get_border_mask(is_included, 'on') == 1, vals == 1], axis=0) )
    return cropped

def get_full_mito_coords(mito_series):
    """
    Extracts complete mitochondria coordinates from a data series.
    
    This function retrieves all voxel coordinates belonging to a mitochondrion,
    handling cases where the mitochondrion might span multiple subvolumes.
    
    Args:
        mito_series: Pandas series containing mitochondria data
    
    Returns:
        Array of mitochondria coordinates
    """
    dr_small = 1
    bodyId = mito_series['bodyId']
    center_xyz = adjust_center(mito_series[['x','y','z']].to_numpy(), dr_small, bodyId, data_instance = 'mito-objects') # ensures the center is actually on a mitochondrion
    center = np.flip(center_xyz) # zyx
    dr = int( ((3 * mito_series['size'] / (np.pi*4))**(1/3)) * 1.5 )
    is_cropped = [True]
    all_box_zyx = []
    mito_coords_zyx = np.array( [[],[],[]], dtype=int)
    while np.any( is_cropped ):
        box_zyx = [ center - dr, center + dr]
        all_box_zyx.append(box_zyx)
        mito_subvol = get_center_object( get_subvol_any_size(box_zyx, 'mito-objects') == bodyId )
        mito_idxs = np.array(np.where(mito_subvol)) # 3xn array of mito indices
        min_coord_zyx = np.array(box_zyx[0]).reshape((3,1)).astype(int)

        # find which coordinates of mito subvol are new
        if mito_coords_zyx.shape[1] > 0:
            coord_not_in = []
            for i_coord in [0,1,2]:
                coord_not_in.append( ~np.isin( box_zyx[0][i_coord] + mito_idxs[i_coord], mito_coords_zyx[i_coord] ) )
            new_coord_bool = np.any( coord_not_in, axis=0 )
            assert len(new_coord_bool) == mito_idxs.shape[1]
        else:
            new_coord_bool = np.ones( ( mito_idxs.shape[1], ), dtype=bool )
        mito_coords_zyx = np.unique( np.concatenate( (mito_coords_zyx,mito_idxs[:,new_coord_bool] + min_coord_zyx), axis=1 ), axis=1 )

        # loop through mito coords to see if all voxels are uncropped
        is_cropped = np.ones( (mito_coords_zyx.shape[1],), dtype=bool)
        for this_box_zyx in all_box_zyx:
            min_zyx, max_zyx = this_box_zyx
            good_bool = np.all([np.all(mito_coords_zyx > min_zyx[:,np.newaxis],axis=0),
                                np.all(mito_coords_zyx < (max_zyx[:,np.newaxis] - 1),axis=0)], axis=0 )
            is_cropped[good_bool] = False # these voxels are uncroped

        if np.any(is_cropped):
            dr = 100
            cropped_coords_zyx = mito_coords_zyx[:,is_cropped]
            center = cropped_coords_zyx[:,np.random.randint(cropped_coords_zyx.shape[1])]

    # consider only return the unique mito coordinates
    mito_coords_xyz = np.flip( mito_coords_zyx.T, axis=1) # nx3 array of mito coordinates
    assert mito_coords_xyz.shape[0] == np.unique(mito_coords_xyz,axis=0).shape[0]
    return mito_coords_xyz.astype(int)

def get_foreground_stats(vals, box_zyx, fill_holes=True, theta=None, phi=None, return_COM=True):
    """
    Calculates statistics about the foreground objects in a boolean array.
    
    Computes various metrics including area/volume, surface area/perimeter,
    and optionally center of mass for foreground objects in 2D or 3D.
    
    Args:
        vals: Boolean array to analyze
        box_zyx: Bounding box coordinates
        fill_holes: Whether to fill holes in objects
        theta: Optional azimuthal angle for cross-sections
        phi: Optional polar angle for cross-sections
        return_COM: Whether to return center of mass
    
    Returns:
        Dictionary of calculated statistics
    """
    if not np.any(vals):
        if not return_COM:
            return 0, 0
        else:
            return 0, 0, None, None

    offset = 1
    kernel_size = 2*offset + 1
    if len(vals.shape) == 3:
        xv, yv, zv = np.meshgrid(np.arange(kernel_size)-offset,
                                 np.arange(kernel_size)-offset,
                                 np.arange(kernel_size)-offset)
        conv_kernel = np.ones( (kernel_size, kernel_size, kernel_size) )
        conv_kernel[np.sqrt( xv**2 + yv**2 + zv**2 ) > offset] = 0
        conv_kernel[offset,offset,offset] = 0
    elif len(vals.shape) == 2:
        xv, yv = np.meshgrid(np.arange(kernel_size)-offset,
                             np.arange(kernel_size)-offset)
        conv_kernel = np.ones( (kernel_size, kernel_size) )
        conv_kernel[np.sqrt( xv**2 + yv**2 ) > offset] = 0
        conv_kernel[offset,offset] = 0
    num_bkg_edges = convolve((vals==0).astype(float),conv_kernel) # number of background pixels within offset radius
    border_mask = get_border_mask(vals, 'on')
    SA = np.sum( np.sqrt( num_bkg_edges[border_mask==1] ) )
    enclosed_space = np.sum(vals==1)

    if not return_COM:
        return enclosed_space, SA

    COM_idx = get_closest_true_idx(vals, np.array(measurements.center_of_mass(vals)).astype(int))
    if theta is None:
        # vals should be 3D
        assert len(vals.shape) == 3 and vals[COM_idx[0],COM_idx[1],COM_idx[2]] == 1
        COM_coords = np.flip(box_zyx[0] + COM_idx)
    else:
        # vals should be 2D
        assert len(vals.shape) == 2 and vals[COM_idx[0],COM_idx[1]] == 1
        plane_coords = calc_plane_coords(theta, phi, box_zyx)
        COM_coords = plane_coords[COM_idx[0], COM_idx[1],:]
    assert np.all(np.flip(COM_coords) >= box_zyx[0]) and np.all(np.flip(COM_coords) < box_zyx[1])

    return enclosed_space, SA, COM_coords, COM_idx

def get_CrossSection_info(bodyId, coord, dr, return_mito=False, return_grayscale=False, 
                         return_plane_coords=False, max_dr=700):
    """
    Retrieves comprehensive information about a cross-section through a neuron.
    
    This function extracts cross-section data and various metrics at a given
    coordinate along a neuron, optionally including mitochondria and grayscale data.
    
    Args:
        bodyId: ID of the neuron to analyze
        coord: Coordinates of the cross-section center
        dr: Radius around center to analyze
        return_mito: Whether to return mitochondria data
        return_grayscale: Whether to return raw image data
        return_plane_coords: Whether to return plane coordinates
        max_dr: Maximum allowed radius
    
    Returns:
        Dictionary containing requested cross-section information
    """
    is_crop = True
    adjusted_coord = adjust_center(coord, 5, bodyId) # coordinates of local center of mass
    #print(adjusted_coord)
    if adjusted_coord is None:
        output = []
        for _ in range(1 + int(return_mito) + int(return_grayscale) + int(return_plane_coords)):
            output.append(None)
        return output
    else:
        while is_crop:
            box_zyx = [ np.flip(adjusted_coord).astype(int) - dr, np.flip(adjusted_coord).astype(int) + dr ]
            skel_subvol = get_subvol_any_size(box_zyx, 'segmentation') == bodyId
            res = find_best_cross_section( skel_subvol )
            theta, phi = res.x
            skel_CS = find_cross_section( theta, phi, skel_subvol)
            is_crop = is_cropped(skel_CS, box_shape = skel_subvol.shape, theta=theta, phi=phi)
            if is_crop and dr == max_dr:
                # even after going to the max_dr, the skeleton is still cropped so just return None
                output = []
                for _ in range(1 + int(return_mito) + int(return_grayscale) + int(return_plane_coords)):
                    output.append(None)
                return output
            if is_crop:
                # going back through the loop, so use a bigger dr
                if dr < 400: dr += 100
                else: dr += 150
                dr = int(dr)
                if dr > max_dr:
                    dr = int(max_dr)
        del skel_subvol
        output = [skel_CS]

        if return_mito:
            mito_subvol = get_subvol_any_size(box_zyx, 'mito-objects') == bodyId
            mito_CS = find_cross_section( theta, phi, mito_subvol, skel_CS = skel_CS)
            del mito_subvol
            output.append(mito_CS)
        if return_grayscale:
            grayscale_subvol = get_subvol_any_size(box_zyx, 'grayscale')
            grayscale_CS = find_cross_section( theta, phi, grayscale_subvol, skel_CS = skel_CS)
            #grayscale_CS = find_cross_section( theta, phi, grayscale_subvol, skel_CS = binary_dilation(skel_CS, iterations=2) )
            del grayscale_subvol
            output.append(grayscale_CS)
        if return_plane_coords:
            output.append( calc_plane_coords(theta, phi, box_zyx) )
    return output
