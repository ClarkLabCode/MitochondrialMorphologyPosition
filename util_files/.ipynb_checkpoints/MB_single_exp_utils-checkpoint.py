from neuprint import Client
from tifffile import imread
import numpy as np
from skimage import data, color, measure, morphology
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, convolve
import pandas as pd 
from os.path import isfile
from os import listdir
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC, MitoCriteria as MC
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import minimize


home_dir = '/Users/gs697/Research/positioning_paper'

def RelaxationLabeling(init_prob, num_iters = 5, delta = 0.5):
    prob_map = np.zeros( init_prob.shape )
    for i in range(num_iters):
        sup_matrix = 2 * init_prob - 1
        prob_map[:,:,:] = 0

        prob_map[1: ,:  ,:  ] += sup_matrix[:-1,:  ,:  ] / 6
        prob_map[:-1,:  ,:  ] += sup_matrix[1: ,:  ,:  ] / 6
        prob_map[:  ,1: ,:  ] += sup_matrix[:  ,:-1,:  ] / 6
        prob_map[:  ,:-1,:  ] += sup_matrix[:  ,1: ,:  ] / 6
        prob_map[:  ,:  ,1: ] += sup_matrix[:  ,:  ,:-1] / 6
        prob_map[:  ,:  ,:-1] += sup_matrix[:  ,:  ,1: ] / 6

        prob_map = init_prob + delta * prob_map

        init_prob = np.where( prob_map > 1, 1, prob_map)
        init_prob = np.where( prob_map < 0, 0, prob_map)
    return prob_map > 0.5

def get_intensity_thresh(image_file):
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        return [2, 3]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        return [2, 2.2]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        return [1.9, 2.2]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        return [1.9, 2.2]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        return [1.9, 2.2]
            

def get_bool_seg_idxs(image_file, seg_idxs, i_side):
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        if i_side == 0:
            bool_seg_idxs = ~np.all([seg_idxs[:,0] > 60, seg_idxs[:,2] < 120],axis=0)
        else:
            bool_seg_idxs = ~np.all([seg_idxs[:,2] > 380, seg_idxs[:,0] > 68],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        if i_side == 0:
            bool_seg_idxs = ~np.all([seg_idxs[:,0] > 70, seg_idxs[:,1] > 270, seg_idxs[:,2] > 350],axis=0)
        else:
            bool_seg_idxs = ~np.all([seg_idxs[:,0] > 70, seg_idxs[:,1] > 230, seg_idxs[:,2] < 120],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        if i_side == 0:
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] > 53, seg_idxs[:,1] < 300, seg_idxs[:,2] > 364],axis=0), 
                                     np.all([seg_idxs[:,0] > 53, seg_idxs[:,2] > 400],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] > 50, seg_idxs[:,1] < 280, seg_idxs[:,2] < 140],axis=0), 
                                     np.all([seg_idxs[:,0] > 50, seg_idxs[:,2] > 400],axis=0)],axis=0)
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        if i_side == 0:
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] > 64, seg_idxs[:,1] < 285, seg_idxs[:,2] < 100],axis=0), 
                                     np.all([seg_idxs[:,0] > 64, seg_idxs[:,2] < 50],axis=0)],axis=0)
        else:
            bool_seg_idxs = np.ones(len(seg_idxs), dtype=bool) # peduncles aren't included on this one
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        if i_side == 0:
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] > 46, seg_idxs[:,2] > 410],axis=0), 
                                     np.all([seg_idxs[:,0] > 51, seg_idxs[:,2] > 375, seg_idxs[:,1] < 325],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] > 42, seg_idxs[:,2] < 90],axis=0), 
                                     np.all([seg_idxs[:,0] > 45, seg_idxs[:,2] < 150, seg_idxs[:,1] < 330],axis=0)],axis=0)
            
    return bool_seg_idxs


def get_is_bp1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        if side == 0: comps[ np.all([idxs[:,2] < 180, idxs[:,2] > 140, idxs[:,1] > 300], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,2] > 340, idxs[:,2] < 375, idxs[:,1] > 300],axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        if side == 1: comps[ np.all([idxs[:,2] < 150, idxs[:,2] > 107, idxs[:,1] > 310], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,2] > 315, idxs[:,2] < 360, idxs[:,1] > 325],axis=0) ] = 1
    
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        if side == 1: comps[ np.all([idxs[:,2] < 170, idxs[:,2] > 130, idxs[:,1] > 280], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,2] > 330, idxs[:,2] < 365, idxs[:,1] > 310],axis=0) ] = 1
    
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        if side == 0: comps[ np.all([idxs[:,2] < 150, idxs[:,2] > 110, idxs[:,1] > 295], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,2] > 310, idxs[:,2] < 365, idxs[:,1] > 305],axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        if side == 1: comps[ np.all([idxs[:,2] < 173, idxs[:,2] > 133, idxs[:,1] > 325], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,2] > 340, idxs[:,2] < 377, idxs[:,1] > 325],axis=0) ] = 1

def get_is_bp2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        if side == 0: comps[ idxs[:,2] > 180 ] = 1
        else: comps[ idxs[:,2] < 340 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        if side == 1: comps[ idxs[:,2] > 150 ] = 1
        else: comps[ idxs[:,2] < 315 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        if side == 1: comps[ idxs[:,2] > 170 ] = 1
        else: comps[ idxs[:,2] < 330 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        if side == 0: comps[ idxs[:,2] > 150 ] = 1
        else: comps[ idxs[:,2] < 310 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        if side == 1: comps[ idxs[:,2] > 173 ] = 1
        else: comps[ idxs[:,2] < 340 ] = 1

def get_is_ap1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        if side == 0: comps[ np.all([idxs[:,1] > 310, idxs[:,2] < 140], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] > 310, idxs[:,2] > 375], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        if side == 1: comps[ np.all([idxs[:,1] > 310, idxs[:,2] < 107], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] > 330, idxs[:,2] > 360], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        if side == 1: comps[ np.all([idxs[:,1] > 280, idxs[:,2] < 130], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] > 310, idxs[:,2] > 365], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        if side == 0: comps[ np.all([idxs[:,1] > 280, idxs[:,2] < 110], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] > 300, idxs[:,2] > 365], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        if side == 1: comps[ np.all([idxs[:,1] > 330, idxs[:,2] < 133, idxs[:,2] > 90], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] > 325, idxs[:,2] > 377], axis=0) ] = 1

def get_is_ap2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        if side == 0: comps[ np.all([idxs[:,1] < 310, idxs[:,1] > 280, idxs[:,2] < 140], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] < 310, idxs[:,1] > 260, idxs[:,2] > 375], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        if side == 1: comps[ np.all([idxs[:,1] < 310, idxs[:,1] > 230, idxs[:,2] < 150], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] < 330, idxs[:,1] > 250, idxs[:,2] > 350], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        if side == 1: comps[ np.all([idxs[:,1] < 280, idxs[:,1] > 250, idxs[:,2] < 150], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] < 310, idxs[:,1] > 280, idxs[:,2] > 350], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        if side == 0: comps[ np.all([idxs[:,1] < 280, idxs[:,1] > 225, idxs[:,2] < 150], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] < 300, idxs[:,1] > 250, idxs[:,2] > 340], axis=0) ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        if side == 1: comps[ np.all([idxs[:,1] < 330, idxs[:,1] > 285, idxs[:,2] < 150], axis=0) ] = 1 # this is the left segmented MB
        else: comps[ np.all([idxs[:,1] < 325, idxs[:,1] > 275, idxs[:,2] > 360], axis=0) ] = 1

def get_is_ap3(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_1_00003.tif":
        if side == 0: comps[ idxs[:,1] < 280 ] = 1 # this is the left segmented MB
        else: comps[ idxs[:,1] < 260 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_2_00003.tif":
        if side == 1: comps[ idxs[:,1] < 230 ] = 1 # this is the left segmented MB
        else: comps[ idxs[:,1] < 250 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_3_00003.tif":
        if side == 1: comps[ idxs[:,1] < 250 ] = 1 # this is the left segmented MB
        else: comps[ idxs[:,1] < 280 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_4_00003.tif":
        if side == 0: comps[ idxs[:,1] < 225 ] = 1 # this is the left segmented MB
        else: comps[ idxs[:,1] < 250 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'/20241224_5_00003.tif":
        if side == 1: comps[ idxs[:,1] < 285 ] = 1 # this is the left segmented MB
        else: comps[ idxs[:,1] < 275 ] = 1
    return comps


def get_is_bp1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 150, idxs[:,2] < 190],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 355, idxs[:,2] < 390, idxs[:,1] > 175],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 148, idxs[:,2] < 190],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,2] < 390, idxs[:,1] > 210],axis=0) ] = 1
    return comps

def get_is_bp2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ idxs[:,2] > 190 ] = 1
        else: comps[ idxs[:,2] < 355 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ idxs[:,2] > 190 ] = 1
        else: comps[ idxs[:,2] < 350 ] = 1
    return comps


def get_is_ap1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,2] > 110, idxs[:,1] > 175],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 425, idxs[:,2] > 390, idxs[:,1] > 170],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 148, idxs[:,2] > 115, idxs[:,1] > 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 420, idxs[:,2] > 390, idxs[:,1] > 205],axis=0) ] = 1
    return comps

def get_is_ap2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 170, idxs[:,1] > 110, idxs[:,1] < 175],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 390, idxs[:,1] < 170, idxs[:,1] > 100],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 170, idxs[:,1] > 148, idxs[:,1] < 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 375, idxs[:,1] < 205, idxs[:,1] > 150],axis=0) ] = 1
    return comps


def get_is_ap3(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ idxs[:,1] < 110 ] = 1
        else: comps[ idxs[:,1] < 100 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/dual_imaging/a'b'_ap/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ idxs[:,1] < 148 ] = 1
        else: comps[ idxs[:,1] < 150 ] = 1
    return comps