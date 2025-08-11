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
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing, binary_erosion, measurements, convolve
from scipy.spatial import ConvexHull
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

# import skel_clean_utils file
spec = importlib.util.spec_from_file_location('skel_clean_utils', home_dir+'/util_files/skel_clean_utils.py')
skel_clean_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skel_clean_utils)

all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()

def main(_):
    i_neuron=_FLAGS.i_neuron
    warnings.filterwarnings("ignore") # ignore all warnings
    bodyId, neuron_type = all_bodyIds[i_neuron, [0,1]]

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

    mito_df = fetch_mitochondria(NC(bodyId=bodyId))
    new_feat_space = []

    for i_mito in range(len(mito_df)):
        if mito_df.iloc[i_mito]['size'] < 15000000:
            feat_vec = []
            coords = voxel_utils.get_full_mito_coords(mito_df.iloc[i_mito])
            box_zyx = [ np.flip( np.min(coords,axis=0) ) - 2, np.flip( np.max(coords,axis=0) ) + 2 ]
            center_idx = np.flip( mito_df.iloc[i_mito][['x','y','z']].to_numpy(dtype=int) ) - box_zyx[0]
            mito_subvol = voxel_utils.get_center_object( voxel_utils.get_subvol_any_size(box_zyx, 'mito-objects') == bodyId , center = center_idx)
            skel_subvol = voxel_utils.get_center_object( voxel_utils.get_subvol_any_size(box_zyx, 'segmentation') == bodyId )
            grayscale_subvol = voxel_utils.get_subvol_any_size(box_zyx, 'grayscale')

            # get cross-sectional area of each mitochondrion at its centroid
            skel_CS, mito_CS = voxel_utils.get_CrossSection_info(bodyId, mito_df.iloc[i_mito][['x','y','z']].to_numpy(), 100, return_mito = True)
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

            new_feat_space.append( np.append( mito_df.iloc[i_mito].to_numpy(), feat_vec) )


    # save features
    new_mito_df = pd.DataFrame( np.array(new_feat_space), columns = np.append( np.array(mito_df.columns), new_feats ) )
    new_mito_df.to_csv(home_dir + f'/saved_data/saved_mito_df_all/{neuron_type}_{bodyId}_mito_df.csv', index = False)

if __name__ == '__main__':
    app.run(main)
