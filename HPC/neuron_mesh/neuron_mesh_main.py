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
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_closing, binary_erosion, measurements, convolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from scipy.spatial.distance import pdist, squareform, cdist
import pickle
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

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# import voxel_utils file
spec = importlib.util.spec_from_file_location('voxel_utils', home_dir+'/util_files/voxel_utils.py')
voxel_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voxel_utils)

def main(_):
    i_neuron=_FLAGS.i_neuron

    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    s_pandas = pd.read_csv( home_dir + f'/saved_clean_skeletons/s_pandas_{bodyId}_{neuron_type}_200nm.csv' )
    s_np = s_pandas.to_numpy()
    node_classes = s_pandas['node_classes'].to_numpy()

    for arbor in ['axon', 'dendrite', 'connecting cable', 'cell body fiber', 'soma']:
        if not isfile(home_dir + f'/saved_data/neuron_meshes/{bodyId}_{neuron_type}_{arbor}.pkl'):
            coords_zyx = np.array( [[],[],[]], dtype=int)
            for idx in np.where( node_classes == node_class_dict[arbor] )[0]:
                dr = np.max([10,s_np[idx,4] * 2])
                this_coord = s_np[idx,[1,2,3]]
                box_zyx = np.array([np.flip(this_coord)-dr, np.flip(this_coord)+dr])
                try:
                    subvol = voxel_utils.get_subvol_any_size(box_zyx, 'segmentation') == bodyId
                except:
                    subvol = np.zeros( 2, dtype=bool )
                if np.any(subvol):
                    idxs = np.array( np.where(subvol) )
                    this_coords_zyx = idxs + box_zyx[0][:,np.newaxis]
                    this_coords_zyx = np.unique( (this_coords_zyx / 4).astype(int), axis=1 )
                    coords_zyx = np.unique( np.concatenate( (coords_zyx,this_coords_zyx), axis=1 ), axis=1 )
            coords_xyz = np.flip(coords_zyx,axis=0)
            bool_skel = np.zeros( np.ptp(coords_xyz,axis=1) + 1, dtype=bool )
            min_coords_xyz = np.min( coords_xyz, axis=1)
            bool_skel[ coords_xyz[0] - min_coords_xyz[0], coords_xyz[1] - min_coords_xyz[1], coords_xyz[2] - min_coords_xyz[2]] = True

            verts, faces, normals, values = measure.marching_cubes(bool_skel)
            verts = (verts + np.min(coords_xyz,axis=1)) * 4

            verts_faces_normals = [verts, faces, normals]
            with open(home_dir + f'/saved_data/neuron_meshes/{bodyId}_{neuron_type}_{arbor}.pkl', 'wb') as f:
                pickle.dump(verts_faces_normals, f)

if __name__ == '__main__':
    app.run(main)
