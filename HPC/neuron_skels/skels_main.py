#!/usr/bin/env python
from absl import app
from absl import flags

# % matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuprint import Client, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
import importlib
import time
import scipy
from skimage import measure
from os.path import isfile
from scipy.ndimage import measurements, binary_fill_holes
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
# uuid of the hemibrain-flattened repository
gray_scale_uuid = config.gray_scale_uuid
uuid = config.uuid
node_class_dict = config.node_class_dict

neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()

os.environ['TENSORSTORE_CA_BUNDLE'] = config.tensorstore_ca_bundle
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.google_application_credentials

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# import skel_clean_utils file
spec = importlib.util.spec_from_file_location('skel_clean_utils', home_dir+'/util_files/skel_clean_utils.py')
skel_clean_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skel_clean_utils)

# import voxel_utils file
spec = importlib.util.spec_from_file_location('voxel_utils', home_dir+'/util_files/voxel_utils.py')
voxel_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voxel_utils)

def main(_):
    i_neuron=_FLAGS.i_neuron
    warnings.filterwarnings("ignore") # ignore all warnings

    min_res = 200.0 # resolution of new skeleton in nanometers
    folder_name_res = int(min_res)
    min_res /= 8.0 # resolution in voxel units

    columns = ['rowId', 'x', 'y', 'z', 'radius', 'link', 'theta', 'phi', 'skel_CA', 'skel_SA', 'mito_CA', 'mito_SA']
    bodyId, neuron_type = all_bodyIds[i_neuron,[0,1]]
    file_name = f'{home_dir}/saved_neuron_skeletons/s_pandas_{bodyId}_{neuron_type}_{folder_name_res}nm.csv'
    assert not isfile(file_name)
    # load pandas dataframe
    s_pandas = c.fetch_skeleton( bodyId, format='pandas', heal=False, with_distances=True) # I will heal the skeleton later
    s_np = s_pandas.to_numpy()
    available_label = np.max(s_np[:,0]) + 1

    new_snp = np.array( [ [] for _ in range(len(columns)) ] ).T
    for idx in np.arange(s_pandas.shape[0]):
        cur_coords = s_np[idx,[1,2,3]].astype(int) # coordinates of root
        dr = int(np.max([50+s_np[idx,4], s_np[idx,4]*2]))
        next_node = s_np[idx,5]
        if next_node == -1:
            # there are no downstream nodes to use as initial guess for plane normal vector
            all_coords = [s_np[idx,[1,2,3]].astype(int)]
        else:
            # use the next node for initial guess for parameters
            next_idx = np.where( s_np[:,0] == next_node )[0][0]
            next_coords = s_np[next_idx,[1,2,3]].astype(int)

            if np.all(next_coords == cur_coords):
                # something weird happened
                all_coords = [s_np[idx,[1,2,3]].astype(int)]
            else:
                r_het = next_coords - cur_coords
                dist = np.sqrt( np.sum( r_het**2 ) )
                num_points = np.ceil( dist / min_res ).astype(int)

                if num_points == 1:
                    # segment is shorter than min_res
                    all_coords = [s_np[idx,[1,2,3]].astype(int)]
                else:
                    # go up to but not including the next node
                    all_coords = []
                    for r_len in np.arange(0,1,1/num_points):
                        all_coords.append( (cur_coords + r_len * r_het).astype(int) )

        dr_small = int(dr/5)
        for i_coord, this_coord in enumerate(all_coords):
            this_coord = voxel_utils.adjust_center(this_coord, dr_small, bodyId) # coordinates of local center of mass
            if this_coord is not None:
                is_cropped = True
                while is_cropped and dr < 600:
                    box_zyx = np.array([np.flip(this_coord)-dr, np.flip(this_coord)+dr])
                    subvol = voxel_utils.get_subvol_any_size(box_zyx, 'segmentation') == bodyId
                    res = voxel_utils.find_best_cross_section( subvol )
                    theta, phi = res.x
                    skel_CS = voxel_utils.find_cross_section( theta, phi, subvol)
                    is_cropped = voxel_utils.is_cropped(skel_CS, box_shape = subvol.shape, theta=theta, phi=phi)
                    if is_cropped:
                        # increase dr for next time
                        if dr > 400:
                            dr += 100
                        else:
                            dr = int(dr*1.5)
                mito_subvol = voxel_utils.get_subvol_any_size(box_zyx, 'mito-objects') == bodyId
                mito_CS = voxel_utils.find_cross_section( theta, phi, mito_subvol , skel_CS = skel_CS)
                skel_CS = binary_fill_holes(np.any([skel_CS==1, mito_CS==1],axis=0))

                skel_CA, skel_SA, COM_coords, COM_idx = voxel_utils.get_foreground_stats(skel_CS, box_zyx, theta=theta, phi=phi)
                mito_CA, mito_SA,          _,       _ = voxel_utils.get_foreground_stats(mito_CS, box_zyx, theta=theta, phi=phi)

                rowId, link, available_label = skel_clean_utils.get_node_labels(idx, i_coord, len(all_coords), next_node, s_np[:,0], available_label)
                feature_vec = np.array( [[ rowId, COM_coords[0], COM_coords[1], COM_coords[2], np.sqrt(skel_CA / np.pi), link, theta, phi, skel_CA, skel_SA, mito_CA, mito_SA]] )
                new_snp = np.append( new_snp, feature_vec, axis=0 )

                init_guess = (theta,phi) # update init guess for the next node
                param_bounds = ( (theta-np.pi/2,theta+np.pi/2), (phi-np.pi/2,phi+np.pi/2) )
                assert len( np.unique(new_snp[:,0]) ) == new_snp.shape[0], 'some node was given the same label'
    new_s_pandas = pd.DataFrame( data=new_snp, columns=columns )
    new_s_pandas.to_csv(file_name, index=False)

if __name__ == '__main__':
    app.run(main)
