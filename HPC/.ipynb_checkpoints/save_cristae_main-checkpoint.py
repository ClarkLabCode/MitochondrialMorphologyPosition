#!/usr/bin/env python


from absl import app
from absl import flags
from neuprint import Client, fetch_roi_hierarchy, skeleton
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.queries import fetch_mitochondria
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import importlib
import random
from os.path import isfile
import neuclease.dvid as dvid
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import warnings

_FLAGS = flags.FLAGS

flags.DEFINE_integer('i_neuron', None, 'ith neuron in neuron_quality', lower_bound=0)

flags.mark_flags_as_required(['i_neuron'])


def main(_):

    i_neuron=_FLAGS.i_neuron
    warnings.filterwarnings("ignore") # ignore all warnings
    token_id = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImdhcnJldHQuc2FnZXJAeWFsZS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpTGNqZXlHYWNnS3NPcTgzdDNfczBoTU5sQUtlTkljRzdxMkU5Rz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgwMTAxNzUwNn0.dzq7Iy01JwSWbKq-Qvi8ov7Hwr0-ozpYeSnOsUD-Mx0"
    np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
    home_dir = '/home/gs697/project/morphology/'

    c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token_id)
    neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
    neuron_quality_np = neuron_quality.to_numpy()
    server = 'http://hemibrain-dvid.janelia.org'
    # uuid of the hemibrain-flattened repository
    gray_scale_uuid = 'a89eb3af216a46cdba81204d8f954786'
    uuid = '15aee239283143c08b827177ebee01b3'

    # import utils file
    spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/utils.py')
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    # import dvid_utils file
    spec = importlib.util.spec_from_file_location('dvid_utils', home_dir+'/util_files/dvid_utils.py')
    dvid_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dvid_utils)

    # import cristae_segmentation file
    spec = importlib.util.spec_from_file_location('cris_seg', home_dir+'/util_files/cristae_segmentation.py')
    cris_seg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cris_seg)

    node_class_dict = {"soma": 1,"axon": 2,"dendrite": 3,"cell body fiber": 4,"connecting cable": 5,"other": 6}

    compute_SA = lambda vols : (np.pi**(1/3)) * ((6 * vols)**(2/3))

    new_fields = []
    new_fields.append( 'number of cristae' )
    new_fields.append( 'cristae volume' )
    new_fields.append( 'cristae surface area' )
    new_fields.append( 'relaxed mito volume' )
    new_fields.append( 'relaxed mito surface area' )

    bodyId, neuron_type = neuron_quality_np[i_neuron,[0,1]]
    skel_file = home_dir + f'/saved_clean_neurons/s_pandas_{bodyId}_{neuron_type}_200nm.csv'
    mito_file_name = home_dir + f'/saved_mito_df_cristae/mito_df_{neuron_type}_{bodyId}.csv'
    #print(neuron_type, bodyId)
    if isfile( skel_file ):
        crisNum_crisVol_crisSA_omVol_omSA = []
        t0 = time.time()
        mito_df = pd.read_csv(home_dir + f'/saved_mito_df/mito_df_{bodyId}.csv')

        s_pandas = pd.read_csv( skel_file )
        s_np = s_pandas.to_numpy()[:,:9].astype(float)

        node_classes = s_pandas['node_classes'].to_numpy()
        mito_classes = utils.find_object_classes(s_np, node_classes, mito_df)
        mito_coords = mito_df[ ['x','y','z'] ].to_numpy()
        mito_idxs = [ np.where(node==s_np[:,0])[0][0] for node in utils.find_closest_nodes(s_np, mito_df) ]
        mito_lengths = mito_df[ 'length along skeleton' ].to_numpy()

        #for i_mito in np.where( mito_classes == node_class_dict['axon'] )[0]:
        for i_mito in range(len(mito_df)): #np.where( np.isin(mito_classes, [node_class_dict['axon'], node_class_dict['dendrite']]) )[0]
            if mito_classes[i_mito] == node_class_dict['soma'] or mito_classes[i_mito] == node_class_dict['other']:
                # don't analyze these mitochondria since they're susceptible to false positives
                crisNum_crisVol_crisSA_omVol_omSA.append( [0,0,0,0,0] )
            else:
                center = mito_coords[i_mito]
                idx = mito_idxs[i_mito]
                mito_subvol, grayscale_subvol, box_zyx = cris_seg.get_subvols( idx, center, s_np, mito_df.iloc[i_mito], bodyId)
                prob_map, bot_val, top_val = cris_seg.get_init_prob_map_lumen(center, grayscale_subvol, mito_subvol)
                IL_bool = cris_seg.RelaxationLabeling(prob_map) # inner lumen is the mitochondrial matrix
                cristae_bool = cris_seg.get_all_cristae(IL_bool, mito_subvol, mito_df.iloc[i_mito]['CA'])
                mito_bool = binary_fill_holes(np.any([cristae_bool, IL_bool],axis=0))
                IL_bool = np.all([cristae_bool==0,mito_bool==1],axis=0) # ensures no issues with noise

                cris_Vol = np.sum(cristae_bool)
                if cris_Vol > 0:
                    cris_mem = cris_seg.get_cristae_mem(IL_bool, cristae_bool)
                    cris_SA = 0
                    for i in range(1,7):
                        cris_SA += np.sum(cris_mem == i) * np.sqrt(i)
                    assert cris_SA > 0 or mito_classes[i_mito] == node_class_dict['other']
                else:
                    cris_SA = 0

                om_mem = cris_seg.get_mito_mem(mito_bool)
                om_SA = 0
                for i in range(1,7):
                    om_SA += np.sum(om_mem == i) * np.sqrt(i)

                crisNum_crisVol_crisSA_omVol_omSA.append( [len(np.unique(measure.label(cristae_bool))),
                                                           np.sum(cristae_bool),
                                                           cris_SA,
                                                           np.sum(mito_bool),
                                                           om_SA] )
        crisNum_crisVol_crisSA_omVol_omSA = np.array(crisNum_crisVol_crisSA_omVol_omSA)
        for i, title in enumerate(new_fields):
            mito_df[title] = crisNum_crisVol_crisSA_omVol_omSA[:,i]

        mito_df.to_csv(mito_file_name, index=False)


if __name__ == '__main__':
    app.run(main)
