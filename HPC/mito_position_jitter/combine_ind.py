

import os
import importlib
from os.path import isfile
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

home_dir = config.home_dir
jitter_strengths = config.jitter_strengths
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()

for neuron_type in config.analyze_neurons:
    for jitter_strength in jitter_strengths:
        X_df_file = home_dir + f'/saved_data/position_feats_jitter/{neuron_type}_X_df_Jitter_{int(jitter_strength)}.csv'
        Y_df_file = home_dir + f'/saved_data/position_feats_jitter/{neuron_type}_Y_df_Jitter_{int(jitter_strength)}.csv'
        bodyId_type_arbor_df_file = home_dir + f'/saved_data/position_feats_jitter/{neuron_type}_bodyId_type_arbor_Jitter_{int(jitter_strength)}.csv'
        mean_dx_df_file = home_dir + f'/saved_data/position_feats_jitter/{neuron_type}_mean_dx_Jitter_{int(jitter_strength)}.csv'
        rms_df_file = home_dir + f'/saved_data/position_feats_jitter/{neuron_type}_rms_Jitter_{int(jitter_strength)}.csv'

        all_X = []
        for i_neuron in np.where( neuron_quality_np[:,1] == neuron_type )[0]:
            this_X_df_file = home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_X_df_Jitter_{int(jitter_strength)}_{i_neuron}.csv'
            this_Y_df_file = home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_Y_df_Jitter_{int(jitter_strength)}_{i_neuron}.csv'
            this_bodyId_type_arbor_df_file = home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_bodyId_type_arbor_Jitter_{int(jitter_strength)}_{i_neuron}.csv'
            this_mean_dx_df_file = home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_mean_dx_Jitter_{int(jitter_strength)}_{i_neuron}.csv'
            this_rms_df_file = home_dir + f'/saved_data/ind_position_feats_jitter/{neuron_type}_rms_Jitter_{int(jitter_strength)}_{i_neuron}.csv'

            if isfile(this_X_df_file):
                if len(all_X) == 0:
                    all_X = pd.read_csv(this_X_df_file).to_numpy()
                    all_Y = pd.read_csv(this_Y_df_file).to_numpy()
                    bodyId_type_arbor = pd.read_csv(this_bodyId_type_arbor_df_file).to_numpy()
                    mean_dx = pd.read_csv(this_mean_dx_df_file).to_numpy()
                    rms = pd.read_csv(this_rms_df_file).to_numpy()

                    X_cols = np.array(pd.read_csv(this_X_df_file).columns)
                    Y_cols = np.array(pd.read_csv(this_Y_df_file).columns)
                    bodyId_cols = np.array(pd.read_csv(this_bodyId_type_arbor_df_file).columns)
                    mean_dx_cols = np.array(pd.read_csv(this_mean_dx_df_file).columns)
                    rms_cols = np.array(pd.read_csv(this_rms_df_file).columns)
                else:
                    all_X = np.append( all_X, pd.read_csv(this_X_df_file).to_numpy(), axis=0 )
                    all_Y = np.append( all_Y, pd.read_csv(this_Y_df_file).to_numpy(), axis=0 )
                    bodyId_type_arbor = np.append( bodyId_type_arbor, pd.read_csv(this_bodyId_type_arbor_df_file).to_numpy(), axis=0 )
                    mean_dx = np.append( mean_dx, pd.read_csv(this_mean_dx_df_file).to_numpy(), axis=0 )
                    rms = np.append( rms, pd.read_csv(this_rms_df_file).to_numpy(), axis=0)
        if len(all_X) > 0:
            X_df = pd.DataFrame( all_X, columns = X_cols)
            Y_df = pd.DataFrame( all_Y, columns = Y_cols)
            bodyId_type_arbor_df = pd.DataFrame( bodyId_type_arbor, columns = bodyId_cols )
            assert np.all( np.mean(mean_dx,axis=0) > -np.inf )
            mean_dx_df = pd.DataFrame( np.mean(mean_dx,axis=0)[np.newaxis,:], columns = mean_dx_cols )
            rms_df = pd.DataFrame( np.mean(rms,axis=0)[np.newaxis,:], columns = rms_cols )

            X_df.to_csv(X_df_file, index=False)
            Y_df.to_csv(Y_df_file, index=False)
            bodyId_type_arbor_df.to_csv(bodyId_type_arbor_df_file, index=False)
            mean_dx_df.to_csv(mean_dx_df_file, index=False)
            rms_df.to_csv(rms_df_file, index=False)
        else:
            print(neuron_type, jitter_strength * 8/1000)
