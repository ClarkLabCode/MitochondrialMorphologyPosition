from neuprint import Client
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, measure, morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy.ndimage import binary_erosion, convolve
import pandas as pd 
from os.path import isfile
from os import listdir
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC, MitoCriteria as MC
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import minimize
import importlib

token_id = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImdhcnJldHQuc2FnZXJAeWFsZS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpTGNqZXlHYWNnS3NPcTgzdDNfczBoTU5sQUtlTkljRzdxMkU5Rz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgwMTAxNzUwNn0.dzq7Iy01JwSWbKq-Qvi8ov7Hwr0-ozpYeSnOsUD-Mx0"
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
home_dir = '/Users/gs697/Research/mitochondriaMorphologyPosition' 
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1', token=token_id)
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
server = 'http://hemibrain-dvid.janelia.org'

all_bodyIds = pd.read_csv( home_dir + '/saved_data/all_bodyIds.csv' ).to_numpy()

# import utils file
spec = importlib.util.spec_from_file_location('utils', home_dir+'/util_files/MB_double_exp_utils.py')
MB_double_exp_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MB_double_exp_utils)

def format_axes(ax, fontsize):
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction = 'inout', length=6, width=0.25)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.25)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    return ax

all_compartments = ["a1(R)", "a2(R)", "a3(R)", "a'1(R)", "a'2(R)", "a'3(R)", "b'1(R)", "b'2(R)", 
                    "b1(R)", "b2(R)","g1(R)", "g2(R)", "g3(R)", "g4(R)", "g5(R)"]
type_VisType_color = [["KCa'b'-ap1", r"$KC\alpha'\beta'-ap1$", np.array([214, 40, 40])/255 ],
                      ["KCa'b'-ap2", r"$KC\alpha'\beta'-ap2$", np.array([214, 40, 40])/255 ],
                      ["KCa'b'-m", r"$KC\alpha'\beta'-m$", np.array([214, 40, 40])/255 ],
                      ['KCab-c', r"$KC\alpha\beta-c$", np.array([0, 48, 73])/255 ],
                      ['KCab-m', r"$KC\alpha\beta-m$", np.array([0, 48, 73])/255 ],
                      ['KCab-p', r"$KC\alpha\beta-p$", np.array([0, 48, 73])/255 ],
                      ['KCab-s', r"$KC\alpha\beta-s$", np.array([0, 48, 73])/255 ],
                      ['KCg-d', r"$KC\gamma-d$", np.array([247, 127, 0])/255 ],
                      ['KCg-m', r"$KC\gamma-m$", np.array([247, 127, 0])/255 ]]

compartments = ["a'1(R)", "a'2(R)", "a'3(R)", "b'1(R)", "b'2(R)"]
vis_compartments = [r"$\alpha '1$", r"$\alpha '2$", r"$\alpha '3$", r"$\beta '1$", r"$\beta '2$"]
synapse_df = pd.read_csv(f'{home_dir}/saved_data/MB_imaging/synapse_df.csv')

# eliminate synapses whose PC3 coordinate is less than -30, because these are on the peduncle
pca = PCA()
bool_synapses = pca.fit_transform(synapse_df[['x','y','z']].to_numpy())[:,2] > -20
synapse_df = synapse_df.iloc[np.where(bool_synapses)[0]]

image_file = home_dir + "/saved_data/MB_imaging/MB463B_a'b'_ap/20250210_6_970nm_00001.tif"

if MB_double_exp_utils.get_intensity_thresh(image_file, 0) is None:
    # intensity threshold have not been specified
    raw_image = imread(image_file)
    if len(raw_image.shape) == 5:
        raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
    raw_image[raw_image < 1] = 1
    vlims = [ np.quantile(np.log10(raw_image[:,0].flatten()), [0.85, 0.95]), np.quantile(np.log10(raw_image[:,1].flatten()), [0.85, 0.95]) ]
    assert False, f'Recommended vlims = {vlims}'

if True:
    # visualize segmentation alongside image
    cropped_raw_images = []
    cropped_seg_images = []
    for i_channel in range(2):
        raw_image = imread(image_file)
        if len(raw_image.shape) == 5:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
        raw_image = raw_image[:,i_channel]
        raw_image[raw_image < 1] = 1
        raw_image = MB_double_exp_utils.crop_img(image_file, raw_image)
        
        image = np.interp(np.log10(raw_image), MB_double_exp_utils.get_intensity_thresh(image_file, i_channel), [0,1])
        image = MB_double_exp_utils.RelaxationLabeling(image)
        labels = measure.label(image)
        unique_labels, counts = np.unique(labels[labels>0], return_counts=True)

        seg_image = np.zeros(image.shape)
        for i_side in range(2):
            image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[i_side]])
            seg_idxs = np.array(np.where(image)).T
            
            #bool_seg_idxs = get_bool_seg_idxs(image_file, seg_idxs, side)
            ab_seg_idxs = seg_idxs#[ bool_seg_idxs ]
            seg_image[ab_seg_idxs[:,0], ab_seg_idxs[:,1], ab_seg_idxs[:,2]] = 1
    
        min_row, max_row = np.where( np.sum(np.sum(seg_image > 0, axis=1), axis=1) > 0)[0][[0,-1]] # axis 0
        min_col, max_col = np.where( np.sum(np.sum(seg_image > 0, axis=0), axis=1) > 0)[0][[0,-1]] + np.array([-5, 6]) # axis 1
        min_page, max_page = np.where( np.sum(np.sum(seg_image > 0, axis=0), axis=0) > 0)[0][[0,-1]] + np.array([-5, 6]) # axis 2
        
        cropped_raw_images.append( raw_image )#[min_row : max_row, min_col : max_col, min_page : max_page]
        cropped_seg_images.append( seg_image )#[min_row : max_row, min_col : max_col, min_page : max_page]
    
    fig, axes = plt.subplots(figsize=(9 * cropped_raw_images[0].shape[2] / cropped_raw_images[0].shape[1],9), ncols = 2, nrows = 2)
    fontsize = 12
    im0s = []
    im1s = []
    seg_image = np.any(cropped_seg_images,axis=0)
    
    labels = measure.label(seg_image)
    unique_labels, counts = np.unique(labels[labels>0], return_counts=True)
    side_0, side_1 = unique_labels[np.flip(np.argsort(counts))[:2]]
    is_L = np.mean( np.where(labels == side_0)[-1] ) < np.mean( np.where(labels == side_1)[-1] )
    
    for i_channel in range(2):
        vlims = np.quantile(np.log10(cropped_raw_images[i_channel].flatten()), [0.25, 0.99])
        im0s.append( axes[i_channel, 0].imshow(np.log10(cropped_raw_images[i_channel][0]), vmin = vlims[0], vmax = vlims[1]) )
        im1s.append( axes[i_channel, 1].imshow(cropped_seg_images[i_channel][0], vmin = 0, vmax = 1) )
        #im1s.append( axes[i_channel, 1].imshow(seg_image[0], vmin = 0, vmax = 1) )

        axes[i_channel, 0].set_title(f'vmin = {np.round(vlims[0],decimals=1)}; vmax = {np.round(vlims[1],decimals=1)}')
        for col in range(2):
            axes[i_channel,col].set_yticks(np.arange(0,seg_image.shape[1], 25))
            axes[i_channel,col].set_xticks(np.arange(0,seg_image.shape[2], 25))
            for label in axes[i_channel,col].get_xticklabels():
              label.set_rotation(270)
            for y in np.arange(0,np.mean(seg_image,axis=0).shape[0],25):
                axes[i_channel,col].plot([0,np.mean(seg_image,axis=0).shape[1]], [y,y], linewidth= 1, color = 'k')
                
            for x in np.arange(0,np.mean(seg_image,axis=0).shape[1],25):
                axes[i_channel,col].plot([x,x], [np.mean(seg_image,axis=0).shape[0], 0], linewidth= 1, color = 'k')
            axes[i_channel,col].set_xlim([0, seg_image.shape[2]])
            axes[i_channel,col].set_ylim([seg_image.shape[1], 0])

        if is_L:
            # side_0 is on the left side
            axes[i_channel, 1].text(0, 0, '0_L', fontsize=fontsize, color = 'white', va = 'top', ha = 'left')
            axes[i_channel, 1].text(cropped_seg_images[0].shape[2], 0, '1_R', fontsize=fontsize, color = 'white', va = 'top', ha = 'right')
        else:
            # side_0 is on the right side
            axes[i_channel, 1].text(0, 0, '1_L', fontsize=fontsize, color = 'white', va = 'top', ha = 'left')
            axes[i_channel, 1].text(cropped_seg_images[0].shape[2], 0, '0_R', fontsize=fontsize, color = 'white', va = 'top', ha = 'right')

    for nframe in range(2 * len(cropped_raw_images[0])):
        frame = nframe % len(cropped_raw_images[0])
        for i_channel in range(2):
            im0s[i_channel].set_data(np.log10(cropped_raw_images[i_channel][frame]))
            im1s[i_channel].set_data(cropped_seg_images[i_channel][frame])
            #im1s[i_channel].set_data(seg_image[frame])
        
        fig.suptitle('Frame: %d' % (frame))
    
        plt.draw()
        plt.pause(0.2)
    plt.show()

elif False:    
    seg_image = None
    for i_channel in range(2):
        raw_image = imread(image_file)
        if len(raw_image.shape) == 5:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
        raw_image = raw_image[:,i_channel]
        raw_image[raw_image < 1] = 1
        raw_image = MB_double_exp_utils.crop_img(image_file, raw_image)
        image = np.interp(np.log10(raw_image), MB_double_exp_utils.get_intensity_thresh(image_file, i_channel), [0,1])
        image = MB_double_exp_utils.RelaxationLabeling(image)
        labels = measure.label(image)
        unique_labels, counts = np.unique(labels[labels>0], return_counts=True)
        for side in ['R', 'L']:
            if MB_double_exp_utils.is_connected(image_file):
                image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[0]])
            else:
                if np.mean( np.where(labels == unique_labels[np.flip(np.argsort(counts))[0]])[2] ) < image.shape[2]/2:
                    # side 0 is on the left side
                    i_side = 0 if side == 'L' else 1
                else:
                    i_side = 1 if side == 'L' else 0
                
                image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[i_side]])
            seg_idxs = np.array(np.where(image)).T
    
            bool_seg_idxs = MB_double_exp_utils.get_bool_seg_idxs(image_file, seg_idxs, side)
            ab_seg_idxs = seg_idxs[ bool_seg_idxs ]
            if seg_image is None:
                seg_image = np.zeros(image.shape) 
            seg_image[ab_seg_idxs[:,0], ab_seg_idxs[:,1], ab_seg_idxs[:,2]] = 1
    
            
    fig = plt.figure(figsize=(12,12))
    plt.imshow(np.mean(seg_image,axis=0))
    #plt.imshow(seg_image[53])
    plt.xticks(np.arange(0,raw_image.shape[2],10), rotation = 270)
    plt.yticks(np.arange(0,raw_image.shape[1],10))
    
    for y in np.arange(0,np.mean(raw_image,axis=0).shape[0],10):
        plt.plot([0,np.mean(raw_image,axis=0).shape[1]], [y,y], linewidth= 1, color = 'k')
        
    for x in np.arange(0,np.mean(raw_image,axis=0).shape[1],10):
        plt.plot([x,x], [np.mean(raw_image,axis=0).shape[0], 0], linewidth= 1, color = 'k')
    
    min_x = np.where( np.sum( np.mean(seg_image,axis=0), axis=0) > 0 )[0][0]
    max_x = np.where( np.sum( np.mean(seg_image,axis=0), axis=0) > 0 )[0][-1]
    plt.xlim([min_x, max_x])
    
    min_y = np.where( np.sum( np.mean(seg_image,axis=0), axis=1) > 0 )[0][0]
    max_y = np.where( np.sum( np.mean(seg_image,axis=0), axis=1) > 0 )[0][-1]
    plt.ylim([max_y, min_y])
    
    plt.show()

elif True:
    # visualize compartment classification
    raw_image = imread(image_file)
    if len(raw_image.shape) == 5:
        raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
    raw_image = raw_image[:,0]
    comp_image = np.zeros(raw_image.shape)
    comp_image_rgb = np.zeros((raw_image.shape[0], raw_image.shape[1], raw_image.shape[2], 3))

    for i_channel in range(2):
        raw_image = imread(image_file)
        if len(raw_image.shape) == 5:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
        raw_image = raw_image[:,i_channel]
        raw_image[raw_image < 1] = 1
        raw_image = MB_double_exp_utils.crop_img(image_file, raw_image)
        image = np.interp(np.log10(raw_image), MB_double_exp_utils.get_intensity_thresh(image_file, i_channel), [0,1])
        image = MB_double_exp_utils.RelaxationLabeling(image)
        labels = measure.label(image)
        unique_labels, counts = np.unique(labels[labels>0], return_counts=True)
        for side in ['R', 'L']:
            if MB_double_exp_utils.is_connected(image_file):
                image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[0]])
            else:
                if np.mean( np.where(labels == unique_labels[np.flip(np.argsort(counts))[0]])[2] ) < image.shape[2]/2:
                    # side 0 is on the left side
                    i_side = 0 if side == 'L' else 1
                else:
                    i_side = 1 if side == 'L' else 0
                
                image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[i_side]])
            seg_idxs = np.array(np.where(image)).T
    
            bool_seg_idxs = MB_double_exp_utils.get_bool_seg_idxs(image_file, seg_idxs, side)
            ab_seg_idxs = seg_idxs[ bool_seg_idxs ]
    
            if "a'b'" in image_file:
                is_ap1 = MB_double_exp_utils.get_is_ap1(image_file, side, ab_seg_idxs) == 1
                is_ap2 = MB_double_exp_utils.get_is_ap2(image_file, side, ab_seg_idxs) == 1
                is_ap3 = MB_double_exp_utils.get_is_ap3(image_file, side, ab_seg_idxs) == 1
                is_bp1 = MB_double_exp_utils.get_is_bp1(image_file, side, ab_seg_idxs) == 1
                is_bp2 = MB_double_exp_utils.get_is_bp2(image_file, side, ab_seg_idxs) == 1
                bool_comps = [is_ap1, is_ap2, is_ap3, is_bp1, is_bp2]
            elif "Gamma" in image_file:
                bool_comps = MB_double_exp_utils.get_is_g12345(side, ab_seg_idxs)
            else:
                is_a1 = MB_double_exp_utils.get_is_a1(image_file, side, ab_seg_idxs) == 1
                is_a2 = MB_double_exp_utils.get_is_a2(image_file, side, ab_seg_idxs) == 1
                is_a3 = MB_double_exp_utils.get_is_a3(image_file, side, ab_seg_idxs) == 1
                is_b1 = MB_double_exp_utils.get_is_b1(image_file, side, ab_seg_idxs) == 1
                is_b2 = MB_double_exp_utils.get_is_b2(image_file, side, ab_seg_idxs) == 1
                bool_comps = [is_a1, is_a2, is_a3, is_b1, is_b2]
        
            for i_comp, bool_comp in enumerate(bool_comps):
                comp_image[ab_seg_idxs[bool_comp,0], ab_seg_idxs[bool_comp,1], ab_seg_idxs[bool_comp,2]] = i_comp + 1
                for i_color in range(3):
                    comp_image_rgb[ab_seg_idxs[bool_comp,0], ab_seg_idxs[bool_comp,1], ab_seg_idxs[bool_comp,2], i_color] = plt.get_cmap("tab10")(i_comp)[i_color]
    min_row, max_row = np.where( np.sum(np.sum(comp_image > 0, axis=1), axis=1) > 0)[0][[0,-1]] # axis 0
    min_col, max_col = np.where( np.sum(np.sum(comp_image > 0, axis=0), axis=1) > 0)[0][[0,-1]] + np.array([-5, 6]) # axis 1
    min_page, max_page = np.where( np.sum(np.sum(comp_image > 0, axis=0), axis=0) > 0)[0][[0,-1]] + np.array([-5, 6]) # axis 2
    
    cropped_raw_image = raw_image[min_row : max_row, min_col : max_col, min_page : max_page]
    cropped_comp_image_rgb = comp_image_rgb[min_row : max_row, min_col : max_col, min_page : max_page]

    fig, axes = plt.subplots(figsize=(4.5 * cropped_raw_image.shape[2] / cropped_raw_image.shape[1],9), nrows = 2)
    fontsize=12
    imgs = []
    frame = 0
    imgs.append( axes[0].imshow(np.log10(cropped_raw_image[frame]), vmin = 1.8, vmax = 3) )
    imgs.append( axes[1].imshow(cropped_comp_image_rgb[frame]) )

    axes[0].axis('off')
    axes[1].axis('off')
    fig.suptitle(frame)
    
    for frame in np.arange(1, cropped_raw_image.shape[0]):
        imgs[0].set_data(np.log10(cropped_raw_image[frame]))
        imgs[1].set_data(cropped_comp_image_rgb[frame])
        fig.suptitle('Frame: %d' % (frame))
        plt.draw()
        plt.pause(0.2)
    plt.show()

elif True:
    # visualize all segmented voxels to find threshold for peduncle
    pca = PCA()

    seg_image = None
    for i_channel in range(2):
        raw_image = imread(image_file)
        if len(raw_image.shape) == 5:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
        raw_image = raw_image[:,i_channel]
        
        raw_image[raw_image < 1] = 1
        raw_image = MB_double_exp_utils.crop_img(image_file, raw_image)
        image = np.interp(np.log10(raw_image), MB_double_exp_utils.get_intensity_thresh(image_file, i_channel), [0,1])
        image = MB_double_exp_utils.RelaxationLabeling(image)
        labels = measure.label(image)
        unique_labels, counts = np.unique(labels[labels>0], return_counts=True)

        if MB_double_exp_utils.is_connected(image_file):
            image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[0]])
        else:
            if np.mean( np.where(labels == unique_labels[np.flip(np.argsort(counts))[0]])[2] ) < image.shape[2]/2:
                # side 0 is on the left side
                i_side = 0 if side == 'L' else 1
            else:
                i_side = 1 if side == 'L' else 0
            
            image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[i_side]])
            
        seg_idxs = np.array(np.where(image)).T
        
        if seg_image is None:
            seg_image = np.zeros(image.shape) 
        seg_image[seg_idxs[:,0], seg_idxs[:,1], seg_idxs[:,2]] = 1
        
    seg_idxs = np.array(np.where(seg_image)).T
    seg_idxs = seg_idxs[np.random.choice(len(seg_idxs), int(len(seg_idxs)/100))]
    print(seg_image.shape)

    colors = MB_double_exp_utils.get_bool_seg_idxs(image_file, seg_idxs, side)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(seg_idxs[colors == 1,0], seg_idxs[colors == 1,1], seg_idxs[colors == 1,2], s = 5, edgecolor = 'none', color = 'k')
    ax.scatter(seg_idxs[colors == 0,0], seg_idxs[colors == 0,1], seg_idxs[colors == 0,2], s = 5, edgecolor = 'none', color = 'r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lim_mins = np.min(seg_idxs,axis=0)
    lim_maxs = np.max(seg_idxs,axis=0)
    ax.set_xlim([lim_mins[0], lim_maxs[0]])
    ax.set_ylim([lim_mins[1], lim_maxs[1]])
    ax.set_zlim([lim_mins[2], lim_maxs[2]])
    
    for angle in range(0, 360*3 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180
    
        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        if angle <= 360:
            elev = angle_norm
        elif angle <= 360*2:
            azim = angle_norm
        elif angle <= 360*3:
            roll = angle_norm
    
        # Update the axis view and title
        ax.view_init(elev, azim, roll)
        plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))
    
        plt.draw()
        plt.pause(.001)
    plt.show()
elif False:
    # compare hemibrain to my final segmentation
    i_side = 0
    
    synapse_coords = synapse_df[['x','y','z']].to_numpy() * 8/1000
    PC_synapse_coords = pca.fit_transform(synapse_coords)
    synapse_i_comps = np.array([ np.where(comp == np.array(compartments))[0][0] for comp in synapse_df['roi']])
    
    raw_image = imread(image_file)
    if 'dual_imaging' in image_file:
        if len(raw_image.shape) == 5:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
    else:
        if len(raw_image.shape) == 4:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], raw_image.shape[2], raw_image.shape[3]) )
    raw_image[raw_image < 1] = 1
    raw_image = MB_double_exp_utils.crop_img(image_file, raw_image)
    init_prob = np.interp(np.log10(raw_image), get_intensity_thresh(image_file), [0,1])
    image = RelaxationLabeling(init_prob)
    labels = measure.label(image)
    unique_labels, counts = np.unique(labels[labels>0], return_counts=True)

    T_matrix_df = pd.read_csv(image_file[:-4] + "_T_Matrices.csv")
    T_matrices = T_matrix_df.to_numpy()

    image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[i_side]])
    seg_idxs = np.array(np.where(image)).T
    PC_seg_idxs = pca.fit_transform(seg_idxs)

    bool_seg_idxs = get_bool_seg_idxs(image_file, PC_seg_idxs, side)
    ab_seg_idxs = seg_idxs[ bool_seg_idxs ] 
    PC_ab_seg_idxs = pca.fit_transform(ab_seg_idxs)  * 0.541551925

    T_matrix = T_matrices[i_side]
    reg_PC_ab_seg_idxs = np.matmul( PC_ab_seg_idxs, T_matrix[:-3].reshape((3,3)) ) + T_matrix[-3:][np.newaxis,:]
    
    exp_compartments = np.zeros(len(reg_PC_ab_seg_idxs)) + np.nan
    exp_init_points = np.append(np.arange(0, reg_PC_ab_seg_idxs.shape[0], int(reg_PC_ab_seg_idxs.shape[0]/100)), reg_PC_ab_seg_idxs.shape[0])
    for i_init in range(len(exp_init_points)-1):
        exp_idxs = np.arange(exp_init_points[i_init], exp_init_points[i_init+1])
        dists = cdist(reg_PC_ab_seg_idxs[exp_idxs], PC_synapse_coords)
        exp_compartments[ exp_idxs ] = synapse_i_comps[np.argmin(dists, axis=1)]


    # only plot certain synapses and exp voxels
    exp_idxs = np.random.choice(len(PC_ab_seg_idxs), 20000, replace = False)
    synapse_idxs = np.random.choice(len(synapse_df), 10000, replace = False)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fontsize=12
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axes = [ax1, ax2]
    for i_comp, comp in enumerate(compartments):
        bool_synapses = synapse_df['roi'].to_numpy()[synapse_idxs] == comp
        if np.any(bool_synapses):
            axes[0].scatter(PC_synapse_coords[synapse_idxs[bool_synapses],0], 
                            PC_synapse_coords[synapse_idxs[bool_synapses],1], 
                            PC_synapse_coords[synapse_idxs[bool_synapses],2], 
                            label = vis_compartments[i_comp], edgecolor = 'none')
            axes[1].scatter(PC_ab_seg_idxs[exp_idxs][exp_compartments[exp_idxs] == i_comp,0], 
                            PC_ab_seg_idxs[exp_idxs][exp_compartments[exp_idxs] == i_comp,1], 
                            PC_ab_seg_idxs[exp_idxs][exp_compartments[exp_idxs] == i_comp,2], 
                            label = vis_compartments[i_comp], edgecolor = 'none')
    for i_axes in range(len(axes)):
        format_axes(axes[i_axes], fontsize)
        axes[i_axes].set_xlim([-80,65])
        axes[i_axes].set_ylim([-55,45])
        axes[i_axes].set_zlim([-40,27])
        axes[i_axes].set_xlabel('PC1 (' + r'$\mu m$' + ')', fontsize=fontsize)
        axes[i_axes].set_ylabel('PC2 (' + r'$\mu m$' + ')', fontsize=fontsize)
        axes[i_axes].set_zlabel('PC3 (' + r'$\mu m$' + ')', fontsize=fontsize)
    axes[0].set_title('Hemibrain Compartments', fontsize=fontsize)
    axes[1].set_title('Experiment Compartments', fontsize=fontsize)
    
    for angle in range(0, 360*3 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180
    
        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        if angle <= 360:
            elev = angle_norm
        elif angle <= 360*2:
            azim = angle_norm
        elif angle <= 360*3:
            roll = angle_norm
    
        # Update the axis view and title
        for i_axes in range(len(axes)):
            axes[i_axes].view_init(elev, azim, roll)
        fig.suptitle('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))
    
        plt.draw()
        plt.pause(.0001)
    plt.show()
else:
    # visualize segmentation alongside image
    synapse_coords = synapse_df[['x','y','z']].to_numpy() * 8/1000
    PC_synapse_coords = pca.fit_transform(synapse_coords)
    synapse_i_comps = np.array([ np.where(comp == np.array(compartments))[0][0] for comp in synapse_df['roi']])
    
    raw_image = imread(image_file)
    if 'dual_imaging' in image_file:
        if len(raw_image.shape) == 5:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], 2, raw_image.shape[3], raw_image.shape[4]) )
    else:
        if len(raw_image.shape) == 4:
            raw_image = raw_image.reshape( (raw_image.shape[0]*raw_image.shape[1], raw_image.shape[2], raw_image.shape[3]) )
    raw_image[raw_image < 1] = 1
    raw_image = MB_double_exp_utils.crop_img(image_file, raw_image)
    image = np.interp(np.log10(raw_image), get_intensity_thresh(image_file), [0,1])
    image = RelaxationLabeling(image)
    labels = measure.label(image)
    unique_labels, counts = np.unique(labels[labels>0], return_counts=True)

    T_matrix_df = pd.read_csv(image_file[:-4] + "_T_Matrices.csv")
    T_matrices = T_matrix_df.to_numpy()

    comp_image = np.zeros(image.shape)
    comp_image_rgb = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3))
    
    for i_side in range(2):
        image = np.isin(labels, unique_labels[np.flip(np.argsort(counts))[i_side]])
        seg_idxs = np.array(np.where(image)).T
        PC_seg_idxs = pca.fit_transform(seg_idxs)
    
        bool_seg_idxs = get_bool_seg_idxs(image_file, PC_seg_idxs, side)
        ab_seg_idxs = seg_idxs[ bool_seg_idxs ]
        
        PC_ab_seg_idxs = pca.fit_transform(ab_seg_idxs) * 0.541551925
    
        T_matrix = T_matrices[i_side]
        reg_PC_ab_seg_idxs = np.matmul( PC_ab_seg_idxs, T_matrix[:-3].reshape((3,3)) ) + T_matrix[-3:][np.newaxis,:]
        
        exp_compartments = np.zeros(len(reg_PC_ab_seg_idxs)) + np.nan
        exp_init_points = np.append(np.arange(0, reg_PC_ab_seg_idxs.shape[0], int(reg_PC_ab_seg_idxs.shape[0]/100)), reg_PC_ab_seg_idxs.shape[0])
        for i_init in range(len(exp_init_points)-1):
            exp_idxs = np.arange(exp_init_points[i_init], exp_init_points[i_init+1])
            dists = cdist(reg_PC_ab_seg_idxs[exp_idxs], PC_synapse_coords)
            exp_compartments[ exp_idxs ] = synapse_i_comps[np.argmin(dists, axis=1)]
        
        for i_comp, comp in enumerate(compartments):
            bool_synapses = synapse_df['roi'].to_numpy() == comp
            if np.any(bool_synapses):
                comp_seg_idxs = seg_idxs[bool_seg_idxs][exp_compartments == i_comp]
                comp_image[comp_seg_idxs[:,0], comp_seg_idxs[:,1], comp_seg_idxs[:,2]] = i_comp
                for i_color in range(3):
                    comp_image_rgb[comp_seg_idxs[:,0], comp_seg_idxs[:,1], comp_seg_idxs[:,2], i_color] = plt.get_cmap("tab10")(i_comp)[i_color]

    min_row, max_row = np.where( np.sum(np.sum(comp_image > 0, axis=1), axis=1) > 0)[0][[0,-1]] # axis 0
    min_col, max_col = np.where( np.sum(np.sum(comp_image > 0, axis=0), axis=1) > 0)[0][[0,-1]] + np.array([-5, 6]) # axis 1
    min_page, max_page = np.where( np.sum(np.sum(comp_image > 0, axis=0), axis=0) > 0)[0][[0,-1]] + np.array([-5, 6]) # axis 2
    cropped_raw_image = raw_image[min_row : max_row, min_col : max_col, min_page : max_page]
    cropped_comp_image_rgb = comp_image_rgb[min_row : max_row, min_col : max_col, min_page : max_page]
    
    fig, axes = plt.subplots(figsize=(10,5), ncols = 2)
    fontsize = 12
    
    im0 = axes[0].imshow(np.log10(cropped_raw_image[0]), vmin = 1.8, vmax = 4)
    im1 = axes[1].imshow(cropped_comp_image_rgb[0])
    
    for i_axes in range(len(axes)):
        axes[i_axes].axis('off')

    for nframe in range(2 * len(cropped_raw_image)):
        frame = nframe % len(cropped_raw_image)
        im0.set_data(np.log10(cropped_raw_image[frame]))
        im1.set_data(cropped_comp_image_rgb[frame])
        
        fig.suptitle('Frame: %d' % (frame))
    
        plt.draw()
        plt.pause(0.5)
    
    plt.show()








