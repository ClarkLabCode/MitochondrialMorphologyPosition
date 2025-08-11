# Configuration file for analyzing data from the hemibrain dataset
# Contains settings, constants, and lookup tables for neuroanatomical analysis

from neuprint import Client
import os
import pandas as pd
import numpy as np
from neuprint.queries import fetch_primary_rois

#------------------------------------------------------------------------------
# Authentication and Environment Setup
#------------------------------------------------------------------------------

# Authentication token for accessing the hemibrain dataset
token_id = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImdhcnJldHQuc2FnZXJAeWFsZS5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hLS9BT2gxNEdpTGNqZXlHYWNnS3NPcTgzdDNfczBoTU5sQUtlTkljRzdxMkU5Rz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTgwMTAxNzUwNn0.dzq7Iy01JwSWbKq-Qvi8ov7Hwr0-ozpYeSnOsUD-Mx0"

# Flag to determine if code is running on HPC cluster
on_HPC = False 

# Flag to determine if to connect to the client, which requires the internet
use_internet = True

# Set paths based on execution environment
if on_HPC:
    # Paths for running on High Performance Computing cluster
    home_dir = '/home/gs697/project/clean_mito_code'
    google_application_credentials = '/home/gs697/project/application_default_credentials.json' 
    tensorstore_ca_bundle = '/home/gs697/project/ca-certificates.crt' 
else:
    # Paths for running on local machine
    home_dir = '/Users/gs697/Research/mitochondriaMorphologyPosition' 
    google_application_credentials = '~/.config/gcloud/application_default_credentials.json'

# Initialize neuPrint client for accessing hemibrain data
c = Client('neuprint.janelia.org', dataset = 'hemibrain:v1.2.1', token=token_id) if use_internet else None
server = 'http://hemibrain-dvid.janelia.org'

#------------------------------------------------------------------------------
# Cell Component Classifications
#------------------------------------------------------------------------------

# Numerical mapping for different cell compartments
node_class_dict = {"soma": 1,"axon": 2,"dendrite": 3,"cell body fiber": 4,"connecting cable": 5,"other": 6}

# Cell compartments included in analysis
analyze_sections = np.array(['dendrite', 'connecting cable', 'axon'])

#------------------------------------------------------------------------------
# Analysis Parameters
#------------------------------------------------------------------------------

# Jitter strength parameters for analysis
jitter_strengths = np.logspace( np.log10(0.1), np.log10(1000), 30) * 1000/8

#------------------------------------------------------------------------------
# Neuron Type Analysis Settings
#------------------------------------------------------------------------------

# Load and filter neuron quality data
neuron_quality = pd.read_csv(home_dir + '/saved_data/neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
unique_neuron_types, counts = np.unique( neuron_quality_np[:,1], return_counts = True )

# Filter for neuron types with at least 10 samples
unique_neuron_types = unique_neuron_types[ counts >= 10 ]
counts = counts[ counts >= 10 ]
analyze_neurons = np.append( unique_neuron_types[-3:], unique_neuron_types[:-3] )
analyze_neurons = list( analyze_neurons[ analyze_neurons != 'LC14' ] )

# Specific neuron groups for different analyses
vis_neurons = ['LC4', 'LC9', 'LC12']  # Neurons used for visualizations in figures for the paper
card_neurons = ['LC18', 'LC21', 'LC11', 'LC25', 'LC15', 'LC4', 'LC17', 'LC12']  # LC neurons types in Klapoetke2022
loom_neurons = ['LC4', 'LC17', 'LC12']  # Looming sensitive neurons

#------------------------------------------------------------------------------
# Visualization Settings
#------------------------------------------------------------------------------

# Color schemes for different LC neuron types - taken from Brewer color scheme
LC_colors = {'LC4': [0.904705882352941, 0.191764705882353, 0.198823529411765],
             'LC6': [0.294117647058824, 0.544705882352941, 0.749411764705882],
             'LC9': [0,0,0],
             'LC10':[255/255, 215/255, 0],
             'LC11':[0.865000000000000, 0.811000000000000, 0.433000000000000],
             'LC12':[0.685882352941177, 0.403529411764706, 0.241176470588235],
             'LC13':[0.971764705882353, 0.555294117647059, 0.774117647058824],
             'LC15':[0.904705882352941, 0.191764705882353, 0.198823529411765],
             'LC16':[0.294117647058824, 0.544705882352941, 0.749411764705882],
             'LC17':[0,0,0],
             'LC18':[255/255, 215/255, 0],
             'LC20':[0.865000000000000, 0.811000000000000, 0.433000000000000],
             'LC21':[0.685882352941177, 0.403529411764706, 0.241176470588235],
             'LC22':[0.971764705882353, 0.555294117647059, 0.774117647058824],
             'LC24':[0.904705882352941, 0.191764705882353, 0.198823529411765],
             'LC25':[0.294117647058824, 0.544705882352941, 0.749411764705882],
             'LC26':[0,0,0],
             'LC27':[255/255, 215/255, 0],
             'LC29':[0.865000000000000, 0.811000000000000, 0.433000000000000],
             'LC31':[0.685882352941177, 0.403529411764706, 0.241176470588235],
             'LC36':[0.971764705882353, 0.555294117647059, 0.774117647058824]}

# Marker styles for different LC neuron types
LC_markers ={'LC4' :'o','LC6' :'o','LC9' :'o','LC10':'o','LC11':'o','LC12':'o','LC13':'o',
             'LC15':'^','LC16':'^','LC17':'^','LC18':'^','LC20':'^','LC21':'^','LC22':'^',
             'LC24':'P','LC25':'P','LC26':'P','LC27':'P','LC29':'P','LC31':'P','LC36':'P'}

# Line styles for different LC neuron types
LC_linestyles = {'LC4' : 'solid','LC6' : 'solid','LC9' : 'solid','LC10':'solid' ,'LC11':'solid' ,'LC12':'solid' ,'LC13':'solid' ,
                 'LC15':'dashed','LC16':'dashed','LC17':'dashed','LC18':'dashed','LC20':'dashed','LC21':'dashed','LC22':'dashed',
                 'LC24':'dotted','LC25':'dotted','LC26':'dotted','LC27':'dotted','LC29':'dotted','LC31':'dotted','LC36':'dotted'}

# Colors for different cell sections and synapses
section_colors = {'axon': [0.371764705882353, 0.717647058823529, 0.361176470588235],
                  'dendrite': [152/255, 78/255, 163/255],
                  'connecting cable': [1, 0.548235294117647, 0.100000000000000],
                  'cell body fiber': [0, 0.30196078431372547, 0.25098039215686274],
                  'soma': [0, 0.6941176470588235, 0.6901960784313725],
                  'other': [0.7686274509803922, 0.4980392156862745, 0.19607843137254902]}

synapse_colors = {'pre': [0.904705882352941, 0.191764705882353, 0.198823529411765],
                  'post': [55/255, 126/255, 184/255]}

#------------------------------------------------------------------------------
# Brain Region Definitions
#------------------------------------------------------------------------------

# Optic lobe regions
optic_lobe = ['ME(R)', 'AME(R)', 'LO(R)', 'LOP(R)']
optic_lobe_names = ['Medulla (R)', 'Accessory Medulla(R)', 'Lobula(R)', 'Lobula Plate(R)']

# Mushroom body regions
mushroom_body = ['CA(L)', 'CA(R)', 'PED(R)', "a'L(R)", "a'L(L)", 'aL(R)', 
                 'aL(L)', 'gL(R)', 'gL(L)', "b'L(R)", "b'L(L)", 'bL(R)', 'bL(L)'] # 'dACA(R)', 'lACA(R)', 'vACA(R)'
mushroom_body_names = ['Calyx(L)', 'Calyx(R)', 'Peduncles(R)', r'$\alpha$' + "' lobe(R)", r'$\alpha$' + "' lobe(L)", r'$\alpha$' + " lobe(R)", 
                 r'$\alpha$' + " lobe(L)", r'$\gamma$' + " lobe(R)", r'$\gamma$' + " lobe(L)", r'$\beta$' + "' lobe(R)", r'$\beta$' + "' lobe(L)", r'$\beta$' + " lobe(R)", r'$\beta$' + " lobe(L)"]

# Additional brain regions and their human-readable names
lateral_complex = ['BU(R)', 'BU(L)', 'LAL(R)', 'LAL(L)'] # 'GA(R)'
lateral_complex_names = ['Bulb(R)', 'Bulb(L)', 'Lateral Accessory Lobe(R)', 'Lateral Accessory Lobe(L)']

ventrolateral_neuropils = ['AOTU(R)', 'AVLP(R)', 'PVLP(R)', 'PLP(R)', 'WED(R)']
ventrolateral_neuropils_names = ['Anterior Optic Tubercle(R)', 'Anterior Ventrolateral Protocerebrum(R)', 'Posterior Ventrolateral Protocerebrum(R)', ' Posterior Lateral Cerebrum(R)', 'Wedge(R)']

central_complex = ['FB', 'EB', 'AB(R)', 'AB(L)', 'PB', 'NO'] # 'IPS(L)'
central_complex_names = ['Fan-shaped body', 'Ellipsoid body', 'Asymmetrical body(R)', 'Asymmetrical body(L)', 'Protocerebral bridge', 'Noduli']

superior_neuropils = ['SLP(R)', 'SIP(R)', 'SIP(L)', 'SMP(R)', 'SMP(L)']
superior_neuropils_names = ['Superior Lateral Protocerebrum(R)', 'Superior Intermediate Protocerebrum(R)', 
                            'Superior Intermediate Protocerebrum(L)', 'Superior Medial Protocerebrum(R)', 'Superior Medial Protocerebrum(L)']

inferior_neuropils = ['CRE(R)', 'CRE(L)', 'SCL(R)', 'SCL(L)', 'ICL(R)', 'ICL(L)', 'IB', 'ATL(R)', 'ATL(L)']
inferior_neuropils_names = ['Crepine(R)', 'Crepine(L)', 'Superior Clamp(R)', 'Superior Clamp(L)', 'Inferior Clamp(R)', 
                                'Inferior ClampL(L)', 'Inferior Bridge', 'Antler(R)', 'Antler(L)']
    
ventromedial_neuropils = ['VES(R)', 'VES(L)', 'EPA(R)', 'EPA(L)', 'GOR(R)', 'GOR(L)', 'SPS(R)', 'SPS(L)', 'IPS(R)']
ventromedial_neuropils_names = ['Vest(R)', 'Vest(L)', 'Epaulette(R)', 'Epaulette(L)', 'Gorget(R)', 'Gorget(L)', 
                                'Superior Posterior Slope(R)', 'Superior Posterior Slope(L)', 'Inferior Posterior Slope(R)']
    
pariesophageal_neuropils = ['SAD', 'FLA(R)', 'CAN(R)', 'PRW']
pariesophageal_neuropils_names = ['Saddle', 'Flange(R)', 'Cantle(R)', 'Prow']
    
other_neuropils = ['LH(R)', 'AL(R)', 'AL(L)', 'GNG']
other_neuropils_names = ['Lateral Horn(R)', 'Antennal Lobe(R)', 'Antennal Lobe(L)', 'Gnathal Ganglia']

# Compile all neuropil regions and their names
all_neuropils = [optic_lobe, mushroom_body, lateral_complex, ventrolateral_neuropils, central_complex, 
                 superior_neuropils, inferior_neuropils, ventromedial_neuropils, 
                 pariesophageal_neuropils, other_neuropils]
all_neuropils_list = np.concatenate(all_neuropils, axis=0)

all_neuropils_names = [optic_lobe_names, mushroom_body_names, lateral_complex_names, 
                      ventrolateral_neuropils_names, central_complex_names,
                      superior_neuropils_names, inferior_neuropils_names, 
                      ventromedial_neuropils_names, pariesophageal_neuropils_names, 
                      other_neuropils_names]
all_neuropils_names_list = np.concatenate(all_neuropils_names, axis=0)

# Fetch and process ROI (Region of Interest) information
if use_internet:
    rois = np.array(fetch_primary_rois())
    rois_names = [ all_neuropils_names_list[ np.where(all_neuropils_list == roi)[0][0] ] for roi in rois ]
else:
    rois, rois_names = None, None

# Classification and coloring for different neuropil types
neuropil_classes = ['Optic Lobe', 'Mushroom Body', 'Lateral Complex', 'Ventrolateral', 
                   'Central Complex', 'Superior', 'Inferior', 'Ventromedial', 
                   'Pariesophageal', 'Other']
neuropil_colors = [f'C{ii}' for ii in range(len(all_neuropils))]

