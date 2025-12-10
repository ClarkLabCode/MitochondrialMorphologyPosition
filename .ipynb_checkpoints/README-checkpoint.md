# Mitochondrial Positioning and Morphology Analysis Project

This repository contains code and analysis for studying mitochondrial positioning and morphology across different brain regions and species.

## Project Structure

### Main Analysis Notebooks

#### Mouse Brain Analysis
- `MB_imaging.ipynb` - Mushroom body imaging and analysis
- `positioning rules - mouse.ipynb` - Analysis of positioning rules in mouse neurons
- `morphology classifier - mouse.ipynb` - Classification of mouse neuron types by their mitochondria morphologies

#### LC Neurons Analysis
- `positioning rules - LC neurons.ipynb` - Analysis of mitochondrial positioning rules in LC neurons
- `positioning rules arbor - LC neurons.ipynb` - Mitochondria positioning rules for all mitochondria in all LC neurons for both arbors
- `positioning rules jitter - LC neurons.ipynb` - Analysis of mitochondrial positioning precision in LC neurons by jittering the mitochondria locations
- `morphology classifier - LC neurons.ipynb` - Classification of LC neuron types by their mitochondria morphologies
- `morphology embeddings - LC neurons.ipynb` - Embedding analysis of mitochondria morphologies in LC neurons

#### Hemibrain Analysis
- `morphology classifier - hemibrain.ipynb` - Classification of Drosophila neuron types in the hemibrain by their mitochondria morphologies
- `mito connectome - hemibrain.ipynb` - Analysis of the mitochondrially conditioned connectome in the hemibrain

#### Kenyon Cells Analysis
- `mito connectome - Kenyon Cells.ipynb` - Analysis of the mitochondrially conditioned connectome in the Kenyon Cells

#### General Analysis
- `visualize_skeleton.ipynb` - Visualizations of neuronal skeletons
- `positioning features correlation function.ipynb` - Correlatation functions for the histogram positioning features in LC neurons
- `neuron morphology and mito density correlation.ipynb` - Correlation between the mitochondria density and various neural morphometrics
- `cdfs.ipynb` - Various cumulative distributions summarizing bulk statistics of the mitochondria in LC neuons.

### Supporting Files and Directories

#### Internal Data Directories
- `saved_data/` - Unless otherwise explicitly stated below in the External Data Directories section, all files in this directory are outputs from scripts throughout this repository. The data are saved to this folder for faster computation later.
- `saved_figures/` - Storage for generated figures from code 
- `saved_clean_skeletons/` - Processed neuronal skeletons with updating radii and trimmed trivial leaves. The code to generate these files is found within the directory.
- `saved_neuron_skeletons/` - Neuronal skeletons with updated radii but the trivial leaves are still present. Code to generate these skeletons is found in `HPC/neuron_skels/`
- `saved_synapse_df/` - Synapse pandas dataframes of LC neurons as computed and saved in `HPC/save_mito_synapse/`
- `saved_mito_df/` - Mitochondrial pandas dataframes of LC neurons as computed and saved in `HPC/save_mito_synapse/`

#### External Data Directories
- `saved_data/Flywire` - List of neurons with their predicted neurotransmitter type from the Flywire dataset. The data are downloaded from Codex in the `Download Data` section (https://codex.flywire.ai/?dataset=fafb). In `neurons.csv`, the "root_id" is the unique id given to the neuron in Flywire, "group" is the neuropil of the neuron, "nt_type" is the predicted neurotransmitter type of the neuron, "nt_type_score" is the predicted probability for the neuron type to be its predicted neurotransmitter, "{neurotransmitter}_avg" is the probability for the neuron to be the respective neurotransmitter type. In `consolidated_cell_types.csv`, the "root_id" is the unique id given to the neuron in Flywire, the "primary_type" is the primary neuron type of the neuron, and the "additional_type(s)" is any additional neuron type assigned to that root id.
- `saved_data/MB_Imaging` - Images of mushroom body neurons expressing a neuronal and mitochondrial marker for various Kenyon Cell neuron types. Details for how the experiment is conducted is found in the Methods section. Each subfolder is named for the driver of the flyline which can be cross-referenced to the drivers in the methods section table in the "Optical imaging of mitochondria in Kenyon cells" section of the Methods.
- `saved_data/tseries`- Imaging experimental data of metabolic markers measured from spontaneous activity in the brain. Data are derived from the "K. Mann, C. L. Gallen, T. R. Clandinin, Whole-Brain Calcium Imaging Reveals an Intrinsic Functional Network in Drosophila. Curr. Biol. 15, 2389-2396. j.cub.2017.06.076 (2017).". The data were distributed via personal communication with the authors, but the Clark lab retains a local copy that can also be distributed up request. Within the tseries folder are subfolders for the three calcium indicators used in the study. Within each of these subfolders are txt files for independent measures of brain activity measured from the indicator. The txt files are matrices where the rows are time indices (sampled at 1.2 Hz) and the columns are the neuropils which can be cross-referenced to the file `turner_data/ito_68_atlas/Original_Index_panda_full.csv`.
- `saved_data/turner_data` - Brain atlases used to define neuropils. Data are derived from "M. H. Turner, K. Mann, T. R. Clandinin, The connectome predicts resting-state functional connectivity across the Drosophila brain. Curr. Biol. 31, 2386-2394. e2383 (2021)." Upon initially downloading the zipped data folder on Dryad associated with this publication, this folder will be empty. To download the necessary data, go to "https://figshare.com/articles/dataset/Drosophila_central_brain_connectivity/13349282" (https://doi.org/10.6084/m9.figshare.13349282.v3) and select the "Download all (240.3 MB)". The file "13349282.zip" should now be downloaded to your computer. After unzipping and opening that file, there should be the zipped file "data_TurnerMannClandinin.tar.gz" within in. Open that file and it should unzip a folder called "data". This is exactly the same folder as "turner_data". Move that folder into the "saved_data" folder in this repository. The files within the "data" folder (called "turner_data" in this repository) should be body_ids.csv, EBR_skels.png, LNO_skels.png, connectome_connectivity, atlas_data, ito_responses, subsample, branson_999_atlas, ito_68_atlas, branson_responses, template_brains, and hemi_2_atlas. However, it issues persist in downloading the data, the Clark lab has a local copy of this folder that can be shared.
- `saved_data/all_bodyIds.csv` - List of bodyIds for neurons in the hemibrain downloaded from `https://dvid.io/blog/release-v1.2/`. The "bodyId" is the unique id assigned to each neuron in the hemibrain, the "neuron_type" is the neuron type of the associated bodyId as defined by the hemibrain, and "is_cropped" is whether the bodyId is cropped in the dataset.

#### Utility Files
- `show_MB.py` - Python script for visualizing the Kenyon Cell experiments
- `util_files/` - Directory containing utility functions for various scripts

#### Other Directories
- `HPC/` - High Performance Computing related files
- `user_queries/` - Scripts to query the user about the existence of trivial leaves

## Project Overview

This project focuses on analyzing mitochondrial positioning and morphology across different brain regions and species. The analysis includes:

1. Classification of neurons into neuron types by their mitochondria's morphometrics
2. Positioning rule analysis fo mitochondria
3. Mitochondrial connectome analysis
4. Skeleton visualization and processing
5. Correlation analyses between different neuronal features

## Data Organization

The project maintains separate directories for different types of data:
- Raw and processed skeletons
- Synapse and mitochondrial data
- Generated figures
- Utility functions

## Getting Started

To work with this project:

1. Ensure you have the required Python packages installed
2. Start with the visualization notebooks to understand the data structure
3. Use the classification notebooks for morphological analysis
4. Explore the positioning rules notebooks for spatial analysis
5. Check the mitochondrial analysis notebooks for connectome studies

## Note

This project contains multiple analysis pipelines for different brain regions and species. Each notebook is designed to be self-contained but may share common utility functions and data processing steps. 

It is recommended to use the pre-processed data, which can be found on our dryad folder. However, this data can all be computed from the relevant script in the HPC folder or the python notebooks in the main folder. The HPC files were built to run on the Yale University HPC clusters, so edits will likely need to be made to run on your local computer or HPC cluster.