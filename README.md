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

#### Data Directories
- `saved_data/` - Storage for pre-processed data
- `saved_figures/` - Storage for generated figures
- `saved_clean_skeletons/` - Processed neuronal skeletons with updating radii and trimmed trivial leaves
- `saved_neuron_skeletons/` - Neuronal skeletons with updated radii but the trivial leaves are still present
- `saved_synapse_df/` - Synapse pandas dataframes of LC neurons
- `saved_mito_df/` - Mitochondrial pandas dataframes of LC neurons
- `unprocessed_skeletons/` - Neuronal skeletons as found in the hemibrain dataset

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