# Microns L2/3 

Dataset of mitochondria and synpases in all excitatory and inhibitory cells segmented from the phase 1 Microns dataset. It combines the published dataset of mitochondria in pyramidal cells with rerunning published analysis scripts from the MicronsBinder repository on the mitochondria in interneurons and additional new morphometrics analysis.

- Website: https://www.microns-explorer.org/phase1
- Repository: https://github.com/AllenInstitute/MicronsBinder, especially the [mitochondrial analysis](https://github.com/AllenInstitute/MicronsBinder/tree/master/notebooks/vignette_analysis/mitochondria) therein
- Publications:
    - [Reconstruction of neocortex: Organelles, compartments, cells, circuits, and activity](https://www.sciencedirect.com/science/article/pii/S0092867422001349)

## Dataset generation

The process of preparing the dataset is documented in the [Makefile](./Makefile).

Note that to rerun the analysis of the mitochondria, we reused the published code from the MicronsBinder repository and for convenience added them to this repository:
- `scripts`
- `lib`

### Package and kernel installation

```
conda env create -f environment.yml
```

Install the jupyter notebook kernel
```
conda activate micronsbinder
python -m ipykernel install --user --name micronsbinder
```

### Downloading base data

```
make base_data
```

### Calculating morphometrics and skeleton associations

```
make all
```


## Output files

All coordinates are in nm converted from the original voxel positions by the EM resolution of 3.58x3.58x40nm. 

After running the dataset generation, the folder `smo_data` contains the following files:

- microns.tar.gz: Folder with csv files for each cell among the pyramidal, inhibitory and astroycte cells with a soma in the reconstructed volume where each row specifies a mitochondria
    - mito id: segmentation id of the mitochondria
	- cellid: segmentation id of the cell 
	- compartment: associated neuronal compartment
	- x,y,z: centroid coordinatess
	- PC1 length, etc: morphometrics of the mitochondrial meshes
- morphologies.tar.gz: Folder with swc-based morphology files according to the [SWC standard](https://swc-specification.readthedocs.io/en/latest/swc.html) except that coordinates are in nm to fit with the synapse and mitochondrial positions and the compartment labels are shifted by one 
    - 0: 'Somatic'
    - 1: 'Axonal',
    - 2: 'Basal dendrite',
    - 3: 'Apical dendrite',
    - 4: 'Unknown/Ambiguous dendrite',
    - 5: 'Unknown/Ambiguous'
- nodes.tar.gz: Folder with json files for each cell named by its segmentation id that reports the nodes in the neurons skeleton covered by its mitochondria <mito\_id>: [<node\_ids>] (best to read via `nodes_df = pandas.read_json("<cell_id>_associated_nodes.json", typ="series")`
- cell\_type\_classification_filtered.csv: all analysed cells with cell ids, centroid positions and annoated cell type and cell subtype (e.g. inhibitory basket cell)
- synapse_filtered.csv: all synapses with each row specifying one synpase by
	- synapse id: segmentation id of the synpase
    - cell_id: associated cell id (note that in this way, synapses between cells in the dataset appear twice once as pre and once as post)
    - synapse_type: pre | post | autapse
    - x,y,z: centroid position in nm
	- pre\_root\_id and post\_root\_id: cell ids of presynaptic and postsynaptic partner
- smo_microns_demonstration.ipynb: plots of cell morphologies with mitochondria and synapses, requires [navis](https://navis-org.github.io/navis/)

### Comments

- While compartment labels are provided for the reconstructed pyramidal and inhibitory neurons, they are missing for the astrocytes. Thus for the astroycytes, compartment labels are inferred from the distance of a mitochondria to the center of the nucleus (Somatic if d<7um and Branch otherwise). In addition, a mitochondria is labelled far if it further than 70um away from the soma as a simple way to mark potential merge errors of the astroyctes with neurites such as axons. 
