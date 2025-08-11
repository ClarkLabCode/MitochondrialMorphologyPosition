# Get base data from Microns Layer 2/3 and analysed pyramidal mitochondria (Turner, 2022)

jupyterrun=jupyter nbconvert --to notebook --execute --inplace 

### Available skeletons with compartment labels
vignette_datadir= skeleton_data
vignette_tarfile= ${vignette_datadir}/211019_smoothed_skeletons.tgz 
vignette_url= https://zenodo.org/record/6363348/files/211019_smoothed_skeletons.tgz
${vignette_tarfile}:
	mkdir -p ${vignette_datadir}
	if [ ! -f ${vignette_tarfile} ]; \
		then printf "\033[1;33m%s\033[0;39m\n" \
			"Downloading base vignette analysis data"; \
		wget -q -O ${vignette_tarfile} ${vignette_url} \
		&& tar -C ${vignette_datadir} -xzf ${vignette_tarfile}; \
	fi
vignette_analysis/data: ${vignette_tarfile} 
clean-vignette_analysis/data:
	if [ -d ${vignette_datadir} ]; then rm -r ${vignette_datadir}; fi

### Processed pyramical mitochondria
mito_datadir= data
mito_tarfile= ${mito_datadir}/vignette_mito_analysis_data.tgz
${mito_tarfile}:
	mkdir -p ${mito_datadir}
	if [ ! -f ${mito_tarfile} ]; \
		then printf "\033[1;33m%s\033[0;39m\n" \
			"Downloading mitochondria vignette analysis data"; \
		gdown --id 1MruDsi7q77G5Mq0b1t95TDFwVoHDK3WB -O ${mito_tarfile} \
		&& tar -C ${mito_datadir} -xzf ${mito_tarfile}; \
	fi
vignette_analysis/mitochondria: ${mito_tarfile}
clean-vignette_analysis/mitochondria:
	if [ -d ${mito_datadir} ]; then rm -r ${mito_datadir}; fi

### Datafile with all mitochondria in the L2/3 dataset and all labelled cells
DATA_DIR=smo_data
ALL_MITOS_URL="https://zenodo.org/records/7510511/files/211019_mitochondria_info.csv?download=1"
CELL_TYPE_URL="https://zenodo.org/records/7510511/files/220309_cell_type_classification.csv?download=1"
ALL_MITO_DF=${DATA_DIR}/mito.csv
CELL_TYPE_DF=${DATA_DIR}/cell_type_classification.csv


${ALL_MITO_DF} ${CELL_TYPE_DF}&:
	mkdir -p ${DATA_DIR}
	wget -O ${ALL_MITO_DF} ${ALL_MITOS_URL}
	wget -O ${CELL_TYPE_DF} ${CELL_TYPE_URL}
additional_mito_and_celltype_data: ${ALL_MITO_DF} ${CELL_TYPE_DF}


### Download pyramidal mito meshes
data/mitomeshes/.download_pyr_mitos_done:
	echo "## Downloading mitochondrial meshes for pyramidal cells"
	mkdir -p data/mitomeshes
	jupyter execute smo_download_pyr_mito_meshes.ipynb \
	 && touch data/mitomeshes/.download_pyr_mitos_done
download_pyr_meshes: data/mitomeshes/.download_pyr_mitos_done


### Download neuronal meshes
download_neuron_meshes:
	mkdir -p data/neuron_meshes_v185
	cd data \
	&& gdown --id 1lPWicWMBsIiXur68Sqf_FMS5ZzHoSqLj -O neuron_meshes_v185.tgz \
	&& tar -xzf neuron_meshes_v185.tgz \


base_data: vignette_analysis/mitochondria additional_mito_and_celltype_data download_pyr_meshes
	
# Process data for mitochondrial morphometrics 

MITO_BASE_INFO=data/pni_mito_cellswskel_v185.csv

INHIBITORY_CELLS=${DATA_DIR}/inh_cell.csv

MITO_SKEL_ASSOC_DIR=data/mitotoskel
ASSOC_PREFIX=assoc
INH_MITO=${DATA_DIR}/inh_mito.csv
INH_MITO_TO_SKEL=${DATA_DIR}/inhmitotoskel.h5
INH_MITO_EXTRA_STATS=${DATA_DIR}/inh_mito_extrastats.csv

### Extract list of inhibitory cells for matching mitos
${INH_MITO} ${INHIBITORY_CELLS}&:
	jupyter execute smo_write_list_of_inhibitory_cellids.ipynb
list_of_inhibitory_cells: ${INH_MITO} ${INHIBITORY_CELLS}

smo_data/cell_type_classification_filtered.csv: ${CELL_TYPE_DF} ${INH_MITO}
	jupyter execute smo_export_celldf.ipynb

### Associate mitos in inhibitory cells to the skeleton to get the compartment they reside in
${INH_MITO_TO_SKEL}: ${INH_MITO} ${INHIBITORY_CELLS} 
	mkdir -p ${MITO_SKEL_ASSOC_DIR}
	python scripts/mitotoskel.py ${INH_MITO} ${INHIBITORY_CELLS} ${MITO_SKEL_ASSOC_DIR}/${ASSOC_PREFIX}
	python scripts/assemblemitotoskel.py ${INH_MITO} ${INHIBITORY_CELLS} ${INH_MITO_TO_SKEL} --no_diam
inh_mito_to_skeleton_association: ${INH_MITO_TO_SKEL} 

### Compute compartment label and other stats for each inh mito
${INH_MITO_EXTRA_STATS}: ${INH_MITO_TO_SKEL}
	python scripts/computeextramitostats.py ${INH_MITO} ${INH_MITO_TO_SKEL} ${INH_MITO_EXTRA_STATS}
inh_compartment_labels: ${INH_MITO_EXTRA_STATS} 

### Compute mitochondrial lengths
INH_MITO_FULL_STATS=${DATA_DIR}/inh_mito_fullstats.csv
${INH_MITO_FULL_STATS}: ${INH_MITO_EXTRA_STATS} 
	python scripts/computemitolengths.py ${INH_MITO_EXTRA_STATS} ${INHIBITORY_CELLS} ${INH_MITO_TO_SKEL} ${INH_MITO_FULL_STATS}
inh_mito_lengths: ${INH_MITO_FULL_STATS}


### Extract list of astrocyte mitos and download their mitochondrial meshes
smo_data/astro_mito.csv:
	jupyter execute smo_write_list_of_astroyctes.ipynb
	juptyer execute smo_download_astrocyte_mitos.ipynb

### Calculate morphometrics for all mitochondria
smo_data/inh_mito_morphometrics.csv smo_data/pyr_mito_morphometrics.csv smo_data/astro_mito_morphometrics.csv&: ${INH_MITO_TO_SKEL} smo_data/astro_mito.csv
	jupyter execute smo_calculate_mito_morphometrics.ipynb
morphometrics: smo_data/inh_mito_morphometrics.csv smo_data/pyr_mito_morphometrics.csv

### Simple compartment labelling for astro mitos
smo_data/astro_mito_fullstats.csv: smo_data/astro_mito.csv smo_data/astro_mito_morphometrics.csv
	jupyter execute smo_compile_astro_mito_compartment_label.ipynb

### Prepare cell files
CELLFILES_FOLDER=smo_data/microns_cells
CELLFILES_RUN=${CELLFILES_FOLDER}/.has_run

${CELLFILES_RUN}: smo_data/inh_mito_morphometrics.csv smo_data/pyr_mito_morphometrics.csv smo_data/astro_mito_morphometrics.csv ${CELL_TYPE_DF} ${INH_MITO_FULL_STATS} smo_data/astro_mito_fullstats.csv
	mkdir -p ${CELLFILES_FOLDER}
	jupyter execute smo_write_cellfiles.ipynb &&\
	touch ${CELLFILES_RUN}

### Pack cell files for upload
smo_data/microns.tar.gz: ${CELLFILES_RUN}
	cd ${DATA_DIR} &&\
	tar -zcvf microns.tar.gz microns_cells/


# Positioning rules

### Export cell morphologies
CELL_MORPHOLOGY_FOLDER=smo_data/cell_morphologies
CELL_MORPHOLOGY_RUN=${CELL_MORPHOLOGY_FOLDER}/.has_run
${CELL_MORPHOLOGY_RUN}:
	mkdir -p ${CELL_MORPHOLOGY_FOLDER}
	${jupyterrun} smo_export_cell_morphologies.ipynb && \
	touch ${CELL_MORPHOLOGY_RUN}

### Get list of all synapses
SYNAPSE_DF=${DATA_DIR}/synapse.csv
${SYNAPSE_DF}:
	mkdir -p ${DATA_DIR}
	wget -O ${SYNAPSE_DF} "https://zenodo.org/records/5579388/files/pni_synapses_v185.csv?download=1"

### Export filtered synapses
smo_data/synapse_filtered.csv: ${SYNAPSE_DF}
	jupyter execute smo_export_synapse_df.ipynb

### Get neuronal skeleton node list for each mito
MITO_NODES_FOLDER=${DATA_DIR}/associated_nodes
MITO_NODES_RUN=${MITO_NODES_FOLDER}/.has_run
${MITO_NODES_RUN}: ${CELL_MORPHOLOGY_RUN} smo_data/synapse_filtered.csv ${CELLFILES_RUN}
	mkdir -p ${MITO_NODES_FOLDER}
	${jupyterrun} smo_associate_mitos_to_skeleton_nodes.ipynb &&\
	touch ${MITO_NODES_RUN}

smo_data/nodes.tar.gz: ${MITO_NODES_RUN}
	cd ${DATA_DIR} &&\
	tar -zcvf nodes.tar.gz associated_nodes/



smo_data/morphologies.tar.gz: ${CELL_MORPHOLOGY_RUN}
	cd ${DATA_DIR} &&\
	tar -zcvf morphologies.tar.gz cell_morphologies/
    
all: smo_data/microns.tar.gz smo_data/nodes.tar.gz smo_data/morphologies.tar.gz smo_data/cell_type_classification_filtered.csv


