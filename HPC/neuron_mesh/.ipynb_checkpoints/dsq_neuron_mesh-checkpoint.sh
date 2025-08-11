#!/bin/bash

module load dSQ

dsq --job-file=joblist_neuron_mesh.txt \
--job-name=save_neuron_mesh \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=1 \
--mem-per-cpu=50G \
--time=24:00:00 \
--mail-type=ALL \
--mail-user=garrett.sager@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist_neuron_mesh-%A_%a-%N.out
