#!/bin/bash

module load dSQ

dsq --job-file=joblist_mito_0.txt \
--job-name=save_mito \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=1 \
--mem-per-cpu=50G \
--time=1- \
--mail-type=ALL \
--mail-user=garrett.sager@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist_mito-%A_%a-%N.out
