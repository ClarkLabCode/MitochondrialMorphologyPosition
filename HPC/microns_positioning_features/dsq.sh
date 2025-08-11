#!/bin/bash

module load dSQ

dsq --job-file=joblist.txt \
--job-name=microns_pos_feats \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=1 \
--mem-per-cpu=8G \
--time=1:00:00 \
--mail-type=ALL \
--mail-user=garrett.sager@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist-%A_%a-%N.out
