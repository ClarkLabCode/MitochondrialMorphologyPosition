#!/bin/bash

module load dSQ

dsq --job-file=joblist.txt \
--job-name=save_feat_dists \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=1 \
--mem-per-cpu=16G \
--time=12:00:00 \
--mail-type=ALL \
--mail-user=garrett.sager@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist-%A_%a-%N.out
