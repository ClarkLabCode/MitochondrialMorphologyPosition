#!/bin/bash

module load dSQ

dsq --job-file=joblist_skels.txt \
--job-name=skels \
--partition=day \
--ntasks=1 \
--nodes=1 \
--cpus-per-task=3 \
--mem-per-cpu=32G \
--time=1- \
--mail-type=ALL \
--mail-user=garrett.sager@yale.edu \
--status-dir=status_files \
--output=dsq_output/dsq-joblist_skels-%A_%a-%N.out
