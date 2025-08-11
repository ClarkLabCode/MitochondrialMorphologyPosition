#!/usr/bin/env python

import os
import importlib
from os.path import isfile
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('config', '/home/gs697/project/clean_mito_code/util_files/config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Save info to a joblist file
log_file = 'joblist.txt'
with open(log_file, 'w') as f:
    f.truncate()
    for i_type in range(len(config.analyze_neurons)):
        f.write(f'module load miniconda; source activate flyem-stuff; python3 main.py --i_type={i_type}\n')
