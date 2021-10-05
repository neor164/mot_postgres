from mot_postgres import dbc
import os
from glob import glob
import re
import numpy as np
pattern = 'gt_data/*/train/*/gt/gt.txt'
sequence_pattern = 'gt_data/(.*?)Labels/train/'
scenario_pattern = 'train/(.*?)/gt'
files = glob(pattern)
for idx, file in enumerate(files):
    detections_array = np.loadtxt(file, delimiter=',', ndmin=2)
    sequnce_name = re.findall(sequence_pattern, file).findall()
    scenario_name = re.findall(scenario_pattern, file).findall()
    for row in detections_array:
        frame_id = row[0]
        target_id = row[1]

    if sequnce_name == "MOT20":
        pass
    else:
        pass
