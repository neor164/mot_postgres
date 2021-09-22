import os
from glob import glob
import re
import numpy as np
from ..database_creator import DatabaseCreator
from ..tables.ground_truth_tables import GroundTruthProps


def read_ground_truth():
    pattern = 'gt_data/*/train/*/gt/gt.txt'
    sequence_pattern = 'gt_data/(.*?)Labels/train/'
    scenario_pattern = 'train/(.*?)/gt'
    dc = DatabaseCreator()
    files = glob(pattern)
    for idx, file in enumerate(files):
        detections_array = np.loadtxt(file, delimiter=',', ndmin=2)
        sequnce_name = re.findall(sequence_pattern, file).findall()
        scenario_name = re.findall(scenario_pattern, file).findall()
        scp = dc.get_scenario_props_by_name(scenario_name)
        for row in detections_array:
            frame_id = row[0]
            target_id = row[1]
            tp = GroundTruthProps(frame_id=frame_id, target_id=target_id,
                                  scenario_id=scp.id, is_hidden=row[6] > 0)
            tp.min_x = row[2]
            tp.min_y = row[3]
            tp.width = row[4]
            tp.height = row[5]
