
import os
from glob import glob
import re
import numpy as np
import pandas as pd
from tables.ground_truth_tables import ScenatioProps
from database_creator import DatabaseCreator, DatabaseProps

pattern = 'mot_postgres/gt_data/*/train/*/gt/gt.txt'
sequence_pattern = 'gt_data/(.*?)Labels/train/'
scenario_pattern = 'train/(.*?)/gt'

db_props = DatabaseProps()
db_props.port = 8888
os.environ['PGPASSWORD'] = 'neor123'
dbc = DatabaseCreator()

files = glob(pattern)
for idx, file in enumerate(files):
    detections_array = np.loadtxt(file, delimiter=',', ndmin=2)
    sequnce_name = re.findall(sequence_pattern, file)[0]
    scenario_name = re.findall(scenario_pattern, file)[0]

    # if sequnce_name == "MOT15":
    #     df = pd.read_csv(file, names=[
    #                      'frame_id', 'target_id', 'min_x', 'min_y', 'width', 'height', 'is_valid', 'x', 'y', 'z'])
    #     df['visibility'] = None
    #     df = df.drop(columns=['x', 'y', 'z'])
    # else:
    #     df = pd.read_csv(file, names=['frame_id', 'target_id', 'min_x',
    #                      'min_y', 'width', 'height', 'is_valid', 'target_id', 'visibility'])
    # df['scenario_id'] = dbc.get_scenario_props_by_name(scenario_name).id
    # df_list = df.to_dict('records')
    # dbc.upsert_bulk_ground_truth(df_list[::4])
    # dbc.upsert_bulk_ground_truth(df_list[1::4])
    # dbc.upsert_bulk_ground_truth(df_list[2::4])
    # dbc.upsert_bulk_ground_truth(df_list[3::4])

    if sequnce_name == "MOT20":

        df = pd.read_csv(file, names=['frame_id', 'target_id', 'min_x',
                                      'min_y', 'width', 'height', 'is_valid', 'target_id', 'visibility'])
        df['scenario_id'] = dbc.get_scenario_props_by_name(scenario_name).id
        df_list = df.to_dict('records')
        dbc.upsert_bulk_ground_truth(df_list[::4])
        dbc.upsert_bulk_ground_truth(df_list[1::4])
        dbc.upsert_bulk_ground_truth(df_list[2::4])
        dbc.upsert_bulk_ground_truth(df_list[3::4])

        print(scenario_name)
