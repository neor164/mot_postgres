from typing import Sequence
from typing_extensions import runtime
from numpy.core.fromnumeric import shape
from mot_postgres.tables.detector_tables import DetectionsFrameEval
from mot_postgres.database_creator import DatabaseCreator
from mot_postgres.database_creator import DatabaseProps
import os
from mot_postgres.database_evaluator import DatabaseEvaluator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import subprocess as sp
import numpy as np
import cv2
dp = DatabaseProps()
os.environ["PGPASSWORD"] = 'neor123'
dp.port = 8888
dbc = DatabaseCreator(dp)
dbe = DatabaseEvaluator(dp)
fps = 25
run_id = 1


def show_miss_detection(seq, output_path):
    challenge = dbc.get_scenario_props_by_name(seq).source
    encoding = cv2.VideoWriter_fourcc(*'MPEG')
    im_dir = f'/home/neor/dataset/{challenge}/train/{seq.upper()}/img1/{1:06d}.jpg'
    im = cv2.imread(im_dir)
    video_file = f'{output_path}/{seq}.avi'
    vid_dim = (im.shape[1], im.shape[0])
    fps = 25
    vid = cv2.VideoWriter(video_file,  encoding, fps, vid_dim)

    frame_ids = dbc.get_frame_ids_by_scenario(seq)
    for frame_id in frame_ids:
        im_dir = f'/home/neor/dataset/{challenge}/train/{seq.upper()}/img1/{frame_id:06d}.jpg'
        im = cv2.imread(im_dir)
        gt_ids = dbc.get_missed_detection_target_ids_by_frame(
            run_id, seq, frame_id)
        gt_df = dbc.get_ground_truth_by_frame_table(seq, frame_id)
        new_gt_df = gt_df.loc[gt_df['target_id'].isin(gt_ids)]
        gts = new_gt_df.values[:, 1:-1]
        det_df = dbc.get_detection_table_by_frame(run_id, seq, frame_id, 0.6)
        dets = det_df.values[:, 4:-1]
        for gt in gts:
            im = cv2.rectangle(
                im, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1]+gt[3])), (0, 0, 255), 2)

        for det in dets:
            im = cv2.rectangle(
                im, (int(det[0]), int(det[1])), (int(det[0]+det[2]), int(det[1]+det[3])), [255, 0, 0], 2)
        vid.write(im)


def show_tracking_vid(seq, output_path):
    challenge = dbc.get_scenario_props_by_name(seq).source
    encoding = cv2.VideoWriter_fourcc(*'MPEG')
    im_dir = f'/home/neor/dataset/{challenge}/train/{seq.upper()}/img1/{1:06d}.jpg'
    im = cv2.imread(im_dir)
    video_file = f'{output_path}/{seq}.avi'
    vid_dim = (im.shape[1], im.shape[0])
    fps = 25
    vid = cv2.VideoWriter(video_file,  encoding, fps, vid_dim)
    frame_ids = dbc.get_frame_ids_by_scenario(seq)
    for frame_id in frame_ids:
        im_dir = f'/home/neor/dataset/{challenge}/train/{seq.upper()}/img1/{frame_id:06d}.jpg'
        im = cv2.imread(im_dir)
        gt_ids = dbc.get_missed_detection_target_ids_by_frame(1, seq, frame_id)
        gt_df = dbc.get_ground_truth_by_frame_table(seq, frame_id)
        new_gt_df = gt_df.loc[gt_df['target_id'].isin(gt_ids)]
        gts = new_gt_df.values[:, 1:-1]
        tt = dbc.get_tracker_table_by_frame(
            run_id=run_id, scenario_name=seq, frame_id=frame_id)
        tracks = tt.values[:, 2:]
        kalmans = dbc.get_kalman_with_no_detection_table_by_frame(
            run_id=run_id, scenario_name=seq, frame_id=frame_id).values[:, 1:]
        for gt in gts:
            im = cv2.rectangle(
                im, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1]+gt[3])), (0, 0, 255), 2)

        for det in tracks:
            im = cv2.rectangle(
                im, (int(det[0]), int(det[1])), (int(det[0]+det[2]), int(det[1]+det[3])), [0, 255, 0], 2)

        for det in kalmans:
            im = cv2.rectangle(
                im, (int(det[0]), int(det[1])), (int(det[0]+det[2]), int(det[1]+det[3])), [0, 255, 255], 2)
        vid.write(im)
        # print(frame_id)


def show_IDSW_vid(seq, output_path):
    challenge = dbc.get_scenario_props_by_name(seq).source
    encoding = cv2.VideoWriter_fourcc(*'MPEG')
    im_dir = f'/home/neor/dataset/{challenge}/train/{seq.upper()}/img1/{1:06d}.jpg'
    im = cv2.imread(im_dir)
    video_file = f'{output_path}/{seq}.avi'
    vid_dim = (im.shape[1], im.shape[0])
    fps = 25
    vid = cv2.VideoWriter(video_file,  encoding, fps, vid_dim)
    target_id = dbc.get_target_nth_highest_IDSW_by_scenario(run_id, seq)
    first_frame = dbc.get_first_frame_for_target_id(seq, target_id)
    last_frame = dbc.get_last_frame_for_target_id(seq, target_id)

    for frame_id in range(first_frame, last_frame + 1):

        im_dir = f'/home/neor/dataset/{challenge}/train/{seq.upper()}/img1/{frame_id:06d}.jpg'
        im = cv2.imread(im_dir)
        gt = dbc.get_gt_props_by_frame_and_id(seq, frame_id, target_id)
        if gt and gt.min_x:
            im = cv2.rectangle(
                im, (int(gt.min_x), int(gt.min_y)), (int(gt.min_x + gt.width), int(gt.min_y + gt.height)), (0, 0, 255), 2)
        tracker_id = dbc.get_current_tracker_id_for_gt(
            run_id, seq, frame_id, target_id)
        if tracker_id is not None:
            tt = dbc.get_tracker_props_by_frame_and_id(
                run_id=run_id, scenario_name=seq, frame_id=frame_id, tracker_id=tracker_id)
            if tt.min_x:
                im = cv2.rectangle(
                    im, (int(tt.min_x), int(tt.min_y)), (int(tt.min_x + tt.width), int(tt.min_y + tt.height)), (0, 255, 0), 2)
            elif tt.kalman_min_x:
                im = cv2.rectangle(
                    im, (int(tt.kalman_min_x), int(tt.kalman_min_y)), (int(tt.kalman_min_x + tt.kalman_width), int(tt.kalman_min_y + tt.kalman_height)), (0, 255, 255), 2)

        ft_ids = dbc.get_future_tracker_id_for_gt(
            run_id, seq, frame_id, target_id, tracker_id)
        for ft_id in ft_ids:
            ft = dbc.get_tracker_props_by_frame_and_id(
                run_id=run_id, scenario_name=seq, frame_id=frame_id, tracker_id=ft_id)
            if ft is not None:

                if ft.min_x:
                    im = cv2.rectangle(
                        im, (int(ft.min_x), int(ft.min_y)), (int(ft.min_x + ft.width), int(ft.min_y + ft.height)), (255, 0, 0), 2)
                    im = cv2.putText(im, str(
                        int(ft.tracker_id)), (int(ft.min_x), int(ft.min_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, [255, 0, 0], 2)
                elif ft.kalman_min_x:
                    im = cv2.rectangle(
                        im, (int(ft.kalman_min_x), int(ft.kalman_min_y)), (int(ft.kalman_min_x + ft.kalman_width), int(ft.kalman_min_y + ft.kalman_height)), (255, 0, 255), 2)
                    im = cv2.putText(im, str(
                        int(ft.tracker_id)), (int(ft.kalman_min_x), int(ft.kalman_min_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, [255, 0, 255], 2)

        vid.write(im)
