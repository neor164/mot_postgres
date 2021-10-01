from .database_creator import DatabaseProps, DatabaseCreator
from typing import Dict, Optional, Union
from .evaluate.objects import CostMatrix
from .tables.tracker_tables import TargetFrameEvalProps, TrackerEvalProps, TrackerScenarioEvalProps
from .evaluate.preprocessing import calculate_similarity_matrix
import numpy as np


class DatabaseEvaluator:

    def __init__(self, db_props: DatabaseProps = DatabaseProps()) -> None:
        self.database: DatabaseCreator = DatabaseCreator(db_props)
        self.threshold = 0.5

    def calculate_detector_frame_evaluation(self, run_id: int, scenario_name: str,  frame_id: int, confidance: Optional[float] = None) -> Dict[str, Union[float, int]]:

        dt = self.database.get_detection_table_by_frame(
            run_id, scenario_name, frame_id, confidance)
        gt = self.database.get_ground_truth_by_frame_table(
            scenario_name, frame_id, visibilty_thresh=0.0)
        scenario_id = self.database.get_scenario_props_by_name(
            scenario_name).id
        eval_dict = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "frame_id": frame_id,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "Recall":  0.0,
            "Precision": 0.0,
            "confidance_level": confidance
        }

        if dt.empty:
            eval_dict["FN"] = gt.shape[0]
            return eval_dict
        if eval_dict['confidance_level'] > 0:
            eval_dict['confidance_level'] = eval_dict['confidance_level'] or self.database.get_confidance_by_run_id(
                run_id)
        if gt.empty:
            eval_dict["FP"] = dt.shape[0]
            return eval_dict

        bb1 = gt.values[:, 1:-1]
        bb2 = dt.values[:, 4:-1]
        cost_matrix_object = CostMatrix(
            cost_matrix=-calculate_similarity_matrix(bb1, bb2), ground_truth_ids=gt['target_id'], prediction_ids=dt['target_index'])

        match_predition, _, unmatched_prediction, unmatched_detection = cost_matrix_object.match()
        eval_dict["TP"] = len(match_predition)
        eval_dict["FP"] = len(unmatched_prediction)
        eval_dict["FN"] = len(unmatched_detection)
        eval_dict["Precision"] = eval_dict["TP"] / \
            (eval_dict["TP"] + eval_dict["FP"])
        eval_dict["Recall"] = eval_dict["TP"] / \
            (eval_dict["TP"] + eval_dict["FN"])
        return eval_dict

    def calculate_tracker_frame_evaluation(self, run_id: int, scenario_name: str,  frame_id: int) -> Dict[str, Union[float, int]]:
        pt = self.database.get_tracker_table_by_frame(
            run_id, scenario_name, frame_id)
        gt = self.database.get_ground_truth_by_frame_table(
            scenario_name, frame_id, visibilty_thresh=0.0)
        scenario_id = self.database.get_scenario_props_by_name(
            scenario_name).id
        # tefp = TargetFrameEvalProps(
        #     run_id=run_id, frame_id=frame_id, scenario_id=scenario_id)
        tep = TrackerEvalProps(
            run_id=run_id, frame_id=frame_id, scenario_id=scenario_id)
        if pt.empty:
            tep.FN = gt.shape[0]
            return tep.dict()

        if gt.empty:
            tep.FP = pt.shape[0]
            return tep.dict()
        bb1 = gt.values[:, 1:-1]
        bb2 = pt.values[:, 4:-1]
        sim_mat = calculate_similarity_matrix(bb1, bb2)

        tfet = self.database.get_target_frame_eval_table_by_frame(
            run_id, scenario_name, frame_id - 1)
        if tfet.empty:
            prev_timestep_tracker_id = np.nan * np.zeros(gt.shape[0])
        else:
            prev_timestep_tracker_id = tfet['tracker_id'].fillna(value=np.nan)

        score_mat = (pt['tracker_id'][np.newaxis, :] ==
                     prev_timestep_tracker_id[:, np.newaxis])
        score_mat[sim_mat < self.threshold - np.finfo('float').eps] = 0
        score_mat = 1000 * score_mat + sim_mat
        cost_matrix_object = CostMatrix(
            cost_matrix=-score_mat, ground_truth_ids=gt['target_id'], prediction_ids=pt['tracker_id'])

        matched_predition, matched_objects, unmatched_prediction, unmatched_detection = cost_matrix_object.match()
        target_frame_eval_list = []
        for target_id in gt['target_id']:
            tfep = TargetFrameEvalProps(
                scenario_id=scenario_id, run_id=run_id, frame_id=frame_id, target_id=target_id)
            tfep.tracker_id = matched_predition[matched_objects == target_id]
            tid = self.database.get_previous_match_by_frame_and_id(
                run_id, scenario_name, frame_id, target_id)
            if tfep.tracker_id is not None:
                tfep.iou = sim_mat[matched_objects == target_id,
                                   matched_predition == matched_objects]

                if tid is not None and tid != tfep.tracker_id:
                    tep.IDSW += 1
            elif tid is None and tfep.tracker_id is not None:
                tep.Frag += 1
            target_frame_eval_list.append(tfep.dict())

        self.database.upsert_bulk_target_frame_eval_props(
            target_frame_eval_list)
        tep.TP = len(matched_predition)
        tep.FP = len(unmatched_prediction)
        tep.FN = len(unmatched_detection)
        tep.GT = gt.shape[0]

        return tep
