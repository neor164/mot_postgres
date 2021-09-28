from .database_creator import DatabaseProps, DatabaseCreator
from typing import Dict, Optional, Union
from .evaluate.objects import CostMatrix
from .evaluate.preprocessing import calculate_cost_matrix


class DatabaseEvaluator:

    def __init__(self, db_props: DatabaseProps = DatabaseProps()) -> None:
        self.database: DatabaseCreator = DatabaseCreator(db_props)

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
            cost_matrix=calculate_cost_matrix(bb1, bb2), ground_truth_ids=gt['target_id'], prediction_ids=dt['target_index'])

        match_predition, _, unmatched_prediction, unmatched_detection = cost_matrix_object.match()
        eval_dict["TP"] = len(match_predition)
        eval_dict["FP"] = len(unmatched_prediction)
        eval_dict["FN"] = len(unmatched_detection)
        eval_dict["Precision"] = eval_dict["TP"] / \
            (eval_dict["TP"] + eval_dict["FP"])
        eval_dict["Recall"] = eval_dict["TP"] / \
            (eval_dict["TP"] + eval_dict["FN"])
        return eval_dict
