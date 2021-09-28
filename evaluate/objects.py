import numpy as np
from pydantic import BaseModel, validator
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from lap import lapjv


class CostMatrix(BaseModel):
    cost_matrix: np.ndarray
    ground_truth_ids: np.ndarray
    prediction_ids: np.ndarray
    cost_unmatched: Optional[float] = 0.5

    @validator('cost_matrix', pre=True)
    def parse_cost_matrix(v):
        return np.array(v, dtype=float)

    @validator('ground_truth_ids', pre=True)
    def parse_ground_truth_ids(v):
        return np.array(v, dtype=float)

    @validator('prediction_ids', pre=True)
    def parse_prediction_ids(v):
        return np.array(v, dtype=float)

    class Config:
        arbitrary_types_allowed = True

    def match(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        matching_scores = self.cost_matrix.copy()
        matching_scores[matching_scores >
                        -0.5 + np.finfo('float').eps] = 0
        match_rows, match_cols = linear_sum_assignment(
            matching_scores)
        actually_matched_mask = matching_scores[match_rows,
                                                match_cols] < 0 - np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]
        match_predition = self.prediction_ids[match_cols]
        matched_ground_truth = self.ground_truth_ids[match_rows]
        unmatched_prediction = np.setdiff1d(
            self.prediction_ids, match_predition)
        unmatched_detection = np.setdiff1d(
            self.ground_truth_ids, matched_ground_truth)

        return match_predition, matched_ground_truth, unmatched_prediction, unmatched_detection
