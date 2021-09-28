import numpy as np
from ..tables.detector_tables import DetectionsProps
from ..tables.ground_truth_tables import GroundTruthProps
from ..tables.tracker_tables import TrackersProps
from typing import List, Union
from copy import deepcopy
from .objects import CostMatrix
from typing import Dict


def calculate_cost_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """ Calculates the IOU (intersection over union) between two arrays of boxes.
    Allows variable box format 'xywh' .
    If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
    used to determine if detections are within crowd ignore region.
    """

    # layout: (x0, y0, w, h)
    bboxes1 = deepcopy(bboxes1)
    bboxes2 = deepcopy(bboxes2)

    bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
    bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
    bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
    bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(
        min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * \
        (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * \
        (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
    intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
    intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
    intersection[union <= 0 + np.finfo('float').eps] = 0
    union[union <= 0 + np.finfo('float').eps] = 1
    ious = intersection / union

    return - ious
