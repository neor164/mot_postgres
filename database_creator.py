from .tables.eval_tables import GroundTruthDetectionMatchesFrame, GroundTruthDetectionMatchesFrameProps
from .tables.tracker_tables import TargetFrameEvalProps, TrackerDistances, TrackerEvalProps, Trackers, TargetFrameEval, TrackerEval, TrackerScenarioEval, TrackersProps
from typing import Optional, List, Dict, Set
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import Session, query
from sqlalchemy.dialects.postgresql import insert
from .tables.ground_truth_tables import GroundTruth, GroundTruthProps, Scenarios, ScenatioProps, TargetTypes
from .tables.detector_tables import Detections, DetectionsFrameEval, Detectors, DetectionsProps
from .tables.run_tables import Run
from pydantic import BaseModel
from sqlalchemy.sql import select
from sqlalchemy import cast, Float
from sqlalchemy.engine import Engine
from sqlalchemy.sql.functions import func
from sqlalchemy.sql.expression import case, distinct
from .tables.tables_base import Base
from glob import glob
import re
import numpy as np
from datetime import datetime
import os
import pandas as pd
from pathlib import Path


class DatabaseProps(BaseModel):
    username: str = 'postgres'
    name: str = 'postgres'
    port: int = 6666
    ip_address = '127.0.0.1'


class DatabaseCreator:
    def __init__(self, db_props: DatabaseProps = DatabaseProps()) -> None:
        self.engine: Engine = create_engine(
            f"postgresql://{db_props.username}:@{db_props.ip_address}:{db_props.port}/{db_props.name}")
        self.session: Session = Session(self.engine)
        self.base = Base
        self.base.metadata.create_all(self.engine)

    def add_scenario(self, scenario_props: ScenatioProps):
        values = scenario_props.dict(exclude_none=True)
        insert_stmt = insert(Scenarios).values(values)
        do_update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['id'], set_=values)
        self.session.execute(do_update_stmt)
        self.session.commit()

    def get_challenges(self):
        return [f.source for f in self.session.query(Scenarios).distinct(Scenarios.source)]

    def get_scenario_names_by_challenge(self, challenge: str) -> List[str]:
        return [f.name for f in self.session.query(Scenarios).filter(func.lower(Scenarios.source) == challenge.lower())]

    def get_scenario_ids_by_challenge(self, challenge: str) -> List[str]:
        return [f.id for f in self.session.query(Scenarios).filter(func.lower(Scenarios.source) == challenge.lower())]

    def get_scenario_props_by_name(self, scenario_name: str) -> Optional[ScenatioProps]:

        resp = self.session.query(Scenarios).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).first()
        if resp is not None:
            return ScenatioProps.from_orm(resp)

    def get_confidance_by_run_id(self, run_id: int) -> Optional[float]:
        resp = self.session.query(Run.confidance).filter(
            Run.id == run_id).first()
        if resp is not None:
            return float(resp[0])

    def get_scenario_props_by_id(self, scenario_id: int) -> Optional[ScenatioProps]:
        resp = self.session.query(Scenarios).filter(
            Scenarios.id == scenario_id).first()
        if resp is not None:
            return ScenatioProps.from_orm(resp)

    def get_frame_ids_by_scenario(self, scenario_name: str) -> List[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        return [r.frame_id for r in self.session.query(GroundTruth.frame_id).distinct(GroundTruth.frame_id).filter(GroundTruth.scenario_id == subquery)]

    def check_has_evaluated(self, run_id: int, challenge: str) -> bool:
        subquery = self.session.query(Scenarios.id).filter(
            Scenarios.source == challenge).scalar_subquery()
        resp = self.session.query(Detections.scenario_id).filter(
            Detections.scenario_id.in_(subquery)).all()
        if len(resp):
            return True
        else:
            return False

    def get_ground_truth_detection_by_scenario_and_frame(self, scenario_name: str, frame_id: int) -> Optional[List[GroundTruthProps]]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        resp = self.session.query(GroundTruth).filter(
            GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id).all()
        if resp is not None:
            gt_list = []
            for gt in resp:
                gt_list.append(GroundTruthProps.from_orm(gt))
            return gt_list

    def get_ground_truth_by_scenario_name(self, scenario_name: str, valid: bool = True) -> Optional[List[GroundTruth]]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        resp = self.session.query(GroundTruth).filter(
            GroundTruth.scenario_id == subquery, GroundTruth.is_valid.is_(valid)).all()
        return resp

    def add_detector(self, detector_name: str) -> int:
        resp = self.session.query(Detectors.id).filter(
            func.lower(Detectors.name) == detector_name.lower()).first()
        if resp is not None:
            return resp.id
        else:
            self.session.add(Detectors(name=detector_name))
            resp = self.session.query(Detectors.id).filter(
                func.lower(Detectors.name) == detector_name.lower()).first()
            self.session.commit()

    def upsert_detection_data(self, det: DetectionsProps):
        values = det.dict(exclude_none=True)
        insert_stmt = insert(Detections).values(values)
        do_update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['target_index', 'frame_id', 'scenario_id', 'run_id'], set_=values)
        self.session.execute(do_update_stmt)
        self.session.commit()

    def get_detections_by_scenario_and_frame(self, run_id: int, name: str, frame_id: int) -> Optional[List[DetectionsProps]]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == name.lower()).scalar_subquery()

        resp = self.session.query(Detections).filter(
            Detections.scenario_id == subquery, Detections.frame_id == frame_id, Detections.run_id == run_id).all()
        if resp is not None:
            det_list = []
            for gt in resp:
                det_list.append(DetectionsProps.from_orm(gt))
            return det_list

    def get_ground_truth_by_frame_table(self, scenario_name: str, frame_id: int, visibilty_thresh: Optional[float] = None, type_ids: List[int] = [TargetTypes.Pedestrian]) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        challege = self.get_scenario_props_by_name(scenario_name)
        if challege is not None:
            challege = challege.source
            if visibilty_thresh is not None:
                if challege.lower() != 'mot15':

                    resp = self.session.query(GroundTruth.target_id, GroundTruth.min_x, GroundTruth.min_y, GroundTruth.width, GroundTruth.height, GroundTruth.visibility, GroundTruth.type_id.in_(type_ids)).filter(
                        GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id, GroundTruth.visibility > visibilty_thresh, GroundTruth.is_valid)

                else:

                    resp = self.session.query(GroundTruth.target_id, GroundTruth.min_x, GroundTruth.min_y, GroundTruth.width, GroundTruth.height, GroundTruth.visibility, GroundTruth.type_id.in_(type_ids)).filter(
                        GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id, GroundTruth.is_valid)
            else:
                resp = self.session.query(GroundTruth.target_id, GroundTruth.min_x, GroundTruth.min_y, GroundTruth.width, GroundTruth.height, GroundTruth.visibility).filter(
                    GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id,  GroundTruth.type_id.in_(type_ids))

            return pd.read_sql(resp.statement, self.engine)

    def get_frames_where_detector_misses(self, run_id: int, scenario_name: str) -> List[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        return [f.frame_id for f in self.session.query(GroundTruthDetectionMatchesFrame).filter(GroundTruthDetectionMatchesFrame.run_id == run_id, GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.target_index.is_(None))]

    def get_missed_detection_target_ids_by_frame(self, run_id: int, scenario_name: str, frame_id: int):
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        return [f.target_id for f in self.session.query(GroundTruthDetectionMatchesFrame).filter(GroundTruthDetectionMatchesFrame.run_id == run_id,

                                                                                                 GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.frame_id == frame_id, GroundTruthDetectionMatchesFrame.target_index.is_(None))]

    def get_number_of_idsw_by_id(self, run_id: int, scenario_name: str, target_id: int) -> int:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        query = self.session.query(func.count(TargetFrameEval.target_id)).distinct().filter(TargetFrameEval.run_id == run_id, TargetFrameEval.scenario_id == subquery,
                                                                                            TargetFrameEval.target_id == target_id).first()
        if query:
            return query[0]

    def get_target_nth_highest_IDSW_by_scenario(self, run_id: int, scenario_name: str, n: int = 1) -> Optional[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(TargetFrameEval.target_id).filter(TargetFrameEval.run_id == run_id, TargetFrameEval.scenario_id == subquery) \
            .group_by(TargetFrameEval.target_id).order_by(func.count(distinct(TargetFrameEval.tracker_id)).desc()).limit(n)

        target_id_list = [f.target_id for f in query]
        if len(target_id_list) >= n:
            return target_id_list[n-1]
        # def get_missed_tracking_target_ids_by_frame(self, run_id: int, scenario_name: str, frame_id: int):
        #     subquery = self.session.query(Scenarios.id).filter(
        #         func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        #     return [f.target_id for f in self.session.query(GroundTruthDetectionMatchesFrame).filter(GroundTruthDetectionMatchesFrame.run_id == run_id,
        #     GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.frame_id == frame_id, GroundTruthDetectionMatchesFrame.target_index.is_(None))]

    def get_current_tracker_id_for_gt(self, run_id: int, scenario_name: str, frame_id: int, target_id: int) -> Optional[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        query = self.session.query(TargetFrameEval.tracker_id).filter(TargetFrameEval.run_id == run_id, TargetFrameEval.scenario_id == subquery,
                                                                      TargetFrameEval.frame_id == frame_id,
                                                                      TargetFrameEval.target_id == target_id).first()
        if query:
            return query[0]

    def get_future_tracker_id_for_gt(self, run_id: int, scenario_name: str, frame_id: int, target_id: int, current_tracker_id: int, n: int = 20) -> List[int]:

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(distinct(TargetFrameEval.tracker_id)).filter(TargetFrameEval.run_id == run_id, TargetFrameEval.scenario_id == subquery,
                                                                                TargetFrameEval.frame_id > frame_id, TargetFrameEval.frame_id <= frame_id + n,
                                                                                TargetFrameEval.target_id == target_id, TargetFrameEval.tracker_id != current_tracker_id).all()
        return [f[0] for f in query]

    def get_first_frame_for_target_id(self, scenario_name: str, target_id: int) -> Optional[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(func.min(GroundTruth.frame_id)).filter(
            GroundTruth.scenario_id == subquery, GroundTruth.target_id == target_id).first()
        if query:
            return query[0]

    def get_last_frame_for_target_id(self, scenario_name: str, target_id: int) -> Optional[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(func.max(GroundTruth.frame_id)).filter(
            GroundTruth.scenario_id == subquery, GroundTruth.target_id == target_id).first()
        if query:
            return query[0]

    def get_all_distances_vector_by_scenario(self, run_id: int, scenario_name: str) -> pd.DataFrame:

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(TrackerDistances).filter(TrackerDistances.run_id == 3,
                                                           TrackerDistances.scenario_id == subquery)
        return pd.read_sql(resp.statement, self.engine)

    def get_detection_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int, confidance: Optional[float] = None) -> pd.DataFrame:

        if confidance is None:
            # confidance = self.session.query(Run.confidance).filter(
            #     Run.id == run_id).scalar_subquery()
            confidance = 0.6
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Detections).filter(
            Detections.scenario_id == subquery, Detections.frame_id == frame_id, Detections.run_id == run_id, Detections.confidance >= confidance)

        return pd.read_sql(resp.statement, self.engine)

    def get_distances_by_scenario(self, run_id, scenario_name: str):
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        resp = self.session.query(Trackers.frame_id, Trackers.tracker_id, Trackers.target_index,  Trackers.embedding_distance.label('embedding_distance'), Trackers.mahalanobis_distance.label('mahalanobis_distance')).filter(
            Trackers.scenario_id == subquery, Trackers.run_id == run_id)
        return pd.read_sql(resp.statement, self.engine)

    def get_kalman_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers.tracker_id, Trackers.target_index,  Trackers.kalman_min_x.label('min_x'), Trackers.kalman_min_y.label('min_y'), Trackers.kalman_width.label('width'), Trackers.kalman_height.label('height')).filter(
            Trackers.scenario_id == subquery, Trackers.frame_id == frame_id, Trackers.run_id == run_id)
        return pd.read_sql(resp.statement, self.engine)

    def get_tracker_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers.tracker_id, Trackers.target_index,  Trackers.min_x, Trackers.min_y, Trackers.width, Trackers.height).filter(
            Trackers.scenario_id == subquery, Trackers.frame_id == frame_id, Trackers.run_id == run_id, Trackers.target_index.isnot(None))
        return pd.read_sql(resp.statement, self.engine)

    def get_tracker_props_by_frame_and_id(self, run_id: int, scenario_name: str, frame_id: int, tracker_id: int) -> TrackersProps:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers).filter(
            Trackers.scenario_id == subquery, Trackers.frame_id == frame_id, Trackers.run_id == run_id, Trackers.tracker_id == tracker_id).first()
        if resp is not None:
            return TrackersProps.from_orm(resp)

    def get_gt_props_by_frame_and_id(self, scenario_name: str, frame_id: int, target_id: int) -> TrackersProps:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(GroundTruth).filter(
            GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id,  GroundTruth.target_id == target_id).first()
        if resp is not None:
            return GroundTruthProps.from_orm(resp)

    def get_kalman_with_no_detection_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers.tracker_id,  Trackers.kalman_min_x.label('min_x'), Trackers.kalman_min_y.label('min_y'), Trackers.kalman_width.label('width'), Trackers.kalman_height.label('height')).filter(
            Trackers.scenario_id == subquery, Trackers.frame_id == frame_id, Trackers.run_id == run_id, Trackers.target_index.is_(None), Trackers.kalman_min_x.isnot(None))
        return pd.read_sql(resp.statement, self.engine)

    def get_detection_eval_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(DetectionsFrameEval.confidance_level,
                                  DetectionsFrameEval.TP, DetectionsFrameEval.FP, DetectionsFrameEval.FN,
                                  DetectionsFrameEval.Recall, DetectionsFrameEval.Precision).filter(
            DetectionsFrameEval.scenario_id == subquery, DetectionsFrameEval.frame_id == frame_id, DetectionsFrameEval.run_id == run_id)
        return pd.read_sql(resp.statement, self.engine)

    def get_detection_eval_table_by_scenario(self, run_id: int, scenario_name: str) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(DetectionsFrameEval.confidance_level.label('confidance'),
                                  func.sum(DetectionsFrameEval.TP).label('TP'),
                                  func.sum(DetectionsFrameEval.FP).label('FP'),
                                  func.sum(DetectionsFrameEval.FN).label('FN'),
                                  ).filter(
            DetectionsFrameEval.scenario_id == subquery, DetectionsFrameEval.run_id == run_id,
            # DetectionsFrameEval.TP + DetectionsFrameEval.FN > 0, DetectionsFrameEval.TP +
            # DetectionsFrameEval.FP > 0
        ).group_by(DetectionsFrameEval.confidance_level).order_by(DetectionsFrameEval.confidance_level.asc()).subquery()
        s = select([resp.c.confidance, resp.c.TP, resp.c.FP, resp.c.FN, case(
            [(resp.c.FN + resp.c.TP > 0, cast(resp.c.TP, Float)/cast((resp.c.FN + resp.c.TP), Float))], else_=None).label('Recall'),
            case(
            [(resp.c.FP + resp.c.TP > 0, cast(resp.c.TP, Float)/cast(resp.c.FP + resp.c.TP, Float))], else_=None).label('Precision')
        ])
        # ,  resp.c.TP / (resp.c.FN + resp.c.TP), resp.c.TP/(resp.c.TP + resp.c.FP)

        return pd.read_sql(s, self.engine)

    def get_target_frame_eval_table_by_frame(self, run_id, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(TargetFrameEval).filter(
            TargetFrameEval.scenario_id == subquery, TargetFrameEval.frame_id == frame_id, TargetFrameEval.run_id == run_id)

        return pd.read_sql(resp.statement, self.engine)

    def get_tracker_eval_by_frame_table(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(TrackerEval).filter(
            TrackerEval.scenario_id == subquery, TrackerEval.frame_id == frame_id, TrackerEval.run_id == run_id)

        return pd.read_sql(resp.statement, self.engine)

    def get_tracker_eval_by_scenario_table(self, run_id: int, scenario_name: str) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(TrackerEval).filter(
            TrackerEval.scenario_id == subquery, TrackerEval.run_id == run_id)

        return pd.read_sql(resp.statement, self.engine)

    def get_tracker_eval_by_scenario(self, run_id: int, scenario_name: str):
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(
            func.sum(TrackerEval.TP).label('TP'),
            func.sum(TrackerEval.FP).label('FP'),
            func.sum(TrackerEval.FN).label('FN'),
            func.sum(TrackerEval.GT).label('GT'),
            func.sum(TrackerEval.IDSW).label('IDSW'),
            func.sum(TrackerEval.Frag).label('Frag')
        ).filter(
            TrackerEval.scenario_id == subquery, TrackerEval.run_id == run_id).group_by(TrackerEval.scenario_id).\
            order_by(TrackerEval.scenario_id.asc()).subquery()

        s = select([resp.c.TP, resp.c.FP, resp.c.FN, resp.c.GT, resp.c.IDSW, resp.c.Frag, case(
            [(resp.c.FN + resp.c.TP > 0, cast(resp.c.TP, Float)/cast((resp.c.FN + resp.c.TP), Float))], else_=None).label('Recall'),
            case(
            [(resp.c.FP + resp.c.TP > 0, cast(resp.c.TP, Float)/cast(resp.c.FP + resp.c.TP, Float))], else_=None).label('Precision'),
            (1 - cast(resp.c.FN + resp.c.FP + resp.c.IDSW, Float) /
             case([(resp.c.GT > 0, cast(resp.c.GT, Float))],  else_=None)).label('MOTA')
        ])
        return pd.read_sql(s, self.engine)

    def get_scenario_name_by_id(self, scenario_id: int) -> Optional[str]:

        resp = self.session.query(Scenarios.name).filter(
            Scenarios.id == scenario_id).first()
        if resp is not None:
            return resp[0]

    def get_detector_error_by_scenario(self, scenario_name: str) -> pd.DataFrame:

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        resp = self.session.query(GroundTruth.frame_id, GroundTruth.target_id,  Detections.target_index.label('detection_id'),
                                  Detections.confidance,  GroundTruth.min_x.label('gt_min_x'), Detections.min_x.label(
            'dt_min_x'), GroundTruth.min_y.label('gt_min_y'), Detections.min_y.label('dt_min_y'),
            GroundTruth.width.label('gt_width'), Detections.width.label(
            'dt_width'), GroundTruth.height.label('gt_height'), Detections.height.label('dt_height'),
            GroundTruthDetectionMatchesFrame.iou,
            (GroundTruth.min_x -
             Detections.min_x).label('diff_min_x'), (GroundTruth.min_y -
                                                     Detections.min_y).label('diff_min_y'),
            (GroundTruth.width -
             Detections.width).label('diff_width'), (GroundTruth.height -
                                                     Detections.height).label('diff_height')).filter(GroundTruth.scenario_id == subquery).order_by(GroundTruth.frame_id). \
            join(GroundTruthDetectionMatchesFrame, and_(GroundTruth.target_id == GroundTruthDetectionMatchesFrame.target_id,
                                                        GroundTruthDetectionMatchesFrame.scenario_id == GroundTruth.scenario_id, GroundTruthDetectionMatchesFrame.frame_id == GroundTruth.frame_id)).join(Detections,
                                                                                                                                                                                                          and_(GroundTruthDetectionMatchesFrame.target_index == Detections.target_index,
                                                                                                                                                                                                               Detections.scenario_id == GroundTruth.scenario_id, Detections.frame_id == GroundTruth.frame_id
                                                                                                                                                                                                               ))

        return pd.read_sql(resp.statement,  self.engine)

    def get_detector_difference_by_challenge(self, challenge: str) -> pd.DataFrame:

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.source) == challenge.lower()).subquery()
        resp = self.session.query(GroundTruth.scenario_id, GroundTruth.frame_id, GroundTruth.target_id,  Detections.target_index.label('detection_id'),
                                  Detections.confidance,  GroundTruth.min_x.label('gt_min_x'), Detections.min_x.label(
            'dt_min_x'), GroundTruth.min_y.label('gt_min_y'), Detections.min_y.label('dt_min_y'),
            GroundTruth.width.label('gt_width'), Detections.width.label(
            'dt_width'), GroundTruth.height.label('gt_height'), Detections.height.label('dt_height'),
            GroundTruthDetectionMatchesFrame.iou,
            (GroundTruth.min_x -
             Detections.min_x).label('diff_min_x'), (GroundTruth.min_y -
                                                     Detections.min_y).label('diff_min_y'),
            (GroundTruth.width -
             Detections.width).label('diff_width'), (GroundTruth.height -
                                                     Detections.height).label('diff_height')).filter(GroundTruth.scenario_id.in_(subquery)).order_by(GroundTruth.frame_id). \
            join(GroundTruthDetectionMatchesFrame, and_(GroundTruth.target_id == GroundTruthDetectionMatchesFrame.target_id,
                                                        GroundTruthDetectionMatchesFrame.scenario_id == GroundTruth.scenario_id,  GroundTruthDetectionMatchesFrame.frame_id == GroundTruth.frame_id)).join(Detections,
                                                                                                                                                                                                           and_(GroundTruthDetectionMatchesFrame.target_index == Detections.target_index,
                                                                                                                                                                                                                Detections.scenario_id == GroundTruth.scenario_id, Detections.frame_id == GroundTruth.frame_id
                                                                                                                                                                                                                ))

        return pd.read_sql(resp.statement,  self.engine)

    def get_detection_matches_by_scenario(self, run_id: int, scenario_name: str) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(GroundTruthDetectionMatchesFrame).filter(
            GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.run_id == run_id)
        return pd.read_sql(query.statement, self.engine)

    def get_detection_matches_by_scenario_and_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(GroundTruthDetectionMatchesFrame).filter(
            GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.run_id == run_id,
            GroundTruthDetectionMatchesFrame.frame_id == frame_id)
        return pd.read_sql(query.statement, self.engine)

    def get_previous_match_by_frame_and_id(self, run_id: int, scenario_name: str, frame_id: int, target_id: int) -> Optional[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(GroundTruthDetectionMatchesFrame.tracker_id).filter(
            GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.frame_id < frame_id, GroundTruthDetectionMatchesFrame.run_id == run_id, GroundTruthDetectionMatchesFrame.target_id == target_id, GroundTruthDetectionMatchesFrame.tracker_id != None).limit(1).first()

        if resp is not None:
            return resp[0]

    def check_if_switched_id(self, run_id: int, scenario_name: str, frame_id: int, tracker_id: int) -> bool:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        subquery_2 = self.session.query(GroundTruthDetectionMatchesFrame.target_id).filter(
            GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.frame_id == frame_id, GroundTruthDetectionMatchesFrame.run_id == run_id, GroundTruthDetectionMatchesFrame.tracker_id == tracker_id).scalar_subquery()

        resp = self.session.query(GroundTruthDetectionMatchesFrame.tracker_id).filter(
            GroundTruthDetectionMatchesFrame.scenario_id == subquery, GroundTruthDetectionMatchesFrame.frame_id < frame_id, GroundTruthDetectionMatchesFrame.run_id == run_id, GroundTruthDetectionMatchesFrame.target_id == subquery_2, GroundTruthDetectionMatchesFrame.tracker_id != None).order_by(GroundTruthDetectionMatchesFrame.frame_id.desc()).limit(1).first()
        if resp is not None:
            return resp[0] != tracker_id
        else:
            return False

    def check_if_tracker_still_exist(self, run_id: int, scenario_name: str, frame_id: int, tracker_id: int) -> bool:

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers).filter(Trackers.run_id == run_id, Trackers.scenario_id == subquery, Trackers.frame_id >= frame_id,
                                                   Trackers.tracker_id == tracker_id).first()
        return resp is not None

    def check_if_tracker_new(self, run_id: int, scenario_name: str, frame_id: int, tracker_id: int) -> bool:

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers).filter(Trackers.run_id == run_id, Trackers.scenario_id == subquery, Trackers.frame_id < frame_id,
                                                   Trackers.tracker_id == tracker_id).first()
        return resp is None

    def add_run(self, detector_name: str, comment: str = None) -> int:
        run = Run()
        run.time_stamp = datetime.now()
        run.detector_id = self.add_detector(detector_name)
        run.comment = comment
        self.session.add(run)
        resp = self.session.query(Run.id).filter(
            Run.time_stamp == run.time_stamp).first()[0]
        self.session.commit()
        return resp

    def upsert_bulk_ground_truth(self, ground_truth: List[dict]):

        # Prepare all the values that should be "upserted" to the DB

        stmt = insert(GroundTruth).values(ground_truth)
        stmt = stmt.on_conflict_do_update(
            index_elements=['target_id', 'frame_id', 'scenario_id'],

            # The columns that should be updated on conflict
            set_={
                "min_x": stmt.excluded.min_x,
                "min_y": stmt.excluded.min_y,
                "width": stmt.excluded.width,
                "height": stmt.excluded.height,
                "visibility": stmt.excluded.visibility,
                'is_valid': stmt.excluded.is_valid,
                'type_id': stmt.excluded.type_id
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_detections(self, detections_list: List[dict]):

        # Prepare all the values that should be "upserted" to the DB

        stmt = insert(Detections).values(detections_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['target_index',
                            'frame_id', 'scenario_id', 'run_id'],

            # The columns that should be updated on conflict
            set_={
                "min_x": stmt.excluded.min_x,
                "min_y": stmt.excluded.min_y,
                "width": stmt.excluded.width,
                "height": stmt.excluded.height,
                "confidance": stmt.excluded.confidance,
                "embedding": stmt.excluded.embedding,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_kalman(self, kalman_list: List[dict]):

        stmt = insert(Trackers).values(kalman_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['tracker_id',
                            'frame_id', 'scenario_id', 'run_id'],

            # The columns that should be updated on conflict
            set_={
                "kalman_min_x": stmt.excluded.kalman_min_x,
                "kalman_min_y": stmt.excluded.kalman_min_y,
                "kalman_width": stmt.excluded.kalman_width,
                "kalman_height": stmt.excluded.kalman_height,
                'track_status': stmt.excluded.track_status,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_tracker(self, kalman_list: List[dict]):

        stmt = insert(Trackers).values(kalman_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['tracker_id',
                            'frame_id', 'scenario_id', 'run_id'],

            # The columns that should be updated on conflict
            set_={
                "target_index": stmt.excluded.target_index,
                "min_x": stmt.excluded.min_x,
                "min_y": stmt.excluded.min_y,
                "width": stmt.excluded.width,
                "height": stmt.excluded.height,
                "embedding": stmt.excluded.embedding,
                "embedding_distance": stmt.excluded.embedding_distance,
                "mahalanobis_distance": stmt.excluded.mahalanobis_distance,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def get_tracker_ids_matched_to_target_ids(self, run_id: int, scenario_name: str, frame_id: int, target_ids: List[int]) -> Optional[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        query = self.session.query(TargetFrameEval).filter(
            TargetFrameEval.run_id == run_id, TargetFrameEval.scenario_id == subquery, TargetFrameEval.frame_id == frame_id,
            TargetFrameEval.target_id.in_(target_ids)
        )
        return [q.tracker_id for q in query]

    def upsert_bulk_detector_frame_eval_data(self,  detector_eval_list: List[dict]):
        stmt = insert(DetectionsFrameEval).values(detector_eval_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['scenario_id', 'run_id',
                            'frame_id', 'confidance_level'],

            # The columns that should be updated on conflict
            set_={
                "TP": stmt.excluded.TP,
                "FP": stmt.excluded.FP,
                "FN": stmt.excluded.FN,
                "Recall": stmt.excluded.Recall,
                "Precision": stmt.excluded.Precision,

            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_distances_data(self,  gt_eval_list: List[dict]):
        stmt = insert(TrackerDistances).values(
            gt_eval_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['scenario_id', 'run_id',
                            'frame_id', 'target_id', 'target_index'],

            # The columns that should be updated on conflict
            set_={
                "embedding_distance": stmt.excluded.embedding_distance,
                "mahalanobis_distance": stmt.excluded.mahalanobis_distance,

            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_gt_match_detector_frame_eval_data(self,  gt_eval_list: List[dict]):
        stmt = insert(GroundTruthDetectionMatchesFrame).values(
            gt_eval_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['scenario_id', 'run_id',
                            'frame_id', 'target_id'],

            # The columns that should be updated on conflict
            set_={
                "target_index": stmt.excluded.target_index,
                "iou": stmt.excluded.iou,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_gt_match_tracker_frame_eval_data(self,  gt_eval_list: List[dict]):
        stmt = insert(GroundTruthDetectionMatchesFrame).values(
            gt_eval_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['scenario_id', 'run_id',
                            'frame_id', 'target_id'],

            # The columns that should be updated on conflict
            set_={
                "tracker_id": stmt.excluded.tracker_id,
                "iou_tracker": stmt.excluded.iou_tracker,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_tracker_eval_data(self,  detector_eval_list: List[dict]):

        stmt = insert(TrackerEval).values(detector_eval_list)
        stmt = stmt.on_conflict_do_update(
            index_elements=['scenario_id', 'run_id',
                            'frame_id'],

            # The columns that should be updated on conflict
            set_={
                "TP": stmt.excluded.TP,
                "FP": stmt.excluded.FP,
                "FN": stmt.excluded.FN,
                "GT": stmt.excluded.GT,
                "IDSW": stmt.excluded.IDSW,
                "Frag": stmt.excluded.Frag
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def upsert_bulk_target_frame_eval_props(self, target_frame_eval: List[dict]):
        stmt = insert(TargetFrameEval).values(target_frame_eval)
        stmt = stmt.on_conflict_do_update(
            index_elements=['scenario_id', 'run_id',
                            'frame_id', 'target_id'],

            # The columns that should be updated on conflict
            set_={
                "tracker_id": stmt.excluded.tracker_id,
                "iou": stmt.excluded.iou,
                "embedding_distance": stmt.excluded.embedding_distance,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

    def remove_run(self, run_id: int):
        self.session.query(Run).filter(Run.id == run_id).delete()

    def export_detection_csv_by_challenge_and_run_id(self,  challenge: str, save_path: str, run_id: int = 1):
        scenarios = self.get_scenarios_from_challenge(challenge)
        if scenarios is not None:
            data_dir = os.path.join(save_path, challenge)
            Path(data_dir).mkdir(
                parents=True, exist_ok=True)
            for scenario in scenarios:
                scenario_id = self.get_scenario_props_by_name(scenario[0])
                if scenario_id is not None:
                    scenario_id = scenario_id.id

                query = select(Detections.frame_id, Detections.target_index,
                               Detections.min_x, Detections.min_y, Detections.width, Detections.height,
                               Detections.confidance, -1, -1, -1).where(
                    Detections.run_id == run_id, Detections.scenario_id == scenario_id)

                table = pd.read_sql(query, self.engine, index_col=None)
                path_to_txt = os.path.join(data_dir, scenario[0] + '.txt')
                table.to_csv(path_to_txt, header=False,
                             index=False)

    def get_kalman_frames_by_scenario(self, run_id: int, scenario_name: str) -> Set[int]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        return {f.frame_id for f in self.session.query(
            Trackers).filter(Trackers.run_id == run_id, Trackers.scenario_id == subquery, and_(Trackers.min_x.isnot(None))).distinct(Trackers.frame_id)}

    def get_kalman_predictions_where_detection_condition(self, run_id: int, scenario_name: str, has_detection: bool = False) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        if has_detection:
            query = self.session.query(Trackers).filter(
                Trackers.run_id == run_id, Trackers.target_index.isnot(None))
        else:
            query = self.session.query(Trackers).filter(
                Trackers.run_id == run_id, Trackers.target_index.is_(None), Trackers.scenario_id == subquery)
        return pd.read_sql(query.statement, self.engine)

    def get_distances_for_matched(self, run_id: int, scenario_name: str):
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        query = self.session.query(TrackerDistances.frame_id, GroundTruthDetectionMatchesFrame.target_id.label('gt_id'), TrackerDistances.target_id,
                                   TrackerDistances.mahalanobis_distance, TrackerDistances.embedding_distance).filter(TrackerDistances.run_id == run_id, TrackerDistances.scenario_id == subquery,).join(GroundTruthDetectionMatchesFrame,
                                                                                                                                                                                                         and_(GroundTruthDetectionMatchesFrame.run_id == TrackerDistances.run_id,
                                                                                                                                                                                                              GroundTruthDetectionMatchesFrame.scenario_id == TrackerDistances.scenario_id,
                                                                                                                                                                                                              GroundTruthDetectionMatchesFrame.frame_id == TrackerDistances.frame_id,
                                                                                                                                                                                                              GroundTruthDetectionMatchesFrame.tracker_id == TrackerDistances.target_id,
                                                                                                                                                                                                              GroundTruthDetectionMatchesFrame.target_index == TrackerDistances.target_index,
                                                                                                                                                                                                              ))

        return pd.read_sql(query.statement, self.engine)

    def get_distances_for_matched_by_challenge(self, run_id: int, challenge: str):

        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.source) == challenge.lower()).subquery()
        query = self.session.query(TrackerDistances.frame_id, GroundTruthDetectionMatchesFrame.target_id.label('gt_id'), TrackerDistances.target_id,
                                   TrackerDistances.mahalanobis_distance, TrackerDistances.embedding_distance).filter(TrackerDistances.run_id == run_id, TrackerDistances.scenario_id.in_(subquery)).join(GroundTruthDetectionMatchesFrame,
                                                                                                                                                                                                          and_(GroundTruthDetectionMatchesFrame.run_id == TrackerDistances.run_id,
                                                                                                                                                                                                               GroundTruthDetectionMatchesFrame.scenario_id == TrackerDistances.scenario_id,
                                                                                                                                                                                                               GroundTruthDetectionMatchesFrame.frame_id == TrackerDistances.frame_id,
                                                                                                                                                                                                               GroundTruthDetectionMatchesFrame.tracker_id == TrackerDistances.target_id,
                                                                                                                                                                                                               GroundTruthDetectionMatchesFrame.target_index == TrackerDistances.target_index,
                                                                                                                                                                                                               ))

        return pd.read_sql(query.statement, self.engine)

    def get_distances_for_unmatched(self, run_id: int, scenario_name: str):
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        query = self.session.query(TrackerDistances.frame_id, TrackerDistances.target_id, Trackers.target_index,
                                   TrackerDistances.mahalanobis_distance, TrackerDistances.embedding_distance).join(Trackers,
                                                                                                                    and_(Trackers.run_id == TrackerDistances.run_id,
                                                                                                                         Trackers.scenario_id == TrackerDistances.scenario_id,
                                                                                                                         Trackers.frame_id == TrackerDistances.frame_id,
                                                                                                                         Trackers.tracker_id == TrackerDistances.target_id,
                                                                                                                         )).filter(
            TrackerDistances.run_id == run_id, TrackerDistances.scenario_id == subquery, Trackers.target_index != TrackerDistances.target_index
        )
        return pd.read_sql(query.statement, self.engine)

    def get_all_distances_by_scenario(self, run_id: int, scenario_name: str):
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        query = self.session.query(TrackerDistances.frame_id, TrackerDistances.target_id,
                                   TrackerDistances.mahalanobis_distance, TrackerDistances.embedding_distance).filter(Trackers.run_id == run_id, TrackerDistances.scenario_id == subquery)


if __name__ == '__main__':

    dc = DatabaseCreator()
    dc.update_ground_truth()
