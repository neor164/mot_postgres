from .tables.tracker_tables import Trackers
from typing import Optional, List, Dict
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql.selectable import subquery
from .tables.ground_truth_tables import GroundTruth, GroundTruthProps, Scenarios, ScenatioProps
from .tables.detector_tables import Detections, DetectionsFrameEval, Detectors, DetectionsProps
from .tables.run_tables import Run
from pydantic import BaseModel
from sqlalchemy.sql import select
from sqlalchemy import cast, Float
from sqlalchemy.engine import Engine
from sqlalchemy.sql.functions import func
from sqlalchemy.sql.expression import case
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

    def update_ground_truth(self):
        pattern = 'gt_data/*/train/*/gt/gt.txt'
        # sequence_pattern = 'gt_data/(.*?)Labels/train/'
        scenario_pattern = 'train/(.*?)/gt'
        dc = DatabaseCreator()
        files = glob(pattern)
        for idx, file in enumerate(files):
            detections_array = np.loadtxt(file, delimiter=',', ndmin=2)
            # sequnce_name = re.findall(sequence_pattern, file)[0]
            scenario_name = re.findall(scenario_pattern, file)[0]
            scp = dc.get_scenario_props_by_name(scenario_name)
            for row in detections_array:
                frame_id = row[0]
                target_id = row[1]
                tp = GroundTruthProps(frame_id=frame_id, target_id=target_id,
                                      scenario_id=scp.id, is_hidden=row[6] == 0)
                tp.min_x = row[2]
                tp.min_y = row[3]
                tp.width = row[4]
                tp.height = row[5]
                values = tp.dict(exclude_none=True)
                insert_stmt = insert(GroundTruth).values(values)
                do_update_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=['target_id', 'frame_id', 'scenario_id'], set_=values)
                self.session.execute(do_update_stmt)
                self.session.commit()

    def check_has_evaluated(self, run_id: int, challenge: str) -> bool:
        subquery = self.session.query(Scenarios.id).filter(
            Scenarios.source == challenge).scalar_subquery()
        resp = self.session.query(Detections.scenario_id).filter(
            Detections.scenario_id.in_(subquery))
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

    def get_ground_truth_by_scenario_name(self, scenario_name: str) -> Optional[List[GroundTruth]]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        resp = self.session.query(GroundTruth).filter(
            GroundTruth.scenario_id == subquery).all()
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

    def get_ground_truth_by_frame_table(self, scenario_name: str, frame_id: int, visibilty_thresh: Optional[float] = None) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()
        if visibilty_thresh is not None:
            resp = self.session.query(GroundTruth.target_id, GroundTruth.min_x, GroundTruth.min_y, GroundTruth.width, GroundTruth.height, GroundTruth.visibility).filter(
                GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id, GroundTruth.visibility > visibilty_thresh)
        else:
            resp = self.session.query(GroundTruth.target_id, GroundTruth.min_x, GroundTruth.min_y, GroundTruth.width, GroundTruth.height, GroundTruth.visibility).filter(
                GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id)

        return pd.read_sql(resp.statement, self.engine)

    def get_detection_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int, confidance: Optional[float] = None) -> pd.DataFrame:

        if confidance is None:
            confidance = self.session.query(Run.confidance).filter(
                Run.id == run_id).scalar_subquery()
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Detections).filter(
            Detections.scenario_id == subquery, Detections.frame_id == frame_id, Detections.run_id == run_id, Detections.confidance >= confidance)

        return pd.read_sql(resp.statement, self.engine)

    def get_kalman_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers.tracker_id, Trackers.target_index,  Trackers.kalman_min_x, Trackers.kalman_min_y, Trackers.kalman_width, Trackers.kalman_height).filter(
            Trackers.scenario_id == subquery, Trackers.frame_id == frame_id, Trackers.run_id == run_id)
        return pd.read_sql(resp.statement, self.engine)

    def get_tracker_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Trackers.tracker_id, Trackers.target_index,  Trackers.min_x, Trackers.min_y, Trackers.width, Trackers.height).filter(
            Trackers.scenario_id == subquery, Trackers.frame_id == frame_id, Trackers.run_id == run_id)
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

    def get_scenario_name_by_id(self, scenario_id: int) -> Optional[str]:

        resp = self.session.query(Scenarios.name).filter(
            Scenarios.id == scenario_id).first()
        if resp is not None:
            return resp[0]

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
                "visibility": stmt.excluded.visibility
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
                "confidance": stmt.excluded.confidance
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
                "target_index": stmt.target_index,
                "min_x": stmt.excluded.min_x,
                "min_y": stmt.excluded.min_y,
                "width": stmt.excluded.width,
                "height": stmt.excluded.height,
            }
        )
        self.session.execute(stmt)
        self.session.commit()

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


if __name__ == '__main__':

    dc = DatabaseCreator()
    dc.update_ground_truth()
