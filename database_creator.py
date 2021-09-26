from mot_postgres.tables.tracker_tables import Trackers
from typing import Optional, List, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql.selectable import subquery
from .tables.ground_truth_tables import GroundTruth, GroundTruthProps, Scenarios, ScenatioProps
from .tables.detector_tables import Detections, Detectors, DetectionsProps
from .tables.run_tables import Run
from pydantic import BaseModel
from sqlalchemy.sql import select
from sqlalchemy.engine import Engine
from sqlalchemy.sql.functions import func
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

    def get_scenario_props_by_name(self, scenario_name: str) -> Optional[ScenatioProps]:

        resp = self.session.query(Scenarios).filter(
            Scenarios.name == scenario_name).first()
        if resp is not None:
            return ScenatioProps.from_orm(resp)

    def get_scenario_props_by_id(self, scenario_id: int) -> Optional[ScenatioProps]:
        resp = self.session.query(Scenarios).filter(
            Scenarios.id == scenario_id).first()
        if resp is not None:
            return ScenatioProps.from_orm(resp)

    def update_ground_truth(self):
        pattern = 'gt_data/*/train/*/gt/gt.txt'
        sequence_pattern = 'gt_data/(.*?)Labels/train/'
        scenario_pattern = 'train/(.*?)/gt'
        dc = DatabaseCreator()
        files = glob(pattern)
        for idx, file in enumerate(files):
            detections_array = np.loadtxt(file, delimiter=',', ndmin=2)
            sequnce_name = re.findall(sequence_pattern, file)[0]
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

    def get_ground_truth_by_frame_table(self, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(GroundTruth.min_x, GroundTruth.min_y, GroundTruth.width, GroundTruth.height, GroundTruth.visibility).filter(
            GroundTruth.scenario_id == subquery, GroundTruth.frame_id == frame_id)

        return pd.read_sql(resp.statement, self.engine)

    def get_detection_table_by_frame(self, run_id: int, scenario_name: str, frame_id: int) -> pd.DataFrame:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == scenario_name.lower()).scalar_subquery()

        resp = self.session.query(Detections).filter(
            Detections.scenario_id == subquery, Detections.frame_id == frame_id, Detections.run_id == run_id)
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

    def get_scenario_name_by_id(self, scenario_id: int) -> Optional[str]:

        resp = self.session.query(Scenarios.name).filter(
            Scenarios.id == scenario_id).first()
        if resp is not None:
            return resp[0]

    def get_scenarios_from_challenge(self, challenge: str) -> Optional[List[str]]:

        return self.session.query(Scenarios.name).filter(
            Scenarios.source == challenge).all()

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
            # Let's use the constraint name which was visible in the original posts error msg
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
            # Let's use the constraint name which was visible in the original posts error msg
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
            # Let's use the constraint name which was visible in the original posts error msg
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
            # Let's use the constraint name which was visible in the original posts error msg
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
