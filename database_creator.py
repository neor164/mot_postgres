from typing import Optional, List
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from .tables.ground_truth_tables import GroundTruth, GroundTruthProps, Scenarios, ScenatioProps
from .tables.detector_tables import Detections, Detectors, DetectionsProps
from .tables.run_tables import RunProps
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.sql.functions import func
from .tables.tables_base import Base
from glob import glob
import re
import numpy as np


class DatabaseProps(BaseModel):
    username: str = 'postgres'
    password: str = 'neor123'
    name: str = 'postgres'
    port: int = 6666
    ip_address = '127.0.0.1'


class DatabaseCreator:
    def __init__(self, db_props: DatabaseProps = DatabaseProps()) -> None:
        self.engine: Engine = create_engine(
            f"postgresql://{db_props.username}:{db_props.password}@{db_props.ip_address}:{db_props.port}/{db_props.name}")
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
            func.lower(Detectors) == detector_name.lower).first()
        if resp is not None:
            return detector_name
        else:
            detector = Detectors(name=detector_name)
            self.session.add(detector)
            self.session.commit()

    def upsert_detection_data(self, det: DetectionsProps):
        values = det.dict(exclude_none=True)
        insert_stmt = insert(Detections).values(values)
        do_update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['target_index', 'frame_id', 'scenario_id', 'run_id'], set_=values)
        self.session.execute(do_update_stmt)
        self.session.commit()

    def get_detections_by_scenario_and_frame(self, name: str, frame_id: int) -> Optional[List[DetectionsProps]]:
        subquery = self.session.query(Scenarios.id).filter(
            func.lower(Scenarios.name) == name.lower()).scalar_subquery()

        resp = self.session.query(Detections).filter(
            Detections.scenario_id == subquery, Detections.frame_id == frame_id).all()
        if resp is not None:
            det_list = []
            for gt in resp:
                det_list.append(DetectionsProps.from_orm(gt))
            return det_list


if __name__ == '__main__':

    dc = DatabaseCreator()
    dc.update_ground_truth()
