from pydantic.errors import FloatError
from sqlalchemy import Column, Integer, Float, String, ForeignKey, Boolean, FLOAT
from sqlalchemy import PrimaryKeyConstraint
from .tables_base import Base
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.dialects.postgresql import ARRAY


class Trackers(Base):
    __tablename__ = "trackers"
    __table_args__ = (PrimaryKeyConstraint('run_id',  'scenario_id',
                                           'frame_id', 'tracker_id'),)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    run_id = Column(Integer,  ForeignKey('run.id'))
    frame_id = Column(Integer)
    tracker_id = Column(Integer)
    target_index = Column(Integer)
    min_x = Column(Float)
    min_y = Column(Float)
    width = Column(Float)
    height = Column(Float)
    kalman_min_x = Column(Float)
    kalman_min_y = Column(Float)
    kalman_width = Column(Float)
    kalman_height = Column(Float)
    track_status = Column(Integer)
    emmbeding = ARRAY(FLOAT)


class TrackersProps(BaseModel):
    scenario_id: int
    run_id: int
    frame_id: int
    tracker_id: int
    target_index: Optional[int]
    min_x: Optional[float]
    min_y: Optional[float]
    width: Optional[float]
    height: Optional[float]
    kalman_min_x: Optional[float]
    kalman_min_y: Optional[float]
    kalman_width: Optional[float]
    kalman_height: Optional[float]
    track_status: Optional[int]
    emmbeding: Optional[List[float]]

    class Config:
        orm_mode = True


class TargetFrameEval(Base):
    __tablename__ = "target_frame_eval"
    __table_args__ = (PrimaryKeyConstraint('run_id',  'scenario_id',
                                           'frame_id', 'target_id'),)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    run_id = Column(Integer,  ForeignKey('run.id'))
    frame_id = Column(Integer)
    target_id = Column(Integer)
    tracker_id = Column(Integer)
    iou = Column(Float)


class TargetFrameEvalProps(BaseModel):
    scenario_id: int
    run_id: int
    frame_id: int
    target_id: Optional[int]
    tracker_id: Optional[int]
    iou: Optional[float]

    class Config:
        orm_mode = True


class TrackerEval(Base):
    __tablename__ = "trackers_eval"
    __table_args__ = (PrimaryKeyConstraint('run_id',  'scenario_id',
                                           'frame_id'),)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    run_id = Column(Integer,  ForeignKey('run.id'))
    frame_id = Column(Integer)
    TP = Column(Integer)
    FP = Column(Integer)
    FN = Column(Integer)
    GT = Column(Integer)
    IDSW = Column(Integer)
    Frag = Column(Integer)


class TrackerEvalProps(BaseModel):
    scenario_id: int
    run_id: int
    frame_id: int
    TP: Optional[int] = 0
    FP: Optional[int] = 0
    FN: Optional[int] = 0
    GT: Optional[int] = 0
    IDSW: Optional[int] = 0
    Frag: Optional[int] = 0

    class Config:
        orm_mode = True


class TrackerScenarioEval(Base):
    __tablename__ = "trackers_scenario_eval"
    __table_args__ = (PrimaryKeyConstraint('run_id',  'scenario_id'),)

    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    run_id = Column(Integer,  ForeignKey('run.id'))
    MOTA = Column(Float)
    MOTP = Column(Float)
    Frag = Column(Integer)
    TP = Column(Integer)
    TP = Column(Integer)
    FN = Column(Integer)
    Recall = Column(Float)
    Precision = Column(Float)
    IDSW = Column(Integer)
    MT = Column(Float)
    PT = Column(Float)
    ML = Column(Float)


class TrackerScenarioEvalProps(BaseModel):
    scenario_id: int
    run_id: int
    MOTA: Optional[float]
    MOTP: Optional[float]
    Frag: Optional[int]
    IDSW: Optional[int]
    MT: Optional[float]
    PT: Optional[float]
    ML: Optional[float]
    TP: Optional[int]
    TP: Optional[int]
    FN: Optional[int]
    Recall: Optional[float]
    Precision: Optional[float]
