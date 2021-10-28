from typing import Optional, List
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy import PrimaryKeyConstraint
from pydantic import BaseModel
from sqlalchemy.sql.sqltypes import FLOAT
from .tables_base import Base
from sqlalchemy.dialects.postgresql import ARRAY


class Detections(Base):
    __tablename__ = "detections"
    __table_args__ = (PrimaryKeyConstraint('scenario_id', 'run_id',
                                           'frame_id', 'target_index'),)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    run_id = Column(Integer,  ForeignKey('run.id'))
    frame_id = Column(Integer)
    target_index = Column(Integer)
    min_x = Column(Float)
    min_y = Column(Float)
    width = Column(Float)
    height = Column(Float)
    confidance = Column(Float(12, 2))
    embedding = Column(ARRAY(FLOAT))


class DetectionsProps(BaseModel):
    scenario_id: int
    run_id: int
    frame_id: int
    target_index: int
    min_x: Optional[float]
    min_y: Optional[float]
    width: Optional[float]
    height: Optional[float]
    confidance: Optional[float]
    embedding: Optional[List[float]]

    class Config:
        orm_mode = True


class Detectors(Base):
    __tablename__ = "detectors"
    id = Column(Integer, primary_key=True)
    name = Column(String(60))


class DetectionsFrameEval(Base):
    __tablename__ = "detections_frame_eval"
    __table_args__ = (PrimaryKeyConstraint('scenario_id', 'run_id',
                                           'frame_id', 'confidance_level'),)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    run_id = Column(Integer,  ForeignKey('run.id'))
    frame_id = Column(Integer)
    confidance_level = Column(Float(12, 2))
    TP = Column(Integer)
    FP = Column(Integer)
    FN = Column(Integer)
    Recall = Column(Float)
    Precision = Column(Float)
