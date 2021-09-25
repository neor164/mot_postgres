from sqlalchemy import Column, Integer, Float, String, ForeignKey, Boolean
from sqlalchemy import PrimaryKeyConstraint
from .tables_base import Base
from pydantic import BaseModel
from typing import Optional


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

    class Config:
        orm_mode = True
