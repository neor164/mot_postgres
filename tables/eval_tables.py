from sqlalchemy import Column, Integer, Float, String, ForeignKey, Boolean
from sqlalchemy import PrimaryKeyConstraint
from .tables_base import Base
from pydantic import BaseModel
from typing import Counter, Optional


class GroundTruthDetectionMatchesFrame(Base):
    __tablename__ = "ground_truth_detection_matches_frame"
    __table_args__ = (PrimaryKeyConstraint('run_id',
                                           'scenario_id', 'frame_id', 'target_id'),)
    run_id = Column(Integer, ForeignKey('run.id'))
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    frame_id = Column(Integer)
    target_id = Column(Integer)
    target_index = Column(Integer)
    iou = Column(Float(10, 2))


class GroundTruthDetectionMatchesFrameProps(BaseModel):
    run_id: int
    scenario_id: int
    frame_id: int
    target_id: int
    target_index: Optional[float]
    iou: Optional[float]
