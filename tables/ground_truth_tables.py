from sqlalchemy import Column, Integer, Float, String, ForeignKey, Boolean
from sqlalchemy import PrimaryKeyConstraint
from .tables_base import Base
from pydantic import BaseModel
from typing import Optional


class GroundTruth(Base):
    __tablename__ = "ground_truth"
    __table_args__ = (PrimaryKeyConstraint(
        'scenario_id', 'frame_id', 'target_id'),)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'))
    target_id = Column(Integer)
    frame_id = Column(Integer)
    min_x = Column(Float)
    min_y = Column(Float)
    width = Column(Float)
    height = Column(Float)
    is_hidden = Column(Boolean)


class GroundTruthProps(BaseModel):
    scenario_id: int
    target_id: int
    frame_id: int
    min_x: Optional[float]
    min_y: Optional[float]
    width: Optional[float]
    height: Optional[float]
    is_hidden: Optional[bool]

    class Config:
        orm_mode = True


class Scenarios(Base):
    __tablename__ = "scenarios"
    id = Column(Integer, primary_key=True)
    name = Column(String(60))
    source = Column(String(60))
    video_width = Column(Integer)
    video_height = Column(Integer)
    is_static = Column(Boolean)
    is_test = Column(Boolean)


class ScenatioProps(BaseModel):
    id: int
    name: Optional[str]
    source: Optional[str]
    is_test: Optional[bool]
    video_width: Optional[int]
    video_height: Optional[int]
    is_static: Optional[bool]

    class Config:
        orm_mode = True
