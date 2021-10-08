from sqlalchemy import Column, Integer, Float, String, ForeignKey, Boolean
from sqlalchemy import PrimaryKeyConstraint
from .tables_base import Base
from pydantic import BaseModel
from typing import Counter, Optional
from enum import Enum


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
    type_id = Column(Integer)
    visibility = Column(Float(12, 2))
    is_valid = Column(Boolean)


class GroundTruthProps(BaseModel):
    scenario_id: int
    target_id: int
    frame_id: int
    min_x: Optional[float]
    min_y: Optional[float]
    width: Optional[float]
    height: Optional[float]
    visibility: Optional[float]
    is_valid: Optional[bool]
    type_id: Optional[int]

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


class TargetTypes(object):
    Pedestrian = 1
    Person_on_vehicle = 2
    Car = 3
    Bicycle = 4
    Motorbike = 5
    Non_motorized_vehicle = 6
    Static_person = 7
    Distractor = 8
    Occluder = 9
    Occluder_on_the_ground = 10
    Occluder_full = 11
    Reflection = 12
    Crowd = 13
