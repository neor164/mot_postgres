from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from .tables_base import Base


class Run(Base):
    __tablename__ = "run"
    id = Column(Integer, primary_key=True)
    detector_id = Column(Integer, ForeignKey('detectors.id'))
    time_stamp = Column(DateTime)
    comment = Column(String(2000))
