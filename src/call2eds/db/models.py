from datetime import datetime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, ForeignKey, BigInteger, Index

Base = declarative_base()


class Call(Base):
    __tablename__ = "calls"
    call_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta_json = Column(JSON, nullable=True)
    runs = relationship("Run", back_populates="call")


class Run(Base):
    __tablename__ = "runs"
    run_id = Column(String, primary_key=True)
    call_id = Column(String, ForeignKey("calls.call_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    pipeline_version = Column(String, nullable=False)
    params_json = Column(JSON, nullable=True)
    status = Column(String, default="created", nullable=False)
    call = relationship("Call", back_populates="runs")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="run", cascade="all, delete-orphan")
    __table_args__ = (Index("idx_runs_call_id", "call_id"),)


class Artifact(Base):
    __tablename__ = "artifacts"
    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    kind = Column(String, nullable=False)
    s3_uri = Column(String, nullable=False)
    sha256 = Column(String, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    run = relationship("Run", back_populates="artifacts")
    __table_args__ = (Index("idx_artifacts_run_id", "run_id"),)


class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    key = Column(String, nullable=False)
    value_num = Column(Float, nullable=True)
    value_json = Column(JSON, nullable=True)
    run = relationship("Run", back_populates="metrics")
    __table_args__ = (Index("idx_metrics_run_id", "run_id"),)
