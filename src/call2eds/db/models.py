from datetime import datetime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, ForeignKey, BigInteger, Index, Boolean

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


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="admin", nullable=False)
    can_api = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    events = relationship("AuthEvent", back_populates="user", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=True)
    key_hash = Column(String, nullable=False)
    prefix = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    user = relationship("User", back_populates="api_keys")
    __table_args__ = (Index("idx_api_keys_user_id", "user_id"),)


class AuthEvent(Base):
    __tablename__ = "auth_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    event = Column(String, nullable=False)
    ip = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user = relationship("User", back_populates="events")
    __table_args__ = (Index("idx_auth_events_user_id", "user_id"), Index("idx_auth_events_event", "event"))
