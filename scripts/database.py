from datetime import datetime, timezone
from typing import Any, Optional, cast, Union
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# Define tipos para as colunas
FloatColumn = Union[Column[float], float]
DateTimeColumn = Union[Column[datetime], datetime]

# Cria engine SQLite
engine = create_engine('sqlite:///training.db')
Base = declarative_base()

class ModelSession(Base):
    """Modelo para sessões de treinamento de cada modelo"""
    __tablename__ = 'model_sessions'
    
    id = Column(Integer, primary_key=True)
    training_session_id = Column(Integer, ForeignKey('training_sessions.id'))
    model_name = Column(String, nullable=False)
    start_time: DateTimeColumn = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time: DateTimeColumn = Column(DateTime)
    status = Column(String, default='running')
    config = Column(JSON)
    
    # Relacionamentos
    training_session = relationship("TrainingSession", back_populates="model_sessions")
    metrics = relationship("TrainingMetrics", back_populates="model_session")
    
    def to_dict(self) -> dict[str, Any]:
        start = cast(Optional[datetime], getattr(self, 'start_time', None))
        end = cast(Optional[datetime], getattr(self, 'end_time', None))
        return {
            'id': self.id,
            'training_session_id': self.training_session_id,
            'model_name': self.model_name,
            'start_time': start.isoformat() if start else None,
            'end_time': end.isoformat() if end else None,
            'status': self.status,
            'config': self.config
        }

class TrainingSession(Base):
    """Modelo para sessões de treinamento"""
    __tablename__ = 'training_sessions'
    
    id = Column(Integer, primary_key=True)
    start_time: DateTimeColumn = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    end_time: DateTimeColumn = Column(DateTime)
    status = Column(String, default='running')
    config = Column(JSON)
    
    # Relacionamentos
    model_sessions = relationship("ModelSession", back_populates="training_session")
    
    def to_dict(self) -> dict[str, Any]:
        start = cast(Optional[datetime], getattr(self, 'start_time', None))
        end = cast(Optional[datetime], getattr(self, 'end_time', None))
        return {
            'id': self.id,
            'start_time': start.isoformat() if start else None,
            'end_time': end.isoformat() if end else None,
            'status': self.status,
            'config': self.config,
            'model_sessions': [ms.to_dict() for ms in self.model_sessions]
        }

class TrainingMetrics(Base):
    """Modelo para métricas de treinamento"""
    __tablename__ = 'training_metrics'
    
    id = Column(Integer, primary_key=True)
    model_session_id = Column(Integer, ForeignKey('model_sessions.id'))
    step = Column(Integer, nullable=False)
    loss: FloatColumn = Column(Float, nullable=True)
    learning_rate: FloatColumn = Column(Float, nullable=True)
    epoch = Column(Integer)
    timestamp: DateTimeColumn = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relacionamentos
    model_session = relationship("ModelSession", back_populates="metrics")
    
    def to_dict(self) -> dict[str, Any]:
        loss_val = cast(Optional[float], getattr(self, 'loss', None))
        lr_val = cast(Optional[float], getattr(self, 'learning_rate', None))
        ts = cast(Optional[datetime], getattr(self, 'timestamp', None))
        return {
            'id': self.id,
            'model_session_id': self.model_session_id,
            'step': self.step,
            'loss': loss_val,
            'learning_rate': lr_val,
            'epoch': self.epoch,
            'timestamp': ts.isoformat() if ts else None
        }

# Cria tabelas
Base.metadata.create_all(engine)

# Sessão global
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session:
    """Retorna uma sessão do banco de dados"""
    return SessionLocal() 