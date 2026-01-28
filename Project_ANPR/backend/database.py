from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional
from datetime import datetime
from fastapi import Depends
from typing import Annotated

DATABASE_URL = "sqlite:///metadata.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

class Detections(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    filepath: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    inference_time_ms: float
    num_detections: int
    classes_detected: str
    confidence_avg: float

def create_db():
    SQLModel.metadata.create_all(engine)

# Dependency Injection Session
def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]
