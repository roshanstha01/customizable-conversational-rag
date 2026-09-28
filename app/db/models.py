from sqlalchemy import Column, Date, DateTime, Integer, String, Text, Time
from sqlalchemy.sql import func

from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=True)
    file_type = Column(String(20), nullable=False)
    chunking_strategy = Column(String(50), nullable=False)
    total_chunks = Column(Integer, nullable=False, default=0)
    raw_text = Column(Text, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())


class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())