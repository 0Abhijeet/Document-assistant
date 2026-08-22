import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from src.database import Base

# all-MiniLM-L6-v2 output dimension. If you change embedding models, this must change too,
# and existing rows become incompatible (dimension mismatch) — you'd need a migration + re-ingest.
EMBEDDING_DIM = 384


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default="processing", nullable=False)  # processing | ready | failed

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
