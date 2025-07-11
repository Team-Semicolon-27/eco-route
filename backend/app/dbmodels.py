from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, Float
from sqlalchemy.sql import func
from app.database import Base

class Graph(Base):
    __tablename__ = "graphs"
    graph_data = Column(String, nullable=False)
    id = Column(Integer, primary_key=True, index=True)


class GraphCache(Base):
    """Table to cache serialized NetworkX graph chunks."""
    __tablename__ = "graph_cache"
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String, unique=True, index=True, nullable=False)
    graph_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ElevationCache(Base):
    """Table to cache elevation data for specific nodes."""
    __tablename__ = "elevation_cache"
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, unique=True, index=True, nullable=False)
    elevation = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TomTomCache(Base):
    """Table to cache TomTom API responses."""
    __tablename__ = "tomtom_cache"
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String, unique=True, index=True, nullable=False)
    # Storing as string is simple; handles numbers and 'None' from the API.
    speed_data = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())