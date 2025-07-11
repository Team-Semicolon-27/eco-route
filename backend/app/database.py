from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from databases import Database
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Async database connection (for FastAPI)
database = Database(DATABASE_URL)

# SQLAlchemy setup for metadata and sync operations
engine = create_engine(DATABASE_URL.replace("asyncpg", "psycopg2"))
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

def create_db_and_tables():
    """Creates all database tables defined in Base."""
    Base.metadata.create_all(bind=engine)