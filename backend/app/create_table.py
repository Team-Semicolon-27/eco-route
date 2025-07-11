from database import Base, engine
from dbmodels import Graph, Elevation, Traffic

Base.metadata.create_all(bind=engine)
