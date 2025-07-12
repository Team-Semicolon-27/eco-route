from database import Base, engine
from dbmodels import Graph, GraphCache, ElevationCache, TomTomCache

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
