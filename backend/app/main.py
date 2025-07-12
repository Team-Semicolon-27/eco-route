import logging
import time
import os
import io # Import the io module

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from app.core import calculate_route_streamed
from app.utils import check_api_keys, load_co2_data, co2_map
from app.models import RouteRequest, RouteResponse

from app.database import SessionLocal, create_db_and_tables
from app.models import RouteRequest, RouteResponse
# Load environment variables from .env file
load_dotenv()

API_KEY_GOOGLE = os.getenv("API_KEY_GOOGLE")
API_KEY_TOMTOM = os.getenv("API_KEY_TOMTOM")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="EcoRoute API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    """Initialize data and check API keys on startup."""
    create_db_and_tables() # Create tables

    check_api_keys()
    load_co2_data()
    logger.info("Application startup complete.")


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on AWS!"}


@app.get("/vehicles")
async def get_vehicles():
    """Get available vehicle types for route calculation."""
    if not co2_map:
        load_co2_data()
    return {
        "vehicles": [
            {"id": vehicle_id, "name": vehicle_id.replace('.', ' ').title()}
            for vehicle_id in co2_map.keys()
        ]
    }


def log_performance(func_name: str, start_time: float, **kwargs):
    """Log performance metrics and additional info"""
    duration = time.time() - start_time
    logger.info(f"{func_name} completed in {duration:.2f}s - {kwargs}")

def log_graph_stats(G, stage: str):
    """Log graph statistics for debugging"""
    logger.info(f"Graph stats at {stage}: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Sample some edge attributes for debugging
    sample_edges = list(G.edges(data=True))[:3]
    for u, v, data in sample_edges:
        logger.debug(f"Sample edge {u}->{v}: {list(data.keys())}")


@app.get("/route/stream")
async def stream_route(origin: str, destination: str, vehicle: str):
    """
    Calculates routes and streams logs and results back to the client using SSE.
    
    Query Params:
    - origin: "lat,lon"
    - destination: "lat,lon"
    - vehicle: "pass. car"
    """
    
    try:
        # ... (coordinate parsing) ...
        request = RouteRequest(
            origin=tuple(map(float, origin.split(','))),
            destination=tuple(map(float, destination.split(','))),
            vehicle=vehicle
        )
        origin_coords = tuple(map(float, origin.split(',')))
        destination_coords = tuple(map(float, destination.split(',')))
        
        if not all(-90 <= lat <= 90 and -180 <= lon <= 180 for lat, lon in [origin_coords, destination_coords]):
            raise ValueError("Invalid coordinates.")

        request = RouteRequest(
            origin=origin_coords,
            destination=destination_coords,
            vehicle=vehicle
        )
        
        # Return a streaming response that calls our async generator
        db = next(get_db())
        return StreamingResponse(calculate_route_streamed(request, db), media_type="text/event-stream")
    
    except ValueError as e:
        # This error is for invalid query parameter format
        raise HTTPException(status_code=400, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in route calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    # For local development, you can set API keys directly or via environment variables
    # These will be picked up by os.getenv in startup_event
    uvicorn.run(app, host="0.0.0.0", port=8000)