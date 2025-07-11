import logging
import time
import os
import math
from typing import Dict, Any, List, Tuple, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import networkx as nx
import osmnx as ox
import googlemaps
import polyline
from fastapi import HTTPException
from geopy.distance import geodesic

from app.utils import *
from app.models import RouteRequest, RouteResponse

from sqlalchemy.orm import Session

ox.settings.use_cache = False
ox.settings.log_console = True

ox.settings.data_folder = "/tmp/osmnx_dummy_data_no_cache" 
ox.settings.logs_folder = "/tmp/osmnx_dummy_logs_no_cache"

logger = logging.getLogger(__name__)


def base_eco_cost(u: int, v: int, data: Dict[str, Any], vehicle: str, G: nx.MultiDiGraph) -> float:
    """Calculate ecological cost for an edge."""
    length = data.get("length", 1.0)
    u_elev = G.nodes[u].get("elevation", 0.0)
    v_elev = G.nodes[v].get("elevation", 0.0)
    elev_gain = max(0.0, v_elev - u_elev)
    mass = vehicle_mass_kg(vehicle)
    base_rate = co2_map.get(vehicle, 180000) / 1_000_000
    base_emissions = length * base_rate
    slope_emissions = mass * 9.81 * elev_gain * (0.074 / 1e6)
    turn_penalty_cost = data.get("turn_penalty", 0.0) * 0.1
    return base_emissions + slope_emissions + turn_penalty_cost

def compute_stats(route: List[int], vehicle: str, G: nx.MultiDiGraph, edge_mapping: Dict[Tuple[int, int], Tuple[int, int, int]]) -> Tuple[float, float, float]:
    """Compute route statistics (distance, time, CO2) for a given path."""
    total_dist, total_time, total_co2 = 0.0, 0.0, 0.0
    if not route:
        return 0.0, 0.0, 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        original_u, original_v, original_k = edge_mapping.get((u, v), (u, v, 0))
        try:
            data = G[original_u][original_v][original_k]
            total_dist += data.get("length", 0.0)
            total_time += data.get("travel_time", 0.0)
            total_co2 += data.get("eco_cost", 0.0)
        except KeyError:
            logger.warning(f"Edge ({original_u}, {original_v}, {original_k}) not found in G. Skipping stats for this segment.")
            continue
    return total_dist / 1000, total_time / 60, total_co2

def download_graph_chunk(center_coords, chunk_size_km=25, retry_count=3):
    """Download a single graph chunk with retry logic."""
    west, south, east, north = calculate_chunk_bounds(center_coords, chunk_size_km)
    for attempt in range(retry_count):
        try:
            logger.info(f"Downloading chunk {attempt + 1}/{retry_count} for bbox: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
            G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive", simplify=True)
            logger.info(f"Downloaded chunk with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            return G
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for chunk {center_coords}: {e}")
            if attempt < retry_count - 1:
                wait_time = (2 ** attempt) * 2
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download chunk {center_coords} after {retry_count} attempts.")
                return None

def merge_graph_chunks(graph_chunks: List[nx.MultiDiGraph]) -> nx.MultiDiGraph:
    """Merge multiple graph chunks into a single graph with memory optimization."""
    if not graph_chunks:
        raise ValueError("No graph chunks provided to merge.")
    logger.info(f"Merging {len(graph_chunks)} graph chunks...")
    # Start with a copy of the first chunk to avoid modifying it in place
    merged_graph = graph_chunks[0].copy()
    # Iterate over the rest of the chunks
    for i in range(1, len(graph_chunks)):
        merged_graph = nx.compose(merged_graph, graph_chunks[i])
    logger.info(f"Merged graph has {len(merged_graph.nodes)} nodes and {len(merged_graph.edges)} edges.")
    return merged_graph

def build_optimized_road_network(db: Session, origin_coords, destination_coords, chunk_size_km=25, use_cache=True):
    """Builds road network using a cached, chunked approach for reliability and performance."""
    logger.info("Building road network graph...")
    graph_build_start = time.time()
    total_distance_km = geodesic(origin_coords, destination_coords).kilometers
    logger.info(f"Total route distance: {total_distance_km:.2f} km")

    if total_distance_km <= 40: # Use simple box for shorter distances
        logger.info("Short distance detected. Using simple rectangular download.")
        return build_simple_rectangular_network(origin_coords, destination_coords)

    # Use chunked approach for longer distances
    logger.info(f"Long distance detected. Using chunked approach with {chunk_size_km}km chunks.")
    num_chunks = max(3, int(total_distance_km / (chunk_size_km * 0.75)) + 2)
    route_points = interpolate_points_along_route(origin_coords, destination_coords, num_chunks)
    logger.info(f"Created {len(route_points)} intermediate points for chunked downloading.")

    graph_chunks, cache_hits, downloads = [], 0, 0
    for i, point in enumerate(route_points):
        cache_key = get_chunk_cache_key(point, chunk_size_km)
        chunk = None
        if use_cache:
            chunk = load_chunk_from_cache(db, cache_key)

        if chunk: # Cache hit
            logger.info(f"Chunk {i+1}/{len(route_points)} loaded from cache: {cache_key}")
            cache_hits += 1
        else: # Cache miss, download required
            logger.info(f"Downloading chunk {i+1}/{len(route_points)}...")
            chunk = download_graph_chunk(point, chunk_size_km)
            if chunk:
                downloads += 1
                if use_cache:
                    save_chunk_to_cache(db, chunk, cache_key)
            else:
                logger.warning(f"Failed to download chunk {i+1}. Skipping.")
                continue

        graph_chunks.append(chunk)

    if not graph_chunks:
        raise Exception("Fatal: Failed to download or load any graph chunks for the route.")

    logger.info(f"Processed {len(graph_chunks)} chunks: {cache_hits} from cache, {downloads} new downloads.")

    # Merge all collected chunks at once
    merged_graph = merge_graph_chunks(graph_chunks)

    # Add necessary graph attributes after merging
    logger.info("Adding edge speeds and travel times to the merged graph...")
    merged_graph = ox.add_edge_speeds(merged_graph)
    merged_graph = ox.add_edge_travel_times(merged_graph)
    
    logger.info(f"Final graph built in {time.time() - graph_build_start:.2f}s with {len(merged_graph.nodes)} nodes and {len(merged_graph.edges)} edges.")
    return merged_graph

def build_simple_rectangular_network(origin_coords, destination_coords, buffer_km=5, width_ratio=0.4):
    """
    Build road network using simple rectangular approach for shorter distances.
    
    Args:
        origin_coords: (lat, lon) tuple for origin
        destination_coords: (lat, lon) tuple for destination
        buffer_km: Buffer distance in kilometers
        width_ratio: Width as ratio of total distance
    
    Returns:
        networkx.MultiDiGraph: Road network graph
    """
    # Calculate distance
    distance_km = geodesic(origin_coords, destination_coords).kilometers
    
    # Calculate center point
    center_lat = (origin_coords[0] + destination_coords[0]) / 2
    center_lon = (origin_coords[1] + destination_coords[1]) / 2
    
    # Calculate bounds with buffer
    total_distance = distance_km + (2 * buffer_km)
    width = max(total_distance * width_ratio, 20)  # Minimum 20km width
    
    # Calculate approximate degree offsets
    lat_offset = total_distance / 111.0 / 2
    lon_offset = width / (111.0 * math.cos(math.radians(center_lat))) / 2
    
    north = center_lat + lat_offset
    south = center_lat - lat_offset
    east = center_lon + lon_offset
    west = center_lon - lon_offset
    
    logger.info(f"Downloading simple rectangular region: "
               f"N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
    
    bbox = (west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True
    )
    
    # Add speeds and travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    return G

async def calculate_route_streamed(request: RouteRequest, db: Session) -> AsyncGenerator[str, None]:
    def sse_format(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        origin_coords = request.origin
        destination_coords = request.destination
        vehicle = request.vehicle

        if vehicle not in co2_map:
            raise ValueError(f"Invalid vehicle type '{vehicle}'.")

        yield sse_format({"type": "log", "message": "Received request. Starting process..."})
        await asyncio.sleep(0.01)

        # --- Graph Building ---
        yield sse_format({"type": "log", "message": "Building road network graph... (this may take a while)"})
        graph_build_start = time.time()
        G = build_optimized_road_network(db, origin_coords, destination_coords, chunk_size_km=25)
        yield sse_format({"type": "log", "message": f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges in {time.time() - graph_build_start:.2f}s."})
        await asyncio.sleep(0.01)

        # --- Elevation Data ---
        yield sse_format({"type": "log", "message": "Adding elevation data..."})
        node_ids = list(G.nodes)
        cached_elevations = get_elevations_from_db(db, node_ids)
        uncached_nodes = [n for n in node_ids if n not in cached_elevations]

        if uncached_nodes:
            yield sse_format({"type": "log", "message": f"Fetching elevation for {len(uncached_nodes)} new nodes..."})
            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in uncached_nodes]
            batches = [coords[i:i+100] for i in range(0, len(coords), 100)]

            fetched = []
            with ThreadPoolExecutor(max_workers=5) as ex:
                futures = {ex.submit(fetch_elevation_batch, b): b for b in batches}
                for f in as_completed(futures):
                    fetched.extend(f.result())

            fetched_map = dict(zip(uncached_nodes, fetched))
            save_elevations_to_db(db, fetched_map)
            cached_elevations.update(fetched_map)

        nx.set_node_attributes(G, cached_elevations, "elevation")
        for n in G.nodes:
            if G.nodes[n].get("elevation") is None:
                G.nodes[n]["elevation"] = 0.0
        yield sse_format({"type": "log", "message": "Elevation data added."})
        await asyncio.sleep(0.01)

        # --- Edge Grades ---
        yield sse_format({"type": "log", "message": "Adding edge grades..."})
        G = ox.add_edge_grades(G)
        yield sse_format({"type": "log", "message": "Edge grades added."})
        await asyncio.sleep(0.01)

        # --- Nearest Nodes ---
        orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
        dest_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])
        if orig_node is None or dest_node is None:
            raise HTTPException(status_code=400, detail="Could not find reachable nodes near origin/destination.")
        yield sse_format({"type": "log", "message": f"Nearest nodes found: Origin {orig_node}, Destination {dest_node}"})
        await asyncio.sleep(0.01)

        # --- Initial Fastest Route ---
        fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
        fastest_route_edges_uv = set(zip(fastest_route_nodes[:-1], fastest_route_nodes[1:]))
        yield sse_format({"type": "log", "message": "Initial fastest route calculated."})
        await asyncio.sleep(0.01)

        # --- TomTom Traffic Data ---
        coords_to_check = {}
        edge_tomtom_keys = []
        for u, v in fastest_route_edges_uv:
            for k in G[u][v]:
                lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
                lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
                key = f"{lat:.5f},{lon:.5f}"
                coords_to_check[key] = (lat, lon)
                edge_tomtom_keys.append((u, v, k, key))

        tomtom_cache = get_tomtom_speeds_from_db(db, list(coords_to_check.keys()))
        coords_to_fetch = {k: v for k, v in coords_to_check.items() if k not in tomtom_cache}

        if coords_to_fetch:
            yield sse_format({"type": "log", "message": f"Fetching {len(coords_to_fetch)} TomTom points..."})
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = {ex.submit(fetch_speed, item): item[0] for item in coords_to_fetch.items()}
                for future in as_completed(futures):
                    key, speed = future.result()
                    tomtom_cache[key] = speed
            save_tomtom_speeds_to_db(db, tomtom_cache)

        for u, v, k, key in edge_tomtom_keys:
            speed_kph = tomtom_cache.get(key, 30.0)
            length_m = safe_get(G[u][v][k], "length", 1.0)
            G[u][v][k]["travel_time"] = length_m / (speed_kph * 1000 / 3600)
        yield sse_format({"type": "log", "message": "TomTom speeds applied."})
        await asyncio.sleep(0.01)

        # --- Turn Penalties ---
        yield sse_format({"type": "log", "message": "Adding turn penalties..."})
        try:
            nx.set_edge_attributes(G, 0, "turn_penalty")
            for i in range(1, len(fastest_route_nodes) - 1):
                u, v, w = fastest_route_nodes[i - 1:i + 2]
                b1 = calculate_bearing(G.nodes[u], G.nodes[v])
                b2 = calculate_bearing(G.nodes[v], G.nodes[w])
                delta = abs(b2 - b1)
                if delta > 180: delta = 360 - delta
                penalty = next(p for a, p in [(170, 90), (150, 60), (120, 45), (90, 30), (45, 15)] if delta > a) if delta > 5 else 5
                for k in G[u][v]:
                    G[u][v][k]["turn_penalty"] += penalty
                    G[u][v][k]["travel_time"] += penalty
        except Exception as e:
            logger.warning(f"Turn penalty error: {e}")
        yield sse_format({"type": "log", "message": "Turn penalties added."})
        await asyncio.sleep(0.01)

        # --- Eco Cost Calculation ---
        yield sse_format({"type": "log", "message": "Calculating eco-costs..."})
        eco_costs = {(u, v, k): base_eco_cost(u, v, d, vehicle, G) for u, v, k, d in G.edges(keys=True, data=True)}
        nx.set_edge_attributes(G, eco_costs, "eco_cost")
        yield sse_format({"type": "log", "message": f"Eco-costs calculated for {len(eco_costs)} edges."})
        await asyncio.sleep(0.01)

        # --- Simplify Graph ---
        yield sse_format({"type": "log", "message": "Simplifying graph for eco-routing..."})
        G_simple = nx.DiGraph()
        edge_mapping = {}
        for u, v in G.edges():
            best_k = min(G[u][v], key=lambda k: G[u][v][k].get("eco_cost", float("inf")))
            G_simple.add_edge(u, v, **G[u][v][best_k])
            edge_mapping[(u, v)] = (u, v, best_k)
        yield sse_format({"type": "log", "message": f"Simplified graph ready with {len(G_simple.edges)} edges."})
        await asyncio.sleep(0.01)

        # --- Eco Route ---
        eco_route_nodes = nx.shortest_path(G_simple, orig_node, dest_node, weight="eco_cost")
        eco_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in eco_route_nodes]

        # --- Fastest Route (final) ---
        final_fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")

        # --- Stats ---
        eco_dist, eco_time, eco_co2 = compute_stats(eco_route_nodes, vehicle, G, edge_mapping)
        fast_dist, fast_time, fast_co2 = compute_stats(final_fastest_route_nodes, vehicle, G, {}) # Use empty mapping for fastest route

        # --- Google ---
        google_client = None
        google_route_coords, google_distance, google_duration, google_co2_estimated = [], 0, 0, 0
        api_key_google = os.getenv("API_KEY_GOOGLE")

        if api_key_google:
            try:
                google_client = googlemaps.Client(key=api_key_google)
                g_resp = google_client.directions(origin_coords, destination_coords, mode="driving", departure_time="now")
                if g_resp:
                    poly = g_resp[0]["overview_polyline"]["points"]
                    google_route_coords = polyline.decode(poly)
                    leg = g_resp[0]["legs"][0]
                    google_distance = leg["distance"]["value"] / 1000
                    google_duration = leg.get("duration_in_traffic", leg["duration"])["value"] / 60
                    google_co2_estimated = google_distance * (co2_map.get(vehicle, 180000) / 1_000_000)
            except Exception as e:
                yield sse_format({"type": "log", "message": f"Google route fetch failed: {e}"})

        eco_google_duration = eco_time
        if google_client and eco_coords:
            try:
                waypoints = sample_waypoints(eco_coords)
                resp = google_client.directions(eco_coords[0], eco_coords[-1], waypoints=waypoints, mode="driving", departure_time="now")
                if resp:
                    eco_google_duration = sum(leg.get("duration_in_traffic", leg["duration"])["value"] for leg in resp[0]["legs"]) / 60
            except:
                pass

        co2_savings = max(fast_co2 - eco_co2, 0)
        co2_savings_pct = round((co2_savings / fast_co2) * 100, 1) if fast_co2 else 0

        response_data = RouteResponse(
            eco_route=eco_coords,
            google_route=google_route_coords,
            eco_stats={
                "distance_km": round(eco_dist, 2),
                "time_minutes": round(eco_time, 1),
                "time_minutes_google_estimated": round(eco_google_duration, 1),
                "co2_kg": round(eco_co2, 2)
            },
            google_stats={
                "distance_km": round(google_distance, 2),
                "time_minutes": round(google_duration, 1),
                "co2_kg": round(google_co2_estimated, 2)
            },
            comparison={
                "co2_savings_kg": round(co2_savings, 2),
                "co2_savings_percent": co2_savings_pct,
                "time_difference_minutes": round(eco_google_duration - google_duration, 1)
            }
        ).model_dump()

        yield sse_format({"type": "result", "data": response_data})
        yield sse_format({"type": "log", "message": "Process complete."})

    except Exception as e:
        logger.error(f"Error during route streaming: {e}", exc_info=True)
        yield sse_format({"type": "error", "message": f"A critical error occurred: {e}"})
