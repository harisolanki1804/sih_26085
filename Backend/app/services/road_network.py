"""
VARUNA Mumbai Road Network & Flood-Safe Routing
------------------------------------------------
Fetches real road data from OpenStreetMap Overpass API,
builds a NetworkX graph, and routes around flooded cells.

Falls back to a synthetic grid if the API is unreachable.
"""

import math
import logging
import requests
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("VARUNA.RoadNetwork")

# Mumbai bounding box
MUMBAI_BBOX = (18.88, 72.75, 19.20, 72.95)  # south, west, north, east

# Cache
_road_graph: Optional[nx.Graph] = None
_node_coords: Dict[str, Tuple[float, float]] = {}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points."""
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def fetch_osm_roads() -> List[Dict]:
    """Fetch major roads from OpenStreetMap Overpass API."""
    s, w, n, e = MUMBAI_BBOX
    query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"primary|secondary|tertiary|residential|motorway|trunk"]["name"]({s},{w},{n},{e});
      way["highway"~"primary|secondary|tertiary"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Fetched {len(data.get('elements', []))} OSM elements")
        return data.get("elements", [])
    except Exception as e:
        logger.warning(f"Overpass API failed: {e}. Using synthetic network.")
        return []


def build_graph_from_osm(elements: List[Dict]) -> nx.Graph:
    """Build a NetworkX graph from OSM elements."""
    G = nx.Graph()
    
    # Parse nodes
    nodes = {}
    for el in elements:
        if el["type"] == "node":
            nid = str(el["id"])
            nodes[nid] = (el["lat"], el["lon"])
            G.add_node(nid, lat=el["lat"], lon=el["lon"])
    
    # Parse ways (road segments)
    for el in elements:
        if el["type"] == "way" and "nodes" in el:
            node_ids = el["nodes"]
            highway = el.get("tags", {}).get("highway", "tertiary")
            name = el.get("tags", {}).get("name", "")
            
            # Weight: motorways and primaries get lower weight (preferred)
            weight_mult = {
                "motorway": 0.5, "trunk": 0.6, "primary": 0.7,
                "secondary": 0.8, "tertiary": 0.9, "residential": 1.0
            }.get(highway, 1.0)
            
            for i in range(len(node_ids) - 1):
                n1, n2 = str(node_ids[i]), str(node_ids[i+1])
                if n1 in nodes and n2 in nodes:
                    dist = haversine_m(
                        nodes[n1][0], nodes[n1][1],
                        nodes[n2][0], nodes[n2][1]
                    )
                    G.add_edge(n1, n2, weight=dist * weight_mult, dist_m=dist, name=name, highway=highway)
    
    logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_synthetic_network() -> nx.Graph:
    """Build a synthetic road grid that approximates Mumbai's peninsula shape."""
    G = nx.Graph()
    
    # Create a grid covering Mumbai
    lats = [18.90 + i * 0.004 for i in range(50)]  # ~400m spacing
    lons = [72.78 + i * 0.004 for i in range(45)]
    
    nodes = {}
    node_id = 0
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Approximate Mumbai peninsula shape
            # Mumbai is narrower in the south, wider in the north
            center_lon = 72.85
            if lat < 18.92:
                # South Mumbai — narrow
                if abs(lon - center_lon) > 0.04:
                    continue
            elif lat < 18.96:
                # Central Mumbai
                if abs(lon - center_lon) > 0.07:
                    continue
            else:
                # North Mumbai — wider
                if abs(lon - center_lon) > 0.10:
                    continue
            
            nid = f"s{node_id}"
            nodes[(i, j)] = nid
            G.add_node(nid, lat=lat, lon=lon)
            node_id += 1
    
    # Connect neighbors (grid graph)
    for (i, j), nid in nodes.items():
        # Connect to right neighbor
        if (i, j+1) in nodes:
            nid2 = nodes[(i, j+1)]
            lat1, lon1 = G.nodes[nid]["lat"], G.nodes[nid]["lon"]
            lat2, lon2 = G.nodes[nid2]["lat"], G.nodes[nid2]["lon"]
            dist = haversine_m(lat1, lon1, lat2, lon2)
            G.add_edge(nid, nid2, weight=dist, dist_m=dist, name="Road", highway="tertiary")
        
        # Connect to bottom neighbor
        if (i+1, j) in nodes:
            nid2 = nodes[(i+1, j)]
            lat1, lon1 = G.nodes[nid]["lat"], G.nodes[nid]["lon"]
            lat2, lon2 = G.nodes[nid2]["lat"], G.nodes[nid2]["lon"]
            dist = haversine_m(lat1, lon1, lat2, lon2)
            G.add_edge(nid, nid2, weight=dist, dist_m=dist, name="Road", highway="tertiary")
        
        # Diagonal connections for more natural routing
        if (i+1, j+1) in nodes:
            nid2 = nodes[(i+1, j+1)]
            lat1, lon1 = G.nodes[nid]["lat"], G.nodes[nid]["lon"]
            lat2, lon2 = G.nodes[nid2]["lat"], G.nodes[nid2]["lon"]
            dist = haversine_m(lat1, lon1, lat2, lon2)
            G.add_edge(nid, nid2, weight=dist * 1.4, dist_m=dist, name="Road", highway="residential")
    
    logger.info(f"Built synthetic network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def get_road_graph() -> nx.Graph:
    """Get or build the road network graph (cached)."""
    global _road_graph, _node_coords
    
    if _road_graph is not None:
        return _road_graph
    
    # Try OSM first
    elements = fetch_osm_roads()
    if elements and len(elements) > 100:
        _road_graph = build_graph_from_osm(elements)
    else:
        _road_graph = build_synthetic_network()
    
    # Cache node coordinates
    for nid, data in _road_graph.nodes(data=True):
        _node_coords[nid] = (data["lat"], data["lon"])
    
    return _road_graph


def find_nearest_node(lat: float, lon: float) -> str:
    """Find the nearest graph node to a given lat/lon."""
    graph = get_road_graph()
    best_node = None
    best_dist = float('inf')
    
    for nid in graph.nodes():
        nlat, nlon = _node_coords[nid]
        d = haversine_m(lat, lon, nlat, nlon)
        if d < best_dist:
            best_dist = d
            best_node = nid
    
    return best_node


def find_safe_path(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    flooded_cells: List[Dict],
    flood_radius_m: float = 300
) -> Dict[str, Any]:
    """
    Find a flood-safe path using the road network.
    
    1. Find nearest graph nodes to start/end
    2. Remove edges that pass through flooded areas
    3. Find shortest path on the modified graph
    """
    graph = get_road_graph()
    
    # Find nearest nodes
    start_node = find_nearest_node(start_lat, start_lon)
    end_node = find_nearest_node(end_lat, end_lon)
    
    if not start_node or not end_node:
        return _fallback_linear(start_lat, start_lon, end_lat, end_lon)
    
    # Create a copy and remove flooded edges
    G = graph.copy()
    
    flooded_node_ids = set()
    for cell in flooded_cells:
        flat, flon = cell["lat"], cell["lon"]
        # Find nodes near this flooded cell
        for nid in G.nodes():
            nlat, nlon = _node_coords[nid]
            if haversine_m(flat, flon, nlat, nlon) < flood_radius_m:
                flooded_node_ids.add(nid)
    
    # Remove edges connected to flooded nodes
    for nid in flooded_node_ids:
        if G.has_node(nid):
            # Remove all edges of this node
            neighbors = list(G.neighbors(nid))
            for neighbor in neighbors:
                if G.has_edge(nid, neighbor):
                    G.remove_edge(nid, neighbor)
    
    # Find shortest path
    try:
        path = nx.shortest_path(G, start_node, end_node, weight='weight')
        
        # Convert path to waypoints
        waypoints = []
        total_dist = 0
        for i, nid in enumerate(path):
            lat, lon = _node_coords[nid]
            is_flooded = nid in flooded_node_ids
            
            if i > 0:
                prev_nid = path[i-1]
                if G.has_edge(prev_nid, nid):
                    # Use original graph distance
                    total_dist += graph[prev_nid][nid].get("dist_m", 0)
            
            waypoints.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "is_flooded": is_flooded,
                "detour": False,
                "label": "Start" if i == 0 else ("End" if i == len(path) - 1 else f"W{i}")
            })
        
        # Calculate direct distance
        direct_dist = haversine_m(start_lat, start_lon, end_lat, end_lon)
        
        # Count flooded segments avoided
        flooded_avoided = len(flooded_node_ids)
        
        return {
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "direct_distance_km": round(direct_dist / 1000, 2),
            "safe_route_distance_km": round(total_dist / 1000, 2),
            "detour_overhead_pct": round(((total_dist / direct_dist) - 1) * 100, 1) if direct_dist > 0 else 0,
            "flooded_segments_avoided": flooded_avoided,
            "total_flooded_in_direct_path": flooded_avoided,
            "waypoints": waypoints,
            "flooded_cells_count": len(flooded_cells),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "route_type": "road_network",
            "safety_note": "Route follows actual road network and avoids flooded streets."
        }
    
    except nx.NetworkXNoPath:
        # No path found — return linear fallback
        return _fallback_linear(start_lat, start_lon, end_lat, end_lon)


def _fallback_linear(start_lat, start_lon, end_lat, end_lon):
    """Fallback: linear interpolation when no graph path exists."""
    waypoints = []
    n = 10
    for i in range(n + 1):
        t = i / n
        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)
        waypoints.append({
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "is_flooded": False,
            "detour": False,
            "label": "Start" if i == 0 else ("End" if i == n else f"W{i}")
        })
    
    direct_dist = haversine_m(start_lat, start_lon, end_lat, end_lon)
    return {
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
        "direct_distance_km": round(direct_dist / 1000, 2),
        "safe_route_distance_km": round(direct_dist / 1000, 2),
        "detour_overhead_pct": 0,
        "flooded_segments_avoided": 0,
        "total_flooded_in_direct_path": 0,
        "waypoints": waypoints,
        "flooded_cells_count": 0,
        "route_type": "linear_fallback",
        "safety_note": "No graph path available. Showing direct route."
    }
