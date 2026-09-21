"""
VARUNA India Grid System
=========================
Divides India into a uniform grid for nationwide flood prediction.

Grid Coverage:
- Latitude:  6°N (Kanyakumari) to 37°N (Jammu)
- Longitude: 68°E (Gujarat coast) to 98°E (Arunachal)
- Resolution: 1° × 1° (≈110km × 110km per cell)
- Total cells: 31 × 30 = 930 cells

Each cell is mapped to the nearest major city/region for display.

For finer resolution (cities), the system automatically zooms into
the active grid using the same 0.02° resolution as Mumbai.
"""

from typing import Dict, List, Tuple, Optional

# India bounding box
INDIA_LAT_MIN = 6.0    # Kanyakumari
INDIA_LAT_MAX = 37.0   # Jammu & Kashmir
INDIA_LON_MIN = 68.0   # Gujarat
INDIA_LON_MAX = 98.0   # Arunachal Pradesh

# Grid resolution
LAT_STEP = 1.0  # degrees
LON_STEP = 1.0  # degrees

INDIA_ROWS = int((INDIA_LAT_MAX - INDIA_LAT_MIN) / LAT_STEP)  # 31
INDIA_COLS = int((INDIA_LON_MAX - INDIA_LON_MIN) / LON_STEP)  # 30

# City lookup for each grid cell (approximate)
# Maps (row, col) → nearest major city
CITY_LOOKUP = {
    # South India
    (0, 10): "Kanyakumari", (0, 11): "Nagercoil",
    (1, 10): "Thiruvananthapuram", (1, 11): "Tirunelveli",
    (2, 10): "Kochi", (2, 11): "Coimbatore",
    (2, 12): "Madurai",
    (3, 10): "Kozhikode", (3, 11): "Mysuru", (3, 12): "Bengaluru",
    (3, 13): "Salem",
    (4, 9): "Mangaluru", (4, 11): "Hubli", (4, 12): "Hyderabad",
    (4, 13): "Chennai",
    (5, 8): "Goa", (5, 11): "Solapur", (5, 12): "Nagpur",
    (5, 13): "Vijayawada",
    # West India
    (6, 7): "Mumbai", (6, 8): "Pune", (6, 9): "Aurangabad",
    (6, 12): "Raipur", (6, 13): "Visakhapatnam",
    (7, 6): "Surat", (7, 7): "Nashik", (7, 8): "Nanded",
    (7, 11): "Jalna", (7, 12): "Jagdalpur",
    (8, 5): "Ahmedabad", (8, 6): "Vadodara", (8, 7): "Indore",
    (8, 8): "Bhopal", (8, 9): "Jabalpur",
    (9, 4): "Rajkot", (9, 5): "Jamnagar", (9, 6): "Udaipur",
    (9, 7): "Gwalior", (9, 8): "Kanpur", (9, 9): "Lucknow",
    # North India
    (10, 3): "Bhuj", (10, 6): "Jaipur", (10, 7): "Agra",
    (10, 8): "Varanasi", (10, 9): "Prayagraj", (10, 10): "Patna",
    (11, 5): "Jodhpur", (11, 6): "Ajmer", (11, 7): "Delhi",
    (11, 8): "Meerut", (11, 9): "Gorakhpur",
    (12, 5): "Bikaner", (12, 6): "Alwar", (12, 7): "Chandigarh",
    (12, 8): "Dehradun", (12, 9): "Kathmandu (Nepal)",
    (13, 4): "Karachi (Pakistan)", (13, 5): "Lahore (Pakistan)",
    (13, 7): "Amritsar", (13, 8): "Jammu",
    # Northeast India
    (6, 15): "Bhubaneswar", (6, 16): "Kolkata",
    (7, 14): "Visakhapatnam", (7, 15): "Cuttack",
    (7, 16): "Dhanbad",
    (8, 14): "Raipur", (8, 15): "Ranchi", (8, 16): "Patna",
    (8, 18): "Guwahati",
    (9, 17): "Siliguri", (9, 18): "Shillong",
    (9, 19): "Agartala",
    (10, 19): "Imphal", (10, 20): "Aizawl",
    (10, 21): "Kohima",
    (11, 21): "Itanagar", (11, 22): "Tawang",
    # Islands
    (3, 5): "Lakshadweep",
    (9, 15): "Andaman (Port Blair)",
}


def latlon_to_cell(lat: float, lon: float) -> Tuple[int, int]:
    """Convert lat/lon to grid (row, col)."""
    row = int((lat - INDIA_LAT_MIN) / LAT_STEP)
    col = int((lon - INDIA_LON_MIN) / LON_STEP)
    row = max(0, min(INDIA_ROWS - 1, row))
    col = max(0, min(INDIA_COLS - 1, col))
    return row, col


def cell_to_latlon(row: int, col: int) -> Tuple[float, float]:
    """Convert grid (row, col) to lat/lon center."""
    lat = INDIA_LAT_MIN + row * LAT_STEP + LAT_STEP / 2
    lon = INDIA_LON_MIN + col * LON_STEP + LON_STEP / 2
    return lat, lon


def cell_to_city(row: int, col: int) -> str:
    """Get nearest city name for a grid cell."""
    return CITY_LOOKUP.get((row, col), f"Cell {row}-{col}")


def cell_index(row: int, col: int) -> int:
    """Flat index for a cell."""
    return row * INDIA_COLS + col


def get_all_cells() -> List[Dict]:
    """Return all cells with their metadata."""
    cells = []
    for r in range(INDIA_ROWS):
        for c in range(INDIA_COLS):
            lat, lon = cell_to_latlon(r, c)
            cells.append({
                "cell_index": cell_index(r, c),
                "row": r,
                "col": c,
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "city": cell_to_city(r, c),
            })
    return cells


def get_active_cells_for_region(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> List[Dict]:
    """Get all cells within a bounding box (for regional zoom)."""
    cells = get_all_cells()
    return [
        c for c in cells
        if lat_min <= c["lat"] <= lat_max
        and lon_min <= c["lon"] <= lon_max
    ]


# Pre-built state-level grid for quick lookup
STATE_CENTERS = {
    "Maharashtra": (19.75, 75.71),
    "Karnataka": (15.32, 75.71),
    "Kerala": (10.85, 76.27),
    "Tamil Nadu": (11.12, 78.37),
    "Gujarat": (22.26, 71.19),
    "Rajasthan": (27.02, 74.22),
    "Uttar Pradesh": (26.85, 80.91),
    "Madhya Pradesh": (22.97, 78.66),
    "West Bengal": (22.99, 87.75),
    "Andhra Pradesh": (15.91, 79.74),
    "Telangana": (17.12, 79.20),
    "Odisha": (20.95, 85.10),
    "Bihar": (25.10, 85.31),
    "Punjab": (31.15, 75.34),
    "Haryana": (29.06, 76.09),
    "Delhi": (28.70, 77.10),
    "Assam": (26.20, 92.94),
    "Jharkhand": (23.61, 85.28),
    "Chhattisgarh": (21.27, 81.87),
    "Uttarakhand": (30.07, 79.02),
    "Himachal Pradesh": (31.10, 77.17),
    "Goa": (15.29, 74.12),
    "Jammu & Kashmir": (33.78, 76.57),
    "Sikkim": (27.53, 88.51),
    "Meghalaya": (25.47, 91.37),
    "Tripura": (23.94, 91.99),
    "Manipur": (24.66, 93.91),
    "Mizoram": (23.16, 92.94),
    "Nagaland": (26.16, 94.56),
    "Arunachal Pradesh": (28.22, 94.73),
}
