/**
 * National grid — single source of truth for the India overview.
 * 31 rows x 30 cols over the country bounding box; the same grid the realtime
 * ingester fills. Named cities let a rainfall value be read as a place.
 */

export const ROWS = 31
export const COLS = 30
export const LAT_MIN = 6.0
export const LAT_MAX = 37.0
export const LON_MIN = 68.0
export const LON_MAX = 98.0
export const CELL_LAT = (LAT_MAX - LAT_MIN) / ROWS
export const CELL_LON = (LON_MAX - LON_MIN) / COLS

// Well-known locations inside each grid cell.
export const CITY_NAMES = {
  '6,10': 'Kanyakumari', '8,10': 'Kochi', '9,12': 'Bengaluru',
  '10,12': 'Hyderabad', '10,13': 'Chennai', '11,8': 'Goa',
  '12,7': 'Pune', '12,8': 'Mumbai', '13,6': 'Ahmedabad',
  '13,9': 'Bhopal', '14,16': 'Kolkata', '15,6': 'Jodhpur',
  '16,7': 'Delhi', '16,8': 'Agra', '17,7': 'Dehradun',
  '18,6': 'Amritsar', '19,19': 'Guwahati',
}

export function cellCenter(r, c) {
  return [LAT_MIN + r * CELL_LAT + CELL_LAT / 2, LON_MIN + c * CELL_LON + CELL_LON / 2]
}

export function cellToRowCol(cellIndex) {
  return [Math.floor(cellIndex / COLS), cellIndex % COLS]
}

/** Grid cell index -> a place name, or a readable fallback. */
export function cellToLabel(cellIndex) {
  const [r, c] = cellToRowCol(cellIndex)
  return CITY_NAMES[`${r},${c}`] || `Grid ${r}-${c}`
}

/** True when the caller should treat this cell as a named location. */
export function isNamedCell(cellIndex) {
  const [r, c] = cellToRowCol(cellIndex)
  return Boolean(CITY_NAMES[`${r},${c}`])
}
