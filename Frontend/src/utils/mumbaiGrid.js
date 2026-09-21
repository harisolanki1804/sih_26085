/**
 * Mumbai pilot grid — single source of truth.
 * ===========================================
 * 10 rows x 9 cols = 90 cells, row-major, matching the backend replay grid
 * (cell_index 0..89, ids C01..C90). The map, the left panel and the alerts
 * list all resolve a cell through this module, so a cell can never be named
 * differently in two places.
 */

export const M_ROWS = 10
export const M_COLS = 9
export const M_LAT_MIN = 18.88
export const M_LAT_MAX = 19.26
export const M_LON_MIN = 72.78
export const M_LON_MAX = 73.00
export const M_CELL_LAT = (M_LAT_MAX - M_LAT_MIN) / M_ROWS
export const M_CELL_LON = (M_LON_MAX - M_LON_MIN) / M_COLS
export const M_CENTER = [19.06, 72.9]

// Row-major locality names, aligned with the backend grid.
export const CELL_NAMES = [
  'Sea', 'Colaba', 'Fort', 'Churchgate', 'Marine Drive', 'Nariman Point', 'Malabar Hill', 'Walkeshwar', 'Haji Ali',
  'Sea', 'Grant Road', 'Tardeo', 'Bhuleshwar', 'Girgaon', 'Parel', 'Mahalaxmi', 'Byculla', 'Mazgaon',
  'Sea', 'Mumbai Central', 'Worli', 'Matunga', 'Sion', 'Wadala', 'Sewri', 'Chinchpokli', 'Reay Road',
  'Mahim', 'Dadar West', 'Dadar East', 'Kurla', 'Vidyavihar', 'Ghatkopar', 'BKC', 'Kalina', 'Santacruz',
  'Sea', 'Bandra', 'Bandra West', 'Khar', 'Chembur', 'Powai', 'Hiranandani', 'Chembur East', 'Navi Mumbai',
  'Juhu Beach', 'Juhu', 'Versova', 'Lokhandwala', 'Saki Naka', 'Ghatkopar East', 'Vikhroli', 'Kanjurmarg', 'Nahur',
  'Amboli', 'Jogeshwari', 'Andheri West', 'Andheri East', 'Marol', 'Powai Lake', 'Chandivali', 'Bhandup', 'Mulund',
  'Malvani', 'Malad West', 'Goregaon', 'Kandivali', 'Borivali', 'Deonar', 'Govandi', 'Mulund East', 'Thane Creek',
  'Erangal', 'Kandivali West', 'Borivali West', 'Dahisar', 'Mira Road', 'Thane West', 'Wagle Estate', 'Thane', 'Kopar Khairane',
  'Madh Island', 'Marve', 'Manori', 'Vasai', 'Nallasopara', 'Vashi', 'Sanpada', 'Nerul', 'Belapur',
]

/**
 * Is this (row, col) cell open sea rather than Mumbai land?
 * Column 0 is the Arabian Sea for the southern rows; row 3 / col 1 sits in
 * Mahim Bay; row 7 / col 8 is the Thane Creek channel.
 */
export function isSeaCell(r, c) {
  if (c === 0 && r <= 6) return true
  if (c === 1 && r === 3) return true
  if (c === 8 && r === 7) return true
  return false
}

/** [lat, lon] centre of a grid cell — the same convention the backend uses. */
export function cellCenter(r, c) {
  return [M_LAT_MIN + (r + 0.5) * M_CELL_LAT, M_LON_MIN + (c + 0.5) * M_CELL_LON]
}

/** "C40" -> 39. Returns null for anything that is not a cell id. */
export function cellIdToIndex(cellId) {
  if (typeof cellId !== 'string' || !/^C\d+$/.test(cellId)) return null
  const idx = parseInt(cellId.slice(1), 10) - 1
  return idx >= 0 && idx < M_ROWS * M_COLS ? idx : null
}

/** 39 -> "Powai". Falls back to a neutral cell label. */
export function cellName(index) {
  if (index === null || index === undefined) return '—'
  return CELL_NAMES[index] || `Cell ${index + 1}`
}
