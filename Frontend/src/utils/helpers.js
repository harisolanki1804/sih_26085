// Shared helpers and the dashboard's colour system.
//
// One pastel palette for the whole app: surfaces are muted, semantic states
// (low / moderate / high / severe) are soft tints, and any text drawn on a
// light surface uses the matching dark shade so it stays readable.

/** Semantic state colours: `fill` for bars and dots, `text` for numbers. */
export const STATE = {
  low: { fill: '#8fd3ae', text: '#2f7d55', tint: '#e8f7ee' },
  moderate: { fill: '#f0d391', text: '#8f6b13', tint: '#fdf4e2' },
  high: { fill: '#f5b98b', text: '#b2561f', tint: '#fdeee2' },
  severe: { fill: '#f09595', text: '#ad3535', tint: '#fdeaea' },
  info: { fill: '#a6c4ea', text: '#32568f', tint: '#eaf1fb' },
  neutral: { fill: '#c7ced8', text: '#5b6472', tint: '#f1f3f7' },
}

/** Risk band from a 0-100 score. */
export function riskState(score) {
  const s = Number(score)
  if (!isFinite(s)) return STATE.neutral
  if (s >= 75) return STATE.severe
  if (s >= 50) return STATE.high
  if (s >= 25) return STATE.moderate
  return STATE.low
}

export function riskFill(score) {
  return riskState(score).fill
}

export function riskLabel(score) {
  const s = Number(score)
  if (!isFinite(s)) return '—'
  if (s >= 75) return 'SEVERE'
  if (s >= 50) return 'HIGH'
  if (s >= 25) return 'MODERATE'
  return 'LOW'
}

/** Hazard families the system forecasts. */
export const HAZARDS = {
  THUNDERSTORM: { label: 'Thunderstorm', short: 'TS', icon: '⛈️', state: STATE.moderate },
  CLOUDBURST: { label: 'Cloudburst', short: 'CB', icon: '🌧️', state: STATE.info },
  FLASH_FLOOD: { label: 'Flash Flood', short: 'FF', icon: '🌊', state: STATE.severe },
  SEVERE_THUNDERSTORM: { label: 'Severe Thunderstorm', short: 'TS', icon: '⛈️', state: STATE.moderate },
}

export function hazardMeta(type) {
  return HAZARDS[type] || { label: String(type || 'Hazard').replace(/_/g, ' '), short: '⚠️', icon: '⚠️', state: STATE.moderate }
}

export function severityState(severity) {
  const s = String(severity || '').toUpperCase()
  if (s === 'CRITICAL' || s === 'SEVERE') return STATE.severe
  if (s === 'HIGH') return STATE.high
  if (s === 'MODERATE' || s === 'MEDIUM') return STATE.moderate
  return STATE.low
}

export function fmt(v, d = 1) {
  if (v === null || v === undefined || isNaN(v)) return '--'
  return Number(v).toFixed(d)
}

/**
 * Clock for a replay step.
 * The replay starts at 2022-07-05T00:00Z and advances one hour per step. The
 * backend's own step timestamps overflow their hour field once past 23:00, so
 * the clock is derived from the step index instead of parsed from that string.
 */
const REPLAY_START_UTC = Date.UTC(2022, 6, 5, 0, 0, 0)

export function stepClock(timestep) {
  const t = Number(timestep)
  if (!isFinite(t) || t < 1) return '--'
  const d = new Date(REPLAY_START_UTC + (t - 1) * 3600 * 1000)
  const day = String(d.getUTCDate()).padStart(2, '0')
  const month = d.toLocaleString('en-GB', { month: 'short', timeZone: 'UTC' })
  const hh = String(d.getUTCHours()).padStart(2, '0')
  return `${day} ${month} · ${hh}:00`
}

/**
 * The six phases of the replay, each with the step it ends on. Every label,
 * colour and timeline marker is derived from this table.
 */
export const PHASES = [
  { label: 'Pre-Event Baseline', upto: 12, state: STATE.low },
  { label: 'Convective Initiation', upto: 24, state: STATE.moderate },
  { label: 'Rapid Intensification', upto: 36, state: STATE.high },
  { label: 'Peak Storm Event', upto: 48, state: STATE.severe },
  { label: 'Sustained Deluge', upto: 60, state: STATE.high },
  { label: 'Recession & Recovery', upto: 72, state: STATE.moderate },
]

export function getPhase(t) {
  const step = Number(t)
  return PHASES.find((p) => step <= p.upto) || PHASES[PHASES.length - 1]
}

export function getPhaseLabel(t) {
  return getPhase(t).label
}

export function getPhaseState(t) {
  return getPhase(t).state
}

export function getPhaseColor(t) {
  return getPhaseState(t).text
}

/**
 * Alerts belonging to one replay step. The topbar counter and the alert list
 * both read this, so the two can never disagree about how many are active.
 */
export function stepAlerts(replayStepData, timestep) {
  const all = replayStepData?.alerts_generated
  if (!Array.isArray(all)) return []
  return all
    .filter((a) => a.timestep === undefined || a.timestep === timestep)
    .sort((a, b) => (b.risk_score_total || 0) - (a.risk_score_total || 0))
}

/**
 * Risk bands — the legend, the map heat field and the KPI tiles all read the
 * thresholds from here, so a score can never be described two ways.
 */
export const RISK_BANDS = [
  { label: 'Severe', range: '75 and above', min: 75, state: STATE.severe },
  { label: 'High', range: '50 – 75', min: 50, state: STATE.high },
  { label: 'Moderate', range: '25 – 50', min: 25, state: STATE.moderate },
  { label: 'Low', range: 'below 25', min: 0, state: STATE.low },
]

/**
 * IMD hourly rainfall classification (mm/hr) — the standard the drainage and
 * alerting thresholds are written against.
 */
export const RAIN_BANDS = [
  { label: 'Extremely heavy', range: 'above 64.5 mm/hr', min: 64.5, state: STATE.severe },
  { label: 'Very heavy', range: '15.5 – 64.5 mm/hr', min: 15.5, state: STATE.high },
  { label: 'Heavy', range: '7.5 – 15.5 mm/hr', min: 7.5, state: STATE.moderate },
  { label: 'Moderate', range: '2.5 – 7.5 mm/hr', min: 2.5, state: STATE.info },
  { label: 'Light', range: 'below 2.5 mm/hr', min: 0, state: STATE.low },
]

/** Rainfall band for a rate in mm/hr. */
export function rainBand(mm) {
  const v = Number(mm)
  if (!isFinite(v)) return RAIN_BANDS[RAIN_BANDS.length - 1]
  return RAIN_BANDS.find((b) => v >= b.min) || RAIN_BANDS[RAIN_BANDS.length - 1]
}

/**
 * Mumbai tide bands (metres). 4.6 m is the spring-tide mark above which the
 * outfalls cannot discharge, so it is the level that matters for flooding.
 */
export const TIDE_BANDS = [
  { label: 'High tide lock', range: 'above 4.6 m', min: 4.6, state: STATE.severe },
  { label: 'Rising', range: '3.6 – 4.6 m', min: 3.6, state: STATE.high },
  { label: 'Normal', range: 'below 3.6 m', min: 0, state: STATE.low },
]

export function tideBand(m) {
  const v = Number(m)
  if (!isFinite(v) || v <= 0) return TIDE_BANDS[TIDE_BANDS.length - 1]
  return TIDE_BANDS.find((b) => v >= b.min) || TIDE_BANDS[TIDE_BANDS.length - 1]
}

/**
 * Response guidance for each replay phase, keyed by the phase label so the
 * topbar clock and the side panel can never name the same step differently.
 */
export const PHASE_GUIDANCE = {
  'Pre-Event Baseline': {
    text: 'Conditions normal. Drains being monitored.',
    action: 'No action required',
  },
  'Convective Initiation': {
    text: 'Storm cells building over the city.',
    action: 'Clear drains, secure loose structures',
  },
  'Rapid Intensification': {
    text: 'Rainfall intensifying. Waterlogging likely.',
    action: 'Move vehicles off low-lying roads',
  },
  'Peak Storm Event': {
    text: 'Severe flooding risk in low-lying areas.',
    action: 'Move to high ground, follow advisories',
  },
  'Sustained Deluge': {
    text: 'Continuous heavy rain. Drains near capacity.',
    action: 'Avoid travel, prepare to evacuate',
  },
  'Recession & Recovery': {
    text: 'Rain easing. Residual waterlogging.',
    action: 'Check waterlogged roads before travel',
  },
}
