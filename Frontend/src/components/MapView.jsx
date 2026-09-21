import GeoMap from './GeoMap'
import { STATE, RISK_BANDS, RAIN_BANDS, stepAlerts } from '../utils/helpers'
import { cellName, M_ROWS, M_COLS, isSeaCell } from '../utils/mumbaiGrid'

export default function MapView({ aiData, replayStepData, selectedCell, onSelectCell, mode, timestep }) {
  const isMumbai = mode !== 'india'
  const bands = isMumbai ? RISK_BANDS : RAIN_BANDS

  const selected = resolveSelected(selectedCell, mode)
  const alerts = stepAlerts(replayStepData, timestep)

  return (
    <div className="map-wrap">
      <GeoMap
        aiData={aiData}
        selectedCell={selectedCell}
        onSelectCell={onSelectCell}
        mode={mode}
      />

      <div className="map-caption">
        <span className="map-caption-title">{isMumbai ? 'Flood risk map' : 'Rainfall intensity map'}</span>
        <span className="map-caption-sub">
          {isMumbai
            ? 'Mumbai pilot grid · 90 land cells'
            : 'National grid · live rainfall'}
        </span>
      </div>

      {selected && (
        <div className="map-selection" style={{ borderColor: STATE.info.fill }}>
          📍 {selected}
          <span className="map-selection-hint">click again to clear</span>
        </div>
      )}

      {isMumbai && alerts.length > 0 && (
        <div className="map-alert-badge" style={{ borderColor: STATE.severe.fill }}>
          🚨 <b>{alerts.length}</b> alert{alerts.length === 1 ? '' : 's'} at this step
        </div>
      )}

      <div className="map-legend">
        <span className="map-legend-title">{isMumbai ? 'Flood risk score' : 'Rainfall rate'}</span>
        <div className="map-legend-items">
          {bands.map((b) => (
            <span className="map-legend-item" key={b.label}>
              <span className="dot" style={{ background: b.state.fill }} />
              <b>{b.label}</b>
              <em>{b.range}</em>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

/** "C40" -> "Powai (C40)". Sea cells have no land locality to report. */
function resolveSelected(cellId, mode) {
  if (!cellId || mode === 'india') return null
  const match = /^C(\d+)$/.exec(cellId)
  if (!match) return null
  const index = parseInt(match[1], 10) - 1
  if (index < 0 || index >= M_ROWS * M_COLS) return null
  const r = Math.floor(index / M_COLS)
  const c = index % M_COLS
  if (isSeaCell(r, c)) return `${cellId} · open sea`
  return `${cellName(index)} · ${cellId}`
}
