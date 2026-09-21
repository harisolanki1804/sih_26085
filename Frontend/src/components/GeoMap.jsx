/**
 * GeoMap — OpenStreetMap base with the hazard layer drawn on top.
 * =============================================================
 * Mumbai replay: a continuous pastel heat field over the 90-cell pilot grid,
 * matching the risk bands used by the panels and the legend.
 *
 * Grid geometry and locality names live in utils/mumbaiGrid, so the map, the
 * panels and the alert list can never disagree about what a cell is called.
 *
 * The view is only repositioned when the map mode changes — clicking a cell or
 * scrubbing the timeline redraws the layer without throwing away the operator's
 * pan and zoom.
 */
import { useEffect, useRef } from 'react'
import { fmt, riskLabel, riskState } from '../utils/helpers'
import { isInsideIndia } from '../utils/indiaBoundary'
import { loadIndiaBoundary } from '../utils/indiaGeoJson'
import {
  M_ROWS, M_COLS, M_CENTER, cellCenter, cellName, isSeaCell,
} from '../utils/mumbaiGrid'
import {
  ROWS, COLS, LAT_MIN, LON_MIN, CELL_LAT, CELL_LON, CITY_NAMES,
} from '../utils/indiaGrid'

// ─── Tiles ───────────────────────────────────────────────────
const TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const TILE_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

// Heat stops (intensity 0–1) — soft tints, readable on the OSM base.
const HEAT_GRADIENT = {
  0.0: '#8fd3ae',
  0.25: '#f0d391',
  0.5: '#f5b98b',
  0.75: '#f09595',
  1.0: '#e07474',
}

function riskToIntensity(risk) {
  if (!risk || risk < 5) return 0
  return Math.min(1, Math.max(0.25, risk / 100))
}

/** One label/value line in the cell hover card. */
function row(label, value) {
  return `<div class="map-tip-row"><span>${label}</span>${value}</div>`
}

/**
 * The cell under the cursor gets a visible ring.
 * Without it the pick targets are invisible and adjacent cells sit ~4 km apart,
 * so the hover card floated over the map with nothing to attach it to.
 */
const HOVER_STYLE = {
  color: '#32568f', weight: 2, opacity: 1, fillColor: '#ffffff', fillOpacity: 0.3,
}
const REST_STYLE = {
  color: '#000', weight: 0, opacity: 0, fillOpacity: 0,
}

/**
 * Add a heat layer only when the map is actually measurable.
 * leaflet.heat reads back its canvas with getImageData, which throws an
 * IndexSizeError while the container has zero width — that exception used to
 * propagate out of the render effect and blank the whole dashboard.
 */
function addHeatLayer(map, L, points, options) {
  const size = map.getSize()
  if (!points.length || !size || size.x < 2 || size.y < 2) return null
  try {
    return L.heatLayer(points, options)
  } catch (err) {
    console.error('Heat layer unavailable:', err)
    return null
  }
}

export default function GeoMap({ aiData, selectedCell, onSelectCell, mode }) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const layersRef = useRef([])
  const drawnModeRef = useRef(null)

  useEffect(() => {
    if (mapRef.current) return
    const el = containerRef.current
    if (!el || !window.L) return
    const L = window.L
    const map = L.map(el, {
      center: M_CENTER,
      zoom: 12,
      zoomControl: true,
      attributionControl: true,
      minZoom: 4,
      maxZoom: 19,
    })
    L.tileLayer(TILE_URL, { maxZoom: 19, attribution: TILE_ATTR }).addTo(map)
    mapRef.current = map
    setTimeout(() => map.invalidateSize(), 300)
  }, [])

  useEffect(() => {
    const map = mapRef.current
    const L = window.L
    if (!map || !L) return

    layersRef.current.forEach((l) => { try { map.removeLayer(l) } catch { /* already gone */ } })
    layersRef.current = []

    // A failing map layer must never take the dashboard down with it.
    try {
      if (mode === 'india') {
        drawIndia(map, L, aiData, layersRef.current)
      } else {
        drawMumbai(map, L, aiData, selectedCell, onSelectCell, layersRef.current)
      }
    } catch (err) {
      console.error('Map layer draw failed:', err)
    }

    // Reposition only when the mode actually changes.
    if (drawnModeRef.current !== mode) {
      if (mode === 'india') map.setView([22.5, 78.0], 5)
      else map.setView(M_CENTER, 12)
      drawnModeRef.current = mode
    }
    map.invalidateSize()
  }, [aiData, selectedCell, mode, onSelectCell])

  useEffect(() => {
    const obs = new ResizeObserver(() => mapRef.current?.invalidateSize())
    if (containerRef.current?.parentElement) obs.observe(containerRef.current.parentElement)
    return () => obs.disconnect()
  }, [])

  return (
    <div className="map-canvas">
      <div ref={containerRef} className="leaflet-host" />
    </div>
  )
}

/* ─── National rainfall view ───────────────────────────────── */
async function drawIndia(map, L, aiData, layers) {
  try {
    const geojson = await loadIndiaBoundary()
    layers.push(L.geoJSON(geojson, {
      style: { color: '#9db6d8', weight: 1.5, fillColor: '#dfe8f5', fillOpacity: 0.35 },
      interactive: false,
    }).addTo(map))
  } catch { /* boundary is decorative */ }

  const cellMap = {}
  ;(aiData?.cells || []).forEach((c) => { cellMap[c.cell_index] = c })

  const heat = []
  const labels = []
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c
      const lat = LAT_MIN + r * CELL_LAT + CELL_LAT / 2
      const lon = LON_MIN + c * CELL_LON + CELL_LON / 2
      if (!isInsideIndia(lat, lon)) continue
      const rain = (cellMap[idx] || {}).rainfall_1h_mm ?? 0
      if (rain < 1) continue
      heat.push([lat, lon, Math.min(1, Math.max(0.15, rain / 80))])
      const city = CITY_NAMES[`${r},${c}`]
      if (city && rain > 3) {
        labels.push(L.marker([lat, lon], {
          icon: L.divIcon({
            className: '',
            html: `<div style="font-family:Inter,sans-serif;text-align:center;pointer-events:none;transform:translate(-50%,-50%);line-height:1.25">
              <div style="font-size:13px;font-weight:700;color:#243244;white-space:nowrap">${city}</div>
              <div style="font-size:12px;font-weight:700;color:#b2561f;white-space:nowrap">${fmt(rain, 0)} mm</div>
            </div>`,
            iconSize: [0, 0], iconAnchor: [0, 0],
          }),
          interactive: false,
          zIndexOffset: 2000,
        }))
      }
    }
  }
  const heatLayer = addHeatLayer(map, L, heat, {
    radius: 28, blur: 26, maxZoom: 20, minOpacity: 0.45, gradient: HEAT_GRADIENT,
  })
  if (heatLayer) { heatLayer.addTo(map); layers.push(heatLayer) }

  labels.forEach((m) => { m.addTo(map); layers.push(m) })
}

/* ─── Mumbai pilot: hazard field + click targets ───────────── */
function drawMumbai(map, L, aiData, selectedCell, onSelectCell, layers) {
  const riskMap = {}
  ;(aiData?.risk_heatmap?.heatmap || []).forEach((h) => { riskMap[h.cell_index] = h })
  const floodMap = {}
  ;(aiData?.flood_depth?.node_estimates || []).forEach((n) => { floodMap[n.cell_index] = n })
  const hazardMap = {}
  ;(aiData?.multi_hazard?.cell_predictions || []).forEach((p) => { hazardMap[p.cell_index] = p })

  const heatPts = []
  const overlays = []

  for (let r = 0; r < M_ROWS; r++) {
    for (let c = 0; c < M_COLS; c++) {
      const idx = r * M_COLS + c
      if (isSeaCell(r, c)) continue
      const [lat, lon] = cellCenter(r, c)

      const hazard = hazardMap[idx]
      const riskRaw = riskMap[idx]?.pixel_risk_score ?? 0
      const risk = riskRaw > 0 ? riskRaw : Math.min(hazard?.severity_score ?? 0, 70)
      const name = cellName(idx)
      const intensity = riskToIntensity(risk)
      const dominant = riskMap[idx]?.dominant_risk_class || hazard?.dominant_hazard || 'SAFE'
      const dominantLabel = String(dominant).toUpperCase() === 'SAFE'
        ? ''
        : dominant.replace(/_/g, ' ').toLowerCase()

      if (intensity > 0) heatPts.push([lat, lon, intensity])

      const picker = L.circleMarker([lat, lon], {
        radius: 17, fillOpacity: 0, opacity: 0, color: '#000', weight: 0, interactive: true,
      })
      // One labelled row per reading — a wrapped paragraph of numbers was
      // unreadable at hover size.
      const band = riskState(risk)
      const rows = [
        row('Risk score', `<b>${fmt(risk, 0)} / 100</b>`),
        dominantLabel ? row('Dominant hazard', `<b>${dominantLabel}</b>`) : '',
        hazard ? row('Thunderstorm', `<b>${fmt((hazard.thunderstorm_prob || 0) * 100, 0)}%</b>`) : '',
        hazard ? row('Cloudburst', `<b>${fmt((hazard.cloudburst_prob || 0) * 100, 0)}%</b>`) : '',
        hazard ? row('Flash flood', `<b>${fmt((hazard.flash_flood_prob || 0) * 100, 0)}%</b>`) : '',
        floodMap[idx] ? '<div class="map-tip-sep"></div>' : '',
        floodMap[idx] ? row('Water depth', `<b>${fmt(floodMap[idx].water_depth_cm, 0)} cm</b>`) : '',
        floodMap[idx] ? row('Drainage', `<b>${floodMap[idx].is_overflow_node ? 'overflowing' : 'within capacity'}</b>`) : '',
      ].join('')

      picker.bindTooltip(
        `<div class="map-tip">`
          + `<div class="map-tip-head"><span>${name}</span>`
          + `<span class="map-tip-band" style="color:${band.text}">${riskLabel(risk)}</span></div>`
          + rows
          + `<div class="map-tip-id">Cell C${idx + 1}</div>`
          + `</div>`,
        { direction: 'top', className: 'hazard-tooltip', opacity: 1, offset: [0, -6] }
      )
      picker.on('mouseover', () => { picker.setStyle(HOVER_STYLE); picker.bringToFront() })
      picker.on('mouseout', () => picker.setStyle(REST_STYLE))
      picker.on('click', () => onSelectCell(selectedCell === `C${idx + 1}` ? null : `C${idx + 1}`))
      overlays.push(picker)

      if (floodMap[idx]?.is_overflow_node && (hazard?.predicted_depth_cm ?? 0) > 20) {
        overlays.push(L.marker([lat + 0.004, lon], {
          icon: L.divIcon({
            className: '',
            html: '<div style="font-size:15px;line-height:1">⚠️</div>',
            iconSize: [20, 20], iconAnchor: [10, 20],
          }),
          interactive: false, zIndexOffset: 3000,
        }))
      }
    }
  }

  const heatLayer = addHeatLayer(map, L, heatPts, {
    radius: 40, blur: 34, maxZoom: 20, minOpacity: 0.55, max: 1, gradient: HEAT_GRADIENT,
  })
  if (heatLayer) { heatLayer.addTo(map); layers.push(heatLayer) }

  if (selectedCell && /^C\d+$/.test(selectedCell)) {
    const idx = parseInt(selectedCell.slice(1), 10) - 1
    const r = Math.floor(idx / M_COLS), c = idx % M_COLS
    if (!isSeaCell(r, c)) {
      const [lat, lon] = cellCenter(r, c)
      layers.push(L.circleMarker([lat, lon], {
        radius: 18, fillColor: 'rgba(255,255,255,0.25)', fillOpacity: 0.8,
        color: '#32568f', weight: 3, dashArray: '5 4',
      }).addTo(map))
    }
  }

  overlays.forEach((l) => { l.addTo(map); layers.push(l) })
}
