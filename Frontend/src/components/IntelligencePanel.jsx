/**
 * IntelligencePanel — Forecast, skill and response tools.
 * ======================================================
 * Four compact cards so the whole view fits without scrolling:
 *   1. the 6-hour nowcast the system exists to produce,
 *   2. how accurate that forecast is on the held-out test window,
 *   3. how confident the alert is and what drove it,
 *   4. the field tools (citizen validation, evacuation, what-if).
 *
 * The three field tools share one card behind tabs: they are used one at a
 * time, so stacking them would only push the forecast off screen.
 */
import { useEffect, useMemo, useState, useCallback } from 'react'
import { STATE, riskState, fmt } from '../utils/helpers'
import { cellIdToIndex, cellCenter } from '../utils/mumbaiGrid'
import { api } from '../utils/api'
import WhatIfChatbot from './WhatIfChatbot'

export default function IntelligencePanel({ timestep, aiData, selectedCell, mode }) {
  // The national view carries live rainfall only; the forecast pipeline,
  // alerts and routing below are the Mumbai pilot's.
  if (mode === 'india') {
    return (
      <div className="panel-stack">
        <AccuracySection />
        <section className="card muted-card">
          <header className="card-head"><h3>🏙️ Mumbai Pilot</h3></header>
          <p className="hint" style={{ marginTop: 0 }}>
            The 6-hour flood nowcast, hazard alerts, citizen validation and evacuation
            routing run on the Mumbai pilot grid. Switch to <b>Mumbai Replay</b> to see them.
          </p>
        </section>
      </div>
    )
  }

  if (!aiData) return <div className="panel-loading">Waiting for the forecast pipeline…</div>

  return (
    <div className="panel-stack">
      <NowcastSection aiData={aiData} />
      <AccuracySection aiData={aiData} />
      <ResponseTools selectedCell={selectedCell} timestep={timestep} />
    </div>
  )
}

/* ═══ 1. NEXT 6 HOURS ═══ */
function NowcastSection({ aiData }) {
  const nowcast = aiData?.nowcast || {}
  const forecasts = nowcast.forecasts || []
  const trend = nowcast.trend || 'steady'

  const hours = useMemo(() => {
    if (forecasts.length === 0) return []
    return forecasts.slice(0, 6).map((f, i) => ({
      hour: i + 1,
      rain: f.predicted_avg_rainfall_mm_hr ?? 0,
      level: f.predicted_risk_level || 'LOW',
    }))
  }, [forecasts])

  if (hours.length === 0) {
    return (
      <section className="card muted-card">
        <header className="card-head"><h3>⏰ Next 6 Hours</h3></header>
        <p className="hint" style={{ marginTop: 0 }}>Not enough history at this step to issue a forecast.</p>
      </section>
    )
  }

  const maxRain = Math.max(10, ...hours.map((h) => h.rain))
  const peakIdx = hours.reduce((best, h, i) => (h.rain > hours[best].rain ? i : best), 0)
  const trendState = trend === 'intensifying' ? STATE.high : trend === 'weakening' ? STATE.low : STATE.neutral
  const trendLabel = trend === 'intensifying' ? 'Intensifying' : trend === 'weakening' ? 'Weakening' : 'Steady'

  return (
    <section className="card">
      <header className="card-head">
        <h3>⏰ Next 6 Hours</h3>
        <span className="tag" style={{ background: trendState.tint, color: trendState.text }}>{trendLabel}</span>
      </header>

      <div className="bar-chart">
        {hours.map((h, i) => {
          const state = riskState(levelToScore(h.level))
          const height = Math.max(6, (h.rain / maxRain) * 100)
          return (
            <div className={`bar-col ${i === peakIdx ? 'is-peak' : ''}`} key={i}>
              <span className="bar-value" style={{ color: state.text }}>{fmt(h.rain, 0)}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ height: `${height}%`, background: state.fill }} />
              </div>
              <span className="bar-label">+{h.hour}h</span>
              <span className="bar-note" style={{ color: state.text }}>
                {i === peakIdx ? 'PEAK' : levelShort(h.level)}
              </span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

// Compact label so the stage name fits under a bar in the panel width.
function levelShort(level) {
  const s = String(level || '').toUpperCase()
  if (s === 'CRITICAL' || s === 'SEVERE') return 'CRIT'
  if (s === 'MEDIUM' || s === 'MODERATE') return 'MED'
  return s || '—'
}

function levelToScore(level) {
  const s = String(level || '').toUpperCase()
  if (s === 'CRITICAL' || s === 'SEVERE') return 90
  if (s === 'HIGH') return 65
  if (s === 'MEDIUM' || s === 'MODERATE') return 40
  return 15
}

/* ═══ 2. ACCURACY & CONFIDENCE ═══
   Skill on the held-out window and the confidence attached to this step are two
   readings of the same question — how much to trust the forecast — so they share
   a card: skill on top, confidence for the current step underneath. */
function AccuracySection({ aiData }) {
  const [showModel, setShowModel] = useState(false)

  return (
    <section className="card">
      <header className="card-head">
        <h3>🎯 Forecast Confidence</h3>
        <span className="tag" style={{ background: STATE.info.tint, color: STATE.info.text }}>this step</span>
      </header>

      {/* Anything that moves between steps comes first. */}
      <ConfidenceBody aiData={aiData} />

      {/* The skill scorecard is a property of the trained models, not of the
          current timestep, so it lives behind a disclosure at the foot of the
          card. As the headline it made the whole card look frozen. */}
      <button
        type="button"
        className={`model-toggle ${showModel ? 'open' : ''}`}
        aria-expanded={showModel}
        onClick={() => setShowModel((v) => !v)}
      >
        <span className="model-toggle-caret">▶</span>
        Model skill · held-out test window
        <span className="card-note">{showModel ? 'hide' : 'fixed · show'}</span>
      </button>
      <ForecastSkillBody expanded={showModel} />
    </section>
  )
}

function ForecastSkillBody({ expanded }) {
  const [skill, setSkill] = useState(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let alive = true
    api.metrics.skill().then((d) => { if (alive) setSkill(d) }).catch(() => { if (alive) setError(true) })
    return () => { alive = false }
  }, [])

  if (error) return <p className="hint" style={{ marginTop: 0 }}>Evaluation report unavailable.</p>
  if (!skill) return <div className="panel-loading">Loading accuracy report…</div>

  const flood = (skill.hazard_skill || []).filter((r) => r.hazard === 'Flood')
  const at2h = flood.find((r) => r.lead_time === '+2h') || flood[0] || {}
  const agg = skill.probabilistic?.aggregate || {}
  const leads = skill.nowcast?.per_lead_time || {}

  return (
    <>
      {/* One compact line, always visible, so the model's skill is never
          hidden behind the disclosure. */}
      <div className="score-summary">
        <span>Detection (POD) <b>{pctOrDash(at2h.pod)}</b></span>
        <span>False alarms <b>{pctOrDash(at2h.far)}</b></span>
        <span>Skill (CSI) <b>{pctOrDash(at2h.csi)}</b></span>
        <span>Brier <b>{numOrDash(agg.brier_score, 3)}</b></span>
        <span>ECE <b>{numOrDash(agg.expected_calibration_error_ece, 3)}</b></span>
        <span>Coverage <b>{pctOrDash(agg.conformal_coverage_pct)}</b> of 90%</span>
      </div>

      {expanded && (
        <>
          <div className="strip-title">Mean absolute error (mm/hr) by lead · vs persistence</div>
          <div className="lead-grid">
            {['+1h', '+2h', '+3h', '+4h', '+5h', '+6h'].map((key) => {
              const row = leads[key]
              if (!row) return null
              const s = row.mae_skill_vs_persistence_pct
              const state = s > 0 ? STATE.low : STATE.neutral
              return (
                <div className="lead-cell" key={key}>
                  <div className="lead-head">
                    <span className="lead-name">{key}</span>
                    <span className="lead-mae" style={{ color: state.text }}>{numOrDash(row.mae_mm_hr, 1)}</span>
                  </div>
                  <span className="lead-sub" title={`Persistence MAE ${numOrDash(row.persistence_mae_mm_hr, 1)} mm/hr`}>
                    vs {numOrDash(row.persistence_mae_mm_hr, 1)}
                    {s !== null && s !== undefined && ` · ${s > 0 ? '+' : ''}${fmt(s, 0)}%`}
                  </span>
                </div>
              )
            })}
          </div>
        </>
      )}
    </>
  )
}

function pctOrDash(v) {
  return v === null || v === undefined ? '—' : `${fmt(v, 0)}%`
}
function numOrDash(v, d) {
  return v === null || v === undefined ? '—' : fmt(v, d)
}

/* ═══ 3. CONFIDENCE & DRIVERS ═══ */

/** The per-cell driver names the model emits, given an icon for the panel. */
const DRIVER_ICONS = {
  'High Rainfall Intensity': '🌧️',
  'Convective Instability (CAPE)': '⚡',
  'Cold Cloud-Top Temperature': '☁️',
  'Soil Super-Saturation': '💧',
  'Low-Elevation Depression': '🏔️',
  'High Tide Drainage Lockout': '🌊',
}

/**
 * Why this step looks the way it does.
 *
 * The per-cell explanations carry a contribution computed from that step's own
 * feature values, so summing them gives a driver ranking that genuinely moves
 * through the event. (The model-level weight map also in the payload is a fixed
 * property of the architecture, which is why it never changed.)
 */
function useStepDrivers(aiData) {
  return useMemo(() => {
    const cells = aiData?.xai_explanation?.cell_explanations || []
    const totals = {}
    for (const cell of cells) {
      for (const d of cell.top_drivers || []) {
        totals[d.factor] = (totals[d.factor] || 0) + (d.contribution || 0)
      }
    }
    const ranked = Object.entries(totals).sort((a, b) => b[1] - a[1])
    const total = ranked.reduce((sum, [, v]) => sum + v, 0) || 1
    return {
      cellsAffected: cells.filter((c) => (c.top_drivers || []).length > 0).length,
      totalCells: cells.length,
      drivers: ranked.slice(0, 3).map(([name, value]) => ({
        name,
        icon: DRIVER_ICONS[name] || '•',
        share: (value / total) * 100,
      })),
    }
  }, [aiData])
}

/** Input agreement, 0–1: a high value means the forecast sources concur. */
const TRUST_COMPONENTS = [
  { key: 'cross_source_agreement', label: 'Source agreement' },
  { key: 'depth_severity_agreement', label: 'Depth vs severity' },
  { key: 'storm_detection_consistency', label: 'Storm cell consistency' },
]

function agreementState(v) {
  if (v >= 0.75) return STATE.low
  if (v >= 0.5) return STATE.moderate
  return STATE.severe
}

/**
 * How sure the model is, and why it said what it said. The calibrated interval,
 * the input agreement and the driver ranking answer the same question from
 * three sides, so they share one card instead of competing for space.
 */
function ConfidenceBody({ aiData }) {
  const trust = aiData?.trust_score || {}
  const score = trust.trust_score ?? 0
  const conformal = trust.conformal_prediction || {}
  const lower = (conformal.lower_bound || [])[0]
  const upper = (conformal.upper_bound || [])[0]
  const margin = trust.uncertainty_margin_pct
  const comp = trust.components || {}

  const state = score >= 80 ? STATE.low : score >= 60 ? STATE.moderate : STATE.high
  const label = score >= 80 ? 'High confidence' : score >= 60 ? 'Moderate confidence' : 'Low confidence'

  const { drivers, cellsAffected, totalCells } = useStepDrivers(aiData)
  const maxShare = Math.max(...drivers.map((d) => d.share), 0.0001)

  if (!trust.trust_score && drivers.length === 0) return null

  return (
    <>
      <div className="sub-head">
        <span>How much to trust the forecast</span>
        <span className="tag" style={{ background: state.tint, color: state.text }}>{label}</span>
      </div>

      <div className="gauge-row">
        <svg viewBox="0 0 120 68" className="gauge-svg" role="img" aria-label={`Trust score ${fmt(score, 0)} percent`}>
          <path d="M12 58 A48 48 0 0 1 108 58" fill="none" stroke="#e6e9f0" strokeWidth="10" strokeLinecap="round" />
          <path
            d="M12 58 A48 48 0 0 1 108 58"
            fill="none"
            stroke={state.fill}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${(score / 100) * 150.8} 151`}
          />
          <text x="60" y="50" textAnchor="middle" className="gauge-value" fill={state.text}>{fmt(score, 0)}</text>
          <text x="60" y="63" textAnchor="middle" className="gauge-caption">trust / 100</text>
        </svg>
        <div className="gauge-side">
          <div className="gauge-line">
            <span className="gauge-line-label">Calibrated risk interval</span>
            <span className="gauge-line-value">
              {lower !== undefined && upper !== undefined ? `${fmt(lower, 0)} – ${fmt(upper, 0)}` : '—'}
            </span>
          </div>
          <div className="gauge-line">
            <span className="gauge-line-label">Uncertainty margin</span>
            <span className="gauge-line-value">{margin !== undefined ? `±${fmt(margin, 0)}%` : '—'}</span>
          </div>
        </div>
      </div>

      {/* These three move from step to step — they are the measured agreement
          between the inputs behind this step's forecast, which is why they
          belong here rather than in the fixed scorecard below. */}
      {TRUST_COMPONENTS.some((c) => comp[c.key] !== undefined) && (
        <div className="trust-grid">
          {TRUST_COMPONENTS.map((c) => {
            const v = comp[c.key]
            if (v === undefined) return null
            const s = agreementState(v)
            return (
              <div className="trust-cell" key={c.key}>
                <span className="trust-cell-label">{c.label}</span>
                <span className="trust-cell-value" style={{ color: s.text }}>{fmt(v * 100, 0)}%</span>
              </div>
            )
          })}
        </div>
      )}

      {drivers.length > 0 && (
        <>
          <div className="strip-title">
            What is driving it · {cellsAffected} of {totalCells} cells showing a risk factor
          </div>
          <div className="driver-list">
            {drivers.map((d, i) => (
              <div className="driver-row" key={d.name}>
                <span className="driver-rank">{i + 1}</span>
                <span className="driver-name">{d.icon} {d.name}</span>
                <div className="driver-track">
                  <div className="driver-fill" style={{ width: `${(d.share / maxShare) * 100}%`, background: riskState((d.share / maxShare) * 100).fill }} />
                </div>
                <span className="driver-value">{fmt(d.share, 0)}%</span>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}

/* ═══ 4. RESPONSE TOOLS (tabbed) ═══ */

const TOOLS = [
  {
    key: 'citizen',
    icon: '📢',
    label: 'Citizen report',
    hint: 'A ground report is checked against the model before it is trusted.',
  },
  {
    key: 'evacuation',
    icon: '🗺️',
    label: 'Evacuation',
    hint: 'Safest driving route out of the chosen area, avoiding the deepest water.',
  },
  {
    key: 'whatif',
    icon: '💬',
    label: 'What-if',
    hint: 'Ask how the forecast changes when an input changes.',
  },
]

function ResponseTools({ selectedCell, timestep }) {
  const [tool, setTool] = useState('citizen')
  const active = TOOLS.find((t) => t.key === tool)

  return (
    <section className="card">
      {/* The tab row is the card header — repeating the active tool name in a
          title above it said the same thing twice. */}
      <div className="tool-tabs" role="tablist">
        {TOOLS.map((t) => (
          <button
            key={t.key}
            role="tab"
            aria-selected={tool === t.key}
            className={`tool-tab ${tool === t.key ? 'active' : ''}`}
            onClick={() => setTool(t.key)}
          >
            <span className="tool-tab-icon">{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>
      <p className="tool-hint">{active.hint}</p>

      {/* The tool body gets its own scroll box so a long route list or chat
          cannot push the forecast cards off screen. */}
      <div className="tools-body">
        {tool === 'citizen' && <CitizenReportTool selectedCell={selectedCell} />}
        {tool === 'evacuation' && <EvacuationTool selectedCell={selectedCell} />}
        {tool === 'whatif' && <WhatIfChatbot timestep={timestep} />}
      </div>
    </section>
  )
}

/* Field report validation */
function CitizenReportTool({ selectedCell }) {
  const [text, setText] = useState('')
  const [state, setState] = useState({ loading: false, result: null, error: null })

  const submit = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    setState({ loading: true, result: null, error: null })
    try {
      // Report coordinates: the selected map cell if there is one, else central Mumbai.
      const idx = cellIdToIndex(selectedCell)
      const [lat, lon] = idx === null ? [19.07, 72.88] : cellCenter(Math.floor(idx / 9), idx % 9)
      const result = await api.ai.crowdReport(text.trim(), lat, lon)
      setState({ loading: false, result, error: null })
    } catch (err) {
      setState({ loading: false, result: null, error: err.message || 'Verification failed' })
    }
  }

  // Short chip labels with the full report in the tooltip: a scrolling row of
  // half-truncated sentences read as broken text at this width.
  const samples = [
    { label: 'Flooded at Kurla stn', text: 'Severe flooding near Kurla station, water above knee level' },
    { label: 'Hindmata 2 ft deep', text: 'Water level 2 feet high at Hindmata, Dadar' },
    { label: 'Bandra West clear', text: 'Roads dry and clear at Bandra West' },
  ]

  const confirmed = state.result?.classification === 'CONFIRMS_FLOOD_ZONE'
  const resultState = confirmed ? STATE.high : STATE.info

  return (
    <>
      <form className="inline-form" onSubmit={submit}>
        <input
          className="text-input"
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe the flooding you can see…"
        />
        <button className="btn-primary" type="submit" disabled={state.loading || !text.trim()}>
          {state.loading ? '…' : 'Check'}
        </button>
      </form>

      <div className="chip-row">
        {samples.map((s) => (
          <button className="chip-btn" type="button" key={s.label} title={s.text} onClick={() => setText(s.text)}>
            {s.label}
          </button>
        ))}
      </div>

      {state.error && <p className="hint hint-warn">⚠️ {state.error}</p>}

      {state.result && (
        <div className="result-box" style={{ background: resultState.tint, borderColor: resultState.fill }}>
          <div className="result-head">
            <span style={{ color: resultState.text }}>
              {confirmed ? '✅ Confirms flooding' : 'ℹ️ No flooding indicated'}
            </span>
            <span className="result-conf">
              {Math.round((state.result.confidence || 0) * 100)}% confidence
            </span>
          </div>
          <p className="result-body">
            Location read as <b>{state.result.detected_location || 'Mumbai (general)'}</b>
            {state.result.flood_keywords?.length > 0 && ` · matched: ${state.result.flood_keywords.join(', ')}`}
          </p>
        </div>
      )}
    </>
  )
}

/* Evacuation routing */
function EvacuationTool({ selectedCell }) {
  const [origin, setOrigin] = useState(40)
  const [routes, setRoutes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  const load = useCallback(async (cell) => {
    setLoading(true)
    setError(false)
    try {
      setRoutes(await api.innovations.evacuationRoutes(cell, 'vehicle'))
    } catch {
      setError(true)
      setRoutes(null)
    }
    setLoading(false)
  }, [])

  useEffect(() => { load(origin) }, [origin, load])

  const primary = routes?.primary_route
  const alternates = (routes?.alternative_routes || []).slice(0, 2)

  return (
    <>
      <div className="inline-form">
        <select className="select-input" value={origin} onChange={(e) => setOrigin(parseInt(e.target.value, 10))}>
          <option value={36}>Juhu</option>
          <option value={27}>Bandra</option>
          <option value={19}>Dadar</option>
          <option value={38}>Andheri East</option>
          <option value={46}>Goregaon</option>
          <option value={30}>Kurla</option>
          <option value={21}>Sion</option>
          <option value={40}>Powai</option>
          <option value={9}>Parel</option>
          <option value={0}>Colaba</option>
        </select>
        <button className="btn-ghost" onClick={() => load(origin)} disabled={loading}>
          {loading ? '…' : 'Refresh'}
        </button>
      </div>

      {error && <p className="hint hint-warn">⚠️ Route service unavailable.</p>}

      <ul className="route-list">
        {primary && (
          <li className="route-row primary">
            <span className="route-main">
              <span className="route-badge">Recommended</span>
              <span className="route-dest">{primary.destination || 'Safe zone'}</span>
              <span className="route-time">{primary.estimated_time_min ?? '—'} min</span>
            </span>
            <span className="route-stats">
              {fmt(primary.distance_km, 1)} km · safety {primary.safety_score ?? '—'}%
              {' · '}max depth {fmt(primary.max_water_depth_cm, 0)} cm
            </span>
          </li>
        )}

        {alternates.map((r, i) => (
          <li className="route-row" key={i}>
            <span className="route-main">
              <span className="route-badge">Alt {i + 2}</span>
              <span className="route-dest">{r.destination || 'Safe zone'}</span>
              <span className="route-time">{r.estimated_time_min ?? '—'} min</span>
            </span>
            <span className="route-stats">
              {fmt(r.distance_km, 1)} km · safety {r.safety_score ?? '—'}%
              {r.max_water_depth_cm !== undefined && ` · max depth ${fmt(r.max_water_depth_cm, 0)} cm`}
            </span>
          </li>
        ))}
      </ul>

      {routes?.recommendation && <p className="hint">💡 {routes.recommendation}</p>}
    </>
  )
}
