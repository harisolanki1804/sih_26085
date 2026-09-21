/**
 * LeftPanel — the operational picture.
 * ====================================
 * Only the indicators an early-warning operator acts on:
 *   1. how bad it is now (risk, water depth, alerts, storm cells),
 *   2. what the forcing is (rainfall, tide, drainage overflow),
 *   3. which localities are alerting,
 *   4. what response phase the city is in.
 *
 * Every band and colour comes from utils/helpers, so a score is described the
 * same way here, on the map legend and in the phase card.
 */
import { useState } from 'react'
import {
  fmt, riskState, riskLabel, severityState, hazardMeta, stepAlerts,
  rainBand, tideBand, getPhase, getPhaseLabel, getPhaseState, PHASE_GUIDANCE, PHASES, HAZARDS,
} from '../utils/helpers'
import { KPI, StatRow } from './Tiles'

export default function LeftPanel({ timestep, aiData, replayStepData, mode, liveSummary }) {
  if (mode === 'india') return <NationalOverview summary={liveSummary} />

  return (
    <>
      <SituationOverview aiData={aiData} replayStepData={replayStepData} timestep={timestep} />
      <HazardOutlook aiData={aiData} />
      <AlertsSection timestep={timestep} alerts={replayStepData?.alerts_generated} />
      <ResponseStatus timestep={timestep} />
    </>
  )
}

/* ═══ 1. CURRENT SITUATION ═══ */

function SituationOverview({ aiData, replayStepData, timestep }) {
  const risk = aiData?.risk_heatmap || {}
  const flood = aiData?.flood_depth || {}
  const storm = aiData?.storm_cells || {}

  const meanRisk = risk.mean_pixel_risk ?? aiData?.physics_risk?.avg_physics_risk ?? 0
  const maxRisk = risk.max_pixel_risk ?? meanRisk
  const maxDepth = flood.max_water_depth_cm ?? 0
  const stormCount = storm.total_cells_detected ?? 0
  const overflowCount = flood.overflow_nodes_count ?? 0

  const alertCount = stepAlerts(replayStepData, timestep).length
  const rainfall = replayStepData?.avg_rainfall_1h_mm ?? 0
  const tideHeight = replayStepData?.tide_height_m ?? 0

  const riskLevel = riskState(maxRisk)
  const depthState = maxDepth >= 50 ? riskState(80) : maxDepth >= 20 ? riskState(55) : riskState(10)
  const alertState = alertCount === 0 ? riskState(5) : alertCount >= 6 ? riskState(80) : riskState(55)
  const stormState = stormCount === 0 ? riskState(5) : stormCount >= 5 ? riskState(80) : riskState(55)
  const rain = rainBand(rainfall)
  const tide = tideBand(tideHeight)

  return (
    <section className="card">
      <header className="card-head">
        <h3>📊 Current Situation</h3>
        <span className="tag" style={{ background: riskLevel.tint, color: riskLevel.text }}>
          Grid avg {fmt(meanRisk, 0)}/100
        </span>
      </header>

      <div className="kpi-grid">
        <KPI
          icon="⚠️"
          label="Flood risk"
          value={fmt(maxRisk, 0)}
          suffix="/100"
          sub={riskLabel(maxRisk)}
          state={riskLevel}
        />
        <KPI
          icon="🌊"
          label="Max water depth"
          value={fmt(maxDepth, 0)}
          suffix="cm"
          sub={maxDepth >= 20 ? 'Above safe level' : 'Within capacity'}
          state={depthState}
        />
        <KPI
          icon="🚨"
          label="Localities alerting"
          value={fmt(alertCount, 0)}
          state={alertState}
        />
        <KPI
          icon="⛈️"
          label="Storm cells"
          value={fmt(stormCount, 0)}
          state={stormState}
        />
      </div>

      <div className="stat-rows">
        <StatRow icon="🌧️" label="Rainfall" value={`${fmt(rainfall, 1)} mm/hr`} band={rain} />
        <StatRow icon="🌊" label="Tide" value={`${fmt(tideHeight, 2)} m`} band={tide} />
        <StatRow
          icon="🚰"
          label="Drainage overflow"
          value={`${fmt(overflowCount, 0)} zones`}
          band={overflowCount > 0
            ? { label: 'Over capacity', state: riskState(80) }
            : { label: 'Within capacity', state: riskState(5) }}
        />
      </div>
    </section>
  )
}

/* ═══ 2. HAZARD OUTLOOK ═══ */

/**
 * City-wide probability of each hazard family the system forecasts. The bands
 * reuse the risk colour scale, so a 60% chance reads the same shade here as a
 * 60 risk score does on the map.
 */
function HazardOutlook({ aiData }) {
  const mh = aiData?.multi_hazard
  if (!mh) return null

  const rows = [
    { key: 'THUNDERSTORM', prob: mh.aggregate_thunderstorm_prob },
    { key: 'CLOUDBURST', prob: mh.aggregate_cloudburst_prob },
    { key: 'FLASH_FLOOD', prob: mh.aggregate_flash_flood_prob },
  ].map((r) => ({ ...r, meta: HAZARDS[r.key], state: riskState((r.prob ?? 0) * 100) }))

  const dominant = rows.reduce((best, r) => ((r.prob ?? 0) > (best.prob ?? 0) ? r : best), rows[0])

  return (
    <section className="card">
      <header className="card-head">
        <h3>🌪️ Hazard Outlook</h3>
        <span className="tag" style={{ background: dominant.state.tint, color: dominant.state.text }}>
          Dominant · {dominant.meta.label}
        </span>
      </header>
      <div className="prob-list">
        {rows.map((r) => (
          <div className="prob-row" key={r.key}>
            <span className="prob-label">{r.meta.icon} {r.meta.label}</span>
            <span className="prob-track">
              <span className="prob-fill" style={{ width: `${Math.min(100, (r.prob ?? 0) * 100)}%`, background: r.state.fill }} />
            </span>
            <span className="prob-value" style={{ color: r.state.text }}>{fmt((r.prob ?? 0) * 100, 0)}%</span>
          </div>
        ))}
      </div>
    </section>
  )
}

/* ═══ 3. ACTIVE ALERTS ═══ */

function AlertsSection({ timestep, alerts }) {
  const [expanded, setExpanded] = useState(false)

  const list = stepAlerts({ alerts_generated: alerts }, timestep)

  if (list.length === 0) {
    return (
      <section className="card">
        <header className="card-head"><h3>🚨 Hazard Alerts</h3></header>
        <p className="hint" style={{ marginTop: 0 }}>
          ✅ No locality is above the alert threshold at this step.
        </p>
      </section>
    )
  }

  const critical = list.filter((a) => severityState(a.severity) === severityState('CRITICAL')).length
  // Three rows is what fits before the panel starts scrolling; the rest are one
  // click away rather than permanently on screen.
  const shown = expanded ? list.slice(0, 12) : list.slice(0, 3)

  return (
    <section className="card">
      <header className="card-head">
        <h3>🚨 Hazard Alerts</h3>
        <div className="tag-row">
          {critical > 0 && (
            <span className="tag" style={{ background: riskState(60).tint, color: riskState(60).text }}>
              {critical} critical
            </span>
          )}
          {list.length > 3 && (
            <button className="tag tag-btn" onClick={() => setExpanded(!expanded)}>
              {expanded ? 'Show fewer' : `Show all ${list.length}`}
            </button>
          )}
        </div>
      </header>

      <ul className="row-list">
        {shown.map((a, i) => {
          const hazard = hazardMeta(a.alert_type)
          const sev = severityState(a.severity)
          return (
            <li className="alert-item" key={a.id || i}>
              <span className="alert-bar" style={{ background: sev.fill }} />
              <div className="alert-main">
                <span className="alert-place">{a.locality_name || a.cell_id || `Cell ${i + 1}`}</span>
                <span className="alert-meta">
                  {hazard.icon} {hazard.label}
                  {a.flood_depth_estimate_cm ? ` · ${fmt(a.flood_depth_estimate_cm, 0)} cm` : ''}
                </span>
              </div>
              <div className="alert-right">
                <span className="alert-score" style={{ color: sev.text }}>{fmt(a.risk_score_total, 0)}</span>
                <span className="alert-sev" style={{ color: sev.text }}>{a.severity}</span>
              </div>
            </li>
          )
        })}
      </ul>

    </section>
  )
}

/* ═══ 4. RESPONSE STATUS ═══ */

/**
 * Where the city is in the event, and what the operator should do about it.
 * The numbers themselves live in Current Situation — this card is the response
 * protocol plus the phase timeline, so nothing is stated twice.
 */
function ResponseStatus({ timestep }) {
  const phase = getPhaseState(timestep)
  const phaseLabel = getPhaseLabel(timestep)
  const current = getPhase(timestep)
  const guidance = PHASE_GUIDANCE[phaseLabel] || { text: '', action: '' }

  return (
    <section className="card" style={{ borderColor: phase.fill }}>
      <header className="card-head">
        <h3>🧭 {current.label}</h3>
        <span className="tag" style={{ background: phase.tint, color: phase.text }}>
          Step {fmt(timestep, 0)} / 72
        </span>
      </header>

      <div className="response-banner" style={{ borderColor: phase.fill, background: phase.tint }}>
        <p className="response-text">{guidance.text}</p>
        <p className="response-action" style={{ color: phase.text }}>Action · {guidance.action}</p>
      </div>

      {/* Six phases, one segment each — the shaded tail is the event already past. */}
      <div className="phase-bar" role="img" aria-label={`Event phase ${current.label}`}>
        {PHASES.map((p) => (
          <span
            key={p.label}
            className={`phase-seg ${p.label === current.label ? 'is-current' : ''}`}
            style={{ background: p.upto <= timestep ? p.state.fill : 'var(--surface-sunk)' }}
            title={`${p.label} · to step ${p.upto}`}
          />
        ))}
      </div>
    </section>
  )
}

/* ═══ NATIONAL OVERVIEW (Live India tab) ═══ */

function NationalOverview({ summary }) {
  if (!summary) return <div className="panel-loading">Loading live national feed…</div>

  const totals = {
    cells: summary.total_cells_monitored ?? 0,
    raining: summary.areas_with_rain ?? 0,
    heavy: summary.areas_heavy_rain ?? 0,
    extreme: summary.areas_extreme_rain ?? 0,
  }
  // The feed returns its ten wettest cells whether or not any of them have
  // rain, so only cells actually getting rain are worth listing.
  const top = (summary.top_areas || [])
    .filter((a) => (a.rainfall_mm_hr ?? 0) > 1)
    .slice(0, 6)
  const wettest = top[0]

  return (
    <>
      <section className="card">
        <header className="card-head">
          <h3>🇮🇳 National Rainfall</h3>
          <span className="tag" style={{ background: riskState(10).tint, color: riskState(10).text }}>live</span>
        </header>

        <div className="kpi-grid">
          <KPI icon="🌧️" label="Areas in rain" value={fmt(totals.raining, 0)} sub="above 1 mm/hr" state={rainBand(3).state} />
          <KPI icon="⛈️" label="Very heavy rain" value={fmt(totals.heavy, 0)} sub="above 25 mm/hr" state={rainBand(30).state} />
          <KPI icon="🚨" label="Extremely heavy" value={fmt(totals.extreme, 0)} sub="above 50 mm/hr" state={rainBand(70).state} />
          <KPI icon="🗺️" label="Grid cells" value={fmt(totals.cells, 0)} sub="national coverage" state={riskState(10)} />
        </div>

        <div className="stat-rows">
          <StatRow
            icon="🌧️"
            label={wettest ? `Wettest · ${wettest.city}` : 'Wettest area'}
            value={wettest ? `${fmt(wettest.rainfall_mm_hr, 1)} mm/hr` : 'no rain'}
            band={wettest ? rainBand(wettest.rainfall_mm_hr) : { label: 'Dry across the grid', state: riskState(5) }}
          />
          <StatRow icon="⛈️" label="Very heavy rain" value={`${fmt(totals.heavy, 0)} cells`} band={{ label: 'IMD very heavy', state: riskState(55) }} />
          <StatRow icon="🚨" label="Extremely heavy" value={`${fmt(totals.extreme, 0)} cells`} band={{ label: 'Continuous rain', state: riskState(80) }} />
        </div>
      </section>

      <section className="card">
        <header className="card-head"><h3>📈 Wettest Areas</h3></header>
        <ul className="row-list">
          {top.map((a, i) => {
            const band = rainBand(a.rainfall_mm_hr)
            return (
              <li className="alert-item" key={a.city || i}>
                <span className="alert-bar" style={{ background: band.state.fill }} />
                <div className="alert-main">
                  <span className="alert-place">{a.city}</span>
                  <span className="alert-meta">{band.label} · wind {fmt(a.wind_kmh, 0)} km/h · {fmt(a.temperature_c, 0)}°C</span>
                </div>
                <div className="alert-right">
                  <span className="alert-score" style={{ color: band.state.text }}>{fmt(a.rainfall_mm_hr, 1)}</span>
                  <span className="alert-sev">mm/hr</span>
                </div>
              </li>
            )
          })}
          {top.length === 0 && (
            <li className="hint" style={{ marginTop: 0 }}>
              No rainfall above 1 mm/hr anywhere in the national grid right now.
              Switch to <b>Mumbai Replay</b> for the event timeline.
            </li>
          )}
        </ul>
      </section>

      <section className="card muted-card">
        <header className="card-head"><h3>🏙️ Mumbai Pilot</h3></header>
        <p className="hint" style={{ marginTop: 0 }}>
          The 6-hour flood nowcast, hazard alerts and evacuation routing run on the
          90-cell Mumbai grid. Switch to <b>Mumbai Replay</b> to see them.
        </p>
      </section>
    </>
  )
}

