import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from './utils/api'
import {
  fmt, getPhaseLabel, getPhaseState, stepAlerts, stepClock, rainBand, tideBand,
} from './utils/helpers'
import MapView from './components/MapView'
import IntelligencePanel from './components/IntelligencePanel'
import LeftPanel from './components/LeftPanel'

const TOTAL_STEPS = 72
// Fallback for the "peak event" jump. The evaluation report records the peak the
// models were scored against; when metrics are reachable we use that step so the
// button can never drift from the report.
const FALLBACK_PEAK_STEP = 39

export default function App() {
  const [health, setHealth] = useState(null)
  const [timestep, setTimestep] = useState(1)
  const [replayStatus, setReplayStatus] = useState(null)
  const [replayStepData, setReplayStepData] = useState(null)
  const [aiData, setAiData] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [selectedCell, setSelectedCell] = useState(null)
  const [mapMode, setMapMode] = useState('mumbai')
  const [liveData, setLiveData] = useState(null)
  const [liveSummary, setLiveSummary] = useState(null)
  const [peakStep, setPeakStep] = useState(FALLBACK_PEAK_STEP)
  const intervalRef = useRef(null)
  const liveIntervalRef = useRef(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
    const iv = setInterval(() => api.health().then(setHealth).catch(() => {}), 15000)
    return () => clearInterval(iv)
  }, [])

  // Peak step from the frozen evaluation report, not a hardcoded guess.
  useEffect(() => {
    api.metrics.skill()
      .then((d) => {
        const peak = d?.operational?.observed_peak_timestep
        if (Number.isFinite(peak) && peak > 0) setPeakStep(peak)
      })
      .catch(() => {})
  }, [])

  const fetchLiveData = useCallback(async () => {
    try {
      const [indiaData, summaryData] = await Promise.all([
        api.realtime.india(),
        api.realtime.summary(),
      ])
      setLiveData(indiaData)
      setLiveSummary(summaryData)
      if (mapMode === 'india') setAiData(indiaData)
    } catch (e) {
      console.error('Live data fetch failed:', e)
    }
  }, [mapMode])

  useEffect(() => {
    fetchLiveData()
    liveIntervalRef.current = setInterval(fetchLiveData, 5 * 60 * 1000)
    return () => { if (liveIntervalRef.current) clearInterval(liveIntervalRef.current) }
  }, [fetchLiveData])

  useEffect(() => {
    api.replayStatus().then((d) => {
      setReplayStatus(d)
      setTimestep(d.current_timestep)
    }).catch(() => {})
  }, [])

  const fetchReplayStep = useCallback(async (ts) => {
    if (mapMode !== 'mumbai') return
    try {
      const step = await api.replayStep(ts)
      setReplayStepData(step)
      if (step.ai_pipeline_results) setAiData(step.ai_pipeline_results)
    } catch (e) {
      console.error('Replay step failed:', e)
    }
  }, [mapMode])

  useEffect(() => { fetchReplayStep(timestep) }, [timestep, fetchReplayStep])

  useEffect(() => {
    if (playing && mapMode === 'mumbai') {
      intervalRef.current = setInterval(() => {
        setTimestep((prev) => {
          if (prev >= TOTAL_STEPS) { setPlaying(false); return TOTAL_STEPS }
          return prev + 1
        })
      }, 1500)
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, mapMode])

  const jumpTo = (ts) => { setPlaying(false); setTimestep(ts) }

  const switchMode = (mode) => {
    setMapMode(mode)
    setPlaying(false)
    setSelectedCell(null)
    if (mode === 'india') fetchLiveData()
    else fetchReplayStep(timestep)
  }

  const isOnline = health?.status === 'ONLINE'
  const isMumbai = mapMode === 'mumbai'
  const mapData = isMumbai ? aiData : liveData
  const phaseState = getPhaseState(timestep)
  // One source for the alert count, shared by the topbar and the side panel.
  const alerts = isMumbai ? stepAlerts(replayStepData, timestep) : []
  const rain = rainBand(replayStepData?.avg_rainfall_1h_mm ?? 0)
  const tide = tideBand(replayStepData?.tide_height_m ?? 0)

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">🌊</span>
          <div className="brand-text">
            <h1>VARUNA</h1>
            <span>Urban Flood Early Warning · Mumbai Pilot</span>
          </div>
        </div>

        <div className="segmented">
          <button className={`segment ${!isMumbai ? 'active' : ''}`} onClick={() => switchMode('india')}>Live India</button>
          <button className={`segment ${isMumbai ? 'active' : ''}`} onClick={() => switchMode('mumbai')}>Mumbai Replay</button>
        </div>

        {isMumbai && (
          <div className="transport">
            <button className="icon-btn" onClick={() => jumpTo(1)} title="Start">⏮</button>
            <button className="icon-btn" onClick={() => jumpTo(Math.max(1, timestep - 1))} disabled={timestep <= 1} title="Back">◀</button>
            <button className="icon-btn accent" onClick={() => setPlaying(!playing)} title="Play">
              {playing ? '❚❚' : '▶'}
            </button>
            <button className="icon-btn" onClick={() => jumpTo(Math.min(TOTAL_STEPS, timestep + 1))} disabled={timestep >= TOTAL_STEPS} title="Forward">▶</button>
            <button className="icon-btn" onClick={() => jumpTo(peakStep)}>Peak event</button>

            <div className="progress">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${(timestep / TOTAL_STEPS) * 100}%`, background: phaseState.fill }} />
              </div>
              <div className="progress-meta">
                <span style={{ color: phaseState.text }}>{getPhaseLabel(timestep)}</span>
                <span>{stepClock(timestep)} · step {timestep}/{TOTAL_STEPS}</span>
              </div>
            </div>
          </div>
        )}

        <div className="status-chips">
          <span className={`chip ${isOnline ? 'chip-ok' : 'chip-off'}`}>
            {isOnline ? 'Online' : 'Offline'}
          </span>
          {isMumbai ? (
            <>
              <span className="chip" style={{ background: tide.state.tint, color: tide.state.text }}>
                🌊 Tide {fmt(replayStepData?.tide_height_m, 2)} m
              </span>
              <span className="chip" style={{ background: rain.state.tint, color: rain.state.text }}>
                🌧 Rain {fmt(replayStepData?.avg_rainfall_1h_mm, 1)} mm/hr
              </span>
              <span className="chip" style={{ background: alerts.length ? 'var(--high-tint)' : 'var(--low-tint)', color: alerts.length ? 'var(--high-text)' : 'var(--low-text)' }}>
                🚨 {alerts.length} alerts
              </span>
            </>
          ) : (
            <>
              <span className="chip chip-info">📍 {liveSummary?.total_cells_monitored ?? 0} grid cells</span>
              <span className="chip chip-alert">🌧 {liveSummary?.areas_with_rain ?? 0} areas in rain</span>
            </>
          )}
        </div>
      </header>

      <div className="workspace">
        <aside className="side-panel">
          <div className="panel-scroll">
            <LeftPanel
              timestep={timestep}
              aiData={mapData}
              replayStepData={replayStepData}
              selectedCell={selectedCell}
              mode={mapMode}
              liveSummary={liveSummary}
            />
          </div>
        </aside>

        <main className="map-pane">
          <MapView
            timestep={timestep}
            aiData={mapData}
            replayStepData={replayStepData}
            selectedCell={selectedCell}
            onSelectCell={setSelectedCell}
            mode={mapMode}
          />
        </main>

        <aside className="side-panel right">
          <div className="panel-scroll">
            <IntelligencePanel
              timestep={timestep}
              aiData={mapData}
              replayStepData={replayStepData}
              selectedCell={selectedCell}
              mode={mapMode}
            />
          </div>
        </aside>
      </div>
    </div>
  )
}
