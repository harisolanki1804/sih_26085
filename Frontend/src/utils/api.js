const BASE = '/api/v1'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`)
  return res.json()
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`)
  return res.json()
}

export const api = {
  health: () => get('/health'),

  // Replay
  replayStatus: () => get('/replay/status'),
  replayStep: (step) => post(`/replay/step${step ? `?step_to=${step}` : ''}`),
  replayReset: () => post('/replay/reset'),

  // Features
  featuresLatest: (region) => get(`/features/latest?region_code=${region || 'IN-MH-BOM-01'}`),
  featuresHistory: () => get('/features/history'),

  // Alerts
  alerts: (params = {}) => {
    const q = new URLSearchParams()
    if (params.is_active !== undefined) q.set('is_active', params.is_active)
    if (params.severity) q.set('severity', params.severity)
    if (params.alert_type) q.set('alert_type', params.alert_type)
    const qs = q.toString()
    return get(`/alerts${qs ? `?${qs}` : ''}`)
  },
  alertExplain: (id) => get(`/alerts/${id}/explain`),
  alertAck: (id, op) => post(`/alerts/${id}/acknowledge?operator_name=${op || 'Operator-1'}`),

  // Regions
  regions: () => get('/regions'),
  hotspots: (code) => get(`/regions/${code || 'IN-MH-BOM-01'}/hotspots`),

  // AI Modules
  ai: {
    inference: (ts) => get(`/ai/inference/${ts}`),
    stormCells: (ts) => get(`/ai/storm-cells/${ts}`),
    riskHeatmap: (ts) => get(`/ai/risk-heatmap/${ts}`),
    nowcast: (ts, h = 6) => get(`/ai/nowcast/${ts}?horizon=${h}`),
    multiHazard: (ts) => get(`/ai/multi-hazard/${ts}`),
    fusedFeatures: (ts) => get(`/ai/fused-features/${ts}`),
    floodDepth: (ts) => get(`/ai/flood-depth/${ts}`),
    trustScore: (ts) => get(`/ai/trust-score/${ts}`),
    xai: (ts) => get(`/ai/xai/${ts}`),
    crowdReport: (text, lat, lon) =>
      post('/ai/crowd-report', { report_text: text, predicted_flood_lat: lat, predicted_flood_lon: lon }),
    modelStatus: () => get('/ai/model-status'),
  },

  // Forecast skill — served from the frozen evaluation artifacts
  metrics: {
    skill: () => get('/metrics/skill'),
    evaluation: () => get('/metrics/evaluation'),
    nowcast: () => get('/metrics/nowcast'),
  },

  // Real-Time India-Wide Data
  realtime: {
    india: () => get('/realtime/india'),
    city: (name) => get(`/realtime/city/${encodeURIComponent(name)}`),
    mumbai: () => get('/realtime/mumbai'),
    summary: () => get('/realtime/summary'),
  },

  // Innovation Modules
  innovations: {
    // INSAT-3D Satellite Data
    insat3d: (ts) => get(`/innovations/satellite/insat3d${ts ? `?timestamp=${ts}` : ''}`),
    cmv: (ts) => get(`/innovations/satellite/cmv?timestep_id=${ts || 1}`),
    superResolution: (ch, scale) => get(`/innovations/satellite/super-resolution?channel=${ch || 'TIR1'}&scale=${scale || 25}`),

    // Physics-Informed Risk
    physicsRisk: (ts) => get(`/innovations/physics/risk/${ts}`),
    physicsCheck: (ts) => get(`/innovations/physics/consistency-check?timestep_id=${ts || 36}`),

    // Evacuation Routes
    evacuationRoutes: (cellIdx, type) => get(`/innovations/evacuation/routes/${cellIdx}?route_type=${type || 'vehicle'}`),
    evacuationCitywide: () => get('/innovations/evacuation/citywide-analysis'),

    // What-If Scenarios
    scenarioList: () => get('/innovations/scenario/list'),
    scenarioRun: (id, ts) => post(`/innovations/scenario/run/${id}?timestep_id=${ts || 36}`),
    scenarioSensitivity: (ts) => get(`/innovations/scenario/sensitivity/${ts || 36}`),

    // Edge Model Export
    edgeExport: (platform) => get(`/innovations/edge/export?platform=${platform || 'mobile_android'}`),
    edgeBenchmark: () => get('/innovations/edge/benchmark'),

    // What-If Chatbot
    chatbotAsk: (question, timestep) => post('/innovations/chatbot/ask', { question, timestep: timestep || 36 }),
  },
}
