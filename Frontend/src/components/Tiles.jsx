/**
 * Tiles — the shared metric primitives used by both side panels.
 * ============================================================
 * One definition per tile means the left and right panels can never render the
 * same kind of number two different ways. All of them take a `state` object
 * from utils/helpers (fill / text / tint) so colours stay in one place too.
 *
 * Layout note: the side panels are ~380 px wide, so tiles never rely on
 * overflow to stay readable — long values get their own line instead of
 * spilling past the tile border.
 */

/** Big headline tile: a number, its unit, and the band it falls in. */
export function KPI({ icon, label, value, suffix, sub, state }) {
  return (
    <div className="kpi" style={{ background: state.tint, borderColor: state.fill }}>
      <span className="kpi-top">
        <span className="kpi-icon">{icon}</span>
        <span className="kpi-label">{label}</span>
      </span>
      <span className="kpi-value" style={{ color: state.text }}>
        {value}
        {suffix && <span className="kpi-suffix">{suffix}</span>}
      </span>
      {sub && <span className="kpi-sub" style={{ color: state.text }}>{sub}</span>}
    </div>
  )
}

/** Three-up compact tile for short values. */
export function MiniStat({ label, value, unit, sub, state }) {
  return (
    <div className="mini-stat" style={{ background: state.tint, borderColor: state.fill }}>
      <span className="mini-stat-label">{label}</span>
      <span className="mini-stat-value" style={{ color: state.text }}>
        {value}
        {unit && <span className="mini-stat-unit">{unit}</span>}
      </span>
      <span className="mini-stat-sub">{sub}</span>
    </div>
  )
}

/**
 * One reading per row — label, band name, value. Kept to a single line so a
 * list of readings cannot grow into a scroll area: the band name sits inline in
 * its own colour instead of taking a second line as a pill.
 */
export function StatRow({ icon, label, value, band }) {
  return (
    <div className="stat-row">
      <span className="stat-label">{icon} {label}</span>
      <span className="stat-band" style={{ color: band.state.text }}>{band.label}</span>
      <span className="stat-value" style={{ color: band.state.text }}>{value}</span>
    </div>
  )
}
