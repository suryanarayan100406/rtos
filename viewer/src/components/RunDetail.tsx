import { useEffect, useState } from 'react'
import { api, fileUrl, reportUrl, type ExportsDoc, type RunSummary } from '../api'
import { CesiumView } from './CesiumView'

const fmt = (n: number | null, digits = 2) => (n == null ? '—' : n.toFixed(digits))

export function RunDetail({ run }: { run: RunSummary }) {
  const [exports, setExports] = useState<ExportsDoc | null>(null)

  useEffect(() => {
    let live = true
    setExports(null)
    api
      .getExports(run.run_id)
      .then((e) => live && setExports(e))
      .catch(() => live && setExports(null))
    return () => {
      live = false
    }
  }, [run.run_id])

  const reportReady = run.stages.some((s) => s.name === 'report' && (s.status === 'done' || s.status === 'degraded'))
  const epsg = run.crs?.epsg ?? exports?.epsg ?? null

  return (
    <>
      <div className="card">
        <h3>
          <span style={{ fontFamily: 'ui-monospace, monospace' }}>{run.run_id}</span>{' '}
          <span className={`pill ${run.state}`}>{run.state}</span>
        </h3>
        <div className="kv">
          <span className="k">Dataset</span>
          <span>{run.dataset_name ?? '—'}</span>
          <span className="k">CRS</span>
          <span className="mono">{epsg ? `EPSG:${epsg}` : '—'}</span>
          <span className="k">Progress</span>
          <span>
            {run.n_stages_done}/{run.n_stages_total} stages
            {run.n_stages_failed > 0 && ` · ${run.n_stages_failed} failed`}
          </span>
          <span className="k">Report</span>
          <span>
            {reportReady ? (
              <a href={reportUrl(run.run_id)} target="_blank" rel="noreferrer">
                open accuracy report ↗
              </a>
            ) : (
              <span className="muted">not generated yet</span>
            )}
          </span>
        </div>
        {run.launch_error && <div className="error-banner" style={{ marginTop: 10 }}>{run.launch_error}</div>}
      </div>

      <div className="card">
        <h3>Stages</h3>
        <table className="stages">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Status</th>
              <th>Where</th>
              <th className="num">Seconds</th>
              <th className="num">Confidence</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            {run.stages.map((s) => (
              <tr key={s.name}>
                <td style={{ fontFamily: 'ui-monospace, monospace' }}>{s.name}</td>
                <td>
                  <span className={`pill ${s.status}`}>{s.status}</span>
                </td>
                <td>
                  <code className="env">{s.environment ?? '—'}</code>
                </td>
                <td className="num">{fmt(s.wall_seconds, 1)}</td>
                <td className="num">{fmt(s.confidence)}</td>
                <td>
                  {s.degraded_reason && <div className="reason">⚠ {s.degraded_reason}</div>}
                  {s.error && <div className="err">✕ {s.error}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>3D model</h3>
        <CesiumView runId={run.run_id} exports={exports} />
      </div>

      <div className="card">
        <h3>Deliverables</h3>
        <Downloads runId={run.run_id} exports={exports} />
      </div>
    </>
  )
}

function Downloads({ runId, exports }: { runId: string; exports: ExportsDoc | null }) {
  if (!exports) return <div className="muted">Exports not written yet (s10_export has not run).</div>
  const entries = Object.entries(exports.results)
  if (!entries.length) return <div className="muted">No formats were requested.</div>
  return (
    <div className="downloads">
      {entries.map(([fmtName, r]) =>
        r.ok && r.path ? (
          <a key={fmtName} href={fileUrl(runId, downloadTarget(fmtName, r.path))} target="_blank" rel="noreferrer">
            {fmtName.toUpperCase()} ↓
          </a>
        ) : (
          <a key={fmtName} className="skipped" title={r.reason ?? 'skipped'}>
            {fmtName.toUpperCase()} — skipped
          </a>
        ),
      )}
    </div>
  )
}

// 3D Tiles / Potree exports are directories; link the entry file rather than the folder.
function downloadTarget(fmtName: string, path: string): string {
  if (fmtName === '3dtiles') return `${path}/tileset.json`
  if (fmtName === 'potree') return `${path}/metadata.json`
  return path
}
