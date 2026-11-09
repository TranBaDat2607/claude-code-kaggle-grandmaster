import type { LiveStats, Progress, Sample } from '../../types'
import type { Reading } from './kaggle'

/** Samples a sparkline draws from: when more arrive, neighbours are merged, so they stay evenly spaced over the run. */
export const HISTORY_MAX = 96
const TAIL_MAX = 3

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)
const STEP_KEYS = ['step', 'global_step', 'iter', 'iteration', 'batch']
const TOTAL_KEYS = ['of', 'total', 'steps', 'total_steps', 'max_steps', 'num_steps', 'num_training_steps']
const EPOCH_TOTAL_KEYS = ['epochs', 'num_epochs', 'max_epochs', 'n_epochs']
const TQDM_RE = /(?:^|\s)([^|:]{0,40}?)[:\s]*\d{1,3}%\|[^|]*\|\s*(\d+)\/(\d+)\b/
const COUNT_RE = (word: string) => new RegExp(`\\b(${word})\\s*[:=#]?\\s*\\[?\\s*(\\d+)\\s*/\\s*(\\d+)\\b`, 'i')
const WORDS = ['step|iter(?:ation)?|batch', 'epoch', 'fold']

/** Where one log line says the training is: a JSON record with step/of or epoch/epochs, a tqdm bar, or
 * `step 820/858`, `epoch 3/10`, `fold 2/5`. undefined when it says nothing countable. */
export function parseProgress(line: string): Progress | undefined {
  const open = line.indexOf('{')
  const close = line.lastIndexOf('}')
  if (open >= 0 && close > open) {
    try {
      const j = JSON.parse(line.slice(open, close + 1)) as Record<string, unknown>
      const epoch = num(j.epoch)
      const step = STEP_KEYS.map(k => num(j[k])).find(v => v !== undefined)
      const total = TOTAL_KEYS.map(k => num(j[k])).find(v => v !== undefined)
      if (step !== undefined && total) return { label: 'step', done: step, total, ...(epoch !== undefined ? { epoch } : {}) }
      const epochs = EPOCH_TOTAL_KEYS.map(k => num(j[k])).find(v => v !== undefined)
      if (epoch !== undefined && epochs) return { label: 'epoch', done: epoch, total: epochs }
    } catch {
      // not JSON: read it as text
    }
  }
  const bar = TQDM_RE.exec(line)
  if (bar) {
    const total = Number(bar[3])
    if (total > 0) return { label: (bar[1] ?? '').trim().replace(/\s+/g, ' ') || 'step', done: Number(bar[2]), total }
  }
  for (const word of WORDS) {
    const m = COUNT_RE(word).exec(line)
    if (m && Number(m[3]) > 0) return { label: (m[1] ?? word).toLowerCase(), done: Number(m[2]), total: Number(m[3]) }
  }
  return undefined
}

function merge(a: Sample, b: Sample): Sample {
  const avg = (x?: number, y?: number) => (x === undefined ? y : y === undefined ? x : (x + y) / 2)
  const avgs = (x?: number[], y?: number[]) =>
    x && y ? x.map((v, i) => avg(v, y[i]) ?? v) : (x ?? y)
  return { t: b.t, cpu: avg(a.cpu, b.cpu), ram: avg(a.ram, b.ram), gpu: avgs(a.gpu, b.gpu), gmem: avgs(a.gmem, b.gmem) }
}

/** What one kernel's log has said so far: history, averages and peaks, progress, the last lines.
 * The log is replayed whole when the pane connects, so a kernel that started hours ago has its whole run. */
export class LiveTracker {
  hist: Sample[] = []
  /** Samples merged into one sparkline point at present (doubles when hist fills). */
  private stride = 1
  private pending: Sample | undefined
  private pendingN = 0
  private sums = { n: 0, cpu: 0, cpuN: 0, gpu: [] as number[], gpuN: 0, ramPeak: undefined as number | undefined, gmemPeak: [] as number[] }
  tail: string[] = []
  progress: Progress | undefined
  private lastT = 0

  reading(r: Reading) {
    const t = r.t ?? this.lastT
    this.lastT = t
    const s: Sample = {
      t,
      cpu: r.cpu,
      ram: r.ram?.[0],
      gpu: r.gpus?.map(g => g[1]),
      gmem: r.gpus?.map(g => g[2]),
    }
    const S = this.sums
    S.n++
    if (r.cpu !== undefined) {
      S.cpu += r.cpu
      S.cpuN++
    }
    if (s.ram !== undefined) S.ramPeak = Math.max(S.ramPeak ?? 0, s.ram)
    if (s.gpu?.length) {
      s.gpu.forEach((u, i) => (S.gpu[i] = (S.gpu[i] ?? 0) + u))
      S.gpuN++
      s.gmem?.forEach((m, i) => (S.gmemPeak[i] = Math.max(S.gmemPeak[i] ?? 0, m)))
    }
    this.pending = this.pending ? merge(this.pending, s) : s
    if (++this.pendingN < this.stride) return
    this.hist.push(this.pending)
    this.pending = undefined
    this.pendingN = 0
    if (this.hist.length >= HISTORY_MAX) {
      const thinned: Sample[] = []
      for (let i = 0; i + 1 < this.hist.length; i += 2) thinned.push(merge(this.hist[i]!, this.hist[i + 1]!))
      if (this.hist.length % 2) thinned.push(this.hist[this.hist.length - 1]!)
      this.hist = thinned
      this.stride *= 2
    }
  }

  /** A log line other than a reading. */
  line(text: string) {
    this.tail = [...this.tail, text].slice(-TAIL_MAX)
    this.progress = parseProgress(text) ?? this.progress
  }

  stats(): LiveStats {
    const S = this.sums
    return {
      n: S.n,
      ...(S.cpuN ? { cpuAvg: S.cpu / S.cpuN } : {}),
      ...(S.ramPeak !== undefined ? { ramPeak: S.ramPeak } : {}),
      ...(S.gpuN ? { gpuAvg: S.gpu.map(x => x / S.gpuN), gmemPeak: [...S.gmemPeak] } : {}),
    }
  }

  /** The history, with the samples still being merged as its last point. */
  history(): Sample[] {
    return this.pending ? [...this.hist, this.pending] : [...this.hist]
  }
}

const BLOCKS = '▁▂▃▄▅▆▇█'

/** A sparkline of `values` (0..1, undefined for a gap) in `width` columns: the run squeezed into them, or drawn
 * left to right and padded when it is shorter. */
export function spark(values: (number | undefined)[], width: number): string {
  if (width <= 0) return ''
  const cols: (number | undefined)[] = []
  if (values.length <= width) cols.push(...values)
  else {
    for (let c = 0; c < width; c++) {
      const from = Math.floor((c * values.length) / width)
      const to = Math.max(from + 1, Math.floor(((c + 1) * values.length) / width))
      const part = values.slice(from, to).filter((v): v is number => v !== undefined)
      cols.push(part.length ? part.reduce((a, b) => a + b, 0) / part.length : undefined)
    }
  }
  const out = cols.map(v => (v === undefined ? ' ' : BLOCKS[Math.max(0, Math.min(7, Math.round(Math.max(0, Math.min(1, v)) * 7)))]))
  return out.join('').padEnd(width, ' ')
}
