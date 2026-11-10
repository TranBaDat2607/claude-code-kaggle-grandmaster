// A running kernel's session as the pane can tell it: the limit Kaggle cancels it at, when it started, how long is
// left, and what its log says of its end. Kaggle's own status lags: it can say RUNNING for minutes after the script
// has ended, the outputs are saved, or the limit has cut the session off. Pure: no `$`, so tests call it directly.

import type { Kernel, KernelLive, KernelRun, Outputs } from '../../types'
import { utc } from './kaggle'

/** Kaggle's longest batch session, by the quota it draws (verified Oct 2026; kgkit.gpu.SESSION_LIMITS). */
export const SESSION_MAX_H: Record<string, number> = { GPU: 12, CPU: 12, TPU: 9 }
/** How long before the limit a card warns: half an hour, or a quarter of a shorter limit (a 1 h smoke run). */
export const NEAR_MS = 30 * 60_000
/** A GPU or CPU session starts within minutes of its push (kgkit counts its hours from the push); one seen running
 * later than this after Kaggle's lastRunTime waited in the queue. */
export const QUEUE_SLACK_MS = 15 * 60_000
/** How far a logged push may be after Kaggle's lastRunTime and still be the push of that run: the log is written
 * once the push command returns, and the two clocks differ. */
const PUSH_SKEW_MS = 10 * 60_000

export function age(s: number | undefined): string {
  if (s === undefined) return '-'
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}`
  return `${Math.floor(s / 86400)}d${Math.floor((s % 86400) / 3600)}h`
}

/** 12h, 1h30, 15m. */
export const hours = (h: number) => (Number.isInteger(h) ? `${h}h` : age(h * 3600))

/** 'TPU', 'CPU' or 'GPU' for an accelerator / machine_shape name, as kgkit.gpu.resource_of; undefined for none. */
export function resourceOf(accelerator: unknown): string | undefined {
  const a = String(accelerator ?? '').trim().toLowerCase()
  if (!a) return undefined
  if (a.startsWith('tpu')) return 'TPU'
  return a === 'none' || a === 'cpu' || a === 'no' ? 'CPU' : 'GPU'
}

/**
 * What the run log (`.kaggle-gm/gpu_runs.jsonl`, its text) holds of the push that started the run Kaggle reports for
 * `ref`, last run at `lastRunTime`: the latest record of that kernel pushed no later than the run (a later one is a
 * newer push). `kgkit gpu push` logs every push; the hook logs each GPU or TPU `kaggle kernels push` Claude runs.
 */
export function runOf(log: string, ref: string, lastRunTime?: string): KernelRun | undefined {
  const ran = utc(lastRunTime)
  let found: Record<string, unknown> | undefined
  for (const line of log.split(/\r?\n/)) {
    if (!line.trim()) continue
    let rec: unknown
    try {
      rec = JSON.parse(line)
    } catch {
      continue
    }
    if (!rec || typeof rec !== 'object' || Array.isArray(rec)) continue
    const r = rec as Record<string, unknown>
    if (String(r.kernel ?? '').toLowerCase() !== ref.toLowerCase()) continue
    const pushed = utc(typeof r.pushed_at === 'string' ? r.pushed_at : undefined)
    if (ran !== undefined && pushed !== undefined && pushed > ran + PUSH_SKEW_MS) continue
    found = r
  }
  if (!found) return undefined
  const cap = Number(found.hours)
  const resource = typeof found.resource === 'string' && found.resource ? found.resource.toUpperCase() : resourceOf(found.accelerator)
  return {
    ...(cap > 0 ? { capH: cap } : {}),
    ...(resource ? { resource } : {}),
    ...(typeof found.started_at === 'string' ? { startedAt: found.started_at } : {}),
  }
}

/**
 * When the session started (local clock, ms). The kernel's own clock when its readings carry it (`t0`). Else, for a
 * GPU or CPU session, Kaggle's lastRunTime (such a session starts within minutes of its push), unless the pane saw it
 * wait in the queue longer; for a TPU session, which can wait for hours, only once it was seen running: by the pane,
 * or by kgkit (`gpu wait`, `gpu status`). undefined when it cannot be told.
 */
export function startOf(k: Kernel, L: KernelLive | undefined, resource: string | undefined): number | undefined {
  const ran = utc(k.lastRunTime)
  if (L?.t0 !== undefined && (ran === undefined || L.t0 * 1000 >= ran - PUSH_SKEW_MS)) return L.t0 * 1000
  if (resource === 'TPU') {
    const seen = [k.seenRunningAt, utc(k.run?.startedAt)].filter((x): x is number => x !== undefined)
    return seen.length ? Math.min(...seen) : undefined
  }
  if (ran === undefined) return k.seenRunningAt ?? utc(k.run?.startedAt)
  return k.seenRunningAt !== undefined && k.seenRunningAt - ran > QUEUE_SLACK_MS ? k.seenRunningAt : ran
}

/** running; near: the limit is close and the script still runs; saving: the script has ended, Kaggle saves the
 * outputs; over: past the limit by the clock; closed: Kaggle has closed the log, the session is over. */
export type Phase = 'running' | 'near' | 'saving' | 'over' | 'closed'

export type Clock = {
  limitH: number
  /** 'cap': the push's -t / --hours (from the run log); 'max': Kaggle's longest session for its accelerator. */
  limitFrom: 'cap' | 'max'
  resource?: string
  /** When the session started and when the limit falls (local clock, ms), when that can be told. */
  startAt?: number
  limitAt?: number
  /** Time to the limit, negative past it. */
  leftMs?: number
  /** How long the session has run: it stops at the limit, and when Kaggle closed the log. */
  ranMs?: number
  isOver: boolean
  /** The last NEAR_MS before the limit (a quarter of a limit under 2 h). */
  isNear: boolean
  isClosed: boolean
  isScriptDone: boolean
  phase: Phase
}

/** A running kernel's session clock at `now`. */
export function clockOf(k: Kernel, L: KernelLive | undefined, now: number): Clock {
  // No run log entry: a session that reported GPUs is a GPU one; otherwise CPU or TPU, and 12 h is the longest.
  const resource = k.run?.resource ?? (L?.gpus?.length ? 'GPU' : undefined)
  const maxH = SESSION_MAX_H[resource ?? 'GPU'] ?? 12
  const cap = k.run?.capH
  const limitH = cap !== undefined && cap > 0 ? Math.min(cap, maxH) : maxH
  const startAt = startOf(k, L, resource)
  const limitAt = startAt === undefined ? undefined : startAt + limitH * 3600_000
  const leftMs = limitAt === undefined ? undefined : limitAt - now
  const closedAt = L?.logClosedAt
  const isClosed = closedAt !== undefined
  // A session whose log closed before its limit ended by itself; the limit ends one still open at it.
  const isOver = leftMs !== undefined && leftMs <= 0 && !(closedAt !== undefined && limitAt !== undefined && closedAt < limitAt)
  const isScriptDone = L?.scriptEnd !== undefined
  const isNear = !isOver && !isClosed && leftMs !== undefined && leftMs <= Math.min(NEAR_MS, (limitH * 3600_000) / 4)
  const phase: Phase = isClosed ? 'closed' : isOver ? 'over' : isScriptDone ? 'saving' : isNear ? 'near' : 'running'
  const endAt = Math.min(now, limitAt ?? now, closedAt ?? now)
  const ranMs = startAt !== undefined ? Math.max(0, endAt - startAt) : L?.t !== undefined ? L.t * 1000 : undefined
  return {
    limitH,
    limitFrom: cap !== undefined && cap > 0 && cap < maxH ? 'cap' : 'max',
    ...(resource ? { resource } : {}),
    ...(startAt !== undefined ? { startAt } : {}),
    ...(limitAt !== undefined ? { limitAt } : {}),
    ...(leftMs !== undefined ? { leftMs } : {}),
    ...(ranMs !== undefined ? { ranMs } : {}),
    isOver,
    isNear,
    isClosed,
    isScriptDone,
    phase,
  }
}

/** A bright blue that reads well on a black terminal. */
export const RUNNING_BLUE = '#5fafff'

/** The card's title word and its colour (undefined: dim). */
export const LOOK: Record<Phase, { label: string; color?: string }> = {
  running: { label: 'RUNNING', color: RUNNING_BLUE },
  near: { label: 'RUNNING', color: RUNNING_BLUE },
  saving: { label: 'SAVING', color: 'success' },
  over: { label: 'PAST LIMIT', color: 'warning' },
  closed: { label: 'ENDED' },
}

/** `limit 12h · 28m left`; `limit 9h` when the start cannot be told; `past the 12h limit`. */
export function limitText(c: Clock): string {
  const limit = hours(c.limitH)
  if (c.isOver) return `past the ${limit} limit`
  if (c.leftMs === undefined || c.isClosed) return `limit ${limit}`
  return `limit ${limit} · ${age(c.leftMs / 1000)} left`
}

/** `isAdvice`: what to do about it, drawn on an open card alone. */
export type Note = { text: string; tone?: 'warning' | 'success'; isAdvice?: true }

const LAGS = "though its status still says RUNNING: Kaggle's status lags"

/** What the card says of the session's end: the limit coming, the script ended, the session over; and its outputs. */
export function endNotes(c: Clock, L: KernelLive | undefined, ref: string, now: number): Note[] {
  const limit = hours(c.limitH)
  const end = L?.scriptEnd
  const ended = end ? `${end.isError ? 'stopped on an error' : 'finished'}${end.at !== undefined ? ` ${age((now - end.at) / 1000)} ago` : ''}` : ''
  const left = c.leftMs !== undefined ? age(Math.max(0, c.leftMs) / 1000) : ''
  const notes: Note[] = []
  if (c.isClosed) {
    notes.push({
      text: c.isOver
        ? `Kaggle has closed its log, past its ${limit} limit: the limit has probably cancelled the session, ${LAGS}.`
        : `Kaggle has closed its log: the session is over${end ? ` (the script ${ended})` : ''}, ${LAGS}.`,
      tone: c.isOver ? 'warning' : undefined,
    })
  } else if (c.isOver) {
    notes.push({
      text:
        `Past its ${limit} limit by ${age(-(c.leftMs ?? 0) / 1000)}: Kaggle has probably cancelled it, ${LAGS}.` +
        (end ? ` The script ${ended}; outputs still being saved at the limit can be lost.` : ''),
      tone: 'warning',
    })
  } else if (end) {
    notes.push({ text: `The script ${ended}: Kaggle is saving the outputs.`, tone: end.isError ? 'warning' : 'success' })
    if (c.isNear) notes.push({ text: `${left} to the ${limit} limit: a save the limit cuts off can lose the outputs.`, tone: 'warning' })
  } else if (c.isNear) {
    notes.push({
      text: `${left} to its ${limit} limit and the script has not finished: Kaggle cancels it at the limit and keeps only what is already in /kaggle/working.`,
      tone: 'warning',
    })
  }
  if (c.isOver) {
    notes.push({
      text: 'A cancelled version usually cannot be attached to another kernel: download its outputs and upload them as a dataset.',
      isAdvice: true,
    })
  }
  const out = outputsNote(L?.outputs, ref, now)
  if (out && (c.isOver || c.isClosed || end)) notes.push(out)
  return notes
}

/** What the last lookup of the session's outputs found. */
export function outputsNote(o: Outputs | undefined, ref: string, now: number): Note | undefined {
  if (!o) return undefined
  if (o.error) return { text: `Outputs: could not list them (${o.error}).` }
  const files = o.files ?? []
  if (!files.length) return { text: `Outputs: none readable yet (looked ${age((now - o.checkedAt) / 1000)} ago).` }
  const n = `${files.length}${o.isMore ? '+' : ''}`
  const names = files.slice(0, 3).join(', ') + (files.length > 3 ? ', ...' : '')
  return { text: `Outputs: ${n} file${files.length === 1 && !o.isMore ? '' : 's'} readable now (${names}): kaggle kernels output ${ref} -p <dir>.`, tone: 'success' }
}

/** A toast for a phase the clock has just reached, or none. */
export function phaseToast(c: Clock, L: KernelLive | undefined, ref: string, alias: string): string | undefined {
  const who = `Kaggle kernel ${ref} (${alias})`
  const limit = hours(c.limitH)
  if (c.phase === 'near') return `${who}: ${age(Math.max(0, c.leftMs ?? 0) / 1000)} to its ${limit} limit, and the script has not finished`
  if (c.phase === 'saving') return `${who}: the script ${L?.scriptEnd?.isError ? 'stopped on an error' : 'finished'}; Kaggle is saving its outputs`
  if (c.phase === 'over') return `${who}: past its ${limit} limit, probably cancelled (Kaggle's status lags)`
  if (c.phase === 'closed') return `${who}: Kaggle has closed its log, the session is over (its status lags)`
  return undefined
}
