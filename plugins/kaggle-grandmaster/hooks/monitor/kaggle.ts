// The Kaggle CLI run per account, and the account registry.
//
// An account is a token file. The CLI is pointed at one with KAGGLE_API_TOKEN set
// to the file's path (kagglesdk reads a path there as the file), so a token is
// never held in this module, the session's environment, or the transcript: the
// one exception is a token pasted into the pane's Add field, which is written
// straight to a private file and dropped.

import type { EngineInterface } from 'claude-code'

import type { Kernel, KernelLive, QuotaRow, Submission, Submissions } from '../../types'
import { oneLine } from './describe'
import type { Platform } from './probe'

export type Cli = { argv: string[]; version: string }

/** An account as the registry holds it, its token path resolved. */
export type Account = { alias: string; user?: string; tokenPath: string }

export type Paths = { home: string; sep: string; kaggleDir: string; registry: string; defaultToken: string }

export async function paths($: EngineInterface, platform: Platform): Promise<Paths> {
  const home =
    platform === 'windows'
      ? ((await $.env.get('USERPROFILE')) ?? (await $.env.get('HOME')) ?? '')
      : ((await $.env.get('HOME')) ?? '')
  const sep = platform === 'windows' ? '\\' : '/'
  const kaggleDir = `${home}${sep}.kaggle`
  return {
    home,
    sep,
    kaggleDir,
    registry: `${kaggleDir}${sep}kg_accounts.json`,
    defaultToken: `${kaggleDir}${sep}access_token`,
  }
}

export function expand(p: string, P: Paths): string {
  const t = p.trim()
  if (t === '~') return P.home
  if (t.startsWith('~/') || t.startsWith('~\\')) return `${P.home}${P.sep}${t.slice(2).replace(/[\\/]/g, P.sep)}`
  return t
}

/** `~/.kaggle/x` for a path under the home folder, as the registry writes it. */
export function tilde(p: string, P: Paths): string {
  return P.home && p.startsWith(P.home + P.sep) ? `~/${p.slice(P.home.length + 1).replace(/\\/g, '/')}` : p
}

// ---------------------------------------------------------------- the CLI

const VERSION_RE = /(\d+)\.(\d+)\.(\d+)/

function versionKey(v: string): number {
  const m = VERSION_RE.exec(v)
  return m ? Number(m[1]) * 1e6 + Number(m[2]) * 1e3 + Number(m[3]) : 0
}

/** Splits a path into its folder and name, on either separator. */
function parent(path: string): string {
  const i = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))
  return i > 0 ? path.slice(0, i) : path
}

async function dirs($: EngineInterface, dir: string): Promise<string[]> {
  try {
    return (await $.fs.list(dir)).filter(e => e.kind === 'dir').map(e => e.name)
  } catch {
    return []
  }
}

/**
 * The version of the `kaggle` package an executable belongs to, read from its
 * install's `kaggle-<version>.dist-info` folder. The CLI itself is never run to
 * ask: it checks the token with Kaggle before printing anything, so `kaggle
 * --version` waits minutes whenever kaggle.com does not answer.
 */
async function installedVersion($: EngineInterface, exe: string, P: Paths): Promise<string | undefined> {
  let real = exe
  try {
    real = (await $.fs.stat(exe, { resolve: true })).realPath ?? exe
  } catch {
    // keep the path as given
  }
  const prefix = parent(parent(real)) // <prefix>/bin/kaggle, <prefix>\Scripts\kaggle.exe
  const sites = [`${prefix}${P.sep}Lib${P.sep}site-packages`, `${prefix}${P.sep}site-packages`]
  for (const lib of ['lib', 'lib64']) {
    for (const py of await dirs($, `${prefix}${P.sep}${lib}`)) {
      if (/^python\d/.test(py)) sites.push(`${prefix}${P.sep}${lib}${P.sep}${py}${P.sep}site-packages`)
    }
  }
  for (const site of sites) {
    const m = (await dirs($, site)).map(d => /^kaggle-(\d+\.\d+\.\d+)[^/\\]*\.dist-info$/.exec(d)).find(Boolean)
    if (m) return m[1]
  }
  return undefined
}

async function fileCli($: EngineInterface, exe: string, P: Paths): Promise<Cli | undefined> {
  if (!(await $.fs.exists(exe).catch(() => false))) return undefined
  const version = await installedVersion($, exe, P)
  return version ? { argv: [exe], version } : undefined
}

/** Where `kaggle` is on PATH, found by the shell, never by running it. */
async function onPath($: EngineInterface, win: boolean): Promise<string | undefined> {
  try {
    const r = win
      ? await $.process.run(['where.exe', 'kaggle'], { timeoutMs: 10000 })
      : await $.process.run(['sh', '-c', 'command -v kaggle'], { timeoutMs: 10000 })
    const first = r.stdout.split(/\r?\n/).map(l => l.trim()).find(Boolean)
    return r.exitCode === 0 && first ? first : undefined
  } catch {
    return undefined
  }
}

/** `python -m kaggle`, when that Python has the package: asked through importlib.metadata, which imports nothing. */
async function moduleCli($: EngineInterface, py: string): Promise<Cli | undefined> {
  try {
    const r = await $.process.run([py, '-c', "import importlib.metadata as m; print(m.version('kaggle'))"], { timeoutMs: 15000 })
    const m = VERSION_RE.exec(r.stdout)
    return r.exitCode === 0 && m ? { argv: [py, '-m', 'kaggle'], version: m[0] } : undefined
  } catch {
    return undefined
  }
}

/**
 * Finds the Kaggle CLI without contacting Kaggle: the path saved from the pane,
 * else `kaggle` on PATH, else the newest one in a conda env or the user's Python
 * scripts, else `python -m kaggle`.
 */
export async function findCli($: EngineInterface, platform: Platform, P: Paths, saved?: string[]): Promise<Cli | undefined> {
  const win = platform === 'windows'
  const savedExe = saved?.[0]
  if (savedExe) {
    const s = await fileCli($, savedExe, P)
    if (s) return s
    if (await $.fs.exists(savedExe).catch(() => false)) return { argv: [savedExe], version: '?' }
  }
  const path = await onPath($, win)
  if (path) {
    const c = await fileCli($, path, P)
    if (c) return c
  }

  const exe = win ? 'kaggle.exe' : 'kaggle'
  const bin = win ? 'Scripts' : 'bin'
  const files: string[] = []
  for (const root of ['anaconda3', 'miniconda3', 'miniforge3', 'mambaforge', '.conda', 'micromamba']) {
    const base = `${P.home}${P.sep}${root}`
    files.push(`${base}${P.sep}${bin}${P.sep}${exe}`)
    for (const env of await dirs($, `${base}${P.sep}envs`)) files.push(`${base}${P.sep}envs${P.sep}${env}${P.sep}${bin}${P.sep}${exe}`)
  }
  if (win) {
    for (const v of ['LOCALAPPDATA', 'APPDATA'] as const) {
      const dir = v === 'LOCALAPPDATA' ? await $.env.get('LOCALAPPDATA') : await $.env.get('APPDATA')
      if (!dir) continue
      const sub = v === 'LOCALAPPDATA' ? `${dir}\\Programs\\Python` : `${dir}\\Python`
      for (const py of await dirs($, sub)) files.push(`${sub}\\${py}\\Scripts\\${exe}`)
    }
  } else {
    files.push(`${P.home}/.local/bin/kaggle`)
  }
  const found = (await Promise.all(files.map(f => fileCli($, f, P)))).filter((c): c is Cli => !!c)
  found.sort((a, b) => versionKey(b.version) - versionKey(a.version))
  if (found[0]) return found[0]
  if (path) return { argv: [path], version: '?' }

  for (const py of win ? ['py', 'python'] : ['python3', 'python']) {
    const c = await moduleCli($, py)
    if (c) return c
  }
  return undefined
}

export type Ran = { ok: true; stdout: string } | { ok: false; error: string }

/** Why a CLI call failed, on one line, never more than the CLI printed (it prints no token without VERBOSE). */
export function failure(stdout: string, stderr: string, code: number): string {
  const all = `${stdout}\n${stderr}`
  if (/\b401\b|Unauthorized|Invalid credentials|token.*(invalid|expired)/i.test(all)) return 'token rejected by Kaggle (401)'
  if (/\b403\b|Forbidden/i.test(all)) return 'not allowed (403): rules not accepted, or not a member'
  if (/\b404\b|Not Found/i.test(all)) return 'not found (404)'
  if (/\b429\b|Too Many Requests/i.test(all)) return 'rate limited by Kaggle (429), retrying later'
  if (/NameResolution|ConnectionError|Max retries|timed out|Network is unreachable/i.test(all)) return 'no connection to Kaggle'
  const lines = all.split('\n').map(l => l.trim()).filter(Boolean)
  return oneLine(lines[lines.length - 1] ?? `exit status ${code}`, 140)
}

/** The CLI's environment as `account`. VERBOSE is blanked: with it on the CLI prints the bearer token. */
export function accountEnv(account: Account): Record<string, string> {
  return {
    KAGGLE_API_TOKEN: account.tokenPath,
    KAGGLE_USERNAME: account.user ?? '',
    PYTHONIOENCODING: 'utf-8',
    PYTHONUNBUFFERED: '1',
    VERBOSE: '',
    VERBOSE_OUTPUT: '',
  }
}

/** Runs the Kaggle CLI as `account`. */
export async function kaggle($: EngineInterface, cli: Cli, account: Account, args: string[], timeoutMs = 60000): Promise<Ran> {
  try {
    const r = await $.process.run([...cli.argv, ...args], { timeoutMs, env: accountEnv(account) })
    return r.exitCode === 0 ? { ok: true, stdout: r.stdout } : { ok: false, error: failure(r.stdout, r.stderr, r.exitCode) }
  } catch (err) {
    return {
      ok: false,
      error: /timeout|still running/i.test(String(err))
        ? `Kaggle did not answer within ${Math.round(timeoutMs / 1000)} s: check that kaggle.com is reachable from this machine`
        : oneLine(String(err), 140),
    }
  }
}

function jsonArray(text: string): unknown[] {
  const at = text.indexOf('[')
  if (at < 0) return []
  try {
    const v = JSON.parse(text.slice(at)) as unknown
    return Array.isArray(v) ? v : []
  } catch {
    return []
  }
}

const hours = (v: unknown): number => {
  const m = /([0-9.]+)/.exec(String(v ?? ''))
  return m ? Number(m[1]) : 0
}

export function parseQuota(text: string): QuotaRow[] {
  return jsonArray(text)
    .filter((r): r is Record<string, unknown> => !!r && typeof r === 'object')
    .map(r => ({
      resource: String(r.resource ?? '?').toUpperCase(),
      usedH: hours(r.used),
      remainingH: hours(r.remaining),
      totalH: hours(r.total),
      refreshAt: typeof r.refreshAt === 'string' ? r.refreshAt : undefined,
    }))
}

export async function quota($: EngineInterface, cli: Cli, a: Account): Promise<QuotaRow[] | string> {
  const r = await kaggle($, cli, a, ['quota', '--format', 'json'])
  if (!r.ok) return r.error
  const rows = parseQuota(r.stdout)
  return rows.length ? rows : 'no quota rows in the CLI output (Kaggle CLI 2.2 or later is needed)'
}

/** `KernelWorkerStatus.RUNNING` and `SubmissionStatus.PENDING` read as RUNNING and PENDING. */
export const status = (s: string): string => s.replace(/^\w+Status\./, '').toUpperCase()

/** Kaggle's times carry no zone; they are UTC. */
export function utc(t: string | undefined): number | undefined {
  if (!t) return undefined
  const ms = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(t) ? t : `${t}Z`)
  return Number.isFinite(ms) ? ms : undefined
}

/** Kernels of `a` that ran in the last `withinH` hours, each with its status. */
export async function kernels($: EngineInterface, cli: Cli, a: Account, now: number, withinH = 24): Promise<Kernel[] | string> {
  const r = await kaggle($, cli, a, ['kernels', 'list', '--mine', '--sort-by', 'dateRun', '--page-size', '8', '--format', 'json'])
  if (!r.ok) return r.error
  const recent = jsonArray(r.stdout)
    .filter((k): k is Record<string, unknown> => !!k && typeof k === 'object' && typeof (k as Record<string, unknown>).ref === 'string')
    .filter(k => {
      const at = utc(typeof k.lastRunTime === 'string' ? k.lastRunTime : undefined)
      return at !== undefined && now - at < withinH * 3600_000
    })
  return Promise.all(
    recent.map(async (k): Promise<Kernel> => {
      const ref = String(k.ref)
      const s = await kaggle($, cli, a, ['kernels', 'status', ref], 30000)
      const m = s.ok ? /has status "([^"]+)"/.exec(s.stdout) : null
      return {
        ref,
        title: oneLine(String(k.title ?? ref), 80),
        status: m ? status(m[1] ?? '') : s.ok ? 'UNKNOWN' : 'UNREADABLE',
        lastRunTime: typeof k.lastRunTime === 'string' ? k.lastRunTime : undefined,
      }
    }),
  )
}

export async function submissions($: EngineInterface, cli: Cli, a: Account, competition: string): Promise<Submissions | string> {
  const r = await kaggle($, cli, a, ['competitions', 'submissions', competition, '--format', 'json'])
  if (!r.ok) return r.error
  const items: Submission[] = jsonArray(r.stdout)
    .filter((s): s is Record<string, unknown> => !!s && typeof s === 'object')
    .slice(0, 5)
    .map(s => ({
      ref: String(s.ref ?? ''),
      fileName: oneLine(String(s.fileName ?? ''), 60),
      date: String(s.date ?? ''),
      description: oneLine(String(s.description ?? ''), 100),
      status: status(String(s.status ?? '')),
      publicScore: s.publicScore != null && s.publicScore !== '' ? String(s.publicScore) : undefined,
    }))
  return { competition, items }
}

/** The username a token belongs to, from `kaggle config view` (it prints no secret). */
export async function whoami($: EngineInterface, cli: Cli, tokenPath: string): Promise<string | undefined> {
  const r = await kaggle($, cli, { alias: '?', tokenPath }, ['config', 'view'], 60000)
  if (!r.ok) return undefined
  const m = /^\s*-\s*username:\s*(\S+)\s*$/m.exec(r.stdout)
  return m && m[1] !== 'None' ? m[1] : undefined
}

// ---------------------------------------------------------------- the registry

export type Registry = { entries: Record<string, Record<string, unknown>>; exists: boolean }

export async function loadRegistry($: EngineInterface, P: Paths): Promise<Registry> {
  try {
    const v = JSON.parse(await $.fs.read(P.registry)) as unknown
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const entries: Record<string, Record<string, unknown>> = {}
      for (const [k, e] of Object.entries(v as Record<string, unknown>)) {
        if (e && typeof e === 'object' && !Array.isArray(e)) entries[k] = e as Record<string, unknown>
      }
      return { entries, exists: true }
    }
  } catch {
    // absent or unreadable: the default token alone
  }
  return { entries: {}, exists: false }
}

export function accountsOf(reg: Registry, P: Paths, hasDefault: boolean): Account[] {
  const out: Account[] = []
  for (const [alias, e] of Object.entries(reg.entries)) {
    const file = typeof e.token_file === 'string' && e.token_file ? expand(e.token_file, P) : P.defaultToken
    out.push({ alias, user: typeof e.user === 'string' ? e.user : undefined, tokenPath: file })
  }
  if (!reg.exists && hasDefault) out.push({ alias: 'main', tokenPath: P.defaultToken })
  return out
}

export async function saveRegistry($: EngineInterface, P: Paths, reg: Registry): Promise<void> {
  await $.fs.write(P.registry, `${JSON.stringify(reg.entries, null, 2)}\n`)
}

/** A registry entry for `a`, keeping fields this mod does not know. */
export function entryFor(a: Account, P: Paths, old?: Record<string, unknown>): Record<string, unknown> {
  const e: Record<string, unknown> = { ...old }
  if (a.user) e.user = a.user
  if (a.tokenPath === P.defaultToken) delete e.token_file
  else e.token_file = tilde(a.tokenPath, P)
  return e
}

// ---------------------------------------------------------------- token files

const WRITE_PRIVATE_POSIX = 'umask 077; mkdir -p -- "$(dirname -- "$1")" && cat > "$1" && chmod 600 -- "$1"'
const COPY_PRIVATE_POSIX = 'umask 077; mkdir -p -- "$(dirname -- "$1")" && tr -d " \\r\\n" < "$2" > "$1" && chmod 600 -- "$1"'
const WRITE_PRIVATE_WIN =
  "$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force -Path (Split-Path $env:KGMON_DST) | Out-Null; " +
  '$t = if ($env:KGMON_SRC) { (Get-Content -Raw -LiteralPath $env:KGMON_SRC).Trim() } else { [Console]::In.ReadToEnd().Trim() }; ' +
  '[IO.File]::WriteAllText($env:KGMON_DST, $t)'

/** Writes `token` to `dst`, readable by the owner alone; through stdin, so it is on no command line. */
export async function writeToken($: EngineInterface, platform: Platform, dst: string, token: string): Promise<boolean> {
  const r =
    platform === 'windows'
      ? await $.process.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', WRITE_PRIVATE_WIN], {
          env: { KGMON_DST: dst, KGMON_SRC: '' },
          stdin: token,
        })
      : await $.process.run(['sh', '-c', WRITE_PRIVATE_POSIX, 'sh', dst], { stdin: token })
  return r.exitCode === 0
}

/** Copies a token file to `dst`, readable by the owner alone. */
export async function copyToken($: EngineInterface, platform: Platform, dst: string, src: string): Promise<boolean> {
  const r =
    platform === 'windows'
      ? await $.process.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', WRITE_PRIVATE_WIN], {
          env: { KGMON_DST: dst, KGMON_SRC: src },
        })
      : await $.process.run(['sh', '-c', COPY_PRIVATE_POSIX, 'sh', dst, src])
  return r.exitCode === 0
}

export async function removeFile($: EngineInterface, platform: Platform, path: string): Promise<void> {
  await (platform === 'windows'
    ? $.process.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', 'Remove-Item -LiteralPath $env:KGMON_DST -Force'], {
        env: { KGMON_DST: path },
      })
    : $.process.run(['rm', '-f', '--', path])
  ).catch(() => undefined)
}

export type TokenInput = { kind: 'file'; path: string } | { kind: 'token'; token: string } | { kind: 'bad'; why: string }

/** What the Add field holds: a token file's path, a pasted token, or neither. */
export async function readTokenInput($: EngineInterface, text: string, P: Paths): Promise<TokenInput> {
  const t = text.trim().replace(/^["']|["']$/g, '')
  if (!t) return { kind: 'bad', why: 'Paste an API token, or the path of a token file.' }
  const path = expand(t, P)
  const looksPath = /^(~|\.{0,2}[\\/]|[A-Za-z]:[\\/])/.test(t) || /[\\/]/.test(t)
  if (looksPath || (await $.fs.exists(path).catch(() => false))) {
    if (!(await $.fs.exists(path).catch(() => false))) return { kind: 'bad', why: `No file at ${t}.` }
    if (/kaggle\.json$/i.test(path)) {
      return {
        kind: 'bad',
        why:
          'kaggle.json holds a legacy API key, which the Kaggle CLI uses only when no access token is set up. ' +
          'Generate an API token at kaggle.com/settings (API, Generate New Token) and paste it here.',
      }
    }
    return { kind: 'file', path }
  }
  if (/^\{/.test(t)) return { kind: 'bad', why: 'That looks like kaggle.json: paste the API token itself instead.' }
  if (/\s/.test(t) || t.length < 20 || t.length > 512) return { kind: 'bad', why: 'That does not look like a Kaggle API token.' }
  return { kind: 'token', token: t }
}

// ---------------------------------------------------------------- live readings

export type Reading = Pick<KernelLive, 't' | 'cpu' | 'cores' | 'ram' | 'disk' | 'tmp' | 'gpus' | 'doing'>

const pair = (v: unknown): [number, number] | undefined =>
  Array.isArray(v) && v.length === 2 && v.every(n => typeof n === 'number') ? [v[0] as number, v[1] as number] : undefined
const n = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)

/** The reading a `[kgmon] {...}` log line holds (the kgkit runner and the notebook cell print them). */
export function parseKgmon(line: string): Reading | undefined {
  const at = line.indexOf('[kgmon] ')
  if (at < 0) return undefined
  let j: Record<string, unknown>
  try {
    j = JSON.parse(line.slice(at + 8).trim()) as Record<string, unknown>
  } catch {
    return undefined
  }
  if (!j || typeof j !== 'object' || j.v !== 1) return undefined
  const gpus = Array.isArray(j.gpus)
    ? j.gpus
        .filter((g): g is unknown[] => Array.isArray(g) && g.length >= 4)
        .map(g => [n(g[0]) ?? 0, n(g[1]) ?? 0, n(g[2]) ?? 0, n(g[3]) ?? 0, oneLine(String(g[4] ?? ''), 40)] as [number, number, number, number, string])
    : undefined
  return {
    t: n(j.t),
    cpu: n(j.cpu),
    cores: n(j.cores),
    ram: pair(j.ram),
    disk: pair(j.disk),
    tmp: pair(j.tmp),
    gpus,
    doing: Array.isArray(j.doing) ? j.doing.slice(0, 8).map(d => oneLine(String(d), 80)) : undefined,
  }
}
