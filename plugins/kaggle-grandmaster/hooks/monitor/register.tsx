import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { AccountView, CliInfo, Fetched, Kernel, KernelLive, Machine, Note, Proc, QuotaRow, Scope, Submissions, Tab } from '../../types'
import {
  type Account,
  type Cli,
  type Paths,
  type Reading,
  type Registry,
  accountEnv,
  accountsOf,
  copyToken,
  entryFor,
  failure,
  findCli,
  kernels as fetchKernels,
  loadRegistry,
  parseKgmon,
  paths,
  quota as fetchQuota,
  readTokenInput,
  removeFile,
  saveRegistry,
  submissions as fetchSubmissions,
  tilde,
  utc,
  whoami,
  writeToken,
} from './kaggle'
import { lastLine } from './describe'
import { type Platform, type Prev, machine as machineOf, nextPrev, platformOf, readGpus, readRaw, readTails, rows } from './probe'

const PANE = 'kaggle-monitor'
const TITLE = 'Kaggle'
const TICK_MS = 3000
const QUOTA_MS = 10 * 60_000
const KERNELS_BUSY_MS = 60_000
const KERNELS_IDLE_MS = 5 * 60_000
const SUBS_PENDING_MS = 20_000
const SUBS_IDLE_MS = 5 * 60_000
const ALIAS_RE = /^[A-Za-z0-9_-]{1,32}$/
const LIVE = new Set(['RUNNING', 'QUEUED', 'NEW', 'PENDING', 'CANCEL_REQUESTED'])

// What the pane draws: host-held, so it outlives a reload. Each tag changes with its value's fields.
const tabA = atom({ plugin: 'kaggle-grandmaster', key: 'tab' } as const, 'activity' as Tab)
const scopeA = atom({ plugin: 'kaggle-grandmaster', key: 'scope' } as const, 'ml' as Scope)
const procsA = atom({ plugin: 'kaggle-grandmaster', key: 'procs' } as const, [] as Proc[], { shape: 'procs-1' })
const machineA = atom({ plugin: 'kaggle-grandmaster', key: 'machine' } as const, null as Machine | null, { shape: 'machine-1' })
const accountsA = atom({ plugin: 'kaggle-grandmaster', key: 'accounts' } as const, [] as AccountView[], { shape: 'accounts-1' })
const quotasA = atom({ plugin: 'kaggle-grandmaster', key: 'quotas' } as const, {} as Record<string, Fetched<QuotaRow[]>>, {
  shape: 'quotas-1',
})
const kernelsA = atom({ plugin: 'kaggle-grandmaster', key: 'kernels' } as const, {} as Record<string, Fetched<Kernel[]>>, {
  shape: 'kernels-1',
})
const subsA = atom({ plugin: 'kaggle-grandmaster', key: 'submissions' } as const, null as Fetched<Submissions> | null, {
  shape: 'subs-1',
})
const cliA = atom({ plugin: 'kaggle-grandmaster', key: 'cli' } as const, {} as CliInfo, { shape: 'cli-1' })
const noteA = atom({ plugin: 'kaggle-grandmaster', key: 'note' } as const, null as Note | null, { shape: 'note-1' })
const confirmA = atom({ plugin: 'kaggle-grandmaster', key: 'confirmRemove' } as const, '')
const liveA = atom({ plugin: 'kaggle-grandmaster', key: 'live' } as const, {} as Record<string, KernelLive>, { shape: 'live-1' })
const showLocalA = atom({ plugin: 'kaggle-grandmaster', key: 'showLocal' } as const, false)

// ---------------------------------------------------------------- formatting

const h1 = (n: number) => (n >= 100 ? n.toFixed(0) : n.toFixed(1))

function age(s: number | undefined): string {
  if (s === undefined) return '-'
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m`
  if (s < 86400) return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}`
  return `${Math.floor(s / 86400)}d${Math.floor((s % 86400) / 3600)}h`
}

function mb(n: number | undefined): string {
  if (n === undefined) return '-'
  return n >= 1024 ? `${(n / 1024).toFixed(1)}G` : `${Math.round(n)}M`
}

function rate(b: number | undefined): string {
  if (b === undefined) return '-'
  if (b < 1024) return b < 1 ? '0' : `${Math.round(b)}B`
  if (b < 1048576) return `${Math.round(b / 1024)}K`
  return `${(b / 1048576).toFixed(1)}M`
}

function bar(frac: number, width: number): string {
  const f = Math.max(0, Math.min(1, frac))
  const full = Math.round(f * width)
  return '█'.repeat(full) + '░'.repeat(width - full)
}

function until(iso: string | undefined, now: number): string | undefined {
  const t = utc(iso)
  return t === undefined ? undefined : t > now ? `resets in ${age((t - now) / 1000)}` : 'reset due'
}

/** A bright blue that reads well on a black terminal. */
const RUNNING_BLUE = '#5fafff'

/** Running blue, failed red, complete green, waiting yellow; anything else (cancelled, unknown) stays dim. */
const statusColor = (s: string) =>
  s === 'RUNNING'
    ? RUNNING_BLUE
    : s === 'COMPLETE'
      ? 'success'
      : s === 'ERROR' || s === 'FAILED'
        ? 'error'
        : s === 'QUEUED' || s === 'PENDING' || s === 'NEW'
          ? 'warning'
          : undefined

// ---------------------------------------------------------------- the mod

// Module state: the module is evaluated afresh at a reload, which runs session.start again.
let platform: Platform = 'other'
let P: Paths | undefined
let cli: Cli | undefined
let reg: Registry = { entries: {}, exists: false }
let accounts: Account[] = []
let active: string | undefined
let isSwitched = false
let competition: string | undefined
let prev: Prev | undefined
let tickN = 0
let isTicking = false
let hasGpu = true
let gpuRetryAt = 0
let formAlias = ''
let formToken = ''
let formGen = 0
let cliPathInput = ''
const busy = new Set<string>()
const last = new Map<string, number>()
const seenKernel = new Map<string, string>()
const seenSub = new Map<string, string>()
/** One `kaggle kernels logs -f` per running kernel, by ref. */
const followers = new Map<string, ReturnType<EngineInterface['process']['spawn']>>()
const FLUSH_MS = 1000
const STALE_MS = 120_000

const activeAccount = () => accounts.find(a => a.alias === active)

async function publishAccounts($: EngineInterface) {
  const views: AccountView[] = []
  for (const a of accounts) {
    views.push({
      alias: a.alias,
      user: a.user,
      tokenPath: P ? tilde(a.tokenPath, P) : a.tokenPath,
      hasToken: await $.fs.exists(a.tokenPath).catch(() => false),
      isActive: a.alias === active,
    })
  }
  await update($, accountsA, () => views)
}

async function say($: EngineInterface, text: string, isError = false) {
  await update($, noteA, () => ({ text, isError }))
}

async function statusLine($: EngineInterface) {
  const a = activeAccount()
  if (!a) {
    $.ui.status(accounts.length ? 'Kaggle: no account chosen' : undefined)
    return
  }
  const q = (await read($, quotasA))[a.alias]?.value?.find(r => r.resource === 'GPU')
  $.ui.status(`Kaggle ${a.alias}${q ? ` · GPU ${h1(q.remainingH)}h left` : ''}`)
}

async function reloadAccounts($: EngineInterface) {
  if (!P) return
  reg = await loadRegistry($, P)
  accounts = accountsOf(reg, P, await $.fs.exists(P.defaultToken).catch(() => false))
  if (active && !activeAccount()) active = undefined
  await publishAccounts($)
}

/** Points this session's Kaggle CLI (and every Bash call after) at the account's token file. */
async function useAccount($: EngineInterface, alias: string, quiet = false): Promise<boolean> {
  const a = accounts.find(x => x.alias === alias)
  if (!a) {
    await say($, `No account named ${alias}.`, true)
    return false
  }
  const set = await $.env.set('KAGGLE_API_TOKEN', a.tokenPath)
  if (set.deny) {
    await say($, `Could not switch: ${set.deny}`, true)
    return false
  }
  await $.env.set('KAGGLE_USERNAME', a.user)
  active = alias
  isSwitched = true
  await $.store.set('active', alias)
  await publishAccounts($)
  await statusLine($)
  last.delete(`subs`)
  last.delete(`quota:${alias}`)
  if (!quiet) {
    $.ui.toast(`Kaggle account: ${alias}${a.user ? ` (${a.user})` : ''}`)
    await say($, `Now using ${alias}${a.user ? ` (${a.user})` : ''}: Kaggle CLI and kgkit calls in this session run as it.`)
  }
  return true
}

// ---------------------------------------------------------- Kaggle polling

async function job($: EngineInterface, key: string, fn: () => Promise<void>) {
  if (busy.has(key)) return
  busy.add(key)
  try {
    await fn()
  } catch (err) {
    $.ui.log(`kaggle-monitor: ${key} failed: ${String(err)}`, { to: 'debug' })
  } finally {
    last.set(key, await $.clock.now())
    busy.delete(key)
  }
}

const due = (key: string, every: number, now: number) => now - (last.get(key) ?? 0) >= every

async function pollQuota($: EngineInterface, a: Account) {
  if (!cli) return
  const c = cli
  await update($, quotasA, all => ({ ...all, [a.alias]: { ...all[a.alias], isLoading: true } }))
  const r = await fetchQuota($, c, a)
  const now = await $.clock.now()
  await update($, quotasA, all => ({
    ...all,
    [a.alias]: typeof r === 'string' ? { ...all[a.alias], error: r, isLoading: false } : { value: r, fetchedAt: now },
  }))
  if (a.alias === active) await statusLine($)
}

/** Follows a running kernel's live log: its `[kgmon]` lines are the session's resources. */
async function follow($: EngineInterface, a: Account, ref: string) {
  if (!cli || followers.has(ref)) return
  const stream = $.process.spawn({ argv: [...cli.argv, 'kernels', 'logs', '-f', ref], env: accountEnv(a) })
  followers.set(ref, stream)
  await update($, liveA, all => ({ ...all, [ref]: { ...all[ref], ref, alias: a.alias, isFollowing: true, error: undefined } }))

  let buf = ''
  let errText = ''
  let reading: Reading | undefined
  let line: string | undefined
  let isDirty = false
  let isScheduled = false
  let wroteAt = 0
  const flush = async () => {
    isScheduled = false
    if (!isDirty) return
    isDirty = false
    const now = await $.clock.now()
    wroteAt = now
    const r = reading
    const l = line
    reading = undefined
    await update($, liveA, all => ({
      ...all,
      [ref]: {
        ...all[ref],
        ref,
        alias: a.alias,
        isFollowing: followers.has(ref),
        heardAt: now,
        ...(r ? { ...r, readAt: now } : {}),
        ...(l ? { lastLine: l } : {}),
      },
    }))
  }

  try {
    for await (const chunk of stream) {
      if (chunk.stream === 'stderr') {
        errText = (errText + chunk.text).slice(-2000)
        continue
      }
      buf += chunk.text
      const lines = buf.split(/\r?\n/)
      buf = lines.pop() ?? ''
      for (const one of lines) {
        const r = parseKgmon(one)
        if (r) reading = r
        else {
          const text = lastLine(one, 200)
          if (text) line = text
        }
        isDirty = true
      }
      // The server replays the whole log on connect: write at most once a second, the latest only.
      const now = await $.clock.now()
      if (isDirty && now - wroteAt >= FLUSH_MS) await flush()
      else if (isDirty && !isScheduled) {
        isScheduled = true
        $.clock.after(FLUSH_MS, () => void flush())
      }
    }
  } catch (err) {
    errText = errText || String(err)
  } finally {
    followers.delete(ref)
  }
  await flush()
  const ended = await stream.result.catch(() => undefined)
  const code = ended?.code ?? 0
  await update($, liveA, all => ({
    ...all,
    [ref]: {
      ...all[ref],
      ref,
      alias: a.alias,
      isFollowing: false,
      error: code !== 0 || (!all[ref]?.heardAt && errText) ? failure('', errText, code) : undefined,
    },
  }))
}

function unfollow(ref: string) {
  const s = followers.get(ref)
  if (s) void s.return(undefined as never).catch(() => undefined)
  followers.delete(ref)
}

async function pollKernels($: EngineInterface, a: Account) {
  if (!cli) return
  const c = cli
  const now = await $.clock.now()
  const r = await fetchKernels($, c, a, now)
  await update($, kernelsA, all => ({
    ...all,
    [a.alias]: typeof r === 'string' ? { ...all[a.alias], error: r } : { value: r, fetchedAt: now },
  }))
  if (typeof r === 'string') return
  for (const k of r) {
    const was = seenKernel.get(k.ref)
    if (was && LIVE.has(was) && !LIVE.has(k.status)) {
      $.ui.toast(`Kaggle kernel ${k.ref} (${a.alias}): ${k.status}`)
      last.delete(`quota:${a.alias}`)
    }
    seenKernel.set(k.ref, k.status)
    if (k.status === 'RUNNING') void follow($, a, k.ref)
    else unfollow(k.ref)
  }
  // Readings of this account's kernels that no longer run are dropped.
  const running = new Set(r.filter(k => k.status === 'RUNNING').map(k => k.ref))
  await update($, liveA, all =>
    Object.fromEntries(Object.entries(all).filter(([ref, l]) => l.alias !== a.alias || running.has(ref) || followers.has(ref))),
  )
}

async function pollSubs($: EngineInterface) {
  const a = activeAccount()
  if (!cli || !a || !competition) return
  const c = cli
  const r = await fetchSubmissions($, c, a, competition)
  const now = await $.clock.now()
  await update($, subsA, s => (typeof r === 'string' ? { ...s, error: r } : { value: r, fetchedAt: now }))
  if (typeof r === 'string') return
  for (const s of r.items) {
    const was = seenSub.get(s.ref)
    if (was === 'PENDING' && s.status !== 'PENDING') {
      $.ui.toast(s.status === 'COMPLETE' ? `Submission scored ${s.publicScore ?? '?'}: ${s.fileName}` : `Submission ${s.status}: ${s.fileName}`)
    }
    seenSub.set(s.ref, s.status)
  }
}

async function kaggleJobs($: EngineInterface, now: number, isShown: boolean) {
  if (!cli) return
  for (const a of accounts) {
    if ((isShown || a.alias === active) && due(`quota:${a.alias}`, QUOTA_MS, now)) void job($, `quota:${a.alias}`, () => pollQuota($, a))
    const ks = (await read($, kernelsA))[a.alias]?.value ?? []
    const isLive = ks.some(k => LIVE.has(k.status))
    if ((isShown || isLive) && due(`kernels:${a.alias}`, isLive ? KERNELS_BUSY_MS : KERNELS_IDLE_MS, now)) {
      void job($, `kernels:${a.alias}`, () => pollKernels($, a))
    }
  }
  if (competition && activeAccount()) {
    const pending = (await read($, subsA))?.value?.items.some(s => s.status === 'PENDING') ?? false
    if ((isShown || pending) && due('subs', pending ? SUBS_PENDING_MS : SUBS_IDLE_MS, now)) void job($, 'subs', () => pollSubs($))
  }
}

// ---------------------------------------------------------- local sampling

async function sampleLocal($: EngineInterface, now: number) {
  const raw = await readRaw($, platform)
  if (!raw) return
  let gpu
  if (hasGpu || now >= gpuRetryAt) {
    gpu = await readGpus($, tickN % 2 === 0)
    hasGpu = !!gpu
    if (!gpu) gpuRetryAt = now + 5 * 60_000
  }
  const scope = await read($, scopeA)
  const list = rows(raw, prev, now, scope, gpu, cmd => /kaggle/.test(cmd) && (busy.size > 0 || followers.size > 0) && /\b(quota|kernels (list|status|logs)|competitions submissions|config view|--version)\b/.test(cmd))
  const shown = list.slice(0, 30)
  const files = new Map<number, string>()
  for (const p of shown) {
    const f = raw.procs.find(r => r.pid === p.pid)?.outFile
    if (f) files.set(p.pid, f)
  }
  const tails = platform === 'linux' ? await readTails($, [...new Set(files.values())].slice(0, 12)).catch(() => new Map<string, string>()) : new Map<string, string>()
  const withDetail = shown.map(p => {
    const f = files.get(p.pid)
    const t = f ? tails.get(f) : undefined
    return t ? { ...p, detail: t } : p
  })
  const m = machineOf(raw, prev, platform, gpu)
  prev = nextPrev(raw, now)
  await update($, procsA, () => withDetail)
  await update($, machineA, () => m)
}

async function tick($: EngineInterface) {
  if (isTicking) return
  isTicking = true
  try {
    tickN++
    const panes = await $.ui.panes()
    const isShown = panes.some(p => p.id === PANE && p.isShown)
    const now = await $.clock.now()
    await kaggleJobs($, now, isShown)
    if (isShown && (await read($, showLocalA))) await sampleLocal($, now)
  } catch (err) {
    $.ui.log(`kaggle-monitor: tick failed: ${String(err)}`, { to: 'debug' })
  } finally {
    isTicking = false
  }
}

async function detectCli($: EngineInterface) {
  if (!P) return
  await update($, cliA, c => ({ ...c, isChecking: true }))
  const saved = await $.store.get('cli')
  cli = await findCli($, platform, P, Array.isArray(saved) ? saved.map(String) : undefined)
  await update($, cliA, () =>
    cli
      ? { argv: cli.argv, version: cli.version }
      : { error: 'Kaggle CLI not found: install it (pip install kaggle) or give its path on the Accounts tab.' },
  )
}

// ---------------------------------------------------------- account changes

async function addAccount($: EngineInterface) {
  if (!P) return
  const paths_ = P
  const alias = formAlias.trim()
  if (!ALIAS_RE.test(alias)) {
    await say($, 'Give the account an alias: letters, digits, - and _, up to 32.', true)
    return
  }
  if (!cli) {
    await say($, 'The Kaggle CLI is needed to check the token: set its path below first.', true)
    return
  }
  const input = await readTokenInput($, formToken, paths_)
  if (input.kind === 'bad') {
    await say($, input.why, true)
    return
  }
  await say($, `Checking the token for ${alias}...`)
  const existing = accounts.find(a => a.alias === alias)
  const inKaggleDir = input.kind === 'file' && /[\\/]access_token[^\\/]*$/.test(input.path) && input.path.startsWith(paths_.kaggleDir + paths_.sep)
  const dst = existing && existing.tokenPath !== paths_.defaultToken ? existing.tokenPath : `${paths_.kaggleDir}${paths_.sep}access_token_${alias}`
  const tmp = `${dst}.pending`
  let check = inKaggleDir && input.kind === 'file' ? input.path : tmp
  if (!inKaggleDir) {
    const ok = input.kind === 'token' ? await writeToken($, platform, tmp, input.token) : await copyToken($, platform, tmp, input.path)
    if (!ok) {
      await say($, `Could not write ${tilde(tmp, paths_)}.`, true)
      return
    }
  }
  formToken = ''
  const user = await whoami($, cli, check)
  if (!user) {
    if (!inKaggleDir) await removeFile($, platform, tmp)
    await say($, 'Kaggle did not accept that token (or could not be reached). Nothing was saved.', true)
    return
  }
  const twin = accounts.find(a => a.user === user && a.alias !== alias)
  if (twin) {
    if (!inKaggleDir) await removeFile($, platform, tmp)
    await say($, `That token is ${user}'s, already registered as ${twin.alias}. Nothing was saved.`, true)
    return
  }
  if (!inKaggleDir) {
    const moved =
      platform === 'windows'
        ? await $.process.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', 'Move-Item -Force -LiteralPath $env:KGMON_SRC -Destination $env:KGMON_DST'], {
            env: { KGMON_SRC: tmp, KGMON_DST: dst },
          })
        : await $.process.run(['mv', '-f', '--', tmp, dst])
    if (moved.exitCode !== 0) {
      await say($, `Could not move the token into ${tilde(dst, paths_)}.`, true)
      return
    }
    check = dst
  }
  const account: Account = { alias, user, tokenPath: check }
  if (!reg.exists) {
    for (const a of accounts) reg.entries[a.alias] = entryFor(a, paths_)
    reg.exists = true
  }
  reg.entries[alias] = entryFor(account, paths_, reg.entries[alias])
  await saveRegistry($, paths_, reg)
  await reloadAccounts($)
  formAlias = ''
  formGen++
  last.delete(`quota:${alias}`)
  last.delete(`kernels:${alias}`)
  await say($, `${existing ? 'Updated' : 'Added'} ${alias} (${user}); its token is in ${tilde(check, paths_)}, readable by you alone.`)
  if (!active) await useAccount($, alias, true)
}

async function removeAccount($: EngineInterface, alias: string) {
  if (!P) return
  if ((await read($, confirmA)) !== alias) {
    await update($, confirmA, () => alias)
    $.clock.after(6000, () => void update($, confirmA, c => (c === alias ? '' : c)))
    return
  }
  await update($, confirmA, () => '')
  const a = accounts.find(x => x.alias === alias)
  if (!reg.exists) {
    for (const x of accounts) reg.entries[x.alias] = entryFor(x, P)
    reg.exists = true
  }
  delete reg.entries[alias]
  await saveRegistry($, P, reg)
  await reloadAccounts($)
  await update($, quotasA, q => Object.fromEntries(Object.entries(q).filter(([k]) => k !== alias)))
  await update($, kernelsA, k => Object.fromEntries(Object.entries(k).filter(([x]) => x !== alias)))
  const lives = await read($, liveA)
  for (const l of Object.values(lives)) if (l.alias === alias) unfollow(l.ref)
  await update($, liveA, all => Object.fromEntries(Object.entries(all).filter(([, l]) => l.alias !== alias)))
  if (active === alias || !active) {
    const next = accounts[0]
    if (next) await useAccount($, next.alias, true)
    else {
      active = undefined
      await $.env.set('KAGGLE_API_TOKEN', undefined)
      await $.env.set('KAGGLE_USERNAME', undefined)
      await $.store.delete('active')
      await statusLine($)
    }
  }
  await say($, `Removed ${alias}. Its token file ${a ? tilde(a.tokenPath, P) : ''} was left in place; delete it if you no longer need it.`)
}

async function setCliPath($: EngineInterface, text: string) {
  const t = text.trim()
  if (!t) return
  await $.store.set('cli', [P ? t.replace(/^~(?=[\\/])/, P.home) : t])
  await detectCli($)
  await say($, cli ? `Using the Kaggle CLI ${cli.version} at ${cli.argv.join(' ')}.` : `No working Kaggle CLI at ${t}.`, !cli)
}

async function refreshAll($: EngineInterface) {
  last.clear()
  await say($, 'Refreshing from Kaggle...')
  const now = await $.clock.now()
  await kaggleJobs($, now, true)
}

/** Everything session.start sets up, run after it has answered. */
async function init($: EngineInterface, cwd: string) {
  platform = await platformOf($)
  P = await paths($, platform)
  try {
    const comp = JSON.parse(await $.fs.read(`${cwd}/.kaggle-gm/competition.json`)) as { slug?: unknown }
    competition = typeof comp.slug === 'string' && comp.slug ? comp.slug : undefined
  } catch {
    competition = undefined
  }
  await reloadAccounts($)
  const stored = await $.store.get('active')
  if (typeof stored === 'string' && accounts.some(a => a.alias === stored)) {
    await useAccount($, stored, true)
  } else {
    const envToken = await $.env.get('KAGGLE_API_TOKEN')
    active = accounts.find(a => a.tokenPath === envToken)?.alias ?? accounts.find(a => a.tokenPath === P?.defaultToken)?.alias
    await publishAccounts($)
    await statusLine($)
  }
  // The pane opens only on /kaggle-monitor: opened here it came back in every new session.
  $.clock.every(TICK_MS, () => void tick($))
  await detectCli($)
  await tick($)
}

export const register: Register = on => {
  // ---------------------------------------------------------- events

  on('session.start', async ($, e, next) => {
    const started = await next(e)
    // The command first: nothing below may keep it from being declared.
    await $.command.register({
      name: 'kaggle-monitor',
      description: 'Kaggle pane: local processes, kernels, submissions, GPU/TPU quotas and accounts',
      argumentHint: '[activity|quota|accounts|use <alias>|refresh]',
    })
    $.ui.status('Kaggle: starting...')
    void init($, e.cwd).catch(async err => {
      $.ui.status('Kaggle: start failed (see /kaggle-monitor)')
      $.ui.log(`kaggle-monitor: start failed: ${String(err)}`, { to: 'debug' })
      await say($, `The monitor could not start: ${String(err)}`, true)
    })
    return started
  })

  on('command.run', { command: 'kaggle-monitor' }, async ($, e) => {
    const [sub = '', arg = ''] = e.args.trim().split(/\s+/)
    if (sub === 'use') {
      if (!arg) return { text: `Usage: /kaggle-monitor use <alias> (${accounts.map(a => a.alias).join(', ') || 'no accounts yet'})` }
      const ok = await useAccount($, arg)
      return { text: ok ? `Kaggle account: ${arg}.` : `No account named ${arg}.` }
    }
    if (sub === 'refresh') {
      await refreshAll($)
      return { text: 'Refreshing from Kaggle.' }
    }
    const tab: Tab = sub === 'quota' || sub === 'quotas' ? 'quota' : sub === 'accounts' || sub === 'account' ? 'accounts' : 'activity'
    await update($, tabA, () => tab)
    const { isPlaced } = await $.ui.open({ id: PANE, title: TITLE, focus: true })
    void tick($)
    return { text: isPlaced ? `Kaggle pane open (${tab}).` : 'Kaggle pane opened; it shows once the window is wide enough.' }
  })

  on('prompt.compose', async ($, e, next) => {
    const composed = await next(e)
    const a = activeAccount()
    if (!isSwitched || !a) return composed
    return {
      sections: [
        ...composed.sections,
        {
          id: 'kaggle-grandmaster:account',
          scope: 'session',
          text:
            `Kaggle account for this session: "${a.alias}"${a.user ? ` (Kaggle user ${a.user})` : ''}. ` +
            'The kaggle-monitor mod set KAGGLE_API_TOKEN to that account\'s token file, so Kaggle CLI and kgkit calls run as it. ' +
            'The person switches accounts in the Kaggle pane (/kaggle-monitor accounts); do not change KAGGLE_API_TOKEN or KAGGLE_USERNAME yourself.',
        },
      ],
    }
  })

  // A submit or a push makes the pane look again soon, rather than at its idle pace.
  on('tool.call', { tool: 'Bash' }, async ($, e, next) => {
    const ran = await next(e)
    if (/competitions\s+submit\b/.test(e.command)) last.delete('subs')
    if (/kernels\s+push\b|kgkit\s+gpu\s+push\b/.test(e.command)) {
      for (const a of accounts) last.delete(`kernels:${a.alias}`)
    }
    return ran
  }).catch(($, e, next) => next(e))

  // ---------------------------------------------------------- the pane

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button, Input } = $.ui.resolve(e)
    const width = e.props.bodyColumns
    const isWide = width >= 90
    const tab = await read($, tabA)
    const note = await read($, noteA)
    const views = await read($, accountsA)
    const quotas = await read($, quotasA)
    const now = await $.clock.now()
    const act = views.find(v => v.isActive)
    const gpuQ = act ? quotas[act.alias]?.value?.find(r => r.resource === 'GPU') : undefined

    const tabButton = (id: Tab, key: string, label: string) => (
      <Button
        key={`tab-${id}`}
        hotkey={key}
        label={label}
        variant={tab === id ? 'primary' : undefined}
        dimColor={tab !== id}
        onPress={() => void update($, tabA, () => id)}
      />
    )

    const header = (
      <Box flexDirection="column">
        <Box flexDirection="row" flexWrap="wrap" columnGap={1}>
          {act ? <Text color="claude" bold>{act.alias}</Text> : <Text color="warning">no Kaggle account</Text>}
          {act?.user ? <Text dimColor>{act.user}</Text> : null}
          {gpuQ ? (
            <Text color={gpuQ.remainingH < gpuQ.totalH * 0.1 ? 'error' : undefined}>· GPU {h1(gpuQ.remainingH)}h left</Text>
          ) : null}
          {competition ? <Text dimColor>· {competition}</Text> : null}
        </Box>
        <Box flexDirection="row" columnGap={1}>
          {tabButton('activity', '1', 'Activity')}
          {tabButton('quota', '2', 'Quotas')}
          {tabButton('accounts', '3', 'Accounts')}
          <Button key="refresh" hotkey="r" label="Refresh" dimColor onPress={() => void refreshAll($)} />
        </Box>
        {note ? (
          <Box flexDirection="row" columnGap={1}>
            <Text color={note.isError ? 'error' : 'success'} wrap="wrap">
              {note.text}
            </Text>
            <Button key="note-x" label="x" plain dimColor onPress={() => void update($, noteA, () => null)} />
          </Box>
        ) : null}
      </Box>
    )

    // ---- Activity
    const activity = async () => {
      const m = await read($, machineA)
      const procs = await read($, procsA)
      const scope = await read($, scopeA)
      const kernels = await read($, kernelsA)
      const subs = await read($, subsA)
      const lives = await read($, liveA)
      const showLocal = await read($, showLocalA)

      const machineRows = m ? (
        <Box flexDirection="column">
          <Box flexDirection="row" flexWrap="wrap" columnGap={2}>
            <Text>
              CPU {m.cpuPct !== undefined ? `${Math.round(m.cpuPct)}%` : '-'}
              <Text dimColor>{m.cores ? ` of ${m.cores} cores` : ''}</Text>
            </Text>
            {m.memTotalMb ? (
              <Text>
                RAM {m.memUsedMb !== undefined ? `${(m.memUsedMb / 1024).toFixed(1)}/` : ''}
                {(m.memTotalMb / 1024).toFixed(1)} GB
              </Text>
            ) : null}
            {m.diskTotalGb ? (
              <Text color={m.diskFreeGb !== undefined && m.diskFreeGb < m.diskTotalGb * 0.05 ? 'error' : undefined}>
                Disk {Math.round(m.diskFreeGb ?? 0)} GB free<Text dimColor> of {Math.round(m.diskTotalGb)}</Text>
              </Text>
            ) : null}
          </Box>
          {m.gpus.map(g => (
            <Text key={`gpu-${g.index}`}>
              GPU{g.index} <Text dimColor>{g.name}</Text> {bar(g.util / 100, 10)} {Math.round(g.util)}%{' '}
              {(g.memUsedMb / 1024).toFixed(1)}/{(g.memTotalMb / 1024).toFixed(1)} GB
            </Text>
          ))}
          {m.gpus.length === 0 ? <Text dimColor>No NVIDIA GPU here (nvidia-smi not found).</Text> : null}
        </Box>
      ) : (
        <Text dimColor>Sampling this machine...</Text>
      )

      const cell = (w: number, text: string, color?: string) => (
        <Box width={w} flexShrink={0}>
          <Text color={color} wrap="truncate">
            {text}
          </Text>
        </Box>
      )
      const procRow = (p: Proc) =>
        isWide ? (
          <Box key={`p-${p.pid}`} flexDirection="column">
            <Box flexDirection="row">
              {cell(8, String(p.pid))}
              {cell(7, p.cpu !== undefined ? `${Math.round(p.cpu)}%` : '-', (p.cpu ?? 0) > 90 ? 'success' : undefined)}
              {cell(7, mb(p.memMb))}
              {cell(12, p.gpuMb !== undefined ? `${mb(p.gpuMb)}${p.gpuPct !== undefined ? ` ${Math.round(p.gpuPct)}%` : ''}` : '-')}
              {cell(13, p.readBps !== undefined || p.writeBps !== undefined ? `${rate(p.readBps)}/${rate(p.writeBps)}` : '-')}
              {cell(7, age(p.ageS))}
              <Box flexGrow={1}>
                <Text wrap="truncate">
                  <Text bold>{p.name}</Text> <Text>{p.what}</Text>
                  {p.isSession ? <Text dimColor> · this session</Text> : null}
                </Text>
              </Box>
            </Box>
            {p.detail ? (
              <Box paddingLeft={8}>
                <Text dimColor wrap="truncate">
                  {'└ '}
                  {p.detail}
                </Text>
              </Box>
            ) : null}
          </Box>
        ) : (
          <Box key={`p-${p.pid}`} flexDirection="column">
            <Text wrap="truncate">
              <Text bold>{p.name}</Text> {p.what}
            </Text>
            <Text dimColor wrap="truncate">
              {'  '}pid {p.pid} · cpu {p.cpu !== undefined ? `${Math.round(p.cpu)}%` : '-'} · mem {mb(p.memMb)}
              {p.gpuMb !== undefined ? ` · gpu ${mb(p.gpuMb)}` : ''}
              {p.readBps !== undefined ? ` · io ${rate(p.readBps)}/${rate(p.writeBps)}/s` : ''} · {age(p.ageS)}
            </Text>
            {p.detail ? (
              <Text dimColor wrap="truncate">
                {'  └ '}
                {p.detail}
              </Text>
            ) : null}
          </Box>
        )

      const kernelRows: { alias: string; k: Kernel }[] = []
      const kernelErrors: { alias: string; error: string }[] = []
      for (const v of views) {
        const f = kernels[v.alias]
        for (const k of f?.value ?? []) kernelRows.push({ alias: v.alias, k })
        if (f?.error) kernelErrors.push({ alias: v.alias, error: f.error })
      }
      kernelRows.sort((a, b) => Number(LIVE.has(b.k.status)) - Number(LIVE.has(a.k.status)) || (utc(b.k.lastRunTime) ?? 0) - (utc(a.k.lastRunTime) ?? 0))

      const running = kernelRows.filter(x => x.k.status === 'RUNNING')
      const others = kernelRows.filter(x => x.k.status !== 'RUNNING')
      const meterW = Math.max(8, Math.min(24, width - 46))
      const gb = (mbs: number) => (mbs / 1024).toFixed(1)
      const tone = (frac: number) => (frac >= 0.9 ? 'error' : frac >= 0.75 ? 'warning' : 'success')
      const meter = (key: string, label: string, frac: number, text: string, color?: string) => (
        <Box key={key} flexDirection="row" columnGap={1}>
          <Box width={5} flexShrink={0}>
            <Text>{label}</Text>
          </Box>
          {width >= 60 ? <Text color={color ?? tone(frac)}>{bar(frac, meterW)}</Text> : null}
          <Text wrap="truncate">{text}</Text>
        </Box>
      )

      const liveCard = ({ alias, k }: { alias: string; k: Kernel }) => {
        const L = lives[k.ref]
        const hasReading = L?.readAt !== undefined
        const isStale = hasReading && now - (L?.readAt ?? 0) > STALE_MS
        const runFor = L?.t !== undefined ? age(L.t) : age((now - (utc(k.lastRunTime) ?? now)) / 1000)
        const sameDisk = L?.tmp && L.disk && L.tmp[1] === L.disk[1]
        return (
          <Box key={`live-${k.ref}`} flexDirection="column" borderStyle="round" borderColor={RUNNING_BLUE} paddingX={1}>
            <Text wrap="truncate">
              <Text color={RUNNING_BLUE} bold>
                RUNNING
              </Text>{' '}
              <Text bold>{k.ref}</Text>
              <Text dimColor>
                {' '}
                · {alias} · {runFor}
              </Text>
            </Text>
            {hasReading && L ? (
              <Box flexDirection="column">
                {L.cpu !== undefined
                  ? meter('cpu', 'CPU', L.cpu / 100, `${Math.round(L.cpu)}%${L.cores ? ` of ${L.cores} cores` : ''}`)
                  : null}
                {L.ram ? meter('ram', 'RAM', L.ram[0] / L.ram[1], `${gb(L.ram[0])} / ${gb(L.ram[1])} GB`) : null}
                {(L.gpus ?? []).map(g =>
                  meter(`gpu${g[0]}`, `GPU${g[0]}`, g[1] / 100, `${g[1]}% · ${gb(g[2])} / ${gb(g[3])} GB${g[4] ? ` · ${g[4].replace(/^Tesla\s+/i, '')}` : ''}`, g[1] < 30 ? 'warning' : 'success'),
                )}
                {L.gpus && L.gpus.length === 0 ? <Text dimColor>No GPU in this session.</Text> : null}
                {L.disk ? meter('disk', 'Disk', L.disk[0] / L.disk[1], `${gb(L.disk[0])} / ${gb(L.disk[1])} GB  /kaggle/working`) : null}
                {L.tmp && !sameDisk ? meter('tmp', 'Tmp', L.tmp[0] / L.tmp[1], `${gb(L.tmp[0])} / ${gb(L.tmp[1])} GB  scratch`) : null}
                {L.doing?.length ? (
                  <Text wrap="truncate">
                    <Text dimColor>Doing </Text>
                    {L.doing.join(' · ')}
                  </Text>
                ) : null}
                {isStale ? <Text color="warning">Last reading {age((now - (L.readAt ?? now)) / 1000)} ago.</Text> : null}
              </Box>
            ) : L?.error ? (
              <Text color="error" wrap="wrap">
                Could not follow its log: {L.error}
              </Text>
            ) : L?.heardAt ? (
              <Text dimColor wrap="wrap">
                Its log has no resource readings. Kernels pushed with kgkit gpu push print them; for any other notebook,
                paste kaggle-grandmaster's templates/kaggle_monitor_cell.py as its first cell.
              </Text>
            ) : (
              <Text dimColor>Connecting to its live log...</Text>
            )}
            {L?.lastLine ? (
              <Text dimColor wrap="truncate">
                {'└ '}
                {L.lastLine}
              </Text>
            ) : null}
          </Box>
        )
      }

      return (
        <Box flexDirection="column" rowGap={1}>
          <Box flexDirection="column" rowGap={1}>
            <Text bold>Running on Kaggle ({running.length})</Text>
            {running.length === 0 ? (
              <Text dimColor>
                {cli ? (Object.keys(kernels).length ? 'No kernel is running on any account.' : 'Reading your kernels...') : 'Waiting for the Kaggle CLI.'}
              </Text>
            ) : null}
            {running.map(liveCard)}
          </Box>
          <Box flexDirection="column">
            <Text bold>Other kernels, last 24 h</Text>
            {others.length === 0 && kernelErrors.length === 0 ? <Text dimColor>None.</Text> : null}
            {others.map(({ alias, k }) => (
              <Box key={`k-${k.ref}`} flexDirection="row" columnGap={1}>
                <Box width={11} flexShrink={0}>
                  <Text color={statusColor(k.status)} dimColor={statusColor(k.status) === undefined}>
                    {k.status}
                  </Text>
                </Box>
                <Text wrap="truncate">
                  {k.ref}
                  <Text dimColor>
                    {' '}
                    · {alias} · {age((now - (utc(k.lastRunTime) ?? now)) / 1000)} ago
                  </Text>
                </Text>
              </Box>
            ))}
            {kernelErrors.map(x => (
              <Text key={`ke-${x.alias}`} color="error" dimColor wrap="truncate">
                {x.alias}: {x.error}
              </Text>
            ))}
          </Box>
          {competition ? (
            <Box flexDirection="column">
              <Text bold>
                Submissions · {competition}
                <Text dimColor>{act ? ` as ${act.alias}` : ''}</Text>
              </Text>
              {subs?.error ? <Text color="error" dimColor>{subs.error}</Text> : null}
              {!subs?.value ? <Text dimColor>{act ? 'Reading...' : 'Choose an account on the Accounts tab.'}</Text> : null}
              {subs?.value?.items.length === 0 ? <Text dimColor>No submissions yet.</Text> : null}
              {(subs?.value?.items ?? []).map(s => (
                <Box key={`s-${s.ref}`} flexDirection="row" columnGap={1}>
                  <Box width={9} flexShrink={0}>
                    <Text color={statusColor(s.status)} dimColor={statusColor(s.status) === undefined}>
                      {s.status}
                    </Text>
                  </Box>
                  <Box width={9} flexShrink={0}>
                    <Text bold={s.status === 'COMPLETE'}>{s.publicScore ?? '-'}</Text>
                  </Box>
                  <Text wrap="truncate">
                    {s.fileName}
                    <Text dimColor>
                      {' '}
                      · {age((now - (utc(s.date) ?? now)) / 1000)} ago{s.description ? ` · ${s.description}` : ''}
                    </Text>
                  </Text>
                </Box>
              ))}
            </Box>
          ) : null}
          <Box flexDirection="column">
            <Button
              key="local"
              hotkey="c"
              label={showLocal ? '▾ This computer' : '▸ This computer (local processes)'}
              dimColor={!showLocal}
              onPress={() => void update($, showLocalA, v => !v)}
            />
            {showLocal ? (
              <Box flexDirection="column" rowGap={1}>
                {machineRows}
                <Box flexDirection="column">
                  <Box flexDirection="row" columnGap={1}>
                    <Text bold>Processes ({procs.length})</Text>
                    <Button
                      key="scope"
                      hotkey="s"
                      label={scope === 'ml' ? 'ML work + this session' : 'this session only'}
                      dimColor
                      onPress={() => void update($, scopeA, x => (x === 'ml' ? 'session' : 'ml'))}
                    />
                  </Box>
                  {isWide ? (
                    <Box flexDirection="row">
                      {cell(8, 'PID')}
                      {cell(7, 'CPU')}
                      {cell(7, 'MEM')}
                      {cell(12, 'GPU mem/sm')}
                      {cell(13, 'DISK r/w /s')}
                      {cell(7, 'AGE')}
                      <Text dimColor>WHAT</Text>
                    </Box>
                  ) : null}
                  {procs.length === 0 ? <Text dimColor>Nothing running: no training, notebook, Kaggle CLI or session task.</Text> : null}
                  {procs.map(procRow)}
                  {m?.note ? <Text dimColor>{m.note}</Text> : null}
                </Box>
              </Box>
            ) : null}
          </Box>
        </Box>
      )
    }

    // ---- Quotas
    const quotaTab = () => {
      const barW = Math.max(8, Math.min(30, width - 56))
      const totals = new Map<string, number>()
      for (const v of views) for (const r of quotas[v.alias]?.value ?? []) totals.set(r.resource, (totals.get(r.resource) ?? 0) + r.remainingH)
      return (
        <Box flexDirection="column" rowGap={1}>
          {views.length === 0 ? <Text dimColor>No accounts yet: add one on the Accounts tab.</Text> : null}
          {totals.size > 1 || views.length > 1 ? (
            <Text>
              Left across {views.length} accounts:{' '}
              {[...totals].map(([res, hrs]) => `${res} ${h1(hrs)}h`).join(' · ')}
            </Text>
          ) : null}
          {views.map(v => {
            const q = quotas[v.alias]
            return (
              <Box key={`q-${v.alias}`} flexDirection="column">
                <Text>
                  <Text color={v.isActive ? 'claude' : undefined} bold>
                    {v.isActive ? '● ' : '  '}
                    {v.alias}
                  </Text>
                  <Text dimColor>
                    {v.user ? ` ${v.user}` : ''}
                    {q?.fetchedAt ? ` · updated ${age((now - q.fetchedAt) / 1000)} ago` : ''}
                    {q?.isLoading ? ' · reading...' : ''}
                  </Text>
                </Text>
                {(q?.value ?? []).map(r => {
                  const left = r.totalH ? r.remainingH / r.totalH : 0
                  const color = left < 0.1 ? 'error' : left < 0.3 ? 'warning' : 'success'
                  return (
                    <Box key={`q-${v.alias}-${r.resource}`} flexDirection="row" columnGap={1} paddingLeft={2}>
                      <Box width={4} flexShrink={0}>
                        <Text>{r.resource}</Text>
                      </Box>
                      {width >= 60 ? <Text color={color}>{bar(r.totalH ? r.usedH / r.totalH : 0, barW)}</Text> : null}
                      <Text>
                        <Text color={color} bold>
                          {h1(r.remainingH)}h left
                        </Text>
                        <Text dimColor>
                          {' '}
                          of {h1(r.totalH)}h{until(r.refreshAt, now) ? ` · ${until(r.refreshAt, now)}` : ''}
                        </Text>
                      </Text>
                    </Box>
                  )
                })}
                {q?.error ? (
                  <Box paddingLeft={2}>
                    <Text color="error" wrap="wrap">
                      {q.error}
                    </Text>
                  </Box>
                ) : null}
                {!q ? (
                  <Box paddingLeft={2}>
                    <Text dimColor>{cli ? 'Reading...' : 'Waiting for the Kaggle CLI.'}</Text>
                  </Box>
                ) : null}
              </Box>
            )
          })}
        </Box>
      )
    }

    // ---- Accounts
    const accountsTab = async () => {
      const confirm = await read($, confirmA)
      const c = await read($, cliA)
      return (
        <Box flexDirection="column" rowGap={1}>
          <Box flexDirection="column">
            {views.length === 0 ? <Text dimColor>No Kaggle account registered yet.</Text> : null}
            {views.map(v => (
              <Box key={`a-${v.alias}`} flexDirection="row" flexWrap="wrap" columnGap={1}>
                <Text color={v.isActive ? 'claude' : undefined} bold>
                  {v.isActive ? '●' : '○'} {v.alias}
                </Text>
                <Text>{v.user ?? '?'}</Text>
                <Text dimColor wrap="truncate">
                  {v.tokenPath}
                </Text>
                {!v.hasToken ? <Text color="error">token file missing</Text> : null}
                {v.isActive ? (
                  <Text color="claude">in use</Text>
                ) : (
                  <Button key={`use-${v.alias}`} label="Use" variant="primary" onPress={() => void useAccount($, v.alias)} />
                )}
                <Button
                  key={`rm-${v.alias}`}
                  label={confirm === v.alias ? 'Press again to remove' : 'Remove'}
                  dimColor={confirm !== v.alias}
                  onPress={() => void removeAccount($, v.alias)}
                />
              </Box>
            ))}
          </Box>
          <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
            <Text bold>Add an account, or replace an account's token</Text>
            <Input
              key={`alias-${formGen}`}
              label="Alias  "
              placeholder="acc4"
              onInput={v => {
                formAlias = v
              }}
              onSubmit={v => {
                formAlias = v
              }}
              submitLabel="next"
            />
            <Input
              key={`token-${formGen}`}
              label="Token  "
              placeholder="paste the API token, or the path of a token file"
              onInput={v => {
                formToken = v
              }}
              onSubmit={v => {
                formToken = v
                void addAccount($)
              }}
              submitLabel="add"
            />
            <Box flexDirection="row" columnGap={1}>
              <Button key="add" label="Add account" variant="primary" onPress={() => void addAccount($)} />
            </Box>
            <Text dimColor wrap="wrap">
              Get a token at kaggle.com/settings, API, Generate New Token. A pasted token shows in the field while you type, is
              checked with Kaggle, then saved to ~/.kaggle/access_token_{'<alias>'} (readable by you alone) and never shown again.
              ~/.kaggle/kg_accounts.json keeps only the alias, the username and the file's path.
            </Text>
          </Box>
          <Box flexDirection="column">
            <Text>
              <Text bold>Kaggle CLI </Text>
              {c.isChecking ? (
                <Text dimColor>looking...</Text>
              ) : c.argv ? (
                <Text dimColor>
                  {c.version} · {c.argv.join(' ')}
                </Text>
              ) : (
                <Text color="error">{c.error ?? 'not found'}</Text>
              )}
            </Text>
            <Input
              key="cli-path"
              label="Path   "
              placeholder="another kaggle executable, e.g. ~/miniconda3/envs/kaggle/bin/kaggle"
              onInput={v => {
                cliPathInput = v
              }}
              onSubmit={v => void setCliPath($, v || cliPathInput)}
              submitLabel="use"
            />
          </Box>
        </Box>
      )
    }

    const body = tab === 'quota' ? quotaTab() : tab === 'accounts' ? await accountsTab() : await activity()

    return (
      <Box flexDirection="column" rowGap={1}>
        {header}
        {body}
      </Box>
    )
  })
}
