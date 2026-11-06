// Local processes, machine load and GPUs, sampled through `$.process.run`.
// The parsers are pure and exported for the tests.

import type { EngineInterface } from 'claude-code'

import type { GpuDev, Machine, Proc, Scope } from '../../types'
import { describe, INFRA_RE, isShell, lastLine, ML_RE } from './describe'

export type Platform = 'linux' | 'darwin' | 'windows' | 'other'

/** One process as read, before rates are known. */
export type Raw = {
  pid: number
  ppid: number
  cmd: string
  /** CPU seconds used so far (Linux, Windows). */
  cpuS?: number
  /** Percent of a core as `ps` reports it (macOS, where it is a recent average). */
  pcpu?: number
  rssKb?: number
  ageS?: number
  readB?: number
  writeB?: number
  /** Where its standard output goes, when that is a regular file (Linux). */
  outFile?: string
}

export type RawSample = {
  /** The Claude Code process: every process this session started descends from it. */
  self: number
  /** The probe's own shell, whose children are the probe's commands. */
  probe: number
  procs: Raw[]
  cpuBusy?: number
  cpuTotal?: number
  cpuPct?: number
  cores?: number
  memTotalMb?: number
  memUsedMb?: number
  diskFreeGb?: number
  diskTotalGb?: number
  note?: string
}

export async function platformOf($: EngineInterface): Promise<Platform> {
  if ((await $.env.get('OS')) === 'Windows_NT') return 'windows'
  try {
    const { stdout } = await $.process.run(['uname', '-s'], { timeoutMs: 5000 })
    const s = stdout.trim().toLowerCase()
    return s === 'linux' ? 'linux' : s === 'darwin' ? 'darwin' : 'other'
  } catch {
    return 'other'
  }
}

const LINUX = `
echo "@SELF $PPID $$"
echo "@HZ $(getconf CLK_TCK 2>/dev/null || echo 100) $(getconf PAGESIZE 2>/dev/null || echo 4096) $(nproc 2>/dev/null || echo 1)"
echo "@UP $(cut -d' ' -f1 /proc/uptime)"
echo "@CPU $(head -1 /proc/stat)"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo | sed 's/^/@MEM /'
df -kP . 2>/dev/null | tail -1 | sed 's/^/@DF /'
echo "@PS"
ps xww -o pid=,ppid=,args=
echo "@STAT"
cat /proc/[0-9]*/stat 2>/dev/null
echo "@IO"
grep -H -E '^(read|write)_bytes' /proc/[0-9]*/io 2>/dev/null
echo "@FD"
ls -l /proc/[0-9]*/fd/1 2>/dev/null
echo "@END"
`

const DARWIN = `
echo "@SELF $PPID $$"
echo "@HZ 100 4096 $(sysctl -n hw.ncpu)"
echo "@MACMEM $(sysctl -n hw.memsize)"
echo "@MACCPU $(ps -A -o pcpu= | awk '{s+=$1} END {print s+0}')"
df -kP . 2>/dev/null | tail -1 | sed 's/^/@DF /'
echo "@PS"
ps xww -o pid=,ppid=,pcpu=,rss=,etime=,args=
echo "@END"
`

const WINDOWS = `
$ErrorActionPreference = 'SilentlyContinue'
$me = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$os = Get-CimInstance Win32_OperatingSystem
$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
$drive = (Get-Item -LiteralPath (Get-Location).Path).PSDrive
$now = Get-Date
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | ForEach-Object {
  $age = -1
  if ($_.CreationDate) { $age = [int]($now - $_.CreationDate).TotalSeconds }
  [pscustomobject]@{ p = [int]$_.ProcessId; pp = [int]$_.ParentProcessId; c = [string]$_.CommandLine;
    t = [double]($_.UserModeTime + $_.KernelModeTime) / 1e7; m = [double]$_.WorkingSetSize;
    r = [double]$_.ReadTransferCount; w = [double]$_.WriteTransferCount; a = $age }
}
[pscustomobject]@{ self = [int]$me.ParentProcessId; me = $PID; cores = [Environment]::ProcessorCount; cpu = $cpu;
  memTotalKb = [double]$os.TotalVisibleMemorySize; memFreeKb = [double]$os.FreePhysicalMemory;
  diskFree = [double]$drive.Free; diskUsed = [double]$drive.Used; procs = @($procs) } | ConvertTo-Json -Compress -Depth 3
`

const num = (s: string | undefined): number | undefined => {
  const n = Number(s)
  return Number.isFinite(n) ? n : undefined
}

function sections(text: string): Map<string, string[]> {
  const out = new Map<string, string[]>()
  let cur = ''
  for (const line of text.split('\n')) {
    const m = /^@(SELF|HZ|UP|CPU|MEM|DF|PS|STAT|IO|FD|MACMEM|MACCPU|END)\b ?(.*)$/.exec(line)
    if (m) {
      cur = m[1] ?? ''
      const rest = m[2] ?? ''
      if (rest) out.set(cur, [...(out.get(cur) ?? []), rest])
      else if (!out.has(cur)) out.set(cur, [])
      continue
    }
    if (cur && line) out.set(cur, [...(out.get(cur) ?? []), line])
  }
  return out
}

/** `df -kP` row: free and total, in GB. */
function df(rows: string[] | undefined): { diskFreeGb?: number; diskTotalGb?: number } {
  const cols = (rows?.[0] ?? '').trim().split(/\s+/)
  const total = num(cols[1])
  const free = num(cols[3])
  return total && free !== undefined ? { diskTotalGb: total / 1048576, diskFreeGb: free / 1048576 } : {}
}

export function parseLinux(text: string): RawSample {
  const s = sections(text)
  const [self = 0, probe = 0] = (s.get('SELF')?.[0] ?? '').split(' ').map(Number)
  const [hz = 100, page = 4096, cores = 1] = (s.get('HZ')?.[0] ?? '').split(' ').map(Number)
  const uptime = num(s.get('UP')?.[0]) ?? 0

  const cpuCols = (s.get('CPU')?.[0] ?? '').trim().split(/\s+/).slice(1).map(Number)
  const cpuTotal = cpuCols.slice(0, 8).reduce((a, b) => a + (b || 0), 0)
  const cpuIdle = (cpuCols[3] ?? 0) + (cpuCols[4] ?? 0)

  const mem: Record<string, number> = {}
  for (const row of s.get('MEM') ?? []) {
    const m = /^(\w+):\s+(\d+)/.exec(row)
    if (m) mem[m[1] ?? ''] = Number(m[2])
  }

  const stat = new Map<number, { cpuS: number; rssKb: number; ageS: number }>()
  for (const row of s.get('STAT') ?? []) {
    const close = row.lastIndexOf(')')
    const pid = Number(row.slice(0, row.indexOf(' ')))
    if (close < 0 || !pid) continue
    const f = row.slice(close + 2).split(' ')
    // Fields after the command's name: 0 is field 3 of proc(5), so field n is f[n - 3].
    const ticks = Number(f[11]) + Number(f[12])
    const start = Number(f[19])
    const rssPages = Number(f[21])
    stat.set(pid, { cpuS: ticks / hz, ageS: Math.max(0, uptime - start / hz), rssKb: (rssPages * page) / 1024 })
  }

  const io = new Map<number, { readB?: number; writeB?: number }>()
  for (const row of s.get('IO') ?? []) {
    const m = /^\/proc\/(\d+)\/io:(read|write)_bytes:\s*(\d+)/.exec(row)
    if (!m) continue
    const pid = Number(m[1])
    const one = io.get(pid) ?? {}
    if (m[2] === 'read') one.readB = Number(m[3])
    else one.writeB = Number(m[3])
    io.set(pid, one)
  }

  const out = new Map<number, string>()
  for (const row of s.get('FD') ?? []) {
    const m = /\/proc\/(\d+)\/fd\/1 -> (.+)$/.exec(row)
    const target = m?.[2] ?? ''
    if (m && target.startsWith('/') && !target.startsWith('/dev/') && !target.endsWith(' (deleted)')) {
      out.set(Number(m[1]), target)
    }
  }

  const procs: Raw[] = []
  for (const row of s.get('PS') ?? []) {
    const m = /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(row)
    if (!m) continue
    const pid = Number(m[1])
    procs.push({ pid, ppid: Number(m[2]), cmd: m[3] ?? '', ...stat.get(pid), ...io.get(pid), outFile: out.get(pid) })
  }

  return {
    self,
    probe,
    procs,
    cpuBusy: cpuTotal - cpuIdle,
    cpuTotal,
    cores,
    memTotalMb: mem.MemTotal ? mem.MemTotal / 1024 : undefined,
    memUsedMb: mem.MemTotal && mem.MemAvailable !== undefined ? (mem.MemTotal - mem.MemAvailable) / 1024 : undefined,
    ...df(s.get('DF')),
  }
}

/** `ps` elapsed time, `[[dd-]hh:]mm:ss`, in seconds. */
export function etime(s: string): number | undefined {
  const m = /^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$/.exec(s.trim())
  if (!m) return undefined
  return Number(m[1] ?? 0) * 86400 + Number(m[2] ?? 0) * 3600 + Number(m[3]) * 60 + Number(m[4])
}

export function parseDarwin(text: string): RawSample {
  const s = sections(text)
  const [self = 0, probe = 0] = (s.get('SELF')?.[0] ?? '').split(' ').map(Number)
  const cores = num((s.get('HZ')?.[0] ?? '').split(' ')[2])
  const memBytes = num(s.get('MACMEM')?.[0])
  const allPcpu = num(s.get('MACCPU')?.[0])
  const procs: Raw[] = []
  for (const row of s.get('PS') ?? []) {
    const m = /^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\S+)\s+(.*)$/.exec(row)
    if (!m) continue
    procs.push({
      pid: Number(m[1]),
      ppid: Number(m[2]),
      pcpu: Number(m[3]),
      rssKb: Number(m[4]),
      ageS: etime(m[5] ?? ''),
      cmd: m[6] ?? '',
    })
  }
  return {
    self,
    probe,
    procs,
    cores,
    cpuPct: allPcpu !== undefined && cores ? Math.min(100, allPcpu / cores) : undefined,
    memTotalMb: memBytes ? memBytes / 1048576 : undefined,
    note: 'macOS does not report per-process disk I/O without root',
    ...df(s.get('DF')),
  }
}

type WinJson = {
  self?: number
  me?: number
  cores?: number
  cpu?: number
  memTotalKb?: number
  memFreeKb?: number
  diskFree?: number
  diskUsed?: number
  procs?: { p: number; pp: number; c: string; t: number; m: number; r: number; w: number; a: number }[]
}

export function parseWindows(text: string): RawSample {
  const j = JSON.parse(text.slice(text.indexOf('{'))) as WinJson
  const total = (j.diskFree ?? 0) + (j.diskUsed ?? 0)
  return {
    self: j.self ?? 0,
    probe: j.me ?? 0,
    cores: j.cores,
    cpuPct: j.cpu,
    memTotalMb: j.memTotalKb ? j.memTotalKb / 1024 : undefined,
    memUsedMb: j.memTotalKb && j.memFreeKb !== undefined ? (j.memTotalKb - j.memFreeKb) / 1024 : undefined,
    diskFreeGb: total ? (j.diskFree ?? 0) / 1073741824 : undefined,
    diskTotalGb: total ? total / 1073741824 : undefined,
    note: 'Windows counts all I/O (disk, network, devices) per process',
    procs: (j.procs ?? []).map(p => ({
      pid: p.p,
      ppid: p.pp,
      cmd: p.c,
      cpuS: p.t,
      rssKb: p.m / 1024,
      readB: p.r,
      writeB: p.w,
      ageS: p.a >= 0 ? p.a : undefined,
    })),
  }
}

export async function readRaw($: EngineInterface, platform: Platform): Promise<RawSample | undefined> {
  if (platform === 'windows') {
    const r = await $.process.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', WINDOWS], {
      timeoutMs: 20000,
    })
    return r.exitCode === 0 ? parseWindows(r.stdout) : undefined
  }
  if (platform === 'linux') {
    const r = await $.process.run(['sh', '-c', LINUX], { timeoutMs: 15000 })
    return r.stdout.includes('@END') ? parseLinux(r.stdout) : undefined
  }
  if (platform === 'darwin') {
    const r = await $.process.run(['sh', '-c', DARWIN], { timeoutMs: 15000 })
    return r.stdout.includes('@END') ? parseDarwin(r.stdout) : undefined
  }
  return undefined
}

// ---------------------------------------------------------------- GPUs

export type GpuSample = { devs: GpuDev[]; memByPid: Map<number, number>; smByPid: Map<number, number> }

export function parseGpus(query: string, apps: string, pmon: string): GpuSample {
  const devs: GpuDev[] = []
  for (const row of query.split('\n')) {
    const c = row.split(',').map(x => x.trim())
    if (c.length < 5 || !/^\d+$/.test(c[0] ?? '')) continue
    devs.push({
      index: Number(c[0]),
      name: (c[1] ?? '').replace(/^NVIDIA\s+/i, ''),
      util: num(c[2]) ?? 0,
      memUsedMb: num(c[3]) ?? 0,
      memTotalMb: num(c[4]) ?? 0,
    })
  }
  const memByPid = new Map<number, number>()
  for (const row of apps.split('\n')) {
    const [pid, mb] = row.split(',').map(x => x.trim())
    const p = num(pid)
    const m = num(mb)
    if (p && m !== undefined) memByPid.set(p, (memByPid.get(p) ?? 0) + m)
  }
  const smByPid = new Map<number, number>()
  for (const row of pmon.split('\n')) {
    if (row.startsWith('#')) continue
    const c = row.trim().split(/\s+/)
    const p = num(c[1])
    const sm = num(c[3])
    if (p && sm !== undefined) smByPid.set(p, (smByPid.get(p) ?? 0) + sm)
  }
  return { devs, memByPid, smByPid }
}

/** The GPUs through `nvidia-smi`; undefined where there is none (no NVIDIA driver). */
export async function readGpus($: EngineInterface, withPmon: boolean): Promise<GpuSample | undefined> {
  const run = (args: string[]) =>
    $.process.run(['nvidia-smi', ...args], { timeoutMs: 10000 }).then(
      r => (r.exitCode === 0 ? r.stdout : ''),
      () => undefined,
    )
  const [query, apps, pmon] = await Promise.all([
    run(['--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits']),
    run(['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']),
    withPmon ? run(['pmon', '-c', '1', '-s', 'u']) : Promise.resolve(''),
  ])
  if (query === undefined || query === '') return undefined
  return parseGpus(query, apps ?? '', pmon ?? '')
}

// ---------------------------------------------------------------- the rows

export type Prev = { at: number; byPid: Map<number, { cpuS?: number; readB?: number; writeB?: number }>; cpu?: [number, number] }

/** The processes the Activity tab lists, from a sample: rates against `prev`, the probe and helpers left out. */
export function rows(
  raw: RawSample,
  prev: Prev | undefined,
  at: number,
  scope: Scope,
  gpu: GpuSample | undefined,
  isOwn: (cmd: string) => boolean,
): Proc[] {
  const kids = new Map<number, Raw[]>()
  for (const p of raw.procs) kids.set(p.ppid, [...(kids.get(p.ppid) ?? []), p])

  const probe = new Set<number>([raw.probe])
  for (const k of kids.get(raw.probe) ?? []) probe.add(k.pid)

  const session = new Set<number>()
  const queue = [...(kids.get(raw.self) ?? [])]
  while (queue.length) {
    const p = queue.shift()
    if (!p || session.has(p.pid) || probe.has(p.pid)) continue
    session.add(p.pid)
    queue.push(...(kids.get(p.pid) ?? []))
  }

  const keep = (p: Raw) =>
    p.pid !== raw.self &&
    !probe.has(p.pid) &&
    !INFRA_RE.test(p.cmd) &&
    !isOwn(p.cmd) &&
    (session.has(p.pid) || (scope === 'ml' && ML_RE.test(p.cmd)) || (gpu?.memByPid.has(p.pid) ?? false))

  const picked = raw.procs.filter(keep)
  const pickedIds = new Set(picked.map(p => p.pid))
  const dt = prev ? (at - prev.at) / 1000 : 0

  // A shell is a wrapper: drawn only where nothing it started is drawn, or what it
  // started only waits (a polling loop's `sleep`).
  const hasShownChild = (p: Raw): boolean =>
    (kids.get(p.pid) ?? []).some(k => (pickedIds.has(k.pid) && !/^\S*sleep\b/.test(k.cmd)) || hasShownChild(k))
  const out: Proc[] = []
  for (const p of picked) {
    if (isShell(p.cmd) && hasShownChild(p)) continue
    const parent = raw.procs.find(q => q.pid === p.ppid)
    if (/^\S*sleep\b/.test(p.cmd) && parent && isShell(parent.cmd) && pickedIds.has(parent.pid)) continue
    const before = prev?.byPid.get(p.pid)
    const rate = (now?: number, then?: number) =>
      dt > 0 && now !== undefined && then !== undefined ? Math.max(0, (now - then) / dt) : undefined
    const cpuRate = rate(p.cpuS, before?.cpuS)
    const d = describe(p.cmd)
    out.push({
      pid: p.pid,
      name: d.name,
      what: d.what,
      cpu: p.pcpu ?? (cpuRate !== undefined ? cpuRate * 100 : undefined),
      memMb: p.rssKb !== undefined ? p.rssKb / 1024 : undefined,
      gpuMb: gpu?.memByPid.get(p.pid),
      gpuPct: gpu?.smByPid.get(p.pid),
      readBps: rate(p.readB, before?.readB),
      writeBps: rate(p.writeB, before?.writeB),
      ageS: p.ageS,
      isSession: session.has(p.pid),
    })
  }
  return out.sort(
    (a, b) => (b.gpuMb ?? 0) - (a.gpuMb ?? 0) || (b.cpu ?? 0) - (a.cpu ?? 0) || (a.ageS ?? 0) - (b.ageS ?? 0),
  )
}

export function machine(raw: RawSample, prev: Prev | undefined, platform: Platform, gpu: GpuSample | undefined): Machine {
  let cpuPct = raw.cpuPct
  if (cpuPct === undefined && raw.cpuBusy !== undefined && raw.cpuTotal !== undefined && prev?.cpu) {
    const total = raw.cpuTotal - prev.cpu[1]
    cpuPct = total > 0 ? ((raw.cpuBusy - prev.cpu[0]) / total) * 100 : undefined
  }
  return {
    platform,
    cpuPct,
    cores: raw.cores,
    memUsedMb: raw.memUsedMb,
    memTotalMb: raw.memTotalMb,
    diskFreeGb: raw.diskFreeGb,
    diskTotalGb: raw.diskTotalGb,
    gpus: gpu?.devs ?? [],
    note: raw.note,
  }
}

export function nextPrev(raw: RawSample, at: number): Prev {
  const byPid = new Map<number, { cpuS?: number; readB?: number; writeB?: number }>()
  for (const p of raw.procs) byPid.set(p.pid, { cpuS: p.cpuS, readB: p.readB, writeB: p.writeB })
  return {
    at,
    byPid,
    cpu: raw.cpuBusy !== undefined && raw.cpuTotal !== undefined ? [raw.cpuBusy, raw.cpuTotal] : undefined,
  }
}

const TAILS = 'for f do printf "\\n@@F %s\\n" "$f"; tail -c 1500 -- "$f" 2>/dev/null; done'

/** The last line each log file shows (Linux, where a process's output file is known). */
export async function readTails($: EngineInterface, files: string[]): Promise<Map<string, string>> {
  const out = new Map<string, string>()
  if (files.length === 0) return out
  const r = await $.process.run(['sh', '-c', TAILS, 'sh', ...files], { timeoutMs: 10000 })
  for (const chunk of r.stdout.split('\n@@F ').slice(1)) {
    const nl = chunk.indexOf('\n')
    const file = chunk.slice(0, nl)
    const line = lastLine(chunk.slice(nl + 1))
    if (line) out.set(file, line)
  }
  return out
}
