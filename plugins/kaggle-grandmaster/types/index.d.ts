export type Tab = 'activity' | 'quota' | 'accounts'

/** Which local processes the Activity tab lists. */
export type Scope = 'ml' | 'session'

/** One local process as the Activity tab draws it. */
export type Proc = {
  pid: number
  /** The program or script, e.g. `train.py`. */
  name: string
  /** What it is doing, from its command line: `Training a model: fold 3, convnext`. */
  what: string
  /** The last line it wrote to its log file, when its output goes to a file. */
  detail?: string
  /** Percent of one core, over the last sample; absent on the first. */
  cpu?: number
  memMb?: number
  gpuMb?: number
  /** SM utilisation of its GPU work, from `nvidia-smi pmon`. */
  gpuPct?: number
  readBps?: number
  writeBps?: number
  ageS?: number
  /** Started from this Claude Code session (a Bash call or a background task). */
  isSession: boolean
}

export type GpuDev = {
  index: number
  name: string
  util: number
  memUsedMb: number
  memTotalMb: number
}

export type Machine = {
  platform: string
  cpuPct?: number
  cores?: number
  memUsedMb?: number
  memTotalMb?: number
  diskFreeGb?: number
  diskTotalGb?: number
  gpus: GpuDev[]
  /** Why part of the sample is missing (no per-process disk on macOS, say). */
  note?: string
}

export type QuotaRow = {
  resource: string
  usedH: number
  remainingH: number
  totalH: number
  refreshAt?: string
}

export type Kernel = {
  ref: string
  title: string
  /** RUNNING, QUEUED, COMPLETE, ERROR, CANCEL_REQUESTED, ... */
  status: string
  lastRunTime?: string
}

/** A running kernel's latest reading, from the `[kgmon]` lines its log carries. */
export type KernelLive = {
  ref: string
  alias: string
  /** When the pane last read a line of its log (local clock, ms). */
  heardAt?: number
  /** When the latest reading arrived (local clock, ms). */
  readAt?: number
  /** Seconds since the session started, as the kernel counts them. */
  t?: number
  cpu?: number
  cores?: number
  /** [used, limit] MB. */
  ram?: [number, number]
  /** /kaggle/working: [used, total] MB. */
  disk?: [number, number]
  /** The scratch disk: [used, total] MB. */
  tmp?: [number, number]
  /** [index, util %, memory used MB, memory total MB, name]. */
  gpus?: [number, number, number, number, string][]
  doing?: string[]
  /** The newest other line of its log. */
  lastLine?: string
  /** The newest other lines of its log, oldest first (up to 3). */
  tail?: string[]
  /** Its readings since the session started, thinned to at most HISTORY_MAX evenly spaced samples. */
  hist?: Sample[]
  /** Averages and peaks over every reading of the session (not thinned). */
  stats?: LiveStats
  /** The newest progress its training log reported. */
  progress?: Progress
  isFollowing: boolean
  error?: string
}

/** One reading, as a sparkline draws it: kernel seconds, CPU %, RAM MB, and per GPU util % and memory MB. */
export type Sample = { t: number; cpu?: number; ram?: number; gpu?: number[]; gmem?: number[] }

export type LiveStats = {
  /** Readings counted. */
  n: number
  cpuAvg?: number
  ramPeak?: number
  /** Per GPU, by position in the reading. */
  gpuAvg?: number[]
  gmemPeak?: number[]
}

/** Where a training log says it is: `step 820/858`. */
export type Progress = {
  /** What is counted: step, epoch, fold, batch, or a tqdm bar's description. */
  label: string
  done: number
  total: number
  /** A JSON log line's fractional epoch, beside its step count. */
  epoch?: number
}

/** Which running-kernel cards the Activity tab draws open. */
export type Cards = {
  /** The last Show all / Hide all; unset: one running kernel opens, two or more stay closed. */
  all?: 'open' | 'closed'
  /** A card opened or closed by hand since, by kernel ref. */
  open: Record<string, boolean>
}

export type Submission = {
  ref: string
  fileName: string
  date: string
  description: string
  status: string
  publicScore?: string
}

export type Submissions = { competition: string; items: Submission[] }

/** A value fetched from Kaggle: the last good one, when, and the last error. */
export type Fetched<T> = {
  value?: T
  fetchedAt?: number
  error?: string
  isLoading?: boolean
}

/** A registered account; never its token, only the token file's path. */
export type AccountView = {
  alias: string
  user?: string
  tokenPath: string
  hasToken: boolean
  isActive: boolean
}

export type CliInfo = {
  /** How the Kaggle CLI is run, e.g. ["/opt/conda/bin/kaggle"]. */
  argv?: string[]
  version?: string
  error?: string
  isChecking?: boolean
}

export type Note = { text: string; isError: boolean }

declare module 'claude-code' {
  interface PluginState {
    'kaggle-grandmaster': {
      tab: Tab
      scope: Scope
      procs: Shaped<Proc[]>
      machine: Shaped<Machine | null>
      accounts: Shaped<AccountView[]>
      quotas: Shaped<Record<string, Fetched<QuotaRow[]>>>
      kernels: Shaped<Record<string, Fetched<Kernel[]>>>
      submissions: Shaped<Fetched<Submissions> | null>
      cli: Shaped<CliInfo>
      note: Shaped<Note | null>
      live: Shaped<Record<string, KernelLive>>
      /** Running-kernel cards drawn open; kept in the plugin's store across sessions. */
      cards: Cards
      /** The Activity tab's "This computer" section drawn open. */
      showLocal: boolean
      /** The alias whose Remove button waits for a second press. */
      confirmRemove: string
    }
  }
}
