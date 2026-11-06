// What a process is doing, in a few words, from its command line alone; and
// the last line of its log. Pure: no `$`, so tests call it directly.

export type Described = { name: string; what: string }

const SHELLS = new Set(['sh', 'bash', 'zsh', 'dash', 'fish', 'ksh', 'cmd', 'pwsh', 'powershell'])
const PY = /^(python|pypy)[\d.]*(w)?$/i

/** Splits a command line into words, honouring simple quotes. */
export function words(cmd: string): string[] {
  const out: string[] = []
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g
  for (let m = re.exec(cmd); m; m = re.exec(cmd)) out.push(m[1] ?? m[2] ?? m[3] ?? '')
  return out
}

export function base(path: string): string {
  const name = path.split(/[\\/]/).pop() ?? path
  return name.replace(/\.exe$/i, '')
}

export function isShell(cmd: string): boolean {
  return SHELLS.has(base(words(cmd)[0] ?? '').toLowerCase())
}

const KGKIT: Record<string, string> = {
  'gpu wait': 'Waiting for a Kaggle kernel to finish',
  'gpu push': 'Pushing a training kernel to Kaggle',
  'gpu collect': "Bringing back a kernel's outputs",
  'gpu build': 'Building a remote training kernel',
  'gpu status': 'Checking remote GPU runs',
  'gpu quota': 'Reading the GPU quota',
  features: 'Automated feature search',
  folds: 'Assigning CV folds',
  eda: 'Running EDA',
  adv: 'Adversarial validation',
  recheck: 'Re-checking a CV gain on new folds',
  blend: 'Blending experiments',
  validate: 'Validating a submission file',
  score: 'Scoring predictions',
  discussions: 'Reading the competition forum',
  kernels: 'Studying public notebooks',
  datasets: 'Downloading a dataset',
  pages: 'Reading competition pages',
  standing: 'Reading the leaderboard',
  final: 'Checking final submissions',
  'audit-repo': 'Auditing the repository',
}

function kaggleCli(args: string[]): string {
  const [group = '', cmd = ''] = args
  const target = flag(args, ['-c', '--competition']) ?? flag(args, ['-p', '--path']) ?? args[2]
  const of = target && !target.startsWith('-') ? ` (${target})` : ''
  const g = group.replace(/s$/, '')
  if (g === 'competition') {
    if (cmd === 'download') return `Downloading competition data${of}`
    if (cmd === 'submit') return `Submitting to Kaggle${of}`
    if (cmd === 'submissions') return 'Listing submissions'
    if (cmd === 'leaderboard') return 'Reading the leaderboard'
  }
  if (g === 'kernel') {
    if (cmd === 'push') return `Pushing a kernel${of}`
    if (cmd === 'output') return `Downloading kernel output${of}`
    if (cmd === 'pull') return `Pulling a kernel${of}`
    if (cmd === 'status') return 'Checking a kernel'
  }
  if (g === 'dataset') {
    if (cmd === 'download') return `Downloading a dataset${of}`
    if (cmd === 'create' || cmd === 'version') return 'Uploading a dataset'
  }
  if (g === 'model') return args.includes('download') ? 'Downloading a model' : 'Uploading a model'
  if (group === 'quota') return 'Reading Kaggle quota'
  return `Kaggle CLI: ${args.slice(0, 2).join(' ')}`.trim()
}

function flag(args: string[], names: string[]): string | undefined {
  for (let i = 0; i < args.length; i++) {
    const a = args[i] ?? ''
    for (const n of names) {
      if (a === n) return args[i + 1]
      if (n.startsWith('--') && a.startsWith(`${n}=`)) return a.slice(n.length + 1)
    }
  }
  return undefined
}

/** `fold 3, convnext_tiny, 20 epochs` from the usual training flags. */
function trainingFlags(args: string[]): string {
  const parts: string[] = []
  const fold = flag(args, ['--fold', '-f', '--folds'])
  if (fold) parts.push(`fold ${fold}`)
  const model = flag(args, ['--model', '--arch', '--backbone', '--model_name', '--model-name', '--encoder'])
  if (model) parts.push(base(model))
  const epochs = flag(args, ['--epochs', '--n_epochs', '--num_epochs', '--max_epochs', '-e'])
  if (epochs) parts.push(`${epochs} epochs`)
  const seed = flag(args, ['--seed'])
  if (seed) parts.push(`seed ${seed}`)
  const config = flag(args, ['--config', '-c', '--cfg'])
  if (config && parts.length === 0) parts.push(base(config))
  return parts.join(', ')
}

function pythonScript(script: string, args: string[]): string {
  const s = base(script).replace(/\.py$/i, '').toLowerCase()
  const extra = trainingFlags(args)
  const tail = extra ? `: ${extra}` : ''
  if (/train|fit|finetune|fine_tune|pretrain/.test(s)) return `Training a model${tail}`
  if (/infer|predict|inference|test_time|tta/.test(s)) return `Running inference${tail}`
  if (/tune|optuna|hpo|sweep|search/.test(s)) return 'Tuning hyperparameters'
  if (/ensemble|blend|stack/.test(s)) return 'Blending models'
  if (/eval|valid|score|oof|metric/.test(s)) return 'Evaluating predictions'
  if (/prep|preprocess|feature|make_data|build_data|dataset|extract|embed/.test(s)) return 'Preparing data'
  if (/download|fetch/.test(s)) return 'Downloading data'
  if (/submit|submission/.test(s)) return 'Writing a submission'
  return `Running ${base(script)}${tail}`
}

function pythonModule(mod: string, args: string[]): string {
  if (mod === 'kgkit') {
    const two = `${args[0] ?? ''} ${args[1] ?? ''}`.trim()
    return KGKIT[two] ?? KGKIT[args[0] ?? ''] ?? `kgkit ${args[0] ?? ''}`.trim()
  }
  if (mod === 'kaggle') return kaggleCli(args)
  if (mod === 'pytest') return 'Running tests'
  if (mod === 'pip' || mod === 'uv') return args[0] === 'install' ? 'Installing Python packages' : `pip ${args[0] ?? ''}`
  if (mod === 'ipykernel_launcher' || mod === 'ipykernel') return 'Jupyter notebook kernel'
  if (mod === 'jupyter' || mod === 'jupyterlab' || mod === 'notebook') return 'Jupyter server'
  if (mod === 'tensorboard' || mod.startsWith('tensorboard.')) return 'TensorBoard server'
  if (mod === 'torch.distributed.run' || mod === 'torch.distributed.launch') return 'Distributed training'
  if (mod === 'http.server') return 'Serving files over HTTP'
  if (mod === 'streamlit') return 'Streamlit app'
  return `Running module ${mod}`
}

/** A process's name and what it does, from its command line. */
export function describe(cmd: string): Described {
  const w = words(cmd)
  const exe = base(w[0] ?? '?')
  const rest = w.slice(1)
  const lower = exe.toLowerCase()

  if (PY.test(exe)) {
    let i = 0
    while (i < rest.length && (rest[i] ?? '').startsWith('-') && rest[i] !== '-m' && rest[i] !== '-c') {
      i += /^-[WX]$/.test(rest[i] ?? '') ? 2 : 1
    }
    const a = rest[i]
    if (a === '-m') {
      const mod = rest[i + 1] ?? '?'
      return { name: mod, what: pythonModule(mod, rest.slice(i + 2)) }
    }
    if (a === '-c') return { name: 'python -c', what: 'Inline Python' }
    if (a) return { name: base(a), what: pythonScript(a, rest.slice(i + 1)) }
    return { name: exe, what: 'Interactive Python' }
  }
  if (lower === 'kaggle') return { name: 'kaggle', what: kaggleCli(rest) }
  if (lower === 'torchrun' || lower === 'deepspeed') {
    const script = rest.find(a => /\.py$/.test(a))
    return { name: script ? base(script) : exe, what: `Distributed training${script ? `: ${base(script)}` : ''}` }
  }
  if (lower === 'accelerate') {
    const script = rest.find(a => /\.py$/.test(a))
    return { name: script ? base(script) : exe, what: `Training with accelerate${script ? `: ${base(script)}` : ''}` }
  }
  if (/^jupyter(-lab|-notebook|-server)?$/.test(lower)) return { name: exe, what: 'Jupyter server' }
  if (lower === 'tensorboard') return { name: exe, what: 'TensorBoard server' }
  if (lower === 'pytest') return { name: exe, what: 'Running tests' }
  if (lower === 'pip' || lower === 'pip3') return { name: exe, what: 'Installing Python packages' }
  if (lower === 'conda' || lower === 'mamba' || lower === 'micromamba') return { name: exe, what: `conda ${rest[0] ?? ''}`.trim() }
  if (lower === 'nvidia-smi') return { name: exe, what: 'Reading GPU state' }
  if (lower === 'wget' || lower === 'curl' || lower === 'aria2c') {
    const url = rest.find(a => /^https?:\/\//.test(a))
    const host = url ? (url.split('/')[2] ?? '') : ''
    return { name: exe, what: `Downloading${host ? ` from ${host}` : ''}` }
  }
  if (lower === 'unzip' || lower === 'tar' || lower === '7z') return { name: exe, what: 'Extracting an archive' }
  if (lower === 'rsync' || lower === 'cp' || lower === 'scp') return { name: exe, what: 'Copying files' }
  if (lower === 'git') return { name: 'git', what: `git ${rest[0] ?? ''}`.trim() }
  if (lower === 'rscript' || lower === 'r') return { name: exe, what: 'Running an R script' }
  if (lower === 'julia') return { name: exe, what: 'Running Julia' }
  if (lower === 'sleep') return { name: exe, what: `Waiting ${rest[0] ?? ''}`.trim() }
  if (lower === 'make') return { name: exe, what: `make ${rest[0] ?? ''}`.trim() }
  if (lower === 'docker') return { name: exe, what: `docker ${rest[0] ?? ''}`.trim() }
  if (SHELLS.has(lower)) {
    const inner = rest.indexOf('-c')
    if (inner >= 0 && rest[inner + 1]) return { name: exe, what: `Shell: ${oneLine(rest[inner + 1] ?? '', 60)}` }
    return { name: exe, what: 'Shell' }
  }
  return { name: exe, what: `Running ${exe}` }
}

/** Matches the programs the Activity tab lists outside this session. */
export const ML_RE =
  /(^|[\\/\s"'])(python[\d.]*|pypy[\d.]*|kaggle|jupyter[\w-]*|torchrun|accelerate|deepspeed|tensorboard|Rscript|julia)(\.exe)?(["'\s]|$)/i

/** Long-lived helpers that are not the person's work: editors' servers, MCP servers, Claude itself. */
export const INFRA_RE =
  /language[-_]?server|pyright|pylance|jedi|pylsp|ruff\s+server|tsserver|gopls|rust-analyzer|\bmcp\b|mcp[-_]server|@modelcontextprotocol|native[-_]host|@anthropic-ai[\\/]claude-code|[\\/]claude[\\/]versions[\\/]|debugpy[\\/]adapter|vscode-server|\.vscode[\\/]extensions/i

// ESC [ ... final, ESC ] ... BEL/ST, and the remaining C0/C1 controls.
// eslint-disable-next-line no-control-regex
const ANSI = /\u001b\[[0-?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(\u0007|\u001b\\)|\u001b[@-_]/g
// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏‪-‮⁦-⁩﻿]/g

export function oneLine(text: string, max: number): string {
  const flat = text.replace(ANSI, '').replace(CONTROL, '').replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

/** The last line a log shows, a progress bar's latest redraw (after `\r`) included. */
export function lastLine(tail: string, max = 160): string | undefined {
  const pieces = tail.replace(ANSI, '').split(/\r\n|\n|\r/)
  for (let i = pieces.length - 1; i >= 0; i--) {
    const line = oneLine(pieces[i] ?? '', max)
    if (line) return line
  }
  return undefined
}
