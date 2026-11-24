import type { CommandSpec, EngineInterface } from 'claude-code'

import { platformOf } from './probe'

/**
 * /kg-update: updates the plugin from the prompt, with no turn of the model. It runs the plugin's own
 * `python -m kgkit.update --apply` (the check `kgkit update-check` prints, then the `claude plugin update` it names),
 * and shows what that printed: the versions, what changed, and that /reload-plugins loads the new version.
 * It is no skill: Claude Code refuses a mod command named like one of the plugin's skills.
 */
export const UPDATE_COMMAND: CommandSpec = {
  name: 'kg-update',
  description: 'Update kaggle-grandmaster to its latest release, then /reload-plugins (check: only look)',
  argumentHint: '[check]',
}

const CHECK_MS = 120_000
/** `claude plugin update` may take 5 min (kgkit's own limit), and the checks before and after it. */
const APPLY_MS = 480_000

export type UpdateAnswer = { text: string; isUpdated: boolean }

/** kgkit.update's answer: its last stdout line, `{"text", "updated"}`; undefined when it printed none. */
export function parseUpdate(stdout: string): UpdateAnswer | undefined {
  const last = stdout.split(/\r?\n/).map(l => l.trim()).filter(Boolean).pop() ?? ''
  try {
    const j = JSON.parse(last) as { text?: unknown; updated?: unknown }
    return typeof j.text === 'string' ? { text: j.text, isUpdated: j.updated === true } : undefined
  } catch {
    return undefined
  }
}

/** The Pythons to run kgkit with, as bin/kgkit tries them: KGKIT_PYTHON first. kgkit.update is standard library only. */
async function pythons($: EngineInterface): Promise<string[]> {
  const set = await $.env.get('KGKIT_PYTHON').catch(() => undefined)
  const names = (await platformOf($)) === 'windows' ? ['py', 'python', 'python3'] : ['python3', 'python']
  return [...(set ? [set] : []), ...names.filter(n => n !== set)]
}

/** What /kg-update prints for `args` ('' updates, 'check' only looks). Never throws. */
export async function kgUpdate($: EngineInterface, args: string): Promise<string> {
  const sub = args.trim().split(/\s+/)[0]?.toLowerCase() ?? ''
  if (sub && sub !== 'check') return 'Usage: /kg-update (updates kaggle-grandmaster) or /kg-update check (only looks).'
  const isApply = sub !== 'check'
  $.ui.toast(isApply ? 'Updating kaggle-grandmaster...' : 'Checking for a kaggle-grandmaster release...')
  const tried: string[] = []
  for (const py of await pythons($)) {
    try {
      const r = await $.process.run([py, '-m', 'kgkit.update', ...(isApply ? ['--apply'] : [])], {
        env: { PYTHONPATH: $.plugin.root, PYTHONIOENCODING: 'utf-8' },
        timeoutMs: isApply ? APPLY_MS : CHECK_MS,
      })
      const answer = parseUpdate(r.stdout)
      // The engine heads a plugin command's output with the plugin's name: kgkit's first line starts with it too.
      if (answer) return answer.text.replace(/^kaggle-grandmaster: /, '')
      const said = `${r.stderr}\n${r.stdout}`.split(/\r?\n/).map(l => l.trim()).filter(Boolean).pop()
      tried.push(`${py}: exit ${r.exitCode}${said ? `, ${said.slice(0, 160)}` : ''}`)
    } catch (err) {
      tried.push(`${py}: ${String(err).slice(0, 160)}`)
    }
  }
  return (
    `Could not ${isApply ? 'update' : 'check for an update'}: no Python ran the plugin's kgkit ` +
    `(${tried.join('; ')}). Set KGKIT_PYTHON to a Python 3 and try again, or update from the /plugin menu.`
  )
}
