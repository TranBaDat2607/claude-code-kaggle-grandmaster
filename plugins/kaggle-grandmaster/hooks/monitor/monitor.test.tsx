import { expect, test } from 'claude-code/testing'

import type { Kernel, KernelLive } from '../../types'
import { describe, lastLine } from './describe'
import { parseKgmon, parseQuota, status, utc } from './kaggle'
import { HISTORY_MAX, LiveTracker, parseProgress, spark } from './live'
import { etime, parseGpus } from './probe'
import { age, clockOf, endNotes, limitText, runOf } from './session'

// The test kit's runtime has timers; its type library does not name them.
const sleep = (ms: number) =>
  new Promise<void>(resolve => (globalThis as unknown as { setTimeout: (fn: () => void, ms: number) => void }).setTimeout(resolve, ms))

const PANE = {
  plugin: 'kaggle-grandmaster',
  component: 'Pane',
  requestId: 'kaggle-monitor',
  props: {
    title: 'Kaggle',
    isFocused: true,
    bodyColumns: 100,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 40 },
    view: {},
  },
} as const

test('a command line reads as what it does', async () => {
  expect(describe('python train.py --fold 3 --model convnext_tiny --epochs 20').what).toBe(
    'Training a model: fold 3, convnext_tiny, 20 epochs',
  )
  expect(describe('/usr/bin/python3 -u -m kgkit gpu wait me/k').what).toBe('Waiting for a Kaggle kernel to finish')
  expect(describe('kaggle competitions submit -c s6e10 -f sub.csv -m x').what).toBe('Submitting to Kaggle (s6e10)')
  expect(describe('"C:\\Py\\python.exe" infer.py').what).toBe('Running inference')
  expect(describe('torchrun --nproc_per_node 2 train_seg.py').name).toBe('train_seg.py')
})

test("a log's last line is its latest progress redraw, controls removed", async () => {
  expect(lastLine('epoch 1\n 10%|#  | 1/10\r 20%|## | 2/10\r\n')).toBe('20%|## | 2/10')
  expect(lastLine('\u001b[32mok\u001b[0m done\n\n')).toBe('ok done')
})

test('quota rows, statuses, times and GPUs parse', async () => {
  const rows = parseQuota(
    '[{"resource":"GPU","used":"29.75h","remaining":"0.25h","total":"30.00h","refreshAt":"2026-11-07T00:00:00+00:00"},' +
      '{"resource":"TPU","used":"0.00h","remaining":"20.00h","total":"20.00h"}]',
  )
  expect(rows.map(r => [r.resource, r.remainingH, r.totalH])).toEqual([
    ['GPU', 0.25, 30],
    ['TPU', 20, 20],
  ])
  expect(status('KernelWorkerStatus.RUNNING')).toBe('RUNNING')
  expect(utc('2026-11-04T22:05:29.263000')).toBe(Date.UTC(2026, 10, 4, 22, 5, 29, 263))
  expect(etime('1-02:03:04')).toBe(93784)
  const g = parseGpus('0, NVIDIA RTX 4090, 87, 18000, 24564\n', '4242, 17000\n', '# gpu pid type sm mem\n 0 4242 C 85 40\n')
  expect(g.devs[0]?.name).toBe('RTX 4090')
  expect(g.memByPid.get(4242)).toBe(17000)
  expect(g.smByPid.get(4242)).toBe(85)
})

test('the pane draws its tabs and switches to Accounts, on every surface', async ($, on) => {
  // Nothing sits beneath the plugin: this hook stands for the engine's clock.
  on('clock.now', () => ({ value: Date.UTC(2026, 10, 6) }))
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    expect((await ui.find({ type: 'Text', text: /no Kaggle account/ }))?.text).toBe('no Kaggle account')
    await ui.press({ key: 'tab-accounts' })
    expect((await ui.find({ type: 'Text', text: /No Kaggle account registered/ }))?.text).toBe(
      'No Kaggle account registered yet.',
    )
    expect(await ui.find({ type: 'Input', key: 'token-0' })).toBeDefined()
    await ui.press({ key: 'tab-quota' })
    expect(await ui.find({ type: 'Text', text: /No accounts yet/ })).toBeDefined()
    await ui.press({ key: 'tab-activity' })
    await ui.unmount()
  }
})

test('session start declares /kaggle-monitor before anything else can fail', async ($, on) => {
  const registered: string[] = []
  on('command.register', (_, e) => {
    registered.push(e.name)
    return { value: { command: e.name } }
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })
  expect(registered).toEqual(['kaggle-monitor'])
})

test('session start in a competition workspace does not open the pane by itself', async ($, on) => {
  const opened: string[] = []
  let isCompetitionRead = false
  let ticked = () => {}
  const firstTick = new Promise<void>(resolve => (ticked = resolve))
  on('command.register', (_, e) => ({ value: { command: e.name } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('ui.open', (_, e) => {
    opened.push(e.id)
    return { value: { isPlaced: true } }
  })
  on('ui.panes', () => {
    ticked()
    return { value: [] }
  })
  on('env.get', () => ({ value: undefined }))
  on('store.get', () => ({ value: undefined }))
  on('clock.now', () => ({ value: Date.UTC(2026, 10, 7) }))
  on('clock.every', () => new Promise(() => {})) // the polling timer never fires here
  on('process.run', () => {
    throw new Error('no processes in this test')
  })
  on('fs.exists', () => ({ value: false }))
  on('fs.read', (_, e) => {
    if (e.path === '/ws/.kaggle-gm/competition.json') {
      isCompetitionRead = true
      return { value: JSON.stringify({ slug: 'playground-s6e11' }) }
    }
    throw new Error(`ENOENT: ${e.path}`)
  })
  await $.session.start({ cwd: '/ws', surface: 'terminal', isInteractive: true })
  await firstTick // the start runs on after it answers; its last step is a first look at the panes
  expect(isCompetitionRead).toBe(true)
  expect(opened).toEqual([]) // the pane waits for /kaggle-monitor
})

test('session start with an account writes nothing to the status line: the pane shows it', async ($, on) => {
  const statuses: (string | undefined)[] = []
  let ticked = () => {}
  const firstTick = new Promise<void>(resolve => (ticked = resolve))
  on('command.register', (_, e) => ({ value: { command: e.name } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('ui.status', (_, e) => {
    statuses.push(e.text)
    return { value: undefined }
  })
  on('ui.panes', () => {
    ticked()
    return { value: [] }
  })
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/u' : undefined }))
  on('env.set', () => ({ value: {} }))
  on('store.get', (_, e) => ({ value: e.key === 'active' ? 'main' : undefined }))
  on('store.set', () => ({ value: undefined }))
  on('clock.now', () => ({ value: Date.UTC(2026, 10, 7) }))
  on('clock.every', () => new Promise(() => {})) // the polling timer never fires here
  on('process.run', () => {
    throw new Error('no processes in this test')
  })
  on('fs.exists', (_, e) => ({ value: e.path === '/home/u/.kaggle/access_token' }))
  on('fs.read', (_, e) => {
    if (e.path === '/ws/.kaggle-gm/competition.json') return { value: JSON.stringify({ slug: 'playground-s6e11' }) }
    throw new Error(`ENOENT: ${e.path}`)
  })
  await $.session.start({ cwd: '/ws', surface: 'terminal', isInteractive: true })
  await firstTick // the start runs on after it answers; its last step is a first look at the panes
  expect(statuses.filter(t => t !== undefined)).toEqual([])
})

test('a [kgmon] line reads as the Kaggle session\'s resources', async () => {
  const r = parseKgmon(
    '[kg   3.0m] [kgmon] {"v":1,"t":180,"cores":4.0,"cpu":97.5,"ram":[12000,30000],"disk":[900,20000],' +
      '"gpus":[[0,88,11000,15360,"Tesla T4"],[1,3,400,15360,"Tesla T4"]],"doing":["exp/f0 on gpu0"]}',
  )
  expect(r?.cpu).toBe(97.5)
  expect(r?.ram).toEqual([12000, 30000])
  expect(r?.gpus?.[1]).toEqual([1, 3, 400, 15360, 'Tesla T4'])
  expect(r?.doing).toEqual(['exp/f0 on gpu0'])
  expect(parseKgmon('epoch 3 loss 0.2')).toBeUndefined()
  expect(parseKgmon('[kgmon] {"v":2}')).toBeUndefined()
})

test('a running kernel draws its CPU, RAM, GPUs and disk', async ($, on) => {
  const now = Date.UTC(2026, 10, 6, 12)
  on('clock.now', () => ({ value: now }))
  const seeded: Record<string, unknown> = {
    accounts: { shape: 'accounts-1', value: [{ alias: 'acc2', user: 'u2', tokenPath: '~/.kaggle/t', hasToken: true, isActive: true }] },
    kernels: {
      shape: 'kernels-2',
      value: { acc2: { value: [{ ref: 'u2/train-b', title: 'train b', status: 'RUNNING', lastRunTime: '2026-11-06T11:00:00' }], fetchedAt: now } },
    },
    live: {
      shape: 'live-3',
      value: {
        'u2/train-b': {
          ref: 'u2/train-b', alias: 'acc2', isFollowing: true, heardAt: now, readAt: now, t: 3600, cpu: 97, cores: 4,
          ram: [12288, 30720], disk: [1024, 20480], gpus: [[0, 88, 11264, 15360, 'Tesla T4']], doing: ['exp/f0 on gpu0'],
          lastLine: 'epoch 3/10 loss 0.21',
        },
      },
    },
  }
  on('state.get', (_, e) => ({ value: e.key in seeded ? { value: seeded[e.key], version: 1 } : { value: undefined, version: 0 } }))
  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    for (const text of [/u2\/train-b/, /97% now · avg - · 4 cores/, /12\.0 \/ 30\.0 GB/, /88% now · avg - · 11\.0 \/ 15\.0 GB · T4/, /1\.0 \/ 20\.0 GB/, /exp\/f0 on gpu0/, /epoch 3\/10/]) {
      expect((await ui.find({ type: 'Text', text }))?.text).toMatch(text)
    }
    expect(await ui.find({ type: 'Text', text: /Processes \(/ })).toBeUndefined()
    await ui.unmount()
  }
})


test('a training log line reads as its progress', async () => {
  expect(parseProgress('{"epoch": 1.911, "step": 820, "of": 858, "loss": 0.0854, "lr": 1.1e-06}')).toEqual({
    label: 'step', done: 820, total: 858, epoch: 1.911,
  })
  expect(parseProgress('[train] {"epoch": 3, "epochs": 10, "val_auc": 0.91}')).toEqual({ label: 'epoch', done: 3, total: 10 })
  expect(parseProgress('Epoch 2:  45%|####5     | 450/1000 [01:02<01:16, 7.2it/s]')).toEqual({ label: 'Epoch 2', done: 450, total: 1000 })
  expect(parseProgress('fold 2/5 epoch 3/10 loss 0.21')).toEqual({ label: 'epoch', done: 3, total: 10 })
  expect(parseProgress('Step [120/400] loss 1.2')).toEqual({ label: 'step', done: 120, total: 400 })
  expect(parseProgress('loss 0.21 auc 0.93')).toBeUndefined()
})

test('a kernel log keeps its whole run: averages, peaks, thinned history, progress', async () => {
  const tr = new LiveTracker()
  for (let i = 0; i < 500; i++) {
    tr.reading({ t: i * 30, cpu: i % 2 ? 100 : 0, ram: [1000 + i, 30000], gpus: [[0, 50, 2000 + i, 15000, 'T4'], [1, 0, 0, 15000, 'T4']] })
    if (i >= 100 && i <= 200) tr.line(`{"step": ${(i - 100) * 10}, "of": 2000}`)
  }
  const st = tr.stats()
  expect(st.n).toBe(500)
  expect(st.cpuAvg).toBe(50)
  expect(st.ramPeak).toBe(1499)
  expect(st.gpuAvg).toEqual([50, 0])
  expect(st.gmemPeak).toEqual([2499, 0])
  const h = tr.history()
  expect(h.length).toBeLessThanOrEqual(HISTORY_MAX)
  expect(h.length).toBeGreaterThan(HISTORY_MAX / 2 - 2)
  expect(h[h.length - 1]!.t).toBe(499 * 30)
  expect(tr.progress).toEqual({ label: 'step', done: 1000, total: 2000 })
  tr.line('saving checkpoint')  // a line that counts nothing keeps the last progress
  expect(tr.progress?.done).toBe(1000)
  expect(tr.tail).toEqual(['{"step": 990, "of": 2000}', '{"step": 1000, "of": 2000}', 'saving checkpoint'])
  expect(spark([0, 0.5, 1, undefined], 6)).toBe('▁▅█   ')
  expect(spark(Array.from({ length: 100 }, (_, i) => i / 99), 4)).toHaveLength(4)
})

const runningKernels = (now: number, refs: string[]) => ({
  accounts: { shape: 'accounts-1', value: [{ alias: 'acc2', user: 'u2', tokenPath: '~/.kaggle/t', hasToken: true, isActive: true }] },
  kernels: {
    shape: 'kernels-2',
    value: { acc2: { value: refs.map(ref => ({ ref, title: ref, status: 'RUNNING', lastRunTime: '2026-11-06T11:00:00' })), fetchedAt: now } },
  },
  live: {
    shape: 'live-3',
    value: Object.fromEntries(
      refs.map(ref => [
        ref,
        {
          ref, alias: 'acc2', isFollowing: true, heardAt: now, readAt: now, t: 1800, cpu: 53, cores: 4, ram: [7270, 30720],
          disk: [0, 19968], gpus: [[0, 0, 6758, 15360, 'Tesla T4'], [1, 0, 0, 15360, 'Tesla T4']],
          hist: [{ t: 0, cpu: 40, ram: 5000, gpu: [80, 0], gmem: [9000, 0] }, { t: 1800, cpu: 53, ram: 7270, gpu: [0, 0], gmem: [6758, 0] }],
          stats: { n: 2, cpuAvg: 46.5, ramPeak: 8192, gpuAvg: [40, 0], gmemPeak: [9318, 0] },
          progress: { label: 'step', done: 820, total: 858, epoch: 1.911 },
          tail: ['[train] step 818 loss 0.0861', '[train] step 820 loss 0.0854'],
        },
      ]),
    ),
  },
})

test('a running kernel opens to its history, progress and log, and closes to one line', async ($, on) => {
  const now = Date.UTC(2026, 10, 6, 12)
  on('clock.now', () => ({ value: now }))
  const seeded: Record<string, unknown> = runningKernels(now, ['u2/train-a'])
  // Seeded until the pane writes its own value; what it writes (the cards) is read back from the engine.
  on('state.get', ($, e, next) => (e.key in seeded ? { value: { value: seeded[e.key], version: 1 } } : next(e)))
  on('store.set', () => ({ value: undefined }))
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  // one running kernel: open
  expect(await ui.find({ type: 'Button', text: /▾ RUNNING u2\/train-a/ })).toBeDefined()
  const border = { borderStyle: 'round', borderColor: '#5fafff' }
  expect((await ui.find({ key: 'live-u2/train-a' }))?.props).toMatchObject(border)
  // the session clock: from Kaggle's lastRunTime (11:00) to now (12:00), and the GPU session's 12 h limit
  for (const text of [/· acc2 · 1h00 ·/, /^limit 12h · 11h00 left$/, /step 820\/858 · epoch 1\.9$/, /53% now · avg 47% · 4 cores/, /7\.1 \/ 30\.0 GB · peak 8\.0/,
    /0% now · avg 40% · 6\.6 \/ 15\.0 GB, peak 9\.1 · T4/, /step 818 loss/]) {
    expect((await ui.find({ type: 'Text', text }))?.text).toMatch(text)
  }
  expect(await ui.find({ type: 'Text', text: /Tmp|scratch/ })).toBeUndefined()
  await ui.press({ key: 'card-u2/train-a' })
  expect((await ui.find({ type: 'Text', text: /· CPU 53% · RAM 7\.1\/30\.0 GB · GPU 0% 0% · step 820\/858/ }))?.text).toBeDefined()
  expect(await ui.find({ type: 'Text', text: /now · avg/ })).toBeUndefined()
  // closed: the newest log line stays, the older ones go
  expect((await ui.find({ type: 'Text', text: /└ \[train\] step 820 loss 0\.0854/ }))?.text).toBeDefined()
  expect(await ui.find({ type: 'Text', text: /step 818 loss/ })).toBeUndefined()
  // closed, the card keeps its border
  expect((await ui.find({ key: 'live-u2/train-a' }))?.props).toMatchObject(border)
  await ui.unmount()
})

test('two running kernels start closed; Show all opens every card', async ($, on) => {
  const now = Date.UTC(2026, 10, 6, 12)
  on('clock.now', () => ({ value: now }))
  const seeded: Record<string, unknown> = runningKernels(now, ['u2/train-a', 'u2/train-b'])
  // Seeded until the pane writes its own value; what it writes (the cards) is read back from the engine.
  on('state.get', ($, e, next) => (e.key in seeded ? { value: { value: seeded[e.key], version: 1 } } : next(e)))
  on('store.set', () => ({ value: undefined }))
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ type: 'Text', text: /now · avg/ })).toBeUndefined()
  await ui.press({ key: 'cards-all' })
  expect((await ui.find({ type: 'Text', text: /53% now · avg 47%/ }))?.text).toBeDefined()
  await ui.press({ key: 'cards-all' })
  expect(await ui.find({ type: 'Text', text: /now · avg/ })).toBeUndefined()
  await ui.unmount()
})

for (const source of ['clear', 'resume'] as const) {
  test(`a /${source} brings the pane back: the account, the poll and the command`, async ($, on) => {
    // The host's state, held for one session: emptied when it ends, as a /clear or /resume does.
    const held = new Map<string, unknown>()
    let registered = 0
    let polls = 0
    let ticked = () => {}
    let nextTick = new Promise<void>(resolve => (ticked = resolve))
    on('state.get', (_, e) => ({ value: held.has(e.key) ? { value: held.get(e.key), version: 1 } : { value: undefined, version: 0 } }))
    on('state.set', (_, e) => {
      held.set(e.key, e.value)
      return { value: { isSet: true, version: 1 } }
    })
    on('command.register', (_, e) => {
      registered++
      return { value: { command: e.name } }
    })
    on('session.start', (_, e) => ({ cwd: e.cwd }))
    on('session.end', (_, e) => {
      held.clear()
      return { sessionId: e.sessionId }
    })
    on('classic.SessionStart', () => ({}))
    on('ui.panes', () => {
      ticked()
      return { value: [] }
    })
    on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/u' : undefined }))
    on('env.set', () => ({ value: {} }))
    on('store.get', (_, e) => ({ value: e.key === 'active' ? 'main' : undefined }))
    on('store.set', () => ({ value: undefined }))
    on('clock.now', () => ({ value: Date.UTC(2026, 10, 9) }))
    on('clock.every', () => {
      polls++
      return new Promise(() => {}) // the polling timer never fires here
    })
    on('process.run', () => {
      throw new Error('no processes in this test')
    })
    on('fs.exists', (_, e) => ({ value: e.path === '/home/u/.kaggle/access_token' }))
    on('fs.read', (_, e) => {
      throw new Error(`ENOENT: ${e.path}`)
    })
    const accountShown = async () => {
      const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
      const found = await ui.find({ type: 'Text', text: /^(main|no Kaggle account)$/ })
      await ui.unmount()
      return found?.text
    }

    await $.session.start({ cwd: '/ws', surface: 'terminal', isInteractive: true })
    await nextTick
    expect(await accountShown()).toBe('main')

    nextTick = new Promise<void>(resolve => (ticked = resolve))
    await $.session.end({ reason: source, sessionId: 's1', resume: { id: 's1' } })
    expect(await accountShown()).toBe('no Kaggle account') // what the person saw before the fix
    await $.classic.SessionStart({ source, cwd: '/ws' })
    await nextTick
    expect(await accountShown()).toBe('main')
    expect(registered).toBe(2)
    expect(polls).toBe(2)
  })
}

test('a session change with no SessionStart after it: /kaggle-monitor sets the pane up again', async ($, on) => {
  const held = new Map<string, unknown>()
  let polls = 0
  on('state.get', (_, e) => ({ value: held.has(e.key) ? { value: held.get(e.key), version: 1 } : { value: undefined, version: 0 } }))
  on('state.set', (_, e) => {
    held.set(e.key, e.value)
    return { value: { isSet: true, version: 1 } }
  })
  on('command.register', (_, e) => ({ value: { command: e.name } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.end', (_, e) => {
    held.clear()
    return { sessionId: e.sessionId }
  })
  on('command.run', () => ({ text: '' }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('ui.panes', () => ({ value: [] }))
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/u' : undefined }))
  on('env.set', () => ({ value: {} }))
  on('store.get', (_, e) => ({ value: e.key === 'active' ? 'main' : undefined }))
  on('store.set', () => ({ value: undefined }))
  on('clock.now', () => ({ value: Date.UTC(2026, 10, 9) }))
  on('clock.every', () => {
    polls++
    return new Promise(() => {})
  })
  on('process.run', () => {
    throw new Error('no processes in this test')
  })
  on('fs.exists', (_, e) => ({ value: e.path === '/home/u/.kaggle/access_token' }))
  on('fs.read', (_, e) => {
    throw new Error(`ENOENT: ${e.path}`)
  })
  await $.session.start({ cwd: '/ws', surface: 'terminal', isInteractive: true })
  await $.session.end({ reason: 'logout', sessionId: 's1', resume: { id: 's1' } })
  await $.command.run({ command: 'kaggle-monitor', args: '', origin: { kind: 'composer' }, presentation: { isFullscreen: false, columns: 100 } })
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect((await ui.find({ type: 'Text', text: /^(main|no Kaggle account)$/ }))?.text).toBe('main')
  await ui.unmount()
  expect(polls).toBe(2)
})

test('a log reader that hears nothing for 15 min is replaced; Refresh replaces every reader', async ($, on) => {
  let now = Date.UTC(2026, 10, 9, 12)
  const held = new Map<string, unknown>()
  // Each `kaggle kernels logs -f`: the server's replay of `log`, then nothing at all, as on a connection that died unheard.
  const readers: { argv: readonly string[]; isRead: boolean; isEnded: boolean }[] = []
  let log = '{"epoch": 0.091, "step": 60, "of": 1314}\n'
  let fire = () => {}
  const until = async (what: string, ok: () => boolean | Promise<boolean>) => {
    for (let i = 0; i < 2000; i++) {
      if (await ok()) return
      await sleep(1)
    }
    throw new Error(`never: ${what}`)
  }
  const tick = async (ms: number) => {
    now += ms
    fire()
  }
  on('state.get', (_, e) => ({ value: held.has(e.key) ? { value: held.get(e.key), version: 1 } : { value: undefined, version: 0 } }))
  on('state.set', (_, e) => {
    held.set(e.key, e.value)
    return { value: { isSet: true, version: 1 } }
  })
  on('command.register', (_, e) => ({ value: { command: e.name } }))
  on('command.run', () => ({ text: '' }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.end', (_, e) => ({ sessionId: e.sessionId }))
  on('ui.panes', () => ({ value: [{ id: 'kaggle-monitor', title: 'Kaggle', isShown: true, isFocused: false, isPlaced: true }] }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/u' : undefined }))
  on('env.set', () => ({ value: {} }))
  on('store.get', (_, e) => ({ value: e.key === 'active' ? 'main' : e.key === 'cli' ? ['/opt/kaggle'] : undefined }))
  on('store.set', () => ({ value: undefined }))
  on('clock.now', () => ({ value: now }))
  on('clock.after', () => ({ value: undefined }))
  on('clock.every', () => new Promise(resolve => (fire = () => resolve({ value: undefined }))))
  on('fs.exists', (_, e) => ({ value: e.path === '/home/u/.kaggle/access_token' || e.path === '/opt/kaggle' }))
  on('fs.read', (_, e) => {
    throw new Error(`ENOENT: ${e.path}`)
  })
  on('process.run', (_, e) => {
    const cmd = e.argv.join(' ')
    const ran = (stdout: string) => ({ value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } })
    if (cmd.startsWith('/opt/kaggle kernels list')) {
      return ran(JSON.stringify([{ ref: 'u/train', title: 'train', lastRunTime: '2026-11-09T11:00:00' }]))
    }
    if (cmd === '/opt/kaggle kernels status u/train') {
      return ran('u/train has status "KernelWorkerStatus.RUNNING"\n')
    }
    throw new Error('no such process in this test')
  })
  on('process.spawn', async function* (_, e, next) {
    const me = { argv: e.argv, isRead: false, isEnded: false }
    readers.push(me)
    try {
      yield { stream: 'stdout' as const, text: log }
      me.isRead = true // the pane asks for more only once it has handled the replay
      await new Promise(resolve => next.signal.addEventListener('abort', () => resolve(undefined), { once: true }))
    } finally {
      me.isEnded = true
    }
    return { value: { code: null, signal: 'SIGTERM' } }
  })
  const live = () => readers.filter(r => !r.isEnded).length
  const card = async (text: RegExp) => {
    const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
    const found = await ui.find({ type: 'Text', text })
    await ui.unmount()
    return found?.text
  }
  const warnings = async () => {
    const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
    const found = await ui.findAll({ type: 'Text' })
    await ui.unmount()
    return found.filter(t => t.props.color === 'warning').map(t => t.text)
  }
  // when the kernel's log last had a new line, as the pane holds it
  const heard = () => (held.get('live') as { value?: Record<string, { heardAt?: number }> } | undefined)?.value?.['u/train']?.heardAt

  await $.session.start({ cwd: '/ws', surface: 'terminal', isInteractive: true })
  await until('a first reader', () => readers.length === 1)
  expect(readers[0]?.argv.join(' ')).toBe('/opt/kaggle kernels logs -f u/train')
  await until('the first line heard', () => heard() !== undefined)
  const heardFirst = heard()

  // a minute on: quiet for a minute is no reason to reconnect
  await tick(60_000)
  await until('a poll', () => (held.get('kernels') as { value?: { main?: { fetchedAt?: number } } })?.value?.main?.fetchedAt === now)
  expect(readers.length).toBe(1)

  // a quiet log is no warning: a slow step can print nothing for a long time
  now += 2 * 3600_000
  expect(await card(/step 60/)).toBeDefined()
  expect(await warnings()).toEqual([])

  // past 15 min without a byte: the next poll ends the reader and starts another, which replays the log
  await tick(60_000)
  await until('a second reader', () => readers.length === 2)
  await until('the first reader ended', () => readers[0]!.isEnded)
  expect(live()).toBe(1)
  // its replay is the same log: nothing new was heard
  await until('the replay read', () => readers[1]!.isRead)
  expect(heard()).toBe(heardFirst)
  expect(await warnings()).toEqual([])

  // Refresh replaces every reader at once; this one's replay has a line more, heard now
  log += '{"epoch": 0.29, "step": 380, "of": 1314}\n'
  await $.command.run({
    command: 'kaggle-monitor',
    args: 'refresh',
    origin: { kind: 'composer' },
    presentation: { isFullscreen: false, columns: 100 },
  })
  await until('a third reader', () => readers.length === 3)
  await until('the second reader ended', () => readers[1]!.isEnded)
  expect(live()).toBe(1)
  await until('the new line read', () => readers[2]!.isRead)
  expect(await card(/step 380/)).toBeDefined()
  expect(heard()).toBe(now)

  // the session's end ends the last one
  await $.session.end({ reason: 'other', sessionId: 's1', resume: { id: 's1' } })
  await until('every reader ended', () => live() === 0)
})

// ---------------------------------------------------------------- the session's limit and end (#104)

test("a [kgmon] line's t0 is when the kernel's clock started", async () => {
  expect(parseKgmon('[kgmon] {"v":1,"t":5,"t0":1762394400,"cores":4}')?.t0).toBe(1762394400)
  expect(parseKgmon('[kgmon] {"v":1,"t":5,"cores":4}')?.t0).toBeUndefined() // a line from before t0
})

test('the run log says which push started the run Kaggle reports, and its -t cap', async () => {
  const log =
    [
      { kernel: 'u/a', version: 3, hours: 6, resource: 'GPU', pushed_at: '2026-11-06T01:00:00Z', status: 'superseded' },
      { kernel: 'u/a', version: 4, hours: 4, resource: 'GPU', pushed_at: '2026-11-06T02:00:40Z', status: 'pushed' },
      // pushed after the run Kaggle reports (a background push still pending): not this run's
      { kernel: 'u/a', hours: 2, resource: 'GPU', pushed_at: '2026-11-06T09:00:00Z', status: 'pending' },
      { kernel: 'u/tpu', hours: 9, accelerator: 'TpuV5E8', pushed_at: '2026-11-05T20:00:00Z', started_at: '2026-11-06T05:00:00Z' },
    ]
      .map(r => JSON.stringify(r))
      .join('\n') + '\nnot json\n'
  expect(runOf(log, 'U/A', '2026-11-06T02:00:05')).toEqual({ capH: 4, resource: 'GPU' })
  expect(runOf(log, 'u/tpu', '2026-11-05T20:00:30')).toEqual({ capH: 9, resource: 'TPU', startedAt: '2026-11-06T05:00:00Z' })
  expect(runOf(log, 'u/other', '2026-11-06T02:00:05')).toBeUndefined()
  expect(runOf('', 'u/a')).toBeUndefined()
})

test('the session clock: the limit, the time left, past it, and a TPU counted from when it was seen running', async () => {
  const k = (over: Partial<Kernel>): Kernel => ({ ref: 'u/a', title: 'a', status: 'RUNNING', lastRunTime: '2026-11-06T02:00:00', ...over })
  const live = (over: Partial<KernelLive>): KernelLive => ({ ref: 'u/a', alias: 'x', isFollowing: true, ...over })
  const at = (h: number, m = 0) => Date.UTC(2026, 10, 6, h, m)
  // a push capped at 4 h (the run log): 28 min left at 05:32, the last half hour
  let c = clockOf(k({ run: { capH: 4, resource: 'GPU' } }), undefined, at(5, 32))
  expect([c.limitH, c.limitFrom, c.phase, limitText(c)]).toEqual([4, 'cap', 'near', 'limit 4h · 28m left'])
  // nothing logged: Kaggle's 12 h maximum; past it at 14:04, the run time stops at the limit
  c = clockOf(k({}), undefined, at(14, 4))
  expect([c.limitFrom, c.phase, age(c.ranMs! / 1000), limitText(c)]).toEqual(['max', 'over', '12h00', 'past the 12h limit'])
  expect(endNotes(c, undefined, 'u/a', at(14, 4))[0]?.text).toBe(
    "Past its 12h limit by 4m: Kaggle has probably cancelled it, though its status still says RUNNING: Kaggle's status lags.",
  )
  // the kernel's own clock: it waited 40 min in the queue, so 36 min are left
  c = clockOf(k({}), live({ t0: at(2, 40) / 1000 }), at(14, 4))
  expect([c.phase, age(c.leftMs! / 1000)]).toEqual(['running', '36m'])
  // the pane saw it queue for an hour: counted from then
  c = clockOf(k({ seenRunningAt: at(3) }), undefined, at(14, 4))
  expect(limitText(c)).toBe('limit 12h · 56m left')
  // a TPU session counts from when it was seen running, not from its push 9 h before; never seen running: no clock
  const tpu = k({ ref: 'u/tpu', lastRunTime: '2026-11-05T20:00:00', run: { capH: 9, resource: 'TPU', startedAt: '2026-11-06T05:00:00Z' } })
  c = clockOf(tpu, undefined, at(6))
  expect([c.limitH, c.phase, limitText(c)]).toEqual([9, 'running', 'limit 9h · 8h00 left'])
  c = clockOf({ ...tpu, run: { resource: 'TPU' } }, undefined, at(6))
  expect([c.startAt, c.phase, limitText(c)]).toEqual([undefined, 'running', 'limit 9h'])
  // the script has ended: saving, its outputs listed; Kaggle closed the log: over, the run time stops there
  const L = live({ scriptEnd: { at: at(13), isError: false }, outputs: { checkedAt: at(13, 9), files: ['submission.csv', 'model.pt'] } })
  c = clockOf(k({}), L, at(13, 10))
  expect(c.phase).toBe('saving')
  expect(endNotes(c, L, 'u/a', at(13, 10))).toEqual([
    { text: 'The script finished 10m ago: Kaggle is saving the outputs.', tone: 'success' },
    { text: 'Outputs: 2 files readable now (submission.csv, model.pt): kaggle kernels output u/a -p <dir>.', tone: 'success' },
  ])
  c = clockOf(k({}), { ...L, logClosedAt: at(13, 20) }, at(13, 40))
  expect([c.phase, age(c.ranMs! / 1000), limitText(c)]).toEqual(['closed', '11h20', 'limit 12h'])
  // closed before the limit, it ended by itself: not cancelled by the limit once the clock passes it
  c = clockOf(k({}), { ...L, logClosedAt: at(13, 20) }, at(14, 5))
  expect([c.phase, c.isOver, limitText(c)]).toEqual(['closed', false, 'limit 12h'])
  expect(endNotes(c, { ...L, outputs: undefined }, 'u/a', at(14, 5))).toEqual([
    { text: "Kaggle has closed its log: the session is over (the script finished 1h05 ago), though its status still says RUNNING: Kaggle's status lags." },
  ])
  // still saving at 13:45: 15 min to the limit, a warning
  c = clockOf(k({}), L, at(13, 45))
  expect(endNotes(c, L, 'u/a', at(13, 45))[1]).toEqual({ text: '15m to the 12h limit: a save the limit cuts off can lose the outputs.', tone: 'warning' })
})

test("a log's end: the notebook converted to HTML, or nbconvert's error; a reading after it means it ran on", async () => {
  const tr = new LiveTracker()
  tr.line('[NbConvertApp] Converting notebook __notebook__.ipynb to notebook') // how a run starts
  expect(tr.scriptEnd).toBeUndefined()
  tr.reading({ t: 600, t0: 1762394400 })
  tr.reading({ t: 630, t0: 1762394500 }) // a later t0 (a second monitor) does not move the start
  expect(tr.t0).toBe(1762394400)
  tr.line('[NbConvertApp] Converting notebook __script__.ipynb to html')
  tr.line('[NbConvertApp] Writing 412093 bytes to __results__.html')
  expect(tr.scriptEnd).toEqual({ t: 630, isError: false })
  tr.reading({ t: 660 }) // the script had converted a notebook of its own: it still runs
  expect([tr.scriptEnd, tr.isResumed]).toEqual([undefined, true])
  tr.line('nbclient.exceptions.CellExecutionError: An error occurred while executing the following cell:')
  expect(tr.scriptEnd).toEqual({ t: 660, isError: true })
})

test('a card in its last half hour warns; past its limit it says Kaggle has probably cancelled it', async ($, on) => {
  // the helper's kernel last ran at 11:00 (UTC) with GPUs: a 12 h session
  let now = Date.UTC(2026, 10, 6, 22, 32)
  on('clock.now', () => ({ value: now }))
  const seeded: Record<string, unknown> = runningKernels(now, ['u2/train-a'])
  on('state.get', ($, e, next) => (e.key in seeded ? { value: { value: seeded[e.key], version: 1 } } : next(e)))
  on('store.set', () => ({ value: undefined }))
  let ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ type: 'Button', text: /▾ RUNNING u2\/train-a/ })).toBeDefined()
  expect((await ui.find({ type: 'Text', text: /^limit 12h · 28m left$/ }))?.props.color).toBe('warning')
  const near = await ui.find({ type: 'Text', text: /28m to its 12h limit and the script has not finished/ })
  expect(near?.props.color).toBe('warning')
  expect((await ui.find({ key: 'live-u2/train-a' }))?.props).toMatchObject({ borderColor: 'warning' })
  await ui.unmount()

  now = Date.UTC(2026, 10, 6, 23, 4)
  ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ type: 'Button', text: /▾ PAST LIMIT u2\/train-a/ })).toBeDefined()
  expect(await ui.find({ type: 'Text', text: /· acc2 · 12h00 ·/ })).toBeDefined() // the run time stopped at the limit
  expect(await ui.find({ type: 'Text', text: /^past the 12h limit$/ })).toBeDefined()
  expect(await ui.find({ type: 'Text', text: /Past its 12h limit by 4m: Kaggle has probably cancelled it/ })).toBeDefined()
  expect(await ui.find({ type: 'Text', text: /upload them as a dataset/ })).toBeDefined()
  // its readings stopped half an hour ago: no "last reading" warning on a session that is over
  expect(await ui.find({ type: 'Text', text: /Last reading/ })).toBeUndefined()
  await ui.unmount()
})

test('a closed card whose script has ended shows that and its outputs, not its last log line', async ($, on) => {
  const now = Date.UTC(2026, 10, 6, 14)
  on('clock.now', () => ({ value: now }))
  const kernels = runningKernels(now, ['u2/train-a', 'u2/train-b'])
  const live = kernels.live.value as unknown as Record<string, KernelLive>
  live['u2/train-a'] = { ...live['u2/train-a']!, scriptEnd: { at: now - 12 * 60_000, isError: false }, outputs: { checkedAt: now, files: ['sub.csv'] } }
  const seeded: Record<string, unknown> = kernels
  on('state.get', ($, e, next) => (e.key in seeded ? { value: { value: seeded[e.key], version: 1 } } : next(e)))
  const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
  expect(await ui.find({ type: 'Button', text: /▸ SAVING u2\/train-a/ })).toBeDefined()
  expect((await ui.find({ type: 'Text', text: /└ The script finished 12m ago: Kaggle is saving the outputs\./ }))?.props.color).toBe('success')
  expect(await ui.find({ type: 'Text', text: /└ Outputs: 1 file readable now \(sub\.csv\)/ })).toBeDefined()
  expect((await ui.find({ key: 'live-u2/train-a' }))?.props).toMatchObject({ borderColor: 'success' })
  expect(await ui.find({ type: 'Button', text: /▸ RUNNING u2\/train-b/ })).toBeDefined()
  expect((await ui.findAll({ type: 'Text', text: /└ \[train\] step 820 loss/ })).length).toBe(1) // train-b's alone
  await ui.unmount()
})

test("a run's cap from the run log: toasts as it nears and passes it, its script's end, its outputs, its closed log", async ($, on) => {
  let now = Date.UTC(2026, 10, 9, 11, 20)
  const held = new Map<string, unknown>()
  const toasts: string[] = []
  const t0 = Date.UTC(2026, 10, 9, 11, 0) / 1000 // the kernel's clock started at its push
  let log = `[kgmon] {"v":1,"t":1200,"t0":${t0},"cores":4,"gpus":[[0,90,9000,15360,"Tesla T4"]]}\n{"step": 60, "of": 1314}\n`
  let status = 'RUNNING'
  let isLogClosed = false // Kaggle closed it: a reader gets the log, then the stream ends
  const readers: { isRead: boolean; isEnded: boolean }[] = []
  const asked: { cwd?: string; stdin?: string; token?: string }[] = []
  let fire = () => {}
  const until = async (what: string, ok: () => boolean | Promise<boolean>) => {
    for (let i = 0; i < 2000; i++) {
      if (await ok()) return
      await sleep(1)
    }
    throw new Error(`never: ${what}`)
  }
  const polled = () => (held.get('kernels') as { value?: { main?: { fetchedAt?: number } } })?.value?.main?.fetchedAt === now
  // The clock moves on and the poll timer fires, again while an earlier poll (Refresh's) still runs and this one is skipped.
  const tick = async (ms: number) => {
    now += ms
    await until(`a poll at +${ms / 60_000} min`, () => {
      if (polled()) return true
      fire()
      return false
    })
  }
  // Refresh: every reader replaced by one that replays the log
  const refresh = async () => {
    const n = readers.length
    await $.command.run({ command: 'kaggle-monitor', args: 'refresh', origin: { kind: 'composer' }, presentation: { isFullscreen: false, columns: 100 } })
    await until('a new reader', () => readers.length > n && readers[readers.length - 1]!.isRead)
  }
  const card = async () => {
    const ui = await $.ui.mount({ ...PANE, surface: 'terminal' })
    const texts = (await ui.findAll({ type: 'Text' })).map(t => t.text)
    const title = (await ui.find({ type: 'Button', text: /u\/train/ }))?.text
    await ui.unmount()
    return { title, texts }
  }
  on('state.get', (_, e) => ({ value: held.has(e.key) ? { value: held.get(e.key), version: 1 } : { value: undefined, version: 0 } }))
  on('state.set', (_, e) => {
    held.set(e.key, e.value)
    return { value: { isSet: true, version: 1 } }
  })
  on('command.register', (_, e) => ({ value: { command: e.name } }))
  on('command.run', () => ({ text: '' }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.end', (_, e) => ({ sessionId: e.sessionId }))
  on('ui.panes', () => ({ value: [{ id: 'kaggle-monitor', title: 'Kaggle', isShown: true, isFocused: false, isPlaced: true }] }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.toast', (_, e) => {
    toasts.push(e.text)
    return { value: undefined }
  })
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/u' : undefined }))
  on('env.set', () => ({ value: {} }))
  on('store.get', (_, e) => ({ value: e.key === 'active' ? 'main' : e.key === 'cli' ? ['/opt/kaggle'] : undefined }))
  on('store.set', () => ({ value: undefined }))
  on('clock.now', () => ({ value: now }))
  on('clock.after', () => ({ value: undefined }))
  on('clock.every', () => new Promise(resolve => (fire = () => resolve({ value: undefined }))))
  on('fs.exists', (_, e) => ({
    value: e.path === '/home/u/.kaggle/access_token' || e.path === '/opt/kaggle' || e.path.endsWith('/kgkit/kaggle_outputs.py'),
  }))
  on('fs.read', (_, e) => {
    // pushed with -t 3600 (`kgkit gpu push --hours 1`)
    const run = { kernel: 'u/train', version: 2, hours: 1, resource: 'GPU', pushed_at: '2026-11-09T11:00:20Z', status: 'running' }
    if (e.path === '/ws/.kaggle-gm/gpu_runs.jsonl') return { value: `${JSON.stringify(run)}\n` }
    if (e.path === '/opt/kaggle') return { value: '#!/opt/env/bin/python3.12\n# -*- coding: utf-8 -*-\nfrom kaggle.cli import main\n' }
    throw new Error(`ENOENT: ${e.path}`)
  })
  on('process.run', (_, e) => {
    const cmd = e.argv.join(' ')
    const ran = (stdout: string) => ({ value: { exitCode: 0, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false } })
    if (cmd.startsWith('/opt/kaggle kernels list')) {
      return ran(JSON.stringify([{ ref: 'u/train', title: 'train', lastRunTime: '2026-11-09T11:00:00' }]))
    }
    if (cmd === '/opt/kaggle kernels status u/train') return ran(`u/train has status "KernelWorkerStatus.${status}"\n`)
    if (cmd === '/opt/env/bin/python3.12 -m kgkit.kaggle_outputs') {
      asked.push({ cwd: e.init?.cwd, stdin: e.init?.stdin, token: e.init?.env?.KAGGLE_API_TOKEN })
      return ran('{"files": ["submission.csv", "oof.npy"], "more": false, "log": "", "error": null}\n')
    }
    throw new Error('no such process in this test')
  })
  on('process.spawn', async function* (_, e, next) {
    const me = { isRead: false, isEnded: false }
    readers.push(me)
    try {
      yield { stream: 'stdout' as const, text: log }
      me.isRead = true
      if (isLogClosed) return { value: { code: 0, signal: null } }
      await new Promise(resolve => next.signal.addEventListener('abort', () => resolve(undefined), { once: true }))
    } finally {
      me.isEnded = true
    }
    return { value: { code: null, signal: 'SIGTERM' } }
  })

  await $.session.start({ cwd: '/ws', surface: 'terminal', isInteractive: true })
  await until('a first reader', () => readers[0]?.isRead === true)
  await until('the reading', async () => (await card()).texts.some(t => t === 'limit 1h · 40m left'))
  expect((await card()).title).toMatch(/^▾ RUNNING u\/train · main · 20m · limit 1h · 40m left$/)

  // 11:40, 20 min left: not yet the last quarter of a 1 h cap
  await tick(20 * 60_000)
  expect(toasts).toEqual([])
  // 11:47, 13 min left, the script still running: a warning on the card, and a toast
  await tick(7 * 60_000)
  expect(toasts).toEqual(['Kaggle kernel u/train (main): 13m to its 1h limit, and the script has not finished'])
  expect((await card()).texts).toContain(
    '13m to its 1h limit and the script has not finished: Kaggle cancels it at the limit and keeps only what is already in /kaggle/working.',
  )

  // the script ends at 2800 s on the kernel's clock (11:46:40): a new reader replays it, ending in the notebook's
  // conversion to HTML
  log +=
    `[kgmon] {"v":1,"t":2800,"t0":${t0},"cores":4,"gpus":[[0,5,9000,15360,"Tesla T4"]]}\n` +
    '[NbConvertApp] Converting notebook __script__.ipynb to html\n[NbConvertApp] Writing 412093 bytes to __results__.html\n'
  await refresh()
  await tick(6 * 60_000) // 11:53: the poll sees it saving, and looks its outputs up
  await until('the outputs looked up', () => toasts.length === 3)
  expect(toasts.slice(1)).toEqual([
    'Kaggle kernel u/train (main): the script finished; Kaggle is saving its outputs',
    'Kaggle kernel u/train (main): 2 output files readable (kaggle kernels output)',
  ])
  expect(asked[0]?.cwd).toBeDefined()
  expect(JSON.parse(asked[0]!.stdin!)).toEqual({ kernel: 'u/train', max_pages: 1, log: false })
  expect(asked[0]?.token).toBe('/home/u/.kaggle/access_token')
  let seen = await card()
  expect(seen.title).toMatch(/^▾ SAVING u\/train · main · 53m · limit 1h · 7m left$/)
  expect(seen.texts).toContain('The script finished 6m ago: Kaggle is saving the outputs.')
  expect(seen.texts).toContain('7m to the 1h limit: a save the limit cuts off can lose the outputs.')
  expect(seen.texts).toContain('Outputs: 2 files readable now (submission.csv, oof.npy): kaggle kernels output u/train -p <dir>.')

  // 12:01: past the cap, Kaggle still says RUNNING
  await tick(8 * 60_000)
  expect(toasts[3]).toBe("Kaggle kernel u/train (main): past its 1h limit, probably cancelled (Kaggle's status lags)")
  seen = await card()
  expect(seen.title).toMatch(/^▾ PAST LIMIT u\/train · main · 1h00 · past the 1h limit$/)

  // Kaggle closes the log: the next reader gets the replay, then the stream ends; one end, one toast
  isLogClosed = true
  await refresh()
  await until('the closed log read', () => readers[readers.length - 1]?.isEnded === true)
  await until('the card ended', async () => /^▾ ENDED u\/train/.test((await card()).title ?? ''))
  expect((await card()).texts).toContain(
    "Kaggle has closed its log, past its 1h limit: the limit has probably cancelled the session, though its status still says RUNNING: Kaggle's status lags.",
  )
  await tick(60_000)
  expect(toasts.length).toBe(4)

  // the status catches up
  status = 'CANCELLED'
  await tick(60_000)
  expect(toasts[4]).toBe('Kaggle kernel u/train (main): CANCELLED; 2 output files were readable')
  await $.session.end({ reason: 'other', sessionId: 's1', resume: { id: 's1' } })
})
