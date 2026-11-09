import { expect, test } from 'claude-code/testing'

import { describe, lastLine } from './describe'
import { parseKgmon, parseQuota, status, utc } from './kaggle'
import { HISTORY_MAX, LiveTracker, parseProgress, spark } from './live'
import { etime, parseGpus } from './probe'

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
      shape: 'kernels-1',
      value: { acc2: { value: [{ ref: 'u2/train-b', title: 'train b', status: 'RUNNING', lastRunTime: '2026-11-06T11:00:00' }], fetchedAt: now } },
    },
    live: {
      shape: 'live-2',
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
    shape: 'kernels-1',
    value: { acc2: { value: refs.map(ref => ({ ref, title: ref, status: 'RUNNING', lastRunTime: '2026-11-06T11:00:00' })), fetchedAt: now } },
  },
  live: {
    shape: 'live-2',
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
  for (const text of [/30m of 12h/, /step 820\/858 · epoch 1\.9$/, /53% now · avg 47% · 4 cores/, /7\.1 \/ 30\.0 GB · peak 8\.0/,
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
