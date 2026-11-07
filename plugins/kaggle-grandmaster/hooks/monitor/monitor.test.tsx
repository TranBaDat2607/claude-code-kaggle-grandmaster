import { expect, test } from 'claude-code/testing'

import { describe, lastLine } from './describe'
import { parseKgmon, parseQuota, status, utc } from './kaggle'
import { etime, parseGpus } from './probe'

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
      shape: 'live-1',
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
    for (const text of [/u2\/train-b/, /97% of 4 cores/, /12\.0 \/ 30\.0 GB/, /88% · 11\.0 \/ 15\.0 GB · T4/, /1\.0 \/ 20\.0 GB/, /exp\/f0 on gpu0/, /epoch 3\/10/]) {
      expect((await ui.find({ type: 'Text', text }))?.text).toMatch(text)
    }
    expect(await ui.find({ type: 'Text', text: /Processes \(/ })).toBeUndefined()
    await ui.unmount()
  }
})
