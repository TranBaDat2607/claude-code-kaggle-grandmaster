"""Several Kaggle accounts registered in the Kaggle pane (#147): a run belongs to the account that owns its kernel
(`<owner>/<slug>`). Its reservation counts against that account's quota only, and Kaggle is asked about it with that
account's token, whichever account the session has active."""

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

from kgkit import gpu as G
from kgkit import state as S


def _cp(stdout="", rc=0, stderr=""):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


def _ago(**kw) -> str:
    return G._iso(G._now() - dt.timedelta(**kw))


@pytest.fixture
def ws(tmp_path):
    S.CompetitionState(slug="demo", metric="auc", deadline=(dt.date.today() + dt.timedelta(days=20)).isoformat()
                       ).save(tmp_path)
    return tmp_path


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    """Two registered accounts, main (acca) and second (accb), with accb active: the pane switched to it."""
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    reg = {}
    for alias, user in (("main", "acca"), ("second", "accb")):
        (tokens / f"access_token_{alias}").write_text("SECRET-TOKEN")
        reg[alias] = {"user": user, "token_file": str(tokens / f"access_token_{alias}")}
    (tmp_path / "kg_accounts.json").write_text(json.dumps(reg))
    monkeypatch.setenv("KGKIT_ACCOUNTS_FILE", str(tmp_path / "kg_accounts.json"))
    monkeypatch.setenv("KAGGLE_USERNAME", "accb")
    return tokens


def _quota(root, left, user=None, fetched=None):
    q = {"used_h": 30 - left, "remaining_h": left, "total_h": 30.0, "refresh_at": "2099-01-03T00:00:00Z",
         "fetched_at": fetched or G._iso(G._now())}
    if user:
        q["user"] = user
    G.quota_path(root).parent.mkdir(parents=True, exist_ok=True)
    G.quota_path(root).write_text(json.dumps(q))


def _quota_json(left):
    return json.dumps([{"resource": "GPU", "used": f"{30 - left}h", "remaining": f"{left}h", "total": "30h",
                        "refreshAt": "2099-01-03T00:00:00"}])


def _issue_runs(ws):
    """The runs of #147: two of acca's still in flight, and one of accb's that has just started."""
    G.log_run(ws, {"kernel": "acca/kernel-1", "version": 1, "hours": 5.5, "status": "running",
                   "pushed_at": _ago(minutes=20), "started_at": _ago(minutes=20)})
    G.log_run(ws, {"kernel": "acca/kernel-2", "version": 1, "hours": 3.6, "status": "pushed",
                   "pushed_at": _ago(minutes=20)})
    G.log_run(ws, {"kernel": "accb/kernel-3", "version": 1, "hours": 11.5, "status": "running",
                   "pushed_at": _ago(seconds=0), "started_at": _ago(seconds=0)})


def test_other_accounts_runs_are_not_counted_against_the_active_accounts_quota(ws, accounts):
    _issue_runs(ws)
    _quota(ws, 29.99, user="accb")
    q = G.cached_quota(ws)
    assert G.in_flight_hours(ws, q) == pytest.approx(11.5, abs=0.02)
    assert G.available_hours(ws, q) == pytest.approx(18.49, abs=0.02)  # 30h less accb's own 11.5h run
    assert G.in_flight_hours(ws, q, user="acca") == pytest.approx(5.5 + 3.6 - 2 * 20 / 60, abs=0.02)
    txt = G.quota_report(ws, refresh=False)
    assert "accb/kernel-3 v1 [running, cap 11.5h]" in txt and "-> 18.49h plannable" in txt
    reserved = txt.split("in flight:")[1].split("other accounts")[0]
    assert "acca/" not in reserved  # listed apart, on their own quota
    assert "other accounts' runs in flight (their own quota): acca/kernel-1 v1, acca/kernel-2 v1" in txt


def test_runs_logged_since_the_snapshot_count_only_for_their_own_account(ws, accounts):
    _quota(ws, 20.0, user="accb", fetched=_ago(hours=3))
    for k in ("acca/done", "accb/done"):
        G.log_run(ws, {"kernel": k, "version": 1, "hours": 2.0, "status": "complete", "pushed_at": _ago(hours=2),
                       "started_at": _ago(hours=2), "finished_at": _ago(hours=1), "gpu_hours": 1.0})
    q = G.cached_quota(ws)
    assert [r["kernel"] for r, _ in G.unseen_spend(ws, q)] == ["accb/done"]
    assert G.available_hours(ws, q) == pytest.approx(19.0, abs=0.01)


def test_without_a_known_active_account_every_run_still_counts(ws, accounts, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME")  # no kaggle.json either (conftest): the old, cautious total
    _issue_runs(ws)
    _quota(ws, 29.99)
    assert G.in_flight_hours(ws, G.cached_quota(ws)) == pytest.approx(11.5 + 5.5 + 3.6 - 2 * 20 / 60, abs=0.02)


def test_quota_all_puts_each_accounts_runs_on_its_own_line(ws, accounts, monkeypatch):
    _issue_runs(ws)
    left = {"acca": 20.0, "accb": 29.99}

    def fake(args, timeout=600, env=None):
        if args[:2] == ["kernels", "status"]:
            return _cp(f'{args[2]} has status "KernelWorkerStatus.RUNNING"')
        return _cp(_quota_json(left[env["KAGGLE_USERNAME"]]))

    monkeypatch.setattr(G, "run_kaggle", fake)
    txt = G.accounts_report(ws)
    # acca/kernel-2 is seen running now: timed from now, its whole cap is reserved
    assert "main: GPU 20.0h of 30h, up to 8.8h in flight -> 11.2h plannable" in txt
    assert "second (active): GPU 30.0h of 30h, up to 11.5h in flight -> 18.5h plannable" in txt
    assert "Total: GPU 50.0h left, 29.7h plannable" in txt
    # the session brief counts each account's runs against its own hours too
    assert "main 11.2h · second 18.5h = 29.7h" in G.quota_line(ws)


def test_each_run_is_checked_with_the_token_of_the_account_that_owns_it(ws, accounts, monkeypatch):
    for k in ("acca/k1", "accb/k3", "stranger/k9"):
        G.log_run(ws, {"kernel": k, "version": 1, "hours": 4.0, "status": "running", "pushed_at": _ago(minutes=5),
                       "started_at": _ago(minutes=5)})
    calls = []

    def fake(args, timeout=600, env=None):
        calls.append((list(args), (env or {}).get("KAGGLE_API_TOKEN")))
        return _cp(f'{args[2]} has status "KernelWorkerStatus.COMPLETE"')

    monkeypatch.setattr(G, "run_kaggle", fake)
    out = G.refresh_runs(ws)
    asked = {a[2]: tok for a, tok in calls if a[:2] == ["kernels", "status"]}
    # acca's kernel with acca's token; accb's (the active account) with the session's own login
    assert asked == {"acca/k1": str(accounts / "access_token_main"), "accb/k3": None}
    assert out == {"acca/k1": "complete", "accb/k3": "complete", "stranger/k9": "unknown"}
    s, text = G.kernel_status("stranger/k9")  # an account that is not registered: not asked, and said why
    assert s == "unknown" and "owned by stranger" in text and "not registered in the Kaggle pane" in text
    assert "slug" not in text
    assert all(a[2] != "stranger/k9" for a, _ in calls)


def test_logs_outputs_and_deletes_go_through_the_owners_token(ws, accounts, monkeypatch):
    seen = []

    def fake(args, timeout=600, env=None):
        seen.append((args[:2], (env or {}).get("KAGGLE_API_TOKEN")))
        return _cp("")

    monkeypatch.setattr(G, "run_kaggle", fake)
    G.kaggle_for_kernel("acca/k1", ["kernels", "logs", "acca/k1"])
    G.kaggle_for_kernel("accb/k3", ["kernels", "output", "accb/k3", "-p", "x"])
    assert seen == [(["kernels", "logs"], str(accounts / "access_token_main")), (["kernels", "output"], None)]
    with pytest.raises(SystemExit, match="owned by stranger"):
        G.kaggle_for_kernel("stranger/k9", ["kernels", "delete", "stranger/k9", "-y"])


def test_a_pending_push_of_another_account_is_confirmed_from_that_accounts_listing(ws, accounts, monkeypatch):
    G.log_run(ws, {"kernel": "acca/k5", "hours": 3.0, "status": G.PENDING, "pushed_at": _ago(minutes=5)})

    def fake(args, timeout=600, env=None):
        mine = (env or {}).get("KAGGLE_API_TOKEN") == str(accounts / "access_token_main")
        rows = [{"ref": "acca/k5", "lastRunTime": _ago(minutes=2)}] if mine else []
        return _cp(json.dumps(rows))

    monkeypatch.setattr(G, "run_kaggle", fake)
    assert [r["kernel"] for r in G.confirm_pending(ws)] == ["acca/k5"]
    assert G.runs(ws)[-1]["status"] == "pushed"


def test_a_quota_cached_for_another_account_is_not_the_active_accounts(ws, accounts, monkeypatch):
    _quota(ws, 5.0, user="acca")  # read before the pane switched to accb
    assert G.cached_quota(ws) is None
    # the reading of `gpu quota --all` for the active account serves instead
    monkeypatch.setattr(G, "run_kaggle", lambda args, timeout=600, env=None:
                        _cp(_quota_json({"acca": 5.0, "accb": 29.0}[env["KAGGLE_USERNAME"]])))
    G.fetch_account_quotas(ws)
    assert G.cached_quota(ws)["remaining_h"] == 29.0
    # a live read records whose quota it is
    monkeypatch.setattr(G, "run_kaggle", lambda args, timeout=600, env=None: _cp(_quota_json(27.0)))
    q, _ = G.live_quota(ws)
    assert json.loads(G.quota_path(ws).read_text())["user"] == "accb" and G.cached_quota(ws)["remaining_h"] == 27.0
    _quota(ws, 8.0, user=None, fetched=G._iso(G._now() + dt.timedelta(minutes=1)))  # a cache from before kgkit
    assert G.cached_quota(ws)["remaining_h"] == 8.0  # recorded the user is still read: the newest reading wins


# ----------------------------------------------------------------------------- gpu push --account
def _kernel(ws, owner="accb", comp="demo", sources=()):
    k = ws / "kernels" / "train-a"
    k.mkdir(parents=True, exist_ok=True)
    meta = {"id": f"{owner}/train-a", "code_file": "run.py", "kernel_type": "script", "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4", "competition_sources": [comp] if comp else [],
            "kernel_sources": list(sources)}
    (k / "kernel-metadata.json").write_text(json.dumps(meta))
    (k / "kg-train.json").write_text(json.dumps({"kernel": meta["id"], "name": "train-a", "jobs": ["train"],
                                                 "hours": 4.0, "accelerator": "NvidiaTeslaT4", "folds": None,
                                                 "cmd": "python train.py"}))
    (k / "run.py").write_text("print('hi')\n")
    return k


class AsAccount:
    """A fake Kaggle CLI that answers as the account whose token the call carries (None: the active one, accb)."""

    def __init__(self, tokens, left=None, entered=("demo",), busy=None, done=()):
        self.by_token = {str(tokens / "access_token_main"): "acca", str(tokens / "access_token_second"): "accb"}
        self.left = left or {"acca": 20.0, "accb": 30.0}
        self.entered, self.busy, self.done, self.calls = set(entered), busy or {}, set(done), []

    def __call__(self, args, timeout=600, env=None):
        who = self.by_token.get((env or {}).get("KAGGLE_API_TOKEN"), "accb")
        self.calls.append((who, list(args)))
        if args[:1] == ["quota"]:
            return _cp(_quota_json(self.left[who]))
        if args[:2] == ["competitions", "list"]:
            slug = args[args.index("--search") + 1]
            return _cp(f"ref,deadline\nhttps://www.kaggle.com/competitions/{slug},2099-01-01\n"
                       if slug in self.entered else "No competitions found\n")
        if args[:2] == ["kernels", "list"]:
            rows = [{"ref": k, "lastRunTime": G._now().strftime("%Y-%m-%dT%H:%M:%S")} for k in self.busy.get(who, [])]
            return _cp(json.dumps(rows))
        if args[:2] == ["kernels", "status"]:
            return _cp(f'{args[2]} has status "KernelWorkerStatus.{"COMPLETE" if args[2] in self.done else "RUNNING"}"')
        if args[:2] == ["kernels", "push"]:
            kid = json.loads((Path(args[args.index("-p") + 1]) / "kernel-metadata.json").read_text())["id"]
            return _cp(f"Kernel version 3 successfully pushed.  Please check progress at https://www.kaggle.com/code/{kid}")
        return _cp("", 1, "not faked")


def test_push_to_another_account_runs_there_and_checks_its_quota(ws, accounts, monkeypatch):
    k = _kernel(ws)
    fake = AsAccount(accounts)
    monkeypatch.setattr(G, "run_kaggle", fake)
    rec = G.push(ws, k, account="main")
    # the folder is pushed as main's kernel, with main's token
    assert rec["kernel"] == "acca/train-a" and G.runs(ws)[-1]["kernel"] == "acca/train-a"
    assert json.loads((k / "kernel-metadata.json").read_text())["id"] == "acca/train-a"
    assert json.loads((k / "kg-train.json").read_text())["kernel"] == "acca/train-a"
    assert [who for who, a in fake.calls if a[:2] == ["kernels", "push"]] == ["acca"]
    assert ("acca", ["quota", "--format", "json"]) in fake.calls  # its own quota, not the active account's
    assert any(who == "acca" and a[:2] == ["kernels", "list"] for who, a in fake.calls)  # its own sessions
    assert G.cached_account_quotas(ws)["main"]["quota"]["remaining_h"] == 20.0
    # the active account's cached quota is not overwritten by main's reading
    assert not G.quota_path(ws).is_file() or json.loads(G.quota_path(ws).read_text()).get("user") != "acca"


def test_push_to_another_account_refuses_what_that_account_cannot_cover(ws, accounts, monkeypatch):
    k = _kernel(ws)
    monkeypatch.setattr(G, "run_kaggle", AsAccount(accounts, left={"acca": 3.0, "accb": 30.0}))
    with pytest.raises(SystemExit, match=r"capped at 4.00h but only 3.00h of GPU quota is plannable on main \(acca\)"):
        G.push(ws, k, account="main")
    G.log_run(ws, {"kernel": "acca/other", "hours": 9.0, "status": "running", "pushed_at": _ago(minutes=1),
                   "started_at": _ago(minutes=1)})
    monkeypatch.setattr(G, "run_kaggle", AsAccount(accounts, left={"acca": 12.0, "accb": 30.0}))
    with pytest.raises(SystemExit, match="only 3.0"):  # its own run in flight counts against it
        G.push(ws, k, account="main")


def test_push_refuses_an_account_that_has_not_joined_the_competition(ws, accounts, monkeypatch):
    k = _kernel(ws)
    fake = AsAccount(accounts, entered=())
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match=r"main \(acca\) has not joined demo.*kaggle.com/competitions/demo/rules"):
        G.push(ws, k, account="main")
    assert not any(a[:2] == ["kernels", "push"] for _, a in fake.calls)
    assert json.loads((k / "kernel-metadata.json").read_text())["id"] == "accb/train-a"  # left as it was


def test_push_refuses_another_accounts_private_kernel_as_an_input(ws, accounts, monkeypatch):
    k = _kernel(ws, sources=["accb/prep-features", "someone/public-notebook"])
    fake = AsAccount(accounts)
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match="accb/prep-features.*second"):
        G.push(ws, k, account="main")
    assert not any(a[:2] == ["kernels", "push"] for _, a in fake.calls)


def test_push_names_the_registered_accounts_when_the_alias_is_unknown(ws, accounts):
    with pytest.raises(SystemExit, match="--account nope: not registered.*main, second"):
        G.push(ws, _kernel(ws), account="nope")


def test_push_to_the_active_account_by_name_uses_the_sessions_login(ws, accounts, monkeypatch):
    k = _kernel(ws, owner="acca")  # built while main was active
    fake = AsAccount(accounts)
    monkeypatch.setattr(G, "run_kaggle", fake)
    rec = G.push(ws, k, account="accb")  # a username works as well as an alias
    assert rec["kernel"] == "accb/train-a"
    assert [who for who, a in fake.calls if a[:2] == ["kernels", "push"]] == ["accb"]


def test_repush_deletes_and_pushes_again_as_the_kernels_owner(ws, accounts, monkeypatch):
    k = _kernel(ws, owner="acca")
    G.log_run(ws, {"kernel": "acca/train-a", "version": 2, "hours": 4.0, "status": "queued",
                   "pushed_at": _ago(minutes=50), "dir": "kernels/train-a"})
    fake = AsAccount(accounts)
    base = fake.__call__

    def calls(args, timeout=600, env=None):
        if args[:2] == ["kernels", "status"]:
            fake.calls.append(("?", list(args)))
            return _cp(f'{args[2]} has status "KernelWorkerStatus.QUEUED"')
        if args[:2] == ["kernels", "delete"]:
            fake.calls.append((fake.by_token.get((env or {}).get("KAGGLE_API_TOKEN"), "accb"), list(args)))
            return _cp("Kernel deleted successfully.")
        return base(args, timeout, env)

    monkeypatch.setattr(G, "run_kaggle", calls)
    monkeypatch.setattr(G, "DELETED_RE", __import__("re").compile("deleted", __import__("re").I))
    rec = G.repush(ws, "acca/train-a", yes=True)
    assert rec["kernel"] == "acca/train-a"  # still main's kernel, though accb is active
    assert [w for w, a in fake.calls if a[:2] in (["kernels", "delete"], ["kernels", "push"])] == ["acca", "acca"]


def test_cli_push_takes_an_account(ws, accounts, monkeypatch, capsys):
    from kgkit import cli
    seen = {}

    def push(root, kdir, **kw):
        seen.update(kw)
        return {"kernel": "acca/train-a", "version": 1, "accelerator": "NvidiaTeslaT4", "hours": 4.0, "jobs": ["train"]}

    monkeypatch.setattr(G, "push", push)
    monkeypatch.chdir(ws)
    cli.main(["gpu", "push", str(_kernel(ws)), "--account", "main"])
    assert seen["account"] == "main" and "pushed acca/train-a v1" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the best-fit account
def _pushed_as(fake):
    return [who for who, a in fake.calls if a[:2] == ["kernels", "push"]]


@pytest.mark.parametrize("cap, left, chosen", [
    (10.0, {"acca": 11.0, "accb": 30.0}, "acca"),  # the tightest fit: accb's full week stays whole
    (12.0, {"acca": 11.0, "accb": 30.0}, "accb"),  # acca cannot cover it
    (10.0, {"acca": 30.0, "accb": 30.0}, "accb"),  # a tie: the active account, no switch of owner
])
def test_a_built_kernel_goes_to_the_account_whose_hours_fit_it_best(ws, accounts, monkeypatch, capsys, cap, left,
                                                                     chosen):
    k = _kernel(ws)
    fake = AsAccount(accounts, left=left)
    monkeypatch.setattr(G, "run_kaggle", fake)
    rec = G.push(ws, k, hours=cap)
    assert rec["kernel"] == f"{chosen}/train-a" and _pushed_as(fake) == [chosen]
    out = capsys.readouterr().out
    assert f"account: {'main' if chosen == 'acca' else 'second'} ({chosen})" in out and "best fit" in out


def test_no_account_with_room_refuses_and_lists_their_hours(ws, accounts, monkeypatch):
    k = _kernel(ws)
    monkeypatch.setattr(G, "run_kaggle", AsAccount(accounts, left={"acca": 3.0, "accb": 2.5}))
    with pytest.raises(SystemExit, match=r"no registered account has 4.00h of GPU quota plannable: main \(acca\) "
                                         r"3.00h, second \(accb\) 2.50h"):
        G.push(ws, k)


def test_the_best_fit_skips_an_account_that_cannot_take_the_run(ws, accounts, monkeypatch, capsys):
    k = _kernel(ws)
    # acca fits best but both its batch GPU sessions are taken
    G.log_run(ws, {"kernel": "acca/x", "accelerator": "NvidiaTeslaT4", "hours": 0.5, "status": "running",
                   "pushed_at": _ago(minutes=1), "started_at": _ago(minutes=1)})
    G.log_run(ws, {"kernel": "acca/y", "accelerator": "NvidiaTeslaT4", "hours": 0.5, "status": "running",
                   "pushed_at": _ago(minutes=1), "started_at": _ago(minutes=1)})
    fake = AsAccount(accounts, left={"acca": 12.0, "accb": 30.0}, busy={"acca": ["acca/x", "acca/y"]})
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.push(ws, k)["kernel"] == "accb/train-a"
    assert "main (acca) skipped: its 2 batch GPU session(s) are taken" in capsys.readouterr().out


def test_the_best_fit_skips_an_account_that_has_not_joined(ws, accounts, monkeypatch, capsys):
    k = _kernel(ws)

    class NotJoined(AsAccount):
        def __call__(self, args, timeout=600, env=None):
            if args[:2] == ["competitions", "list"] and (env or {}).get("KAGGLE_API_TOKEN"):
                self.calls.append(("acca", list(args)))
                return _cp("No competitions found\n")
            return super().__call__(args, timeout, env)

    fake = NotJoined(accounts, left={"acca": 5.0, "accb": 30.0})
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.push(ws, k)["kernel"] == "accb/train-a"
    assert "main (acca) skipped: it has not joined demo" in capsys.readouterr().out


def test_a_kernel_reading_another_accounts_output_stays_with_that_account(ws, accounts, monkeypatch):
    k = _kernel(ws, sources=["acca/prep-features"])
    # accb fits best, but cannot read acca's output
    fake = AsAccount(accounts, left={"acca": 25.0, "accb": 5.0}, done=["acca/prep-features"])
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.push(ws, k)["kernel"] == "acca/train-a"


def test_a_hand_written_notebook_folder_stays_on_the_active_account(ws, accounts, monkeypatch):
    k = _kernel(ws)
    (k / "kg-train.json").unlink()  # not built by `gpu build`: a submission or inference notebook, say
    fake = AsAccount(accounts, left={"acca": 5.0, "accb": 30.0})
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.push(ws, k, hours=4.0)["kernel"] == "accb/train-a" and _pushed_as(fake) == ["accb"]
    # unless asked: --account auto picks the best fit for any folder
    assert G.push(ws, k, hours=4.0, account="auto")["kernel"] == "acca/train-a"


def test_with_one_account_nothing_changes(ws, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "accb")  # no registry (conftest): no quota of other accounts is read
    k = _kernel(ws)
    fake = AsAccount(G.Path(ws), left={"acca": 30.0, "accb": 30.0})
    monkeypatch.setattr(G, "run_kaggle", fake)
    assert G.push(ws, k)["kernel"] == "accb/train-a"
    assert [a[:1] for _, a in fake.calls].count(["quota"]) == 1


# ----------------------------------------------------------------------------- private dataset copies
class WithDatasets(AsAccount):
    """AsAccount, with accb's private dataset accb/feats (one file) and its copies made under other accounts."""

    def __init__(self, tokens, updated="2026-11-20T10:00:00", fail_create=False, **kw):
        super().__init__(tokens, **kw)
        self.updated, self.fail_create, self.made = updated, fail_create, {}

    def __call__(self, args, timeout=600, env=None):
        who = self.by_token.get((env or {}).get("KAGGLE_API_TOKEN"), "accb")
        if args[:2] == ["datasets", "list"]:
            self.calls.append((who, list(args)))
            owner = args[args.index("--user") + 1]
            return _cp(json.dumps([{"ref": f"{owner}/feats", "lastUpdated": self.updated}] if owner == "accb" else []))
        if args[:2] == ["datasets", "download"]:
            self.calls.append((who, list(args)))
            if who != "accb":
                return _cp("", 1, "403 Client Error: Forbidden")
            dest = Path(args[args.index("-p") + 1])
            (dest / "sub").mkdir(parents=True, exist_ok=True)
            (dest / "train_feats.parquet").write_bytes(b"PAR1")
            (dest / "sub" / "more.npy").write_bytes(b"NPY")
            return _cp("Downloading feats.zip\n")
        if args[:2] in (["datasets", "create"], ["datasets", "version"]):
            folder = Path(args[args.index("-p") + 1])
            meta = json.loads((folder / "dataset-metadata.json").read_text())
            self.calls.append((who, list(args), meta["id"], sorted(p.name for p in folder.rglob("*") if p.is_file())))
            if self.fail_create:
                return _cp("", 1, "400 Client Error: Bad Request")
            self.made[meta["id"]] = who
            return _cp("Dataset version is being created. Please check progress at https://www.kaggle.com/datasets/"
                       + meta["id"])
        if args[:2] == ["datasets", "status"]:
            self.calls.append((who, list(args)))
            return _cp("ready")
        return super().__call__(args, timeout, env)


def test_a_private_dataset_of_another_account_is_copied_to_the_pushing_one(ws, accounts, monkeypatch, capsys):
    k = _kernel(ws)
    meta = json.loads((k / "kernel-metadata.json").read_text())
    meta["dataset_sources"] = ["accb/feats", "someone/public-ds"]
    (k / "kernel-metadata.json").write_text(json.dumps(meta))
    fake = WithDatasets(accounts)
    monkeypatch.setattr(G, "run_kaggle", fake)
    rec = G.push(ws, k, account="main")
    assert rec["kernel"] == "acca/train-a"
    # downloaded as its owner, uploaded as main: private (no --public), with its folders, the same slug
    down = [w for w, a, *_ in fake.calls if a[:2] == ["datasets", "download"]]
    up = [c for c in fake.calls if c[1][:2] == ["datasets", "create"]]
    assert down == ["accb"] and len(up) == 1
    who, args, ref, files = up[0]
    assert who == "acca" and ref == "acca/feats" and "--public" not in args and "--dir-mode" in args
    assert files == ["dataset-metadata.json", "more.npy", "train_feats.parquet"]
    assert [w for w, a, *_ in fake.calls if a[:2] == ["datasets", "status"]] == ["acca"]  # ready before the push
    sources = json.loads((k / "kernel-metadata.json").read_text())["dataset_sources"]
    assert sources == ["acca/feats", "someone/public-ds"]  # only the registered account's dataset is copied
    assert "copied accb/feats to acca/feats (private)" in capsys.readouterr().out
    assert not list((ws / ".kaggle-gm").glob("dataset-copies/*/*"))  # the download is not kept
    # pushed again unchanged: the copy is reused; once the source changes, a new version of the copy is made
    meta["dataset_sources"] = ["accb/feats"]
    (k / "kernel-metadata.json").write_text(json.dumps({**meta, "id": "accb/train-a"}))
    fake.calls.clear()
    G.push(ws, k, account="main")
    assert not any(c[1][:2] in (["datasets", "download"], ["datasets", "create"]) for c in fake.calls)
    fake.updated = "2026-11-25T08:00:00"
    (k / "kernel-metadata.json").write_text(json.dumps({**meta, "id": "accb/train-a"}))
    fake.calls.clear()
    G.push(ws, k, account="main")
    assert [c[2] for c in fake.calls if c[1][:2] == ["datasets", "version"]] == ["acca/feats"]


def test_a_failed_copy_stops_the_push(ws, accounts, monkeypatch):
    k = _kernel(ws)
    meta = json.loads((k / "kernel-metadata.json").read_text())
    (k / "kernel-metadata.json").write_text(json.dumps({**meta, "dataset_sources": ["accb/feats"]}))
    fake = WithDatasets(accounts, fail_create=True)
    monkeypatch.setattr(G, "run_kaggle", fake)
    with pytest.raises(SystemExit, match="could not copy accb/feats to main"):
        G.push(ws, k, account="main")
    assert not any(a[:2] == ["kernels", "push"] for _, a, *_ in fake.calls)
    assert json.loads((k / "kernel-metadata.json").read_text())["dataset_sources"] == ["accb/feats"]
