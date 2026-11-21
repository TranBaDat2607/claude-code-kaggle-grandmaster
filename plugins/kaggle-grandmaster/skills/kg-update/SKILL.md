---
name: kg-update
description: "Updates the kaggle-grandmaster plugin to its latest release from inside the conversation. Use when the user wants the newest version."
allowed-tools:
  - Read
  - Glob
  - Grep
  - 'Bash(kgkit update-check *)'
disable-model-invocation: true
---

# /kg-update

Run kgkit with the Bash tool as `kgkit <command>`: the plugin puts it on PATH, also on Windows (Git Bash),
and this command pre-approves the kgkit calls it makes. Older text that says `python -m kgkit` means the same
command. The update itself (`claude plugin update`) still asks, every time: it changes what the user has installed.

1. **Check**: `kgkit update-check`. It reads the latest release from
   GitHub and this session's install from `claude plugin list --json`, and prints what to do. Show the user the
   versions and the "What changed" lines.
   - *up to date*: say so and stop.
   - *development checkout*, *a copy made by kgkit vendor*, *no installed plugin points to …*, or *the claude CLI
     was not found*: relay what it says (`git pull`, the /plugin menu, …) and stop. Do not guess a command.
   - *already installed, waiting for /reload-plugins*: go to step 3.
2. **Update**: run the exact `Run:` command it printed (`claude plugin update kaggle-grandmaster@… --scope …`) with
   the Bash tool (PowerShell on Windows), from the folder it names when it names one. The user approves it.
   If it fails, show its output and stop. Then run `update-check` again: it should now say the new version is
   installed and waits for /reload-plugins.
3. **Reload**: ask the user to type `/reload-plugins` (a built-in command; Claude cannot run it). It loads the new
   version into this session; the conversation stays. Until then the session runs the old version.
4. **After the reload**, on the user's next message: `kgkit` is the new version's (the reload puts its `bin/` on
   PATH). Plugin paths earlier in this conversation (past commands) are the old version's: do not reuse them. Run
   `kgkit update-check` to confirm it says up to date.
5. **Vendored copies**: when `update-check` lists older kgkit copies in the workspace (made by `kgkit vendor`, e.g.
   `src/kgkit` or a Kaggle dataset folder), offer to refresh them with the commands it prints, after the reload. A
   `gpu build` or an offline kernel ships the copy, not the plugin. Ask before refreshing a dataset folder that is
   uploaded to Kaggle: uploading a new version is a separate step the user approves.
