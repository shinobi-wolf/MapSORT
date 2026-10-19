# MapSORT project instructions (shared by all workspaces)

Machines: **SO** = shinobi-owl (laptop, has a GPU and GitHub login), **SW** = shinobi-wolf (always-on Ubuntu PC, CPU only, no GitHub credentials, reached with `ssh shinobi-wolf` over Tailscale). Apply only the machine section matching `hostname` (`shinobi-owl` = SO, `shinobi-wolf` = SW).

- Don't read docs/.~lock.LocalizeSORTv2_Project_Tracker.xlsx# or docs/LocalizeSORTv2_Project_Tracker.xlsx unless specifically asked.
- Task tracking is the GitHub Project for this repo (project #5, "MapSORT", owner shinobi-wolf). Use `gh project item-list 5 --owner shinobi-wolf` (and `gh issue view <n>`) to find the next task. `gh` is authenticated on SO only.
- Only edit and execute/launch scripts when the user's chat message starts with the prefix "act:". On every machine, including SW work done over ssh or the mount. Otherwise just brainstorm; read-only commands are fine.
- Use conda env `nerfly` (SO and SW).
- Ignore docs/modules.yaml and docs/schema.yaml.
- Ignore the tests dir. Testing work is paused.
- Verbosity level: low. Keep docstrings, comments, and explanations minimal: one-line docstrings at most, comments only where the why is non-obvious, no restating what the code does.
- Two boxmots at the repo root: `boxmot/` is the vendored upstream master (259625c8), which upstream stripped of SAM2MOT on 2026-09-18 (commit 94e1cec7). `boxmot-v25.0.0/` is a git-ignored clone of the v25.0.0 tag, kept only for its SAM2MOT tracker (`boxmot/trackers/multimodal/sam2mot/`) as the paper baseline. Don't pip-install it; `import boxmot` must keep resolving to the vendored copy. It carries local patches, needed on every machine's clone, all in `boxmot/trackers/multimodal/sam2mot/tracker.py`: `propagated[(0, 2)]` / `propagated[(1, 3)]` -> `propagated[[0, 2]]` / `propagated[[1, 3]]` (v25.0.0 crashes on the first unmatched AABB track with a velocity), and the AABB minimum-size epsilon `+ 1e-4` -> `+ 1e-2` for `propagated[2]`, `propagated[3]`, `predicted[:, 2]`, `predicted[:, 3]` (in float32 at 4K coordinates `x + 1e-4 == x`, so a box collapses and fails validation).
- Run artifacts stay on the machine that made them; code and run manifests/metrics move via git. scp an artifact over when needed.
- Sync: SW commits (no push); from SO run `git pull --ff-only shinobi-wolf:ISU/sem3/ME_5770/assignments/MapSORT dev && git push`.

## SO only (shinobi-owl, laptop)

- On a shared/public machine (e.g. a university lab rig), `gh auth logout` before ending the session rather than leaving a persistent GitHub login.
- SW remote option: SW has bigger storage, so some dataset evaluations run there. Its project root is `~/ISU/sem3/ME_5770/assignments/MapSORT`.
- For SW-bound work from SO, edit SW's checkout through the sshfs mount at `~/IISc/LocalizeSORTv2-sw` (not persistent: `sshfs shinobi-wolf:ISU/sem3/ME_5770/assignments/MapSORT ~/IISc/LocalizeSORTv2-sw -o reconnect,ServerAliveInterval=15`). Run git, grep and jobs on SW over ssh, not through the mount.
- Non-interactive ssh to SW needs `source ~/anaconda3/etc/profile.d/conda.sh && conda activate nerfly`. Launch long jobs in tmux.
- GrapeMOTS lives only on SW; its `data/grapemots/raw` symlink doesn't resolve through the mount.

## SW only (shinobi-wolf)

- No GPU: run everything on CPU with the CPU torch build.
- The `nerfly` env is Python 3.10 and was not built by setup.sh, so pins differ from SO where py3.10 forces it (scipy 1.15.3). numpy is held at 1.26.4 despite boxmot/trackeval pip-check complaints; re-pin it after any reinstall of boxmot.
- Non-interactive shells need `source ~/anaconda3/etc/profile.d/conda.sh && conda activate nerfly`. Launch long jobs in tmux.
- No GitHub credentials, by choice: never `gh`/push. Commit here; SO pulls and pushes.
- Project root: `~/ISU/sem3/ME_5770/assignments/MapSORT`. GrapeMOTS raw: `~/ISU/sem3/ME_5770/assignments/data/grapemots` (`data/grapemots/raw` symlinks to it).
