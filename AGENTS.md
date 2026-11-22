# Instructions for coding agents

This file is for AI coding agents (Codex, Claude Code, ...) working in this repository. People start with
`README.md`.

## Read first
1. `README.md`: what the pipeline does, quick start.
2. `docs/AGENTS.md`: the agent layer (verification, benchmark, tool registry, knowledge store, improvement loops).
3. `spec/labels.yaml`: what the labels are (the person's; see "Protected" below).
4. `docs/DATASET.md`: the training data format. `docs/ANNOTATION.md`: acceptance. `docs/MAINTENANCE.md`: code map,
   conventions, baselines.
5. If present (local, not versioned): `configs/local/OPERATIONS.md` for this machine's data locations, resource
   rules, external-AI permissions and known problems; goal files in `configs/local/`. Without it, do not guess paths,
   start long GPU jobs or send data to external services; ask the person.
6. The knowledge store before starting a task: `./autolabel kb search "<words>"` (what was tried, measured, rejected).

## Commands
```bash
./autolabel run [config.yaml] [--output DIR]     # bags -> dataset + debug; finished steps are skipped
./autolabel status [config.yaml]                 # what ran, what failed (and its log)
./autolabel validate output/<name>/dataset/dataset.json
./autolabel test                                 # pytest without a ROS PYTHONPATH (do not call pytest directly)
ruff check . && ruff format --check . && pyflakes hdlabel tests   # the CI lint
./autolabel kb search "..." | kb add --kind ... --title ... --text ...
python -m hdlabel.bevnet.train | .evaluate | .video   # reference BEV model (docs/DATASET.md)
```
Check exit codes directly; do not judge a test run through `| tail`.

## Rules
- No HD / vector map (Lanelet2, `/map/vector_map`) anywhere: not for labels, correction, verification or evaluation.
  A LiDAR point cloud map, localization, cameras, LiDAR and calibration are allowed.
- Never use one vehicle's camera mounting for another vehicle. Check that `/tf_static` really chains every camera to
  `base_link` (its content, not just the topic); if not, stop and ask for that vehicle's calibration.
- The model under development is an online model: its inputs are what the vehicle has at inference time (camera
  images, past frames, ego motion). Labels may use LiDAR, future frames and several drives.
- Never turn model predictions into labels without independent evidence from the sensor data.
- Unobserved cells are not background (class 0 / abstain, see `docs/DATASET.md`).
- Do not interfere with processes you did not start (GPUs and disks may be shared). Check free GPU memory before a
  GPU job and do not run two of your own GPU jobs that do not fit together.

## Protected (a person decides)
- `spec/labels.yaml`, `hdlabel/agents/criteria.yaml`, the locked benchmark (`bench/`), source bags, sign-offs.
  Propose changes with the reason and the impact; do not edit them. After the person changes the spec,
  `./autolabel spec <config> apply` aligns tools, tests and docs.
- Publishing: commit locally as needed; never push or release without the person's confirmation.

## Maintainer duties
You are the implementer, architect, maintainer and reviewer of this repository.
- Understand the existing code before changing it; extend it instead of re-implementing it under another name.
  Do not rewrite parts that work without a measured reason.
- Tool / model changes: in an isolated git worktree, adopted only after the benchmark gate and regression tests pass
  (commit the main checkout first: worktrees start from HEAD).
- Before each commit, review the whole diff: correctness, dead code, duplication, consistency with the spec, the data
  format and the docs, tests, private information, large files.
- A data format change updates `docs/DATASET.md` and its `format` number; a behaviour change updates the baselines
  in `docs/MAINTENANCE.md`. Keep docs minimal: no progress reports or design notes in the repository.
- Docs, code comments and commit messages are in English. No AI co-author or signature trailers in commits.

## Private data
This is a public Apache-2.0 repository. Site data, vehicle names, place names, map coordinates, local paths, bags,
checkpoints, review files and the knowledge store stay in the git-ignored `configs/local/`, `review/`, `bench/`,
`knowledge/` and `output/`. Local git hooks reject terms listed in `configs/local/private_terms.txt` when it exists.
Check licenses of new dependencies and models for Apache-2.0 compatibility.

## Pitfalls
- `pgrep` / `pkill` patterns also match your own shell command line: write them as `"[h]dlabel"`.
- Step outputs are reused by a stamp that includes the code hash of the step and the hdlabel modules it imports; a
  code change reruns the affected steps.
