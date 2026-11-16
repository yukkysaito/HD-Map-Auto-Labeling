# Agents: verification, improvement loops, benchmark

The labelling tools (`hdlabel/steps/`) are the agents' tools. AI agents (Claude, ChatGPT, ... through
`hdlabel/agents/llm.py`) verify their output, find what is systematically wrong, research better methods, implement
them as new tool variants and adopt them only when a protected benchmark shows they are better. A person accepts the
final result (`debug/audit.html`) and owns the benchmark.

## Roles and models (`agents` in `configs/default.yaml`)
| role | does | default |
|---|---|---|
| verifier | judges each evidence item (`prompts/verifier.md`): ok / drop / off / unsure | claude_cli sonnet |
| confirmer | must also call an item wrong before it is deleted (another vendor) | codex_cli |
| adjudicator | extra judge when the benchmark reference is built | claude_cli sonnet |
| researcher | finds papers / OSS / models for a problem (web search; `prompts/researcher.md`) | claude_cli opus |
| engineer | implements a candidate as a tool variant in a git worktree (`prompts/engineer.md`) | claude_cli opus |
| default | anything else | claude_cli sonnet |

A role is `{provider, model, timeout}`. Providers: `claude_cli`, `codex_cli`, `openai`, `anthropic`, `replay` (tests);
CLI paths in `agents.bin`. Calls are cached (a re-run costs nothing and repeats the answers), logged with their cost
in `output/<name>/agent/ledger.jsonl`, and stop at `agents.budget_usd`. Change a model by editing one line, then
measure it: `./autolabel bench verifier <config> --role verifier` (accuracy on the locked reference).

## Data loop (inside `./autolabel run`)
label -> export -> debug (evidence packs: `hdlabel/evidence.py`) -> **verify** -> deletions applied (label / export /
debug again) -> verify again, up to `verification.max_rounds`. Per scene `debug/scenes/<scene>/verification.json`:
- state: `accepted` | `needs_revision` (deletions not applied yet) | `rejected` (systematic: >= 5 of 10 spot checks
  wrong, curbs off the LiDAR steps) | `uncertain` (verifier unavailable, too many unsure items)
- issues: `{kind, at, type (false_positive | offset | missed | uncertain | lidar_conflict), evidence, source, action}`
The state rules are code (`verify.scene_state`), not model opinion; LiDAR checks (`geom_check.py`) are combined with
the visual judgement. `dataset/dataset.json` lists only accepted scenes (others under `excluded`) and the `lineage`
of the data (commit, tool variants + code hashes, agent models, prompt versions, benchmark lock). The person sees
everything flagged on `debug/audit.html` (docs/ANNOTATION.md).

## Protected benchmark (`hdlabel/agents/bench.py`, `bench/<name>/`, local)
```bash
./autolabel bench build <config> --by person:NAME   # reference = majority of independent judges per reviewed element
./autolabel bench lock <config> --by person:NAME    # sha256 of reference, evidence, criteria and evaluator
./autolabel bench status <config>                   # contents + each judge's accuracy (leave-one-out)
./autolabel bench verifier <config> [--role R]      # accuracy of the production verifier / one role
```
Every evaluation checks the lock. A changed reference, criteria (`hdlabel/agents/criteria.yaml`) or evaluator stops
it; only a person rebuilds / re-locks. Scenes are split dev / holdout; the gate needs no regression on both.
Metrics are computed on RAW tool output (no review fixes): recall of reference-ok elements, share of reference-wrong
elements still produced (fp), LiDAR step share of all curbs (needs no reference), lengths.

## Tool loop
```bash
./autolabel tools <config>                          # registry: tools, active variants, code versions, history
./autolabel experiment <config> --id ID --variant TOOL=NAME [--overlay JSON] [--code WORKTREE] --target curb.fp [--adopt]
./autolabel impact <config> [--variant TOOL=NAME]   # which steps / scenes a change would redo (dry run)
./autolabel tools <config> rollback TOOL            # previous active variant
```
`tools/registry.yaml` describes each tool (purpose, interface, algorithm / model, good / bad conditions, cost,
license, dependencies) and its variants (config overlays). An experiment runs the candidate in
`output/<name>/experiments/<id>/` (drive data linked read-only, scene data copied, unchanged steps skipped by their
stamps), measures it with the main checkout's evaluator and applies the gate; `--adopt` makes the variant active and
records the evidence. The next `run` redoes exactly the affected steps (stamps: arguments + code hash).

## Orchestrator (`./autolabel evolve <config>`)
One pass picks what is due: unverified scenes -> data loop; a problem with >= `evolve.min_cases` issues and fewer
than `evolve.max_attempts` failed attempts -> research -> engineer (worktree of HEAD; protected paths are checked) ->
experiment + gate -> merge + adopt, or a recorded failure; new accepted data -> model loop; otherwise a technology
scan of every tool. `--dry` shows the state and the ranked problems, `--scan` forces a scan, `--loop N` repeats.
Periodic runs: e.g. `crontab -e` -> `0 3 * * 1 cd /path/to/repo && ./autolabel evolve my_area.yaml --scan`.

## Model loop (`./autolabel train <config>`)
Trains the small multi-camera BEV network (`hdlabel/eval/train_smoke.py`) on the accepted scenes and analyses it:
weak classes (IoU not above the prior), weak val scenes, and places where the model confidently disagrees with the
labels (`debug/model_suspects.json`). Those places become evidence items / map tiles of the next debug step, so the
verifier decides them from the sensor data; a model prediction never becomes a label.

## Knowledge (`./autolabel kb search|list|show|add`, `knowledge/kb.sqlite`, local)
Problems, research results, attempts (with the gate result), evaluations, verifier accuracies and training analyses,
full-text searchable. Agents get the matching records in their prompts; nothing in it is final (records can be
superseded).
