# CLAUDE.md

The classifier side of [rails49/occupancy](https://github.com/rails49/occupancy):
patch export, training, the `rails49-classifier` Python package, and the
`@rails49/classifier` browser package. The plan is #12.

The classifier sometimes reports a car that is not there, and sometimes misses
one. Present nothing here as a safety interlock.

gate: bin/test.sh

## Where things are

What each kind of doc holds: org [ADR-0025](https://github.com/rails49/.github/blob/main/docs/adr/0025-each-fact-has-one-owner-and-an-adr-is-written-after-its-decision-has-run.md).

- Terms: `GLOSSARY.md`. Decisions: `docs/adr/`. Research notes: `docs/research/`.
- The `.r49` format is `rails49-r49`'s, pinned to one version (org [ADR-0022](https://github.com/rails49/.github/blob/main/docs/adr/0022-the-r49-format-is-r49s-and-each-consumer-pins-it.md)).

## Commands

- `bin/test.sh`: Ruff, Black, Pyright, pytest and `bin/check-wheel.sh`. `bin/install-hooks.sh` runs it before each push. The sandbox blocks OpenVINO's JIT and the process aborts, so run it, and any test that scores on OpenVINO, outside the sandbox.
- `uv run export ~/rails49/r49/archives/iot49/*.r49 --dpt 16`: writes `cache/index.csv`,
  `cache/counts.json` and `cache/dpt16/`. `src/classifier/export.py` describes the output.
- `uv run train`: trains and scores MobileNetV3-Small in 3 folds from `cache/` and writes
  `runs/baseline/`. `src/classifier/train.py` describes the output. It uses MPS when
  it can, and the sandbox blocks MPS, so run it outside the sandbox.
- `pnpm -C bench bench <threads> runs/baseline/*.onnx`: times a batch of 50 patches in
  onnxruntime-web WASM under Node.

## Agent skills

The rules every rails49 repository shares are the organisation's
[`docs/agents/`](https://github.com/rails49/.github/tree/main/docs/agents):
`issue-tracker.md`, `triage-labels.md` and `domain.md`.

- Issue tracker: GitHub Issues on rails49/classifier, via the `gh` CLI.
- Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`,
  `ready-for-human`, `wontfix`.
- Domain docs: `GLOSSARY.md` and `docs/adr/`. The terms `GLOSSARY.md` links to
  are owned by rails49/occupancy and rails49/r49.
