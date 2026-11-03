# CLAUDE.md

The classifier side of [rails49/occupancy](https://github.com/rails49/occupancy):
patch export, training, and the `@rails49/classifier` browser package. The
plan is rails49/occupancy#272.

The classifier sometimes reports a car that is not there, and sometimes misses
one. Present nothing here as a safety interlock.

## Where things are

What each kind of doc holds: org [ADR-0021](https://github.com/rails49/.github/blob/main/docs/adr/0021-each-fact-has-one-owner-and-the-glossary-is-glossary-md.md). See `docs/agents/domain.md`.

- Terms: `GLOSSARY.md`. Decisions: `docs/adr/`. Research notes: `docs/research/`.
- The `.r49` format is `rails49-r49`'s, pinned to one version (org [ADR-0022](https://github.com/rails49/.github/blob/main/docs/adr/0022-the-r49-format-is-r49s-and-each-consumer-pins-it.md)).

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `rails49/classifier`, driven via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, used verbatim as label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single context: `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.
