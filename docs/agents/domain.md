# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

Single context. The terms `GLOSSARY.md` links to are owned by [rails49/occupancy](https://github.com/rails49/occupancy) and [rails49/r49](https://github.com/rails49/r49), whose docs own their terms and decisions. What each kind of doc holds, and how ADRs change, is set by org [ADR-0021](https://github.com/rails49/.github/blob/main/docs/adr/0021-each-fact-has-one-owner-and-the-glossary-is-glossary-md.md).

## Before exploring, read these

- **`GLOSSARY.md`**, for the vocabulary.
- **`docs/adr/`**: the ADRs that touch the area you're about to work in.

## File structure

```
/
├── GLOSSARY.md          ← glossary
├── CLAUDE.md            ← pointers, conventions and traps
└── docs/
    ├── adr/             ← decisions
    ├── agents/          ← these files
    └── research/        ← dated research notes
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
