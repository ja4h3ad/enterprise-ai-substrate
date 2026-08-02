# Domain Docs

This is a single-context repository. Engineering skills should consume its domain documentation as follows.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- Read applicable ADRs under `docs/adr/` when they exist.
- If these files do not exist, proceed without flagging their absence. Domain-modeling workflows create them when terminology or decisions need to be recorded.

## Vocabulary

Use terms as defined in `CONTEXT.md` in issue titles, specifications, tests, implementation plans, and code. Do not drift to synonyms that the glossary explicitly avoids.

If a needed concept is absent, reconsider whether the term belongs or note the gap for domain modeling.

## Architectural decisions

Surface conflicts with existing ADRs explicitly rather than silently overriding them.

The expected single-context layout is:

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```
