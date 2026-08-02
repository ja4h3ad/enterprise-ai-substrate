# Issue tracker: GitHub

Issues and PRDs for this repository live as GitHub issues. Use the `gh` CLI for all operations and infer the repository from `git remote -v`.

## Conventions

- Create specs and tickets as GitHub issues.
- Read an issue and its discussion with `gh issue view <number> --comments`.
- Apply the repository's canonical triage labels when a workflow requires them.
- Do not treat pull requests as a feature-request or triage surface.
- Publish tracer-bullet tickets in dependency order so blockers can reference existing issues.
- Represent blocking edges with GitHub's native issue dependencies when available.
- If native dependencies are unavailable, include a `Blocked by: #<number>` reference in the issue body.
- Do not close or modify a parent issue when publishing child tickets.

## Skill instructions

When a skill says to publish to the issue tracker, create a GitHub issue. When it says to fetch a ticket, read the complete issue body, labels, and comments.
