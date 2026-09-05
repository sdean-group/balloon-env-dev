# balloon-env-dev

Research code for **Unbounded Wind Fields for Stratospheric Balloon Simulation** (Sarah Dean lab, Cornell; Shaurya Sen & Rohan Shankar). Public at github.com/sdean-group/balloon-env-dev.

## Context lives elsewhere — read it

Project objectives, decisions, and the current sprint live in Shaurya's planning workspace, which is the single source of truth:

- `/Users/shauryasen/Dev/shaurya-hub/research/PROJECT.md` — project overview and decisions
- `/Users/shauryasen/Dev/shaurya-hub/research/SPRINT.md` — the current 2-3 week sprint and milestones
- `/Users/shauryasen/Dev/shaurya-hub/research/LOG.md` — running log of results and decisions

At the start of a session that involves planning or prioritizing (not pure mechanical edits), read PROJECT.md and SPRINT.md. When a session produces a result or decision that will matter later (a finding, a go/no-go, a changed direction), append it to LOG.md there — do not let it live only in chat.

## Practical notes

- GPU work runs on the Kahan cluster (SLURM + Podman): see `/Users/shauryasen/Dev/shaurya-hub/research/docs/kahan-reference.md`.
- The public README currently describes only the simplified grid environment; it is slated for an update to cover the diffusion/InfiniteDiffusion work before being linked from Shaurya's resume.
