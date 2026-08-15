# AGENTS.md

## Project Context

EEPY is an edge-first driver fatigue monitoring system intended to run alongside dedicated in-vehicle hardware.

During development, the laptop webcam, microphone, and speakers act as substitutes for the eventual camera, microphone, and speaker hardware.

Do not redesign EEPY as a conventional cloud-hosted webcam application unless explicitly requested.

Safety-critical detection and intervention must remain local and deterministic. AI-enhanced behaviour may use internet-dependent services, but loss of internet, OpenAI, speech recognition, or another external API must never prevent a local fatigue alert.

Before roadmap work, read `docs/roadmap.md`.

Only work on the roadmap step explicitly requested. Do not automatically continue to the next step.

---

## Working Rules

Before changing code:

1. Inspect the relevant existing implementation.
2. Explain the current behaviour and how the relevant code fits together.
3. Identify likely bugs, edge cases, regressions, or unsafe assumptions related to the requested task.
4. Propose the smallest sensible implementation for the requested scope.
5. Avoid modifying unrelated functionality.

When changing code:

1. Preserve existing behaviour unless the requested change intentionally replaces it.
2. Do not silently change fatigue thresholds, timing constants, calibration bounds, or other safety-related behaviour.
3. Explain any safety-related constant or behavioural change explicitly.
4. Prefer simple, readable code over unnecessary abstractions.
5. Avoid new dependencies unless they materially improve the implementation.
6. Add or update tests for deterministic behaviour introduced by the change.
7. Keep safety-critical monitoring independent from optional network/API functionality.
8. If an assumption is uncertain, state it clearly rather than presenting it as established fact.

After changing code:

1. Explain what changed.
2. Explain how the new runtime flow works.
3. List the important files, functions, and classes changed.
4. Explain important algorithms, formulas, state transitions, or data structures.
5. Explain every new library or important API used:
   - what it is
   - what it does
   - why it was used here
   - relevant tradeoffs or alternatives where useful
6. Explain failure and fallback behaviour.
7. Report tests added or changed.
8. Report commands run and whether they passed.
9. Identify remaining risks, limitations, and important untested cases.
10. State whether the current change is a good Git commit boundary.
11. If it is, suggest a concise Conventional Commit-style message.
12. Do not commit, push, merge, or delete branches unless explicitly asked.

---

## Explanation Standard

Code explanations should be detailed enough that the project owner can explain the implementation to another engineer or interviewer.

Do not only describe what a line of code does. Explain why the implementation exists and how it interacts with the rest of the system.

For non-obvious Python or library features, explain the underlying concept. For example, if using `collections.deque`, explain why a bounded deque is appropriate for a rolling history rather than only naming it.

For mathematical logic, include both:

- the formula or condition
- the intuition behind it

For concurrency or asynchronous behaviour, explain:

- what runs independently
- what state is shared
- how race conditions or blocking are avoided

---

## Safety Principles

When uncertain, prefer the safer fallback.

For passive calibration specifically:

- never assume startup measurements represent an awake driver
- do not calibrate from suspected fatigue
- reject uncertain samples rather than contaminating a personal baseline
- keep the global fatigue model active as a fallback
- calibration may adapt facial geometry but must not redefine unsafe eye-closure duration
- freeze or reject calibration when fatigue becomes plausible

For intervention behaviour:

- local alerts must occur before network-dependent AI behaviour
- OpenAI or internet availability must never be required to trigger the initial fatigue warning

---

## Git and Scope

Prefer small, coherent milestones over large multi-feature changes.

A good commit should represent one logical change that can be described clearly in one sentence.

Do not begin the next roadmap milestone merely because the current one is complete.

Current roadmap source: `docs/roadmap.md`.

---

## Documentation Maintenance

After making code changes, determine whether the change affects the documented system architecture, module responsibilities, data flow, or repository structure.

If it does, update `docs/architecture.md` as part of the same change.

Do not update `docs/architecture.md` for implementation details that do not materially change the architecture.