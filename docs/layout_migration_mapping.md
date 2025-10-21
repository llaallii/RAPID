# Rapid Layout Migration – Phase 0 Mapping

This document captures how the existing `rapid_irl_nav` scaffold maps onto the planned `rapid_v2` layout. It records decisions agreed during the Layout Migration Kickoff so future phases can build on a shared vocabulary.

## Legacy → `rapid_v2` module mapping

| Legacy module | Planned destination | Notes / Actions |
| ------------- | ------------------- | --------------- |
| `rapid_irl_nav.cli.collect_expert` | `rapid_v2/tools/smoke_env.py` (short-term), later `rapid_v2/tools/collect_expert.py` | CLI entry point stays available via `rapid_irl_nav` while the new builder evolves. |
| `rapid_irl_nav.sim.toy_world` | `rapid_v2/env_builder.py` (`ToyWorldBackend` adapter) and `rapid_v2/randomization_utils.py` | Toy backend becomes a drop-in backend for the new builder to keep smoke tests green. |
| `rapid_irl_nav.sim.controller` | `rapid_v2/randomization_utils.py` (control perturbations) & `rapid_v2/tools/smoke_env.py` | Controller gains feed into randomisation helper during migration. |
| `rapid_irl_nav.sim.traj_utils` | `rapid_v2/env_builder.py` (trajectory helpers) | Limited subset wrapped by the new builder for toy backend rollouts. |
| `rapid_irl_nav.data.record` | `rapid_v2/dataset_builder.py` (bridge) | Writer is reused by the bridge test to ensure episode layout compatibility. |
| `rapid_irl_nav.data.schemas` | `rapid_v2/dataset_builder.py` & `rapid_v2/qc_checks.py` | Schema validation remains the source of truth for QC scaffolding. |
| `rapid_irl_nav.data.transforms` | `rapid_v2/dataloaders/hdf5_loader.py` | Normalisation utilities ported when training pipeline lands. |
| `rapid_irl_nav.irl` (policies, losses) | `rapid_v2/models/` & `rapid_v2/losses/` | No Phase 0 action; tracked so Phase 4 can reuse components. |
| `rapid_irl_nav.model` | `rapid_v2/models/` | Phase 4 milestone. |
| `rapid_irl_nav.utils.logging` | `rapid_v2/tools/`, `rapid_v2/dataset_builder.py` | Shared logging adapters to keep CLI UX consistent. |
| `rapid_irl_nav.utils.seeding` | `rapid_v2/randomization_utils.py` | Deterministic seeds across builders. |

## Directory decisions

- Keep `src/rapid_irl_nav` intact until Phase 3 completes to avoid breaking existing CLI workflows.
- Stage new code under `src/rapid_v2` with thin adapters during the migration window.
- Place all migration notes in `docs/` so future contributors can trace the rationale.

## Open questions

1. Should the toy backend remain in-tree once Isaac Sim integration lands, or move to `examples/`?
2. How will we expose Isaac Sim-only dependencies so users without Isaac Sim can still run the toy smoke test?

