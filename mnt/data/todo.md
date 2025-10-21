## Completed
- None to date.

## Phase 0 - Layout Migration Kickoff
**Objectives**
- Align the existing scaffold with the planned `rapid_v2` directory structure and prep for Isaac Sim integration.

**Tasks**
- [x] Document a mapping between current modules (`sim/toy_world.py`, `data/record.py`, etc.) and their eventual homes in the roadmap (`env_builder`, `dataset_builder`, ...); capture decisions in `docs/`.
- [x] Create the new top-level package skeleton (`rapid_v2/...`) with placeholder modules (`config/__init__.py`, `env_builder.py`, `randomization_utils.py`, `tools/smoke_env.py`) and TODOs referencing the plan.
- [x] Wire the existing toy simulator through the new `env_builder` and `randomization_utils` so the CLI continues to run while Isaac Sim work is in progress; add temporary adapters/tests.
- [x] Update packaging (`pyproject.toml`, `README.md`) to surface both the legacy `rapid_irl_nav` CLI and the emerging `rapid_v2` structure, including migration notes.
- [x] Set up a short-term validation script (`tests/test_layout_bridge.py`) that exercises the new builder pipeline with the toy backend to guard against regressions.

**Dependencies**
- Existing `rapid_irl_nav` functionality must stay green (collect/train CLI commands, toy-world smoke tests).
- No Isaac Sim requirement yet; just ensure hooks/interfaces are ready.

**Outputs**
- Documented module mapping, new directory skeleton matching the roadmap, adapters keeping current workflows alive, updated project docs, and a bridge test that validates the interim wiring.

## Phase 1 - Simulation Environment Setup
**Objectives**
- Stand up Isaac Sim procedural scene generation with deterministic seeds, baseline drone model, and domain randomization knobs.
- Deliver reusable configuration schemas for environments, vehicles, and randomization parameters.

**Tasks**
- [ ] Draft `config/env.yml`, `config/vehicle.yml`, and `config/randomization.yml` schemas covering scene families, physical constants, and sensor specs.
- [ ] Implement `env_builder.py` to load configs, spawn Isaac Sim assets, and attach physics-based sensors and controllers.
- [ ] Create procedural generation scripts (`scenes/indoor_generator.py`, `scenes/outdoor_generator.py`) aligning with Table 3 parameters.
- [ ] Add `randomization_utils.py` supporting lighting, weather, material sampling, Dryden wind injection, and controller gain perturbations.
- [ ] Build `tools/smoke_env.py` to launch headless scenes and export basic frames and ESDF snapshots for validation.

**Dependencies**
- Isaac Sim and Replicator Python APIs.
- Finalized configuration schema decisions.

**Outputs**
- `env_builder.py`, config templates, randomization utilities, and smoke-test tooling that produce seeded Isaac Sim scenes with expected sensor streams.

## Phase 2 - Expert Trajectory & Data Logging
**Objectives**
- Integrate Fast-Planner with the simulated environment, including ESDF updates, replanning loop, and controller command streaming.
- Record expert trajectories with terminal annotations and structured step metadata.

**Tasks**
- [ ] Implement `planner_interface/fast_planner.py` bindings (ROS 2 or direct C++ bridge) exposing global and local planning calls.
- [ ] Add incremental ESDF publisher (`mapping/esdf_builder.py`) consuming depth and VIO feeds from Phase 1.
- [ ] Create `expert_policy.py` managing replanning at 10 Hz with 1-2 s horizons and enforcing command feasibility gates.
- [ ] Develop `episode_recorder.py` plus `terminal.py` to log 20 Hz states and actions, terminal reasons, and planner/controller diagnostics.
- [ ] Stand up message layer (`message_bus/`) for synchronized sensor, planner, and controller data, including unit tests with mocked streams.

**Dependencies**
- Phase 1 environment builders and sensor streams.
- Fast-Planner binaries or libraries and interface documentation.

**Outputs**
- Expert rollout loop producing stepwise logs with QC-ready metadata and archived planner/controller telemetry.

## Phase 3 - Dataset Generation & QC
**Objectives**
- Automate large-scale episode collection, sharding, and QC evaluation and reporting.
- Ensure datasets capture both successes and controlled failures with rigorous flagging.

**Tasks**
- [ ] Implement `dataset_builder.py` orchestrating batch episode runs, seeding, and failure-rate targets.
- [ ] Build `storage/hdf5_writer.py` handling shard rollover (<=10k episodes or 2 GB) and manifest generation (`manifest_index.csv`).
- [ ] Add step-level QC computations in `qc_checks.py` (feasible_ok, sensor_ok, temporal_ok, tracking_ok, esdf_ok, collision_ok) with thresholds from Section 9.
- [ ] Create `qc_report.py` to emit Markdown and JSON summaries, histograms, and alert rules when pass rates drop below thresholds.
- [ ] Run mini integration test (`tests/test_dataset_pipeline.py`) generating >=10 episodes to validate sharding plus QC flow.

**Dependencies**
- Phase 2 logging format and metadata schema.
- Libraries: `h5py`, `lz4`, and numerical stack.

**Outputs**
- Sample shard plus manifest, QC reports in `reports/qc_report.md`, and scripts to reproduce batch collection.

## Phase 4 - Training Pipeline Implementation
**Objectives**
- Establish IL/IRL training loop consuming the sharded dataset with curriculum-aware sampling and absorbing reward handling.
- Provide evaluation harness for simulated rollouts and metric tracking.

**Tasks**
- [ ] Implement `dataloaders/hdf5_loader.py` with curriculum filters, failure sampling ratios, and on-the-fly depth preprocessing.
- [ ] Scaffold model modules (`models/vision_encoder.py`, `models/policy_head.py`) matching state and action definitions.
- [ ] Code `losses/ls_iq.py` (or equivalent) incorporating absorbing rewards {0, -0.5, -2} and gamma 0.99.
- [ ] Assemble `train.py` with config-driven experiment management, checkpointing, and QC flag monitoring hooks.
- [ ] Build `eval/eval_rollouts.py` to replay policies in Isaac Sim scenes and summarize success and collision metrics.

**Dependencies**
- Final dataset schema and QC outputs from Phase 3.
- PyTorch and experiment tracking stack.

**Outputs**
- Runnable training and evaluation suite with baseline configs and logging ready for future datasets.

## Phase 5 - Real Data Integration & Sim-to-Real Testing
**Objectives**
- Ingest teleoperated and motion-capture flight logs into the canonical format and mix them into training.
- Validate trained policies on hardware and quantify the sim-to-real gap.

**Tasks**
- [ ] Implement `real_data_adapter.py` to parse field log formats (depth, IMU, pose, commands) and emit standardized HDF5 episodes with QC flags.
- [ ] Add `samplers/mixed_sampler.py` enforcing staged real-data ratios (1%, 3%, 5%) and weighting (w_real ~ 2.0).
- [ ] Develop `deploy/flight_runner.py` for on-vehicle policy execution, telemetry capture, and safety monitors.
- [ ] Create `eval/sim2real_report.py` comparing sim versus real metrics (success rate, collision rate, tracking error) and generating plots.
- [ ] Document hardware validation checklist (`docs/sim2real_validation.md`) aligning with safety and logging requirements.

**Dependencies**
- Field data interfaces (teleop, motion capture, outdoor runs).
- Trained models from Phase 4 and hardware communication stacks.

**Outputs**
- Mixed dataset sampler, deployment tooling, and sim-to-real evaluation reports with reproducible pipelines.

## Repo Layout (proposed)
```text
rapid_v2/
  config/
    env.yml
    vehicle.yml
    randomization.yml
  scenes/
    indoor_generator.py
    outdoor_generator.py
  env_builder.py
  randomization_utils.py
  planner_interface/
    fast_planner.py
  mapping/
    esdf_builder.py
  expert_policy.py
  replanning_loop.py
  episode_recorder.py
  terminal.py
  message_bus/
  dataset_builder.py
  storage/
    hdf5_writer.py
  manifest_index.csv
  qc_checks.py
  qc_report.py
  reports/
    qc_report.md
  dataloaders/
    hdf5_loader.py
  models/
  losses/
    ls_iq.py
  train.py
  eval/
    eval_rollouts.py
    sim2real_report.py
  samplers/
    mixed_sampler.py
  real_data_adapter.py
  deploy/
    flight_runner.py
  docs/
    sim2real_validation.md
  tools/
    smoke_env.py
  tests/
    test_dataset_pipeline.py
```

## Acceptance Criteria (minimal)
- Phase 1: `tools/smoke_env.py` generates at least one deterministic scene per family with valid depth frames and ESDF export.
- Phase 2: Logged expert episodes show 10 Hz replanning, 20 Hz state logs, and terminal annotations for goal, collision, and timeout cases.
- Phase 3: Pipeline produces at least one shard with manifest plus QC report covering flag pass rates and rejection breakdown.
- Phase 4: `train.py` consumes a sample shard, runs one epoch end-to-end, and reports baseline success and collision metrics from `eval_rollouts.py`.
- Phase 5: Mixed sampler yields requested real-data proportions and `sim2real_report.py` plots sim versus real metrics from a pilot hardware run.
