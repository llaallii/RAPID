## Title & Status
- Title: RAPID v2 - Project Context
- Status: Design/Planning only - no dataset collected, no training performed

## Purpose & Scope
- Framework for training autonomous multirotors via inverse RL and imitation learning with emphasis on sim-to-real robustness.
- Targets high-speed navigation in complex 3D indoor/outdoor scenes using expert trajectories generated in Isaac Sim and later a small fraction of real flights.
- Focuses on reproducible data generation, structured logging, and quality gates before any policy optimization.

## System Architecture (Planned)
### Simulation Environment (Isaac Sim)
- Isaac Sim plus Replicator pipelines generate 2k-5k procedural maps across 10 scene families (Office, Warehouse, Forest, Urban, Cave, Maze, Mine/Tunnel, Industrial Plant, Ruins, Jungle) grouped into Easy/Medium/Hard curricula.
- Geometry via Poisson-disk obstacle placement with per-object scale 0.7-1.5x, random rotations, and deterministic seeding for replay.
- Domain randomization per episode: lighting (dawn/noon/dusk/night), fog/rain ranges, PBR material swaps, Dryden wind up to 5 m/s, and controller gain perturbations (+/-15%).
- Physics-based Oak-D Pro depth sensor emulation at 640x480 @20 Hz (downsampled to 64x64 or 128x128), optional stereo RGB, IMU @200 Hz, and noise models (distance-dependent dropout, rolling-shutter latency ~30 ms).
- Controller-in-the-loop headless runs with geometric controller (50 Hz) and first-order rotor dynamics (~0.02 s time constant) ensure trajectories respect dynamics.

### Expert Policy Stack
- Fast-Planner mapping builds 0.1-0.2 m ESDF grids from depth plus VIO, exposing distance gradients for kinodynamic search.
- Global planner triggers on invalidation, local replanning executes at 10 Hz with 1.0-2.0 s receding horizon to adapt to newly observed obstacles.
- Actions are 10-20 cylindrical waypoints sampled at 0.1 s increments; feasibility checks enforce vmax=8 m/s, amax=10 m/s^2, jmax=30 m/s^3, thrust<=35 N, body-rate<=4 rad/s.
- Expert outputs include commanded waypoints and logged controller tracking errors to capture realistic execution.

### Data Logging & Storage
- Episode logging targets 20 Hz synchronized depth, odometry, and control streams; actions aligned to planner updates (0.1 s cadence).
- State tuple per step: stacked depth window (up to 35 frames), body-frame velocity, attitude quaternion, relative goal vector, motion history buffer, and rolling ESDF slice (~0.2 m voxels, 100x100x30 volume).
- Metadata: scene family, difficulty score, seed, terminal reason, QC flags, controller gains.
- Sharded HDF5 storage (<=10k episodes or ~2 GB per shard) with LZ4 compression, deterministic naming by scene family and difficulty; manifest tracks shard offsets for random access.

### Quality Control & Validation
- Step-level flags: feasible_ok (dynamics limits), sensor_ok (depth dropout<20%), temporal_ok (depth stack complete), tracking_ok (pos err<0.5 m, vel err<1 m/s), esdf_ok (map updates succeed), collision_ok (clearance>=0.3 m).
- Episode gates: terminal state reached (goal<1 m, collision<0.3 m, timeout at 60 s, emergency stop), min length 5 s (~100 steps @20 Hz), valid timestep ratio>=0.95, consistent metadata.
- QC pipeline outputs rejection breakdowns, flag histograms, and alerts when flag pass-rates drop below 95% or terminal mix shifts.
- Split management prevents leakage between train/val and enforces diversity quotas (target ~10% failure episodes).

### Training (Planned)
- Planned IL/IRL stage uses LS-IQ-style objectives with absorbing rewards {goal:0, timeout:-0.5, collision/emergency:-2} and discount gamma=0.99.
- Data loaders stream HDF5 shards with curriculum-aware sampling and optional failure weighting.
- Model stubs: vision encoders for depth stacks, motion/ESDF heads, waypoint policy decoder; evaluation harness replays policies in Isaac Sim.
- Metrics planned: success/collision rates, tracking error, sim-vs-real gap indicators once data exists.

### Real-Data Integration (Planned)
- Adapters will normalize teleoperated, motion-capture, and outdoor flight logs into the episode schema (depth, IMU, commands, metadata).
- Training batches will sample 1%-5% real trajectories as difficulty increases, with loss weighting w_real ~ 2.0 to emphasize rare real data.
- Deployment scripts envisioned for hardware rollouts, logging real vs sim comparisons, and updating QC dashboards with field data.

## Key Design Details
- Episode loop: sensor sync at 20 Hz, depth-to-pointcloud/ESDF update, planner replans every 0.1 s, controller runs at 50 Hz, actions executed via commanded waypoints.
- Episode duration targets 20-60 s, capturing full approaches including failures; terminal labeling drives absorbing rewards.
- State stacking: 35-frame depth windows (~0.5-1.0 s history) plus motion history buffer (5-10 timesteps) and rolling ESDF slice for spatial memory.
- Dataset scale goals: 50k-250k episodes (40M-200M step samples), overall storage 150-750 GB with shard manifests and global indices.
- Validation scripts expected to generate Markdown/JSON QC reports and TensorBoard metrics for flag rates and terminal distribution.

## Deliverables (Planned Code/Artifacts)
- `env_builder.py`, `scene_configs/`, `randomization_utils.py`
- `expert_policy.py`, `planner_interface/fast_planner.py`, `replanning_loop.py`
- `episode_recorder.py`, `terminal.py`, `message_bus/`
- `dataset_builder.py`, `hdf5_writer.py`, `manifest_index.csv`
- `qc_checks.py`, `qc_report.py`, `qc_dashboard.ipynb`
- `dataloaders/hdf5_loader.py`, `models/`, `losses/ls_iq.py`, `train.py`, `eval_rollouts.py`
- `real_data_adapter.py`, `mixed_sampler.py`, `deploy/flight_runner.py`, `sim2real_report.py`

## Non-Goals (for now)
- Publishing benchmark results or sim-to-real metrics before pipeline validation.
- Designing bespoke reward shaping beyond the planned IRL/IL formulations.
- Building photo-realistic sensors without noise; focus remains on physics-based realism with tunable perturbations.
- Supporting non-multirotor platforms or outdoor GPS-denied autonomy beyond defined scene families.

## Open Questions
- Confirm final logging cadence (section 3.7 cites 10 Hz, later tables use 20 Hz); need single source of truth.
- Decide configuration schema format (YAML vs JSON vs Python dataclasses) for environment, planner, and QC parameters.
- Define Fast-Planner API surface and ROS 2 message contracts for Isaac Sim integration.
- Clarify shard rollover policy (episode count vs byte size) and manifest metadata fields.
- Establish randomization seeding and reproducibility strategy across Isaac Sim, planner, and controller components.
