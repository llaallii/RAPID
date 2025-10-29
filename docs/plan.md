# RAPID v2 — Comprehensive Project Plan (Phases 1–11)

> Full end-to-end roadmap from environment setup to Inverse Reinforcement Learning (IRL), sim-to-real fine-tuning, and continuous data refinement.

---

## Phase 1 — Environment Setup (Isaac Lab Foundation)

### Objectives
- Set up Isaac Lab (Isaac Sim 5.0.0 via pip) and configure reproducible simulation environments.
- Prepare procedural scenes and ROS 2 integration.
- Establish Ubuntu 24.04 LTS development workflow with ROS 2 Jazzy.

### Key Actions
1. **Environment Setup:** ✅ COMPLETED
   - ✅ Verified Isaac Sim 5.0.0 installation in `env_isaaclab` conda environment.
   - ✅ Installed ROS 2 Jazzy on Ubuntu 24.04 with Isaac Sim ROS 2 bridge support.
   - ✅ Documented activation sequence and dependency versions in `config/env/`.

2. **Workspace Configuration:**
   - Using monorepo structure at current project root.
   - Organize modules: `src/sim/`, `src/planning/`, `src/control/`, `src/analysis/`.
   - Configure `config/env/` for environment manifests and sensor definitions.
   - Configure `config/ros2/` for ROS 2 bridge launch files and topic specifications.

3. **Scene Generation:** IN PROGRESS
   - ✅ Configured 10 scene families (Office, Warehouse, Forest, Urban, Cave, Maze, Mine, Shipyard, Ruins, Jungle).
   - Leverage Isaac Lab assets + NVIDIA Omniverse assets for procedural generation.
   - Implement custom scene randomization scripts extending Isaac Replicator.
   - Store scene generation seeds in `./data/raw/runtime/scenes.jsonl`.

4. **Sensor & ROS 2 Integration:** IN PROGRESS
   - ✅ Configured sensor specifications in `config/env/sensors.yaml`
   - Configure stereo depth cameras (20 Hz) using Isaac Lab sensor framework.
   - Set up IMU and odometry sensors with physics-based noise models.
   - Verify ROS 2 bridge topics: `/depth`, `/odom`, `/imu` with proper message types.
   - Test sensor data pipeline and logging to `./data/raw/runtime/`.

5. **Isaac Replicator Randomization:** PLANNED
   - Configure lighting randomization (HDR, intensity, color temperature).
   - Material/texture randomization for domain diversity.
   - Dynamic obstacle placement for curriculum complexity.

6. **Validation & Documentation:** ✅ COMPLETED
   - ✅ Created reproducible verification script `scripts/setup_sim.py`.
   - Create sensor validation script `scripts/test_sensors.py`.
   - ✅ Documented complete setup in `config/env/setup_instructions.md`.

### Deliverables
- ✅ Isaac Lab functional and reproducible (verified via `isaacsim` command).
- ✅ ROS 2 Jazzy installed with operational bridge to Isaac Sim on Ubuntu 24.04.
- Scene seeds stored under `./data/raw/runtime/scenes.jsonl`. (IN PROGRESS)
- ✅ Sensor configuration YAMLs committed: `config/env/sensors.yaml`.
- ✅ ROS 2 topic specifications: `config/ros2/bridge_topics.yaml`.
- ✅ Environment setup documentation: `config/env/setup_instructions.md`.
- ✅ Phase 1 completion checklist: `docs/phase1_checklist.md`.

---

## Phase 2 — Expert Planner Integration

### Objectives
Integrate Fast-Planner with kinodynamic trajectory generation.

### Components
- Mapping module builds ESDF map from depth + odometry.
- Global planner performs kinodynamic A* search and B-spline optimization.
- Local replanner (10 Hz) refines trajectory in-flight.
- Topological diversity via homotopy class enumeration.

### Deliverables
- Feasible trajectories verified (velocity ≤ 30 m/s, accel ≤ 10 m/s²).
- Planner nodes in ROS 2 operational.
- Logs of trajectory metrics (jerk, clearance, feasibility).

---

## Phase 3 — Controller-in-the-Loop Execution

### Objectives
Embed geometric controller into Isaac Sim to close the control loop.

### Implementation
- 50 Hz control, 10 Hz replanning.
- ±15% gain randomization, motor lag τ = 0.02 s.
- Apply dynamic disturbances (wind 0–5 m/s, mass/inertia ±10%).
- Evaluate tracking error (< 0.5 m RMS).

### Deliverables
- Stable closed-loop controller.
- Executed episodes saved as raw logs for inspection.

---

## Phase 4 — Data Logging & Sharding

### Objectives
Record full expert trajectories and organize them efficiently.

### Specification
- Sampling rate: 20 Hz.
- Logged data: depth sequences (K = 4), odometry, velocities, quaternions, actions (Δr, Δψ), QC flags.
- Shard size: 100–200 episodes (LZ4 compression).
- Global index: Parquet `episodes.parquet`.

### Deliverables
- `.\data\dataset\shards\{easy,medium,hard}\` populated.
- Global index and shard metadata created.
- Run configuration stored as `config.yaml`.

---

## Phase 5 — Quality Control (QC) & Validation

### QC Levels
- **Timestep:** feasibility, sensor validity < 20%, tracking error < threshold.
- **Episode:** duration ≥ 5 s, completeness ≥ 95%.
- **Dataset:** scene diversity, homotopy coverage, failure ≈ 10–15%.

### Deliverables
- QC reports (per-run scorecards).
- Invalid episodes filtered out.
- Metrics dashboard ready for continuous monitoring.

---

## Phase 6 — Diversity Enhancement & Curriculum Expansion

### Objectives
Expand dataset diversity and control task complexity.

### Enhancements
- Geometric perturbations (roll/yaw ± 0.3 rad).
- Dynamic obstacles (10% of episodes).
- Wind and sensor noise (40% of episodes).
- Difficulty tagging (Easy/Medium/Hard).

### Deliverables
- Multi-tiered dataset suitable for curriculum training.
- Sampling scripts for difficulty progression.

---

## Phase 7 — Sim-to-Real Alignment

### Objectives
Reduce domain gap through realistic sensors and real-world anchors.

### Implementation
- Enable physics-based depth sensors (blur, noise, HDR).
- Randomize actuation and sensor response.
- Collect 1–5% real-world MoCap/teleop trajectories.
- Match performance metrics (success > 85%, collision < 10%).

### Deliverables
- Mixed sim+real dataset with anchor episodes.
- Alignment report (gap metrics).

---

## Phase 8 — Dataset Finalization & Release

### Objectives
Freeze validated dataset for IRL.

### Tasks
- Consolidate validated shards.
- Finalize Parquet global index.
- Tag release `v2-data-final`.
- Archive raw data for reproducibility.

### Deliverables
- Final dataset under `.\data\dataset\`.
- Metadata, QC, and changelogs archived.

---

## Phase 9 — Inverse Reinforcement Learning (IRL)

### Objectives
Learn a latent reward function from expert data and derive a policy that reproduces it.

### Pipeline
1. **Data Preparation:**
   - Load expert episodes, normalize states.
   - Split into train/validation/test.
2. **Reward Model:**
   - CNN + MLP architecture → scalar reward rθ(s,a).
   - Training: MaxEnt IRL or AIRL/GAIL.
3. **Policy Optimization:**
   - Train policy πφ with PPO/SAC using rθ.
   - Alternate reward ↔ policy updates until convergence.
4. **Validation:**
   - Evaluate on held-out scenes (success ≥ 85%, collision ≤ 10%).

### Deliverables
- Reward model weights `r_theta.pth`.
- Policy weights `pi_phi.pth`.
- IRL training and evaluation reports.

---

## Phase 10 — Sim-to-Real Fine-Tuning

### Objectives
Adapt learned policy to real-world conditions.

### Methods
- Domain randomization + curriculum transfer.
- Mixed-data fine-tuning (sim + real).
- Evaluate on motion-capture testbed.

### Deliverables
- Fine-tuned policy `pi_phi_real.pth`.
- Deployment-ready parameters and configs.
- Validation metrics (success gap < 3%).

---

## Phase 11 — Continuous Learning Loop

### Objectives
Close the loop by incorporating deployment failures into dataset updates.

### Workflow
1. Collect failure episodes during real deployment.
2. Relabel and regenerate expert trajectories.
3. Merge new data → update reward model rθ → retrain policy πφ.
4. Continuous validation and dashboard monitoring.

### Deliverables
- Incremental dataset updates (`v2.1`, `v2.2`, etc.).
- Versioned reward/policy checkpoints.
- Automated regression tests.

---

## Phase Completion Criteria

| Phase | Success Criteria |
|:------|:------------------|
| 1 | Isaac Lab fully functional; reproducible scenes; ROS 2 bridge operational |
| 2 | Feasible trajectories generated by planner |
| 3 | Controller closed-loop stable (< 0.5 m error) |
| 4 | Dataset shards + global index built |
| 5 | QC scorecard all green |
| 6 | Curriculum coverage achieved |
| 7 | Sim-real gap < 5% |
| 8 | Final dataset tagged v2-data-final |
| 9 | IRL reward/policy validated |
| 10 | Fine-tuned real policy (< 3% gap) |
| 11 | Continuous retraining operational |

---

## Final Outcome
- Full Isaac Lab-based expert data collection pipeline.
- IRL-trained reward + policy models.
- Fine-tuned real-world capable controller.
- Continuous improvement loop for ongoing releases.
