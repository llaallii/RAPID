# Rapid IRL Navigation Dataset Layout

The dataset is a directory tree containing metadata, shards of episodes, and a tabular index to support fast lookup. The structure is deterministic so downstream code can rely on consistent schemas.

```
<dataset_root>/
  dataset.json                       # global metadata (schema version, sensor spec)
  shards/
    shard-000000/
      ep-000000/
        meta.json
        timestamps.npy               # (T,) float64, seconds since epoch or monotonic
        depth.npy                    # (T, 1, 64, 64) float32, meters; NaN if missing
        vel.npy                      # (T, 3) float32, body-frame linear vel (m/s) + yawrate (rad/s) -> use (vx, vy, wz)
        quat.npy                     # (T, 4) float32, [w,x,y,z]
        pos_xy.npy                   # (T, 2) float32, world meters (x,y) for plotting/sanity
        goal.npy                     # (T, 3) float32, goal in world (x,y) + desired yaw (rad)
        action_raw.npy               # (T, 10, 2) float32, (delta_r, delta_yaw) cylindrical (actor target); NaN on the first step
        done.npy                     # (T,) uint8, terminal flags
      ep-000001/
        ...
    shard-000001/
      ...
  index.parquet                      # fast lookup: columns [shard, episode, T, t0, t1, success, mean_depth_ok]
```

## Episode Metadata

Each episode directory contains a `meta.json` file with the following fields:

| Field           | Type      | Description                                               |
|-----------------|-----------|-----------------------------------------------------------|
| `episode_id`    | `str`     | Identifier matching the episode directory name.          |
| `env`           | `str`     | Name of the simulator/environment used for collection.   |
| `success`       | `bool`    | Whether the episode achieved its goal.                   |
| `collision_count` | `int`   | Number of collisions recorded (0 if none).               |
| `goal_xy`       | `[float, float]` | Target position in world coordinates.          |
| `seed`          | `int`     | Episode RNG seed.                                         |
| `notes`         | `str`     | Optional free-form comment.                              |

All NumPy arrays in an episode folder must have the same first dimension `T`, representing timesteps. The `timestamps.npy` array is monotonically increasing and shares the same length `T`.

## Shards

Episodes are grouped into shard directories to keep folder sizes manageable. Each shard contains at most 200 episodes. After reaching the limit the collector creates a new shard directory with a monotonically increasing zero-padded suffix.

## Index

The dataset root holds an `index.parquet` file summarizing all episodes for disk-efficient loading. Mandatory columns:

- `shard`: shard directory name (e.g. `shard-000000`)
- `episode`: episode directory name (e.g. `ep-000123`)
- `T`: number of timesteps
- `t0`: timestamp of the first frame (float64 seconds)
- `t1`: timestamp of the last frame
- `success`: boolean success flag
- `mean_depth_ok`: boolean flag indicating the depth channel contained finite values

The index is rebuilt by walking the shard layout, so it never needs manual editing. Downstream dataloaders can rely on the index for fast sampling without scanning the full directory tree.
