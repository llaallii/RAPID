# rapid_irl_nav

Prototype scaffold for the Rapid IRL Navigation project. The package provides
placeholders for data handling, simulation interfaces, inverse reinforcement
learning algorithms, and command line utilities so we can iterate quickly on
research ideas.

## Conda Setup

Create the environment from `environment.yml`:

```powershell
PS C:\Users\you\RAPID> conda env create -f environment.yml
```

```bash
$ conda env create -f environment.yml
```

Activate the environment:

```powershell
PS C:\Users\you\RAPID> conda activate rapid
```

```bash
$ conda activate rapid
```

Upgrade `pip` and perform the editable install:

```powershell
PS C:\Users\you\RAPID> python -m pip install --upgrade pip
PS C:\Users\you\RAPID> python -m pip install -e .
```

```bash
$ python -m pip install --upgrade pip
$ python -m pip install -e .
```

> Optional: you can bake the editable install into the environment by adding the following to `environment.yml`, but note that the relative `-e .` is resolved at creation time and hides the explicit post-step:
>
> ```yaml
> dependencies:
>   - pip
>   - pip:
>       - -e .
> ```

### Why Conda

- Works without administrator privileges on shared machines.
- Avoids Windows policies that may block direct `pip.exe` execution.

### Troubleshooting

- If corporate AppLocker blocks `pip.exe`, always call `python -m pip ...`.
- When the solve step is slow, install `mamba` once and reuse it:

  ```powershell
  PS C:\Users\you> conda install -n base -c conda-forge mamba
  PS C:\Users\you\RAPID> mamba env create -f environment.yml
  ```

  ```bash
  $ conda install -n base -c conda-forge mamba
  $ mamba env create -f environment.yml
  ```

Then inspect the available commands:

```bash
python -m rapid_irl_nav.cli.train --help
python -m rapid_irl_nav.cli.evaluate --help
python -m rapid_irl_nav.cli.collect_expert --help
```

The modules ship with stubbed implementations and TODOs that we will flesh out
in subsequent iterations.

## Quickstart: Collect Data

With the `rapid` environment active:

```powershell
PS C:\Users\you\RAPID> conda activate rapid
PS C:\Users\you\RAPID> python -m rapid_irl_nav.cli.collect_expert --dataset-root data/rapid_demo --episodes 50
PS C:\Users\you\RAPID> python -m rapid_irl_nav.cli.validate_dataset --dataset-root data/rapid_demo
PS C:\Users\you\RAPID> python -m rapid_irl_nav.cli.view_episode --dataset-root data/rapid_demo --idx 0
```

```bash
$ conda activate rapid
$ python -m rapid_irl_nav.cli.collect_expert --dataset-root data/rapid_demo --episodes 50
$ python -m rapid_irl_nav.cli.validate_dataset --dataset-root data/rapid_demo
$ python -m rapid_irl_nav.cli.view_episode --dataset-root data/rapid_demo --idx 0
```
