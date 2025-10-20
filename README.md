# rapid_irl_nav

Prototype scaffold for the Rapid IRL Navigation project. The package provides
placeholders for data handling, simulation interfaces, inverse reinforcement
learning algorithms, and command line utilities so we can iterate quickly on
research ideas.

## Getting Started

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -e .
```

Then inspect the available commands:

```bash
python -m rapid_irl_nav.cli.train --help
python -m rapid_irl_nav.cli.evaluate --help
python -m rapid_irl_nav.cli.collect_expert --help
```

The modules ship with stubbed implementations and TODOs that we will flesh out
in subsequent iterations.
