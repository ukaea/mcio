# Badges

[![Tests](https://github.com/ukaea/mcio/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ukaea/mcio/actions/workflows/tests.yml)

[![Coverage Score](https://ukaea.github.io/mcio/coverage.svg)](https://github.com/ukaea/mcio/actions/workflows/publish-docs.yml)

[![Ruff Score](https://ukaea.github.io/mcio/ruff.svg)](https://github.com/ukaea/mcio/actions/workflows/lint.yml)


# Docs

[https://ukaea.github.io/mcio/](https://ukaea.github.io/mcio/)


# Quick start

> [!WARNING]
>  - MCIO uses CM for length and MeV for energy.
>  - MCIO requires STEP files to use the AP242 schema.

The MCIO CAD-to-CSG package can be installed for the end user and run via the command line. 

```bash
python -m mcio decompose -i my_model.stp -m my_materials.json -t my_tallies.json --cores 20
```

The above will convert a step file (AP242 standard) to CSG then to MCNP file, writing the output
to MCNP6 input format to output/mcnp_model.i and to OpenMC XML format if installed to output/openmc.

# Installation

## End user installation (dependencies via Pip - Windows, Mac and latest debian Linux)

MCIO can be installed from source using the below commands, with the dependencies installed for cad and jupyter using the extras [cad,jupyter]. The CAD and Jupyter dependencies are not required to run the non-CAD-to-CSG.

```bash

git clone https://github.com/ukaea/mcio.git

python -m pip install .[cad,jupyter]

```

OpenMC is not published on PyPI. To write OpenMC models, install its Python API from source, see the [OpenMC installation guide](https://docs.openmc.org/en/stable/quickinstall.html).

## Dev installation

For those wishing to contribute to development, you can install all dependencies for docs, 

```bash

git clone -b dev https://github.com/ukaea/mcio.git

python -m pip install .[dev,docs,cad,jupyter]

```

The test suite also needs OpenMC; CI installs it with:

```bash
python -m pip install "openmc @ git+https://github.com/openmc-dev/openmc.git@83a30f6860c79fac8fd58584489cc0d1abcb0376"
```

## Running MCIO - command line

Running MCIO via the command line is the easiest way to produce a model with materials and some common tally types. The below example runs via 20 core multiprocess and assigns materials by RGB and some basic tally types. To assign materials by _m1_ type labels use -ml.

```bash
python -m mcio decompose -c 20 -i my_model.stp -m my_mats_rgb.json --tally my_tallies.json

```

```json
{
    "_m1_": {
        "density": 0.997,
        "composition": {
            "H": 0.666667,
            "O": 0.333333
        },
        "number": 100,
        "reference": "https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev2.pdf",
        "fraction_type": "ao",
        "density_type": "g/cc",
        "name": "Water, Liquid",
        "mcnp": {
            "sab_cards": [],
            "libs": {},
            "library": ""
        }
    }
}
```

The open cascade based OCP modules (the CAD-to-CSG decomposer and its helpers) are loaded automatically by `import mcio` when OCP is installed.

This allows users who don't need the CAD-to-CSG decomposer not to need OCP installed, which can require Conda on non Windows/modern Debian systems. Without OCP, `import mcio` logs a warning and the non-CAD modules (surfaces, cells, materials, tallies, sources and MCNP/OpenMC output) work as normal.

```python
import mcio
```

# Development

## Branches

Work is done on `feature/<name>` branches, which are the only branches that can be pushed to directly.

1. Branch from `dev` and push your changes to a `feature/` branch.
2. Open a pull request from the feature branch into `dev`.
3. Releases are made by a pull request from `dev` into `main`; pull requests into `main` from any other branch are rejected.

Pull requests need an approval and passing checks before they can be merged.

## Tests

Tests are run with pytest, with coverage collected in the same run:

```bash
python -m pytest --cov
```

## Linting and formatting

Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/). CI checks but does not change files, so run these before pushing:

```bash
ruff format mcio tests
```

```bash
ruff check --fix mcio tests
```

The pylint-style score shown on the Ruff badge can be produced locally with:

```bash
python scripts/ruff_score.py mcio
```
