# MCIO

[![Publish docs and badges](https://github.com/ukaea/mcio/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/ukaea/mcio/actions/workflows/publish-docs.yml)
[![Tests](https://github.com/ukaea/mcio/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ukaea/mcio/actions/workflows/tests.yml)
[![Coverage](https://ukaea.github.io/mcio/coverage.svg)](https://github.com/ukaea/mcio/actions/workflows/publish-docs.yml)
[![Ruff score](https://ukaea.github.io/mcio/ruff.svg)](https://github.com/ukaea/mcio/actions/workflows/publish-docs.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ukaea.github.io/mcio/)
[![Release](https://img.shields.io/github/v/release/ukaea/mcio?include_prereleases)](https://github.com/ukaea/mcio/releases)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/ukaea/mcio/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MCIO is a Python package for building Monte Carlo radiation transport models. It lets you:

- **Build CSG models in Python**: define surfaces, cells, materials, sources and tallies once, then write them to
  both MCNP and OpenMC input formats.
- **Convert CAD to CSG**: decompose STEP (AP242) CAD models into CSG cells, with materials assigned by colour or
  label, using multiple cores.
- **Convert CSG to CAD**: turn CSG volumes back into OCP solids and STEP files for inspection.
- **Debug models**: extract lost particles from MCNP output into a STEP file, to load into your CAD software.

**Documentation:** [https://ukaea.github.io/mcio/](https://ukaea.github.io/mcio/)

> [!WARNING]
>  - MCIO uses cm for length and MeV for energy.
>  - MCIO requires STEP files to use the AP242 schema.

## Installation

### End user installation (dependencies via Pip - Windows, Mac and latest Debian Linux)

MCIO can be installed from source using the below commands. The `cad` and `jupyter` extras install the dependencies
for CAD-to-CSG decomposition and Jupyter; they are not needed for building CSG models directly.

```bash
git clone https://github.com/ukaea/mcio.git
cd mcio
python -m pip install ".[cad,jupyter]"
```

To write OpenMC models, install its Python API, see the [OpenMC installation guide](https://docs.openmc.org/en/stable/quickinstall.html).

### Dev installation

For those wishing to contribute to development, install the dependencies for testing, linting and building the docs.

```bash
git clone -b dev https://github.com/ukaea/mcio.git
cd mcio
python -m pip install ".[dev,docs,cad,jupyter]"
```

The test suite also needs OpenMC; CI installs it with:

```bash
python -m pip install "openmc @ git+https://github.com/openmc-dev/openmc.git@83a30f6860c79fac8fd58584489cc0d1abcb0376"
```

## Quick start

Convert a STEP file to CSG from the command line, using 20 cores:

```bash
python -m mcio decompose -i my_model.stp -m my_materials.json -t my_tallies.json --cores 20 --xsdir /path/to/xsdir
```

This writes an MCNP6 input to `output/model_mcnp.i` and, if OpenMC is installed, OpenMC XML files to
`output/model_openmc`. The `--xsdir` option is optional.

## Command line usage

Running MCIO via the command line is the easiest way to produce a model with materials and some common tally types. The below example runs on 20 cores, assigns materials from a JSON file and adds tallies from a second JSON file. Materials are assigned by RGB colour or by `_m1_` type labels, depending on the keys used in the materials JSON; an example label-keyed materials JSON is shown below.

```bash
python -m mcio decompose -c 20 -i my_model.stp -m my_materials.json --tallies my_tallies.json
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

## Python usage

The Open Cascade based OCP modules (the CAD-to-CSG decomposer and its helpers) are loaded automatically by `import mcio` when OCP is installed.

This means users who don't need the CAD-to-CSG decomposer don't need OCP installed, which can require Conda on non Windows/modern Debian systems. Without OCP, `import mcio` logs a warning and the non-CAD modules (surfaces, cells, materials, tallies, sources and MCNP/OpenMC output) work as normal.

```python
import mcio
```

See the [user guide](https://ukaea.github.io/mcio/userguide.html) for worked examples of building CSG models and running the decomposer from Python.

## Development

### Branches

Work is done on `feature/<name>` branches, which are the only branches that can be pushed to directly.

1. Branch from `dev` and push your changes to a `feature/` branch.
2. Open a pull request from the feature branch into `dev`.
3. Releases are made by a pull request from `dev` into `main`; pull requests into `main` from any other branch are rejected.

Pull requests need an approval and passing checks before they can be merged.

### Tests

Tests are run with pytest, with coverage collected in the same run:

```bash
python -m pytest --cov
```

### Linting and formatting

Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/). CI only blocks a pull request on real bugs (syntax errors and invalid constructs); all other findings and formatting lower the ruff score but do not fail the check. CI does not change files, so to tidy up before pushing run:

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
