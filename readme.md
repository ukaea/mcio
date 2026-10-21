# Badges

[![Pylint Score](https://gitlab.ukaea.uk/rworrall/mcio/-/jobs/artifacts/main/raw/pylint_out/pylint.svg?job=lint_and_coverage)](https://gitlab.com/NAMESPACE/PROJECT_NAME/-/pipelines)

[![Coverage Score](https://gitlab.ukaea.uk/rworrall/mcio/-/jobs/artifacts/main/raw/coverage_out/coverage.svg?job=lint_and_coverage)](https://gitlab.com/NAMESPACE/PROJECT_NAME/-/pipelines)

> [!WARNING]
> MCIO uses MCNP default units: cm for length, MeV for energy etc.

# Docs

[https://rworrall.pages.gitlab.ukaea.uk/mcio/index.html](https://rworrall.pages.gitlab.ukaea.uk/mcio/index.html)


# Quick start

The MCIO CAD-to-CSG package can be installed for the end user and run via the command line. 

```bash
python -m mcio decompose -i my_model.stp -m my_materials.json -t my_tallies.json --cores 20
```

The above will conver a step file (AP242 standard) to CSG then to MCNP file, writing the output
to MCNP6 input format to output/mcnp_model.i and to OpenMC XML format if installed to output/openmc.

# Installation

## End user installation (dependencies via Pip - Windows, Mac and latest debian Linux)

MCIO can be installed from source using the below commands, with the dependencies installed for cad and jupyter using the training [cad,jupyter]. The CAD and Jupyter dependencies are not required to run the non-CAD-to-CSG.

```bash

git clone git@gitlab.ukaea.uk:rworrall/mcio.git

python -m pip install .[cad,jupyter]

```

## Dev installation

For those wishing to contribute to development, you can install all dependencies for docs, 

```bash

git clone -b dev git@gitlab.ukaea.uk:rworrall/mcio.git

python -m pip install .[dev,docs,cad,jupyter]

```

## Running MCIO - command line

Running MCIO via the command line is the easliest way to produce a model with materials and some common tally types. The below example runs via 20 core multiprocess and assigns materails by RGB and some basic tally types. To assign materials by _m1_ type labels use -ml.

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

To use the open cascade based OCP module or Jupyterlab CAD plug in, you will need to manually import these using the "mcio.load_decomposer()" and "mcio.load_jupyter()". 

This was done to allow used who don't need to use the CSG-to-CAD decomposer not to need OCP installed, which can require Conda on non Windows/modern Debian systems.

```python
# import mcio 
import mcio



```

