Decomposition
*************

# Decomposing CAD-to-CSG

The MCIO code uses the Open Cascade thin wrapped Python package OCP, and can take a STEP CAD file and convert it into a CSG model, which can then be written to an MCNP input deck.

## Running from the command line

The MCIO CAD-to-CSG decomposer can be run with the default values via the command line, running with single or multiprocess, and optionally with debugging and/or allowing failures to convert. 

```bash
python -m mcio decompose -i my_model.stp --cores 8 --materials materials.json
```

To check if a model converts, it is useful to use the command line method using multiple cores. This will try to convert all bodies, with any bodies which fail to convert being saved to individual step files in the analysis/failed_to_convert directory. Typically, this will be cause of bodies being too complex or having possible geometry errors within them.

When solids fail to convert, they are saved to the analysis/failed_to_convert directory.

Possible reasons for failures:
 - Inexact edges (most common errors), to fix or split on edges, particularly with pipes
 - Solids too complex, try splitting them up multiple bodies
 - Bad geometry, try to fix geometry errors or de-draw solids
 - Contain faces which don't currently convert, such as off axis tori or splines

## Running via the Python API

The MCIO CAD-to-CSG decomposer allows for either single core user or multiple core use using the "multiprocess" package. The latter provides significant speed increases during conversion but requires some boiler plate code to run, an example of which is given below.
### Single core example

```Python
# import the decomposer class from mcio
import mcio 

# define your list of step files to combined into a single model
step_files = ['my_model_1.stp', 'my_model_2.stp']

# create an instance of the decomposer
decomposer = mcio.decompose.Decomposer(step_files, materials_file_path='materials.json')

# run the decomposer with default settings
decomposer.run()

# save the models to mcnp input deck
decomposer.write_mcnp('model.i')

# save the model to a folder of openmc xml files
decomposer.write_openmc('openmc_model')
```

### Multiprocess example

Increasing the number of cores requested in the .run(cores=20) method, allows
the decomposer to run via multiprocess, but this requires the 
__name__="__main__" guard to be placed around the code, otherwise an infinitely 
recursive import will be triggered.

```python
import mcio

# set up guard to stop recursive imports
if __name__ == '__main__':
    # define path to step files
    step_files = ['my_model_1.stp', 'my_model_2.stp']
    
    # refactored multiprocess
    decomposer = mcio.decompose.Decomposer(step_files, materials_file_path='materials.json')
    decomposer.run(cores=20, debug=False)

    # save the models to mcnp or openmc
    decomposer.write_mcnp('model.i')
    decomposer.write_openmc('openmc_model')
```

### Assigning materials

Materials can be assigned within the CAD file either via setting the RGB colour of the solid or via the addition of a label of the form _m<n>_, with <n> being an integer. The materials can be defined within a JSON which the user is required to provide. It should be noted that single solids within a component will lose their strings as OCP will assign the parent components string to the solid, however if there are two or more solids within a component then each solid maintains their labels.

```bash
python -m mcio decompose -i my_model.stp --cores 8 --materials materials.json
```

```python
# create an instance of the decomposer, and provide path to json for materials
decomposer = mcio.decompose.Decomposer(['my_model.stp'], materials_file_path='materials.json')
```

#### Assign by label (_m<n>_)

The below image shows material assignment via the label method, with an example of the required accompanying material definition in JSON format.

```{warning}
A single solid within a component, such as sphere_4 in the example below, will lose its string as OCP will assign the parent component to the solid.
```

```{figure} _static/material_assignment_by_label.png
:alt: Example of a structure tree with material assignment
:width: 350px
:align: center
:name: my_figure_reference
Figure 1: this is my images
```

```json
{
    "m1": {
        "density": 0.997,
        "composition": {
            "H": 0.666667,
            "O": 0.333333
        },
        "number": 100,
        "reference": "",
        "fraction_type": "ao",
        "density_type": "g/cc",
        "name": "Water, Liquid",
        "mcnp": {
            "sab_cards": [],
            "libs": {},
            "library": ".32c",
            "library_by_isotopes": {"O18": ".80c"}
        }
    }
}
```

#### Assign by RGB colour

When assigning by RGB, the JSON should contain the RGB value as a string in the format <red>_<green>_<blue> as the key. 

```json
{
    "000_128_064": {
        "density": 0.997,
        "composition": {
            "H": 0.666667,
            "O": 0.333333
        },
        "number": 100,
        "reference": "",
        "fraction_type": "ao",
        "density_type": "g/cc",
        "name": "Water, Liquid",
        "mcnp": {
            "sab_cards": [],
            "libs": {},
            "library": ".32c",
            "library_by_isotopes": {"O18": ".80c"}
        }
    }
}
```

### Debugging problem solids

CAD-to-CSG decomposition can fail or cause problems with lost particles, which can also have a wide range of reasons for these issues. The failed cad solids will be output to the analysis/failed_to_conver/ directory, which can then be loaded into
your chosen CAD software. A text file containing the exception for each failed solid is output to the analysis/trackbacks/ directory.

Some common issues are:
1. Inexact edges on connected 2dn order faces, particularly cylinders joined at an angle
2. Rounding errors in coincident surface
3. Rounding errors introduced during the splitting / cut / common operations in OCP
4. Overlapping geometry
5. Failure to generate correct void space
6. Incorrect auxiliary surface generation
7. Solid slow to convert as too complex for recursive splitting

There is also a command line tool for extracting the lost particles from an MCNP 
output file, which produces a step file with spherical points and lines as vectors
for the particles directions. This can be accessed via the command line tool.

```bash 
python -m mcio lost_particles -i my_mcnp_run.o -o lost_particles.stp --max_particles 100 --radius 0.5 --vector_length 200.0
```

Or can be scripted using the utilities function "lost_particles_from_mcnp_run"

```python
from mcio.diagnostic import lost_particles_from_mcnp_run

lost_particles_from_mcnp_run(
    'my_mcnp_run.o',
    output_file_path='lost_particles.stp,
    max_particles=100,
    radius=0.5,
    vector_length=200.0
)

### Altering the decomposition parameters

The Decomposer() has a set of default parameters for carrying out decomposition
and conversion to CSG. These parameters can be printed using a python script.

```python
# import mcio
import mcio

# create instance of DecompositionParameters containing default values
decom_params = mcio.decomposeparams.DecompositionParameters()

# wring these to screen
print(decom_params)

# write to json
decom_params.write(args.write_decompose_parameters)

# read in decom params
decom_params.read('decompose_parameters.json')

```{decomposition-params}

The parameters used in the decomposition parameters stored within the decomposer
can be printed as a table, which provides a description of each parameter.

The command line tool for decomposition can also read in a set of decomposition
parameters. 

```bash 
python -m mcio decompose -i my_model.stp -rdp decom_parameters.json
```
