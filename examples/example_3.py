"""
example_2.py

Decomposition using multiprocess, of a dummy bunker model with materials
assigned by colour.

"""

import sys
from pathlib import Path

import mcio

# the dummy bunker model lives in the models folder of the repository
MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
sys.path.insert(0, str(MODELS_DIR))

from referencebunker import write_bunker_model_to_step


# set up guard for using multiprocess
if __name__ == "__main__":
    # set up dummy model and save to stp (AP242)
    write_bunker_model_to_step('bunker.stp')

    # set up a decomposer and provide the model path and material json path
    decomposer = mcio.decompose.Decomposer(
        step_file_paths=['bunker.stp'],
        materials_file_path=MODELS_DIR / 'referencebunkermats.json')

    # run decomposer, with 8 cores
    decomposer.run(cores=8)

    # get the concrete walls form the model
    decomposition_solid_concrete = decomposer.filter_by_label('concrete_walls')[0]

    # apply some custom meshes
    tally_neutron_mesh = mcio.tally.FluenceMeshTally.from_bounding_box(
        bounding_box=decomposition_solid_concrete.bounding_box,
        resolution=10.0,
        extension=200.0, 
        particles=['neutron'])

    tally_photon_mesh = mcio.tally.FluenceMeshTally.from_bounding_box(
        bounding_box=decomposition_solid_concrete.bounding_box,
        resolution=10.0,
        extension=200.0, 
        particles=['photon'])

    # add tallies to model
    decomposer.model.tallies += [tally_neutron_mesh, tally_photon_mesh]

    # set some run parameters for mcnp
    decomposer.model.mcnp.nps = 1e7
    decomposer.model.mcnp.lost = 50
    decomposer.model.mcnp.modes = ['neutron', 'photon']

    # save files for mcnp and openmc
    decomposer.model.write_mcnp('model_mcnp.i')
    decomposer.model.write_openmc('model_openmc')
