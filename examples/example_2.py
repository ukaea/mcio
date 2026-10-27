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

    # save files for mcnp and openmc
    decomposer.model.write_mcnp('model_mcnp.i')
    decomposer.model.write_openmc('model_openmc')
