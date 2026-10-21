""" 
example_2.py

Decomposition using single core mode

"""

import mcio
mcio.load_decomposer()

# set user defined parameters
mcio.decompose.DECOM_PARAMS.SPLIT_MAX_VOL_DEVIATION = 1e-2
mcio.decompose.DECOM_PARAMS.SPLIT_MIN_AREA_TO_VOLUME_RATIO = 1e-3

mcio.decompose.DECOM_PARAMS.VOID_MAX_DECOMPOSITION_SOLIDS_PER_OCTANT = 10
mcio.decompose.DECOM_PARAMS.VOID_MAX_SPLIT_SOLIDS_PER_OCTANT = 15
mcio.decompose.DECOM_PARAMS.VOID_MAX_SURFACES_BOUNDING_BOX_PER_OCTANT = 50
mcio.decompose.DECOM_PARAMS.VOID_MAX_SURFACES_BY_DISTANCE_PER_OCTANT = 50
mcio.decompose.DECOM_PARAMS.VOID_MIN_WIDTH = 20.0
mcio.decompose.DECOM_PARAMS.VOID_MAX_RECURSION_DEPTH = 15

mcio.decompose.DECOM_PARAMS.VOID_USE_FACE_DISTANCE_CHECK = False
mcio.decompose.DECOM_PARAMS.VOID_USE_ALL_FACES_IN_SOLID_SOLID = True

mcio.decompose.DECOM_PARAMS.BOOL_CUTTER_FUZZY_VALUE = 1e-6


# set up decomposer
decomposer = mcio.decompose.Decomposer(
    step_file_paths=['model.stp'],
    materials_file_path='materials.json')

# run decomposer
decomposer.run()