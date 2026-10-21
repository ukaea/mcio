"""Store the decomposition parameters used in the decompose module and OCP operations.

Contains the :class:`DecompositionParameters` slots dataclass which holds every
tolerance, flag, rounding precision and debugging path used during CAD to CSG
decomposition, together with a human-readable definition of each parameter.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any
import math


from .paramlookup import ParameterLookup


@dataclasses.dataclass(slots=True)
class DecompositionParameters(ParameterLookup):
    """Data class used to store all constants used in decomposition.

    Parameter names are lower case (PEP 8) and start with a category key word
    (e.g. ``surf``, ``void``) so that the parameters can be printed in separate
    tables. Each parameter also has a counterpart with ``_def_`` prepended to
    its name, which is a string holding its definition.

    Notes
    -----
    Every class-level annotated attribute is a dataclass field and may be
    overridden via the constructor keyword of the same name. The former upper
    case names (e.g. ``VOID_MIN_WIDTH``) are still accepted as attributes, with
    a ``DeprecationWarning``.

    Examples
    --------
    >>> params = DecompositionParameters(void_min_width=20.0)
    >>> params.void_use_obb_check = False
    """
    
    # document settings
    doc_unit_in_metres: float = 0.01
    doc_xcaf_formatting: str = 'BinXCAF' # 'MDTV-XCAF'
    
    _def_doc_unit_in_metres: str = 'input unit length (m)'
    _def_doc_xcaf_formatting: str = 'XCAF Formatting'

    # pre-splitting
    presplit_pipes: bool = True
    presplit_pipes_cyl_angle_tol: float = 1e-2
    presplit_concentril_2nd_order: bool = True
    presplit_about_axis: str = 'z'
    presplit_about_axis_n_times: int = 4
    presplit_rectilinear_voxels: bool = False
    presplit_rectilinear_voxels_width: float = 50.0

    _def_presplit_pipes: str = 'pre-split: use pipe splitting tools for connected cylinders of equal radius'
    _def_presplit_pipes_cyl_angle_tol: str = 'pre-split on pipe cylinders: angle tolerance'
    _def_presplit_concentril_2nd_order: str = 'pre-split: split connected 2nd order surfaces which are on the same axix with circular edge'
    _def_presplit_about_axis: str = 'pre-split: split using planes about axis, either x, y or z' 
    _def_presplit_about_axis_n_times: str = 'pre-split: number of planes about axis to split, n * 2 = segments, e.g. 4 = 8 octants'
    _def_presplit_rectilinear_voxels: str = 'pre-split: split all solids using a rectilinear grid'
    _def_presplit_rectilinear_voxels_width: str = 'pre-split: width of voxels in rectilinear grid split'

    # pre-cleanup
    cleanup_unify_same_domains: bool = False
    cleanup_shape_fix: bool = False
    cleanup_shape_fix_tol_min: float = 1e-8
    cleanup_shape_fix_tol_max: float = 1e-6
    cleanup_inexact_edges: bool = False
    cleanup_inexact_edges_tol: float = 1e-4
    cleanup_inexact_edge_fix_tol_min: float = 1e-8
    cleanup_inexact_edge_fix_tol_max: float = 1e-6

    _def_cleanup_unify_same_domains: str = 'unify domains can reduce number of preiodic boundaries'
    _def_cleanup_shape_fix: str = 'apply clean up using ShapeFix_Solid before decomposition'
    _def_cleanup_shape_fix_tol_min: str = 'ShapeFix_ShapeTolerance minimum tolerance'
    _def_cleanup_shape_fix_tol_max: str = 'ShapeFix_ShapeTolerance maximum tolerance'
    _def_cleanup_inexact_edges: str = 'check for inexact edges and cleanup with tolerance bounds'
    _def_cleanup_inexact_edges_tol: str = 'threshold tolerance for inexact edge fix'
    _def_cleanup_inexact_edge_fix_tol_min: str = 'inexact edges minimum tolerance'
    _def_cleanup_inexact_edge_fix_tol_max: str = 'inexact edges maximum tolerance'

    # decomposition settings
    split_max_distnace_between_face_and_halfspace: float = 1e-4
    split_unify_same_domains: bool = False
    split_sew: bool = False
    split_skip_non_cutting_planes: bool = True
    split_skip_non_cutting_planes_rel_tol: float = 1e-7

    _def_split_max_distnace_between_face_and_halfspace: str = 'max dist between face and halfspace used to recursively split'
    _def_split_unify_same_domains: str = 'apply unify same domains to merge periodic or duplicated conicident faces following splitting'
    _def_split_sew: str = 'split sew to solids following recursive splitting'
    _def_split_skip_non_cutting_planes: str = 'skip the half space and split of planar faces whose plane leaves the whole solid on the material side (they cannot split it)'
    _def_split_skip_non_cutting_planes_rel_tol: str = 'tolerance, relative to the solid size, for a solid to count as lying behind a planar face'
    
    # boolean operation settings
    bool_common_fuzzy_value: float = 1e-6
    bool_cutter_fuzzy_value: float = 1e-6
    bool_splitter_fuzzy_value: float = 1e-6

    bool_degenerate_min_area_to_vol_ratio: float = 1e-4
    bool_degenerate_min_vol_fraction_of_original: float = 1e-10
    bool_degenerate_min_ab_vol: float = 1e-10
        
    _def_bool_common_fuzzy_value: str = 'fuzzy logic value passed to common boolean operation'
    _def_bool_cutter_fuzzy_value: str = 'fuzzy logic value passed to the cut boolean operation'
    _def_bool_splitter_fuzzy_value: str = 'fuzzy logic value passed to the split boolean operation'

    _def_bool_degenerate_min_area_to_vol_ratio: str = 'area to vol ration limit, used to filter degenerate solids in boolean operations'
    _def_bool_degenerate_min_vol_fraction_of_original: str = 'minumum volume required as a fraction of the original solid'
    _def_bool_degenerate_min_ab_vol: str = 'minimum volume of any split solid produced via boolean operation'

    # tolerances
    make_face_tol: float = 1e-6
    _def_make_face_tol: str = 'tolerance used when making a face'

    # surface tolerances and prcision / rounding
    surf_plane_tol_normal_angle: float = 1e-6
    surf_plane_tol_on_axis: float = 1e-6
    surf_plane_tol_distance: float = 1e-6
    surf_plane_rounding_lateral: int = 5
    surf_plane_rounding_vector: int = 8

    surf_cylinder_tol_axis_angle: float = 1e-6
    surf_cylinder_tol_radius: float = 1e-6
    surf_cylinder_tol_axis_distance: float = 1e-6
    surf_cylinder_rounding_lateral: int = 5
    surf_cylinder_rounding_radius: int = 5

    surf_sph_radius_tol: float = 1e-6
    surf_sph_position_tol: float = 1e-6

    surf_cone_semi_angle_tol: float = 1e-6
    surf_cone_axis_angle_tol: float = 1e-6
    surf_cone_axis_distance_tol: float = 1e-6
    surf_cone_rounding_lateral: int = 5
    surf_cone_rounding_t_sq: int = 8

    surf_tol_tori_minor_radius: float = 1e-6
    surf_tol_tori_major_radius: float = 1e-6
    surf_tol_tori_axis_angle: float = 1e-6
    surf_tol_tori_centre: float = 1e-6
    surf_tori_rounding_lateral: int = 8
    surf_tori_rounding_radii: int = 8

    surf_general_quadratic_rounding_abc: int = 8 
    surf_general_quadratic_rounding_def: int = 8
    surf_general_quadratic_rounding_ghj: int = 8
    surf_general_quadratic_rounding_k: int = 8

    _def_surf_plane_tol_normal_angle: str = 'coincident planes: normal angle tolerance (degrees)'
    _def_surf_plane_tol_on_axis: str = 'ocp to mcio plane: on axis tol angle (degrees)'
    _def_surf_plane_tol_distance: str = 'ocp to mcio plane: distance between planes with same normal'
    _def_surf_plane_rounding_lateral: str = 'ocp to mcio plane: decimal place rounding of lateral x, y, z and d parameters'
    _def_surf_plane_rounding_vector: str = 'ocp to mcio plane: decimal place rounding of vector a, b, c parameters'

    _def_surf_cylinder_tol_axis_angle: str = 'ocp to mcio cyl: on axis tol angle (degrees)'
    _def_surf_cylinder_tol_radius: str = 'coincident cyl: radius tolerance'
    _def_surf_cylinder_tol_axis_distance: str = 'coincident cyl: axis distance'
    _def_surf_cylinder_rounding_lateral: str = 'ocp to mcio cyl: decimal place rounding of lateral x, y, and z parameters'
    _def_surf_cylinder_rounding_radius: str = 'ocp to mcio cyl: decimal place rounding of radius parameter'

    _def_surf_sph_radius_tol: str = 'coincident sph: radius tolerance'
    _def_surf_sph_position_tol: str = 'coincident sph: distance of centres tolerance'

    _def_surf_cone_semi_angle_tol: str = 'coincident cone: semi angle tolerance (degrees)'
    _def_surf_cone_axis_angle_tol: str = 'ocp to mcio cone: on axis angle tolerance (degrees)'
    _def_surf_cone_axis_distance_tol: str = 'ocp to mcio cone: on axis tolerance'
    _def_surf_cone_rounding_lateral: str = 'ocp to mcio cone: rounding number of dp for apex cartesian x,y,z'
    _def_surf_cone_rounding_t_sq: str = 'ocp to mcio cone: rounding of t_sq number of dp, t_sq = tan(semi_angle)**2'

    _def_surf_tol_tori_minor_radius: str = 'coincident tori: minor radius tolerance'
    _def_surf_tol_tori_major_radius: str = 'coincident tori: major radius tolerance'
    _def_surf_tol_tori_axis_angle: str = 'ocp to mcio tori: on axis angle tolerance (degrees)'
    _def_surf_tol_tori_centre: str = 'coincident tori: distance of centres tolerance'
    _def_surf_tori_rounding_lateral: str = 'ocp to mcio tori: decimal place rounding of lateral x, y, and z parameters'
    _def_surf_tori_rounding_radii: str = 'ocp to mcio tori: decimal place rounding of major and minor radii parameters'

    _def_surf_general_quadratic_rounding_abc: str = 'decimal places to round GQ a, b and c params'
    _def_surf_general_quadratic_rounding_def: str = 'decimal places to round GQ d, e and f params'
    _def_surf_general_quadratic_rounding_ghj: str = 'decimal places to round GQ g, h and k params'
    _def_surf_general_quadratic_rounding_k: str = 'decimal places to round GQ a, b and c params'

    # closed face
    wire_is_closed_tol: float = 1e-6

    _def_wire_is_closed_tol: str = 'tolerance checking if wire is closed'

    # void generation
    void_max_recursion_depth: int = 30
    void_min_width: float = 10.0
    void_max_split_solids_per_octant: int = 50
    void_max_faces_per_octant: int = 250
    void_use_split_solid_distance_check: bool = False
    void_split_solid_fractoin_of_bounding_box: float = 0.50
    void_split_solid_distance_tol: float = 1e-2
    void_filter_by_face: bool = False
    void_face_distance_tol: float = 1e-2
    void_bounding_box_use_optimal: bool = False
    void_use_obb_check: bool = True
    void_use_halfspace_pruning: bool = True

    _def_void_max_recursion_depth: str = 'maximum recursive depth'
    _def_void_min_width: str = 'minimum void width of octant'
    _def_void_max_split_solids_per_octant: str = 'maximum split solids in octant'
    _def_void_max_faces_per_octant: str = 'maximum split solid faces per octant (excludes auxiliaries)'
    _def_void_use_split_solid_distance_check: str = ''
    _def_void_split_solid_fractoin_of_bounding_box: str = 'maximum fraction of the bounding box a split solid can contain before distance checks carried out'
    _def_void_split_solid_distance_tol: str = ''
    _def_void_filter_by_face: str = ''
    _def_void_face_distance_tol: str = 'distance tolerance for BRepExtrema_DistanceShapeShapes between void bounding box and face'
    _def_void_bounding_box_use_optimal: str = 'during void generation use the AddOptimal method for tighter bounding boxes'
    _def_void_use_obb_check: str = 'filter split solids per octant by oriented bounding box (separating axis test) after the axis aligned check'
    _def_void_use_halfspace_pruning: str = 'classify split solid half spaces against each octant: drop solids proven outside, drop half spaces proven to contain the octant, skip octants inside a solid'
    
    # half spaces
    bounding_box_multiplier_half_space: float = 10

    _def_bounding_box_multiplier_half_space: str = 'expansion factor for bounding box rpp solid used for halfspace'

    # auxiliary surfaces
    aux_generate: bool = True
    aux_bounding_box_analysis: float = 1.50
    aux_surface_face_distace_tol: float = 1e-4
    aux_surface_cylinder_angle_tol: float = 1e-6
    aux_surface_cylinder_small_angle: float = (0.1/360) * math.tau
    aux_surface_cylinder_simple: bool = True
    aux_surface_cone_angle_tol: float = 1e-6
    aux_surface_cone_small_angle: float  = (0.1/360) * math.tau
    aux_surface_cone_simple: bool = True
    aux_surface_sphere_angle_tol: float = 1e-6
    aux_surface_sphere_small_angle: float = (0.1/360) * math.tau
    aux_surface_sphere_simple: bool = True
    aux_surface_tori_angle_tol: float = 1e-6
    aux_surface_tori_small_angle: float = (0.1/360) * math.tau
    aux_surface_tori_aux_surface_simple: bool = True

    _def_aux_generate: str = 'generate auxiliary surfaces for second order faces'
    _def_aux_bounding_box_analysis: str = 'extension of bounding box for aux surface in analysis save to cad'
    _def_aux_surface_face_distace_tol: str = 'distance tolerance for faces to be grouped fo coincident surfaces'
    _def_aux_surface_cylinder_angle_tol: str = 'aux surface cylinder angle tolerance (degrees)'
    _def_aux_surface_cylinder_small_angle: str = 'aux surface cylinder small angle tolerance (degrees)'
    _def_aux_surface_cylinder_simple: str = 'apply simple auxiliary surface method (true or false)'
    _def_aux_surface_cone_angle_tol: str = 'aux surface cone angle tolerance (degrees)'
    _def_aux_surface_cone_small_angle: str = 'aux surface cone small angle tolerance (degrees)'
    _def_aux_surface_cone_simple: str = 'apply simple auxiliary surface method (true or false)'
    _def_aux_surface_sphere_angle_tol: str = 'aux surface sphere angle tolerance (degrees)'
    _def_aux_surface_sphere_small_angle: str = 'aux surface sphere small angle tolerance (degrees)'
    _def_aux_surface_sphere_simple: str = 'apply simple auxiliary surface method (true or false)'
    _def_aux_surface_tori_angle_tol: str = 'auxiliary surface: tori angle tolerance (degrees)'
    _def_aux_surface_tori_small_angle: str = 'auxiliary surface: tori small angle tolerance (degrees)'
    _def_aux_surface_tori_aux_surface_simple: str = 'apply simple auxiliary surface method (true or false)'

    # model
    model_set_output_cell_vols: bool = True
    model_implicit_void_for_no_mateirals: bool = True

    _def_model_set_output_cell_vols: str = 'assign volume to output csg file'
    _def_model_implicit_void_for_no_mateirals: str = 'Where no material is assigned, cells set to void be default. Otherwise require material assignment'

    # debugging paths
    debug_cut_fatal_path: str = 'analysis/cut_fatal/cut_fatal.stp'
    debug_cut_warning_path: str = 'analysis/cut_warnings/cut_warning.stp'
    debug_cut_error_path: str = 'analysis/cur_errors/cut_error.stp'
    
    debug_common_fatal_path: str = 'analysis/common_fatal/common_fatal.stp'
    debug_common_warning_path: str = 'analysis/common_warnings/common_warning.stp'
    debug_common_error_path: str = 'analysis/common_errors/common_error.stp'

    debug_splitter_fatal_path: str = 'analysis/splitter_fatal/splitter_fatal.stp'
    debug_splitter_warning_path: str = 'analysis/splitter_warnings/splitter_warning.stp'
    debug_splitter_error_path: str = 'analysis/splitter_errors/splitter_error.stp'

    debug_filter_degenerate_solid_path: str = 'analysis/degenerate_solids/degenerate_solid.stp'

    debug_cut_common_path: str = 'analysis/cut_andcut_common/cut_common.stp'
    debug_volume_conserved_path: str = 'analysis/vol_not_conserved/vol_not_conserved.stp'

    debug_pipe_splitter: str = 'analysis/pre_pipe_splitter/pre_pipe_split.stp'
    debug_concentric_splitter: str = 'analysis/concentric_splitter/pre_concentric_split.stp'
    debug_rectilinear_splitter: str = 'analysis/rectilinear_splitter/pre_rectilinear_split.stp'

    debug_invalid_solid_path: str = 'analysis/invalid_solid/invalid_solid.stp'
    _def_debug_invalid_solid_path: str = 'debugging: path of solids failing the validity check after clean up'

    debug_half_space_failed_to_generate_path: str = 'analysis/half_space_fatal/half_space_fatal.stp'

    debug_failed_to_convert_directory: str = 'analysis/failed_to_convert'

    debug_save_solid: str = 'analysis/solids'

    debug_write_added_solids: bool = False
    _def_debug_write_added_solids: str = 'debugging: write each solid added to the decomposer to a STEP file in added_solids/'

    debug_cad_output_verbosity: int = 1
    debug_logger_verbosity: int = 1

    debug_long_conversion_time: float = 20.0
    debug_long_conversion_time_directory: str = 'analysis/long_conversion_time'
    debug_large_number_of_split_solids: int = 50
    debug_large_number_of_split_solids_directory: str = 'analysis/large_number_of_split_solids'

    debug_trackback_directory: str = 'analysis/trackbacks'

    _def_debug_cut_fatal_path: str = 'debugging: path of failed cut operations, dumps solids to step file'
    _def_debug_cut_warning_path: str = 'debugging: path of cut operation with warning(s)'
    _def_debug_cut_error_path: str = 'debugging: path of cut with error(s)'

    _def_debug_common_fatal_path: str = 'debugging: path of failed common operations, dumps solids to step file'
    _def_debug_common_warning_path: str = 'debugging: path of common operation with warning(s)'
    _def_debug_common_error_path: str = 'debugging: path of common operation with error(s)'

    _def_debug_splitter_fatal_path: str = 'debugging: path of failed splitter operations, dumps solids to step file'
    _def_debug_splitter_warning_path: str = 'debugging: path of splitter operation with warning(s)'
    _def_debug_splitter_error_path: str = 'debugging: path of splitter operation with error(s)'

    _def_debug_filter_degenerate_solid_path: str = 'debugging: path of filtered degenerate solids'

    _def_debug_cut_common_path: str = 'debugging: path of cut and common operation where 2+ commons created'
    _def_debug_volume_conserved_path: str = 'debugging: path of failed conserved volume operations, dumps solids to step file'

    _def_debug_pipe_splitter: str = 'debugging: path of step output following successful split of pipe'
    _def_debug_concentric_splitter: str = 'debugging: path to step output following successful split of concentril second order surfaces'
    _def_debug_rectilinear_splitter: str = 'debugging: path to step output following successful rectilinear pre-split'

    _def_debug_half_space_failed_to_generate_path: str = 'debugging: path of failed to generate half space, dumps to step file'

    _def_debug_failed_to_convert_directory: str = 'debugging: each failed decomposition solid will be saved to this directory'

    _def_debug_save_solid: str = 'debugging: save the solid before attempting to convert'

    _def_debug_cad_output_verbosity: str = 'level of debugging output to step files, either 0 for nothing, 1 for printing and 2 for saving to step'
    _def_debug_logger_verbosity: str = 'level of debugging output to logger, either 0 for nothing, 1 for repoting issues or 2 for step by step reporting'

    _def_debug_long_conversion_time: str = 'debug: save output file to stp showing long conversion times above this value (seconds)'
    _def_debug_long_conversion_time_directory: str = 'debuig: output dir for solids with long conversion times'
    _def_debug_large_number_of_split_solids: str = 'debug: save output file to stp for solids with large numbers of split solids produced'
    _def_debug_large_number_of_split_solids_directory: str = 'debug: save dir for output of solids with large numbers of split solids'

    _def_debug_trackback_directory: str = 'debug: save trackback for multiprocessing to text files for each failed solid to this directory'

    other_nominal_gap: float = 0.0
    _def_other_nominal_gap: str = ''

    def __post_init__(self) -> None:
        """Initialise the table row lookup and table titles used for printing."""
        # define sub category string (first word of the parameter name, upper cased)
        self._rows_lookup: dict[str, list[list[Any]]] = {'DOC': [],
                             'PRESPLIT': [],
                             'CLEANUP': [],
                             'SPLIT': [],
                             'BOOL': [],
                             'SURF': [],
                             'VOID': [],
                             'AUX': [],
                             'MODEL': [],
                             'DEBUG': [],
                             'OTHERS': []}
        
        # table titles
        self._titles: dict[str, str] = {'DOC': 'xcaf document',
                        'PRESPLIT': 'pre-splitting',
                        'CLEANUP': 'pre-decomposition cleanup',
                        'SPLIT': 'recursive splitting of original solids',
                        'BOOL': 'boolean operations: cut, common and split',
                        'SURF': 'surfaces',
                        'VOID': 'void generation',
                        'AUX': 'auxiliary surface generation',
                        'MODEL': 'model',
                        'DEBUG': 'debugging',
                        'OTHERS': 'other parameters'}

    def __getattr__(self, name: str) -> Any:
        """Resolve deprecated upper case parameter names.

        Only called when normal attribute lookup fails.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        Any
            Value of the lower case parameter ``name.lower()``.

        Raises
        ------
        AttributeError
            If ``name`` is not a (deprecated spelling of a) parameter.
        """
        lower = name.lower()
        if lower != name and lower in _FIELD_NAMES:
            warnings.warn(
                f"DecompositionParameters.{name} is deprecated, use .{lower}",
                DeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, lower)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, mapping deprecated upper case parameter names.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Value to set.
        """
        lower = name.lower()
        if lower != name and lower in _FIELD_NAMES:
            warnings.warn(
                f"DecompositionParameters.{name} is deprecated, use .{lower}",
                DeprecationWarning,
                stacklevel=2,
            )
            name = lower
        object.__setattr__(self, name, value)


#: names of all DecompositionParameters fields, used to map deprecated names
_FIELD_NAMES: frozenset[str] = frozenset(
    field.name for field in dataclasses.fields(DecompositionParameters)
)
