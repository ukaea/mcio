"""ADVANTG input file generation.

Contains classes for writing ADVANTG text input files: :class:`AdvantgCard`,
:class:`AdvantgHeader` and the main :class:`Advantg` parameter container.
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Union

from .checkvals import numbers_to_strings
from .formatting import formatting


class AdvantgCard:
    "advantg card, which writes generic advantg cards as a string"

    def __init__(self, designator: str, parameters: list, comment: str = None) -> None:
        """advantg card for text input file for advantg

        Parameters
        ----------
        designator : str
            parameter label designator
        parameters : list
            parameter value(s)
        comment : str, optional
            comment for parameter, by default None
        """
        self.designator: str = designator
        self.parameters: list = parameters
        self.comment: str = comment

    @property
    def as_string(self) -> str:
        """returns the string of the advantg card

        Returns
        -------
        str
            string of the advantg card
        """

        # combine strings for card output
        output: str = f"{self.designator:<{formatting.ADVANTG_INDENT}}"
        output += " ".join(numbers_to_strings(self.parameters))
        output += "\n"

        # return string
        return output


class AdvantgHeader:
    """advantg header, which writes a text comment header for advantg files"""

    def __init__(self, header: str) -> None:
        """advantg header, which represents a text comment header for advantg files

        Parameters
        ----------
        header : str
            the header string to be added to the header object.
        """
        self.header: str = header

    @property
    def as_string(self) -> str:
        """returns the string of the header

        Returns
        -------
        str
            the string of the header
        """

        # define a line break
        line_break = "# " + "=" * (formatting.ADVANTG_MAX_CHARS - 2) + "\n"

        # define the header
        header = line_break[: formatting.ADVANTG_INDENT] + f" {self.header.upper()} "
        header += line_break[formatting.ADVANTG_INDENT + len(self.header) + 2 :]

        # return the combined header and line breaks above and below
        return "".join([line_break, header, line_break])


class Advantg:
    """ADVANTG parameters, written as a text input file for ADVANTG.

    Parameter names match the ADVANTG input designators. ``None``, ``True``
    and ``False`` are written as ``none``, ``true`` and ``false``.

    Parameters
    ----------
    model : str, optional
        Transport model type; only 'mcnp' writes model options, by default mcnp
    method : str, optional
        Variance reduction method; 'fwcadis' writes FW-CADIS options, by default fwcadis
    outputs : list[str], optional
        Output formats to generate, by default None (``['mcnp']``)
    mcnp_input : str, optional
        Path to the MCNP input file, by default None
    mcnp_tallies : list[int], optional
        MCNP tally numbers used as the response, by default None
    mcnp_material_names : dict[int, str], optional
        Display names per MCNP material number, by default None (``{0: 'void'}``)
    mcnp_min_source_samples : float, optional
        Minimum number of source samples, by default 1e6
    mcnp_max_source_samples : float, optional
        Maximum number of source samples, by default 1e8
    mcnp_target_source_density : float, optional
        Target source sample density, by default 100.0
    mcnp_max_point_sources : float, optional
        Maximum number of point sources, by default 20.0
    mcnp_force_point_source : bool, optional
        Whether to force a point source treatment, by default False
    mcnp_min_rays_per_face : int, optional
        Minimum number of rays per mesh face, by default 10
    mcnp_ray_directions : str, optional
        Ray tracing directions, by default x y z
    mcnp_num_rays : int, optional
        Number of rays, by default 1
    mcnp_lost_rays : int, optional
        Number of lost rays allowed (not currently written), by default 10
    mcnp_mix_tolerance : float, optional
        Material mixing tolerance (not currently written), by default 0.01
    mcnp_unfolding_origin : int, optional
        Unfolding origin option (not currently written), by default 3
    mcnp_unfolding_safe : bool, optional
        Unfolding safe option (not currently written), by default False
    fwcadis_spatial_treatment : str, optional
        FW-CADIS spatial treatment, by default pathlength
    fwcadis_response_weighting : bool, optional
        Whether to apply FW-CADIS response weighting, by default True
    fwcadis_min_response : float, optional
        FW-CADIS minimum response, by default 0.0
    fwcadis_max_response : float or str, optional
        FW-CADIS maximum response, by default inf
    dx_adjoint : bool, optional
        Whether to write adjoint DX output, by default False
    dx_forward : bool, optional
        Whether to write forward DX output, by default False
    mesh_refinement : str, optional
        Mesh refinement method; 'uniform' also writes the width options, by default mcnp
    mesh_x : list[float], optional
        Coarse mesh x boundaries, by default None
    mesh_y : list[float], optional
        Coarse mesh y boundaries, by default None
    mesh_z : list[float], optional
        Coarse mesh z boundaries, by default None
    mesh_x_ints : int, optional
        Number of x intervals per coarse mesh division, by default 1
    mesh_y_ints : int, optional
        Number of y intervals per coarse mesh division, by default 1
    mesh_z_ints : int, optional
        Number of z intervals per coarse mesh division, by default 1
    mesh_max_width : float, optional
        Maximum mesh width, by default 1.0
    mesh_x_max_width : float, optional
        Maximum mesh width in x, by default 1.0
    mesh_y_max_width : float, optional
        Maximum mesh width in y, by default 1.0
    mesh_z_max_width : float, optional
        Maximum mesh width in z, by default 1.0
    mesh_min_width : float, optional
        Minimum mesh width, by default 1.0
    mesh_x_min_width : float, optional
        Minimum mesh width in x, by default 1.0
    mesh_y_min_width : float, optional
        Minimum mesh width in y, by default 1.0
    mesh_z_min_width : float, optional
        Minimum mesh width in z, by default 1.0
    anisn_library : str, optional
        Multigroup ANISN library name, by default 27n19g
    anisn_upscatter : bool, optional
        Whether to include upscatter, by default False
    anisn_zaid_map : dict[int, str], optional
        ZAID mapping for the multigroup library, by default None
    denovo_discretization : str, optional
        Denovo spatial discretization, by default sc
    denovo_x_blocks : int, optional
        Number of Denovo blocks in x, by default 1
    denovo_y_blocks : int, optional
        Number of Denovo blocks in y, by default 1
    denovo_z_blocks : int, optional
        Number of Denovo blocks in z, by default 1
    denovo_energy_sets : int, optional
        Number of Denovo energy sets, by default 1
    denovo_partition_upscatter : bool, optional
        Whether to partition upscatter, by default False
    denovo_quadrature : str, optional
        Denovo quadrature type, by default qr
    denovo_quad_order : int, optional
        Quadrature order, by default 10
    denovo_ldfe_order : int, optional
        LDFE quadrature order, by default 1
    denovo_quad_num_azi : int, optional
        Number of azimuthal quadrature angles, by default 4
    denovo_quad_num_polar : int, optional
        Number of polar quadrature angles, by default 4
    denovo_quad_polar_axis : str, optional
        Quadrature polar axis, by default z
    denovo_quad_file : str, optional
        Quadrature file (ADVANTG 3.0 only), by default no_file_required
    denovo_pn_order : int, optional
        Legendre (Pn) scattering order, by default 3
    denovo_transport_correction : str, optional
        Transport correction, by default diagonal
    denovo_mc_first_collision : bool, optional
        Whether to use Monte Carlo first collision source (ADVANTG 3.0 only), by default
        False
    denovo_mc_num_particles : int, optional
        Number of first collision particles (ADVANTG 3.0 only), by default 10000
    denovo_solver : str, optional
        Within-group solver, by default gmres
    denovo_multigroup_solver : str, optional
        Multigroup solver, by default gauss_seidel
    denovo_preconditioner : bool, optional
        Preconditioner option (ADVANTG 3.0 only), by default None
    denovo_two_grid : bool, optional
        Whether to use two-grid acceleration (ADVANTG 3.0 only), by default False
    denovo_krylov_space : int, optional
        Krylov subspace size, by default 20
    denovo_max_iterations : int, optional
        Maximum number of iterations, by default 100
    denovo_tolerance : float, optional
        Convergence tolerance, by default 0.001
    denovo_upscatter_tolerance : float, optional
        Upscatter convergence tolerance, by default 0.01
    denovo_upscatter_inner_iterations : int, optional
        Upscatter inner iterations, by default 10
    denovo_upscatter_inner_tolerance : float, optional
        Upscatter inner tolerance, by default 0.01
    denovo_first_group : int, optional
        First energy group solved, by default 0
    denovo_verbose : bool, optional
        Whether Denovo output is verbose, by default True
    mcnp_input_template : str, optional
        MCNP input template for output (not currently written), by default None
    mcnp_mxspln : int, optional
        MCNP MXSPLN value (not currently written), by default 100
    mcnp_ww_ratio : float, optional
        Weight window ratio (not currently written), by default 5.0
    mcnp_sb_type : str, optional
        Source biasing type (not currently written), by default space_energy
    mcnp_min_sb_samples : float, optional
        Minimum source biasing samples (not currently written), by default 1e6
    mcnp_max_sb_samples : float, optional
        Maximum source biasing samples (not currently written), by default 1e8
    mcnp_target_sb_density : float, optional
        Target source biasing sample density (not currently written), by default 10000.0
    mcnp_user_sb_sampling : bool, optional
        Whether to use user source biasing sampling (not currently written), by default
        False
    mcnp_num_wgt_samples : float, optional
        Number of weight samples (not currently written), by default 1e7
    mcnp_ww_collapse_factor : float, optional
        Weight window collapse factor (not currently written), by default 1.0
    silo_response_ids : bool, optional
        Whether to write response ids to silo, by default True
    silo_source_ids : bool, optional
        Whether to write source ids to silo, by default True
    silo_source_strength : bool, optional
        Whether to write source strength to silo, by default True
    silo_ww : bool, optional
        Whether to write weight windows to silo, by default False
    silo_edit_reactions : str, optional
        Silo edit reactions; written only if set, by default None
    advantg_version : str, optional
        ADVANTG version; '3.0' adds the 3.0-only Denovo options, by default 3.2.0
    """

    def __init__(
        self,
        model: str = "mcnp",
        method: str = "fwcadis",
        outputs: Optional[List[str]] = None,
        mcnp_input: Optional[str] = None,
        mcnp_tallies: Optional[List[int]] = None,
        mcnp_material_names: Optional[Dict[int, str]] = None,
        mcnp_min_source_samples: float = 1e6,
        mcnp_max_source_samples: float = 1e8,
        mcnp_target_source_density: float = 100.0,
        mcnp_max_point_sources: float = 20.0,
        mcnp_force_point_source: bool = False,
        mcnp_min_rays_per_face: int = 10,
        mcnp_ray_directions: str = "x y z",
        mcnp_num_rays: int = 1,
        mcnp_lost_rays: int = 10,
        mcnp_mix_tolerance: float = 0.01,
        mcnp_unfolding_origin: int = 3,
        mcnp_unfolding_safe: bool = False,
        fwcadis_spatial_treatment: str = "pathlength",
        fwcadis_response_weighting: bool = True,
        fwcadis_min_response: float = 0.0,
        fwcadis_max_response: Union[float, str] = "inf",
        dx_adjoint: bool = False,
        dx_forward: bool = False,
        mesh_refinement: str = "mcnp",
        mesh_x: Optional[List[float]] = None,
        mesh_y: Optional[List[float]] = None,
        mesh_z: Optional[List[float]] = None,
        mesh_x_ints: int = 1,
        mesh_y_ints: int = 1,
        mesh_z_ints: int = 1,
        mesh_max_width: float = 1.0,
        mesh_x_max_width: float = 1.0,
        mesh_y_max_width: float = 1.0,
        mesh_z_max_width: float = 1.0,
        mesh_min_width: float = 1.0,
        mesh_x_min_width: float = 1.0,
        mesh_y_min_width: float = 1.0,
        mesh_z_min_width: float = 1.0,
        anisn_library: str = "27n19g",
        anisn_upscatter: bool = False,
        anisn_zaid_map: Optional[Dict[int, str]] = None,
        denovo_discretization: str = "sc",
        denovo_x_blocks: int = 1,
        denovo_y_blocks: int = 1,
        denovo_z_blocks: int = 1,
        denovo_energy_sets: int = 1,
        denovo_partition_upscatter: bool = False,
        denovo_quadrature: str = "qr",
        denovo_quad_order: int = 10,
        denovo_ldfe_order: int = 1,
        denovo_quad_num_azi: int = 4,
        denovo_quad_num_polar: int = 4,
        denovo_quad_polar_axis: str = "z",
        denovo_quad_file: str = "no_file_required",
        denovo_pn_order: int = 3,
        denovo_transport_correction: str = "diagonal",
        denovo_mc_first_collision: bool = False,
        denovo_mc_num_particles: int = 10000,
        denovo_solver: str = "gmres",
        denovo_multigroup_solver: str = "gauss_seidel",
        denovo_preconditioner: Optional[bool] = None,
        denovo_two_grid: bool = False,
        denovo_krylov_space: int = 20,
        denovo_max_iterations: int = 100,
        denovo_tolerance: float = 0.001,
        denovo_upscatter_tolerance: float = 0.01,
        denovo_upscatter_inner_iterations: int = 10,
        denovo_upscatter_inner_tolerance: float = 0.01,
        denovo_first_group: int = 0,
        denovo_verbose: bool = True,
        mcnp_input_template: Optional[str] = None,
        mcnp_mxspln: int = 100,
        mcnp_ww_ratio: float = 5.0,
        mcnp_sb_type: str = "space_energy",
        mcnp_min_sb_samples: float = 1e6,
        mcnp_max_sb_samples: float = 1e8,
        mcnp_target_sb_density: float = 1e4,
        mcnp_user_sb_sampling: bool = False,
        mcnp_num_wgt_samples: float = 1e7,
        mcnp_ww_collapse_factor: float = 1.0,
        silo_response_ids: bool = True,
        silo_source_ids: bool = True,
        silo_source_strength: bool = True,
        silo_ww: bool = False,
        silo_edit_reactions: Optional[str] = None,
        advantg_version: str = "3.2.0",
    ) -> None:
        """Initialise the ADVANTG parameters."""
        # Defaults for mutable inputs
        if outputs is None:
            outputs = ["mcnp"]
        if mcnp_material_names is None:
            mcnp_material_names = {0: "void"}

        # advantg version
        self.advantg_version: str = advantg_version

        # driver options
        self.model: str = model
        self.method: str = method
        self.outputs: List[str] = outputs

        # model options (mcnp)
        self.mcnp_input: Optional[str] = mcnp_input
        self.mcnp_tallies: Optional[List[int]] = mcnp_tallies
        self.mcnp_material_names: Dict[int, str] = mcnp_material_names
        self.mcnp_min_source_samples: float = mcnp_min_source_samples
        self.mcnp_max_source_samples: float = mcnp_max_source_samples
        self.mcnp_target_source_density: float = mcnp_target_source_density
        self.mcnp_max_point_sources: float = mcnp_max_point_sources
        self.mcnp_force_point_source: bool = mcnp_force_point_source
        self.mcnp_min_rays_per_face: int = mcnp_min_rays_per_face
        self.mcnp_ray_directions: str = mcnp_ray_directions
        self.mcnp_num_rays: int = mcnp_num_rays
        self.mcnp_lost_rays: int = mcnp_lost_rays
        self.mcnp_mix_tolerance: float = mcnp_mix_tolerance
        self.mcnp_unfolding_origin: int = mcnp_unfolding_origin
        self.mcnp_unfolding_safe: bool = mcnp_unfolding_safe

        # fw cadis
        self.fwcadis_spatial_treatment: str = fwcadis_spatial_treatment
        self.fwcadis_response_weighting: bool = fwcadis_response_weighting
        self.fwcadis_min_response: float = fwcadis_min_response
        self.fwcadis_max_response: Union[float, str] = fwcadis_max_response

        # dx
        self.dx_adjoint: bool = dx_adjoint
        self.dx_forward: bool = dx_forward

        # spatial mesh
        self.mesh_refinement: str = mesh_refinement
        self.mesh_x: Optional[List[float]] = mesh_x
        self.mesh_y: Optional[List[float]] = mesh_y
        self.mesh_z: Optional[List[float]] = mesh_z
        self.mesh_x_ints: int = mesh_x_ints
        self.mesh_y_ints: int = mesh_y_ints
        self.mesh_z_ints: int = mesh_z_ints
        self.mesh_max_width: float = mesh_max_width
        self.mesh_x_max_width: float = mesh_x_max_width
        self.mesh_y_max_width: float = mesh_y_max_width
        self.mesh_z_max_width: float = mesh_z_max_width
        self.mesh_min_width: float = mesh_min_width
        self.mesh_x_min_width: float = mesh_x_min_width
        self.mesh_y_min_width: float = mesh_y_min_width
        self.mesh_z_min_width: float = mesh_z_min_width

        # multigroup library options
        self.anisn_library: str = anisn_library
        self.anisn_upscatter: bool = anisn_upscatter
        self.anisn_zaid_map: Optional[Dict[int, str]] = anisn_zaid_map

        # solver options (denovo)
        self.denovo_discretization: str = denovo_discretization
        self.denovo_x_blocks: int = denovo_x_blocks
        self.denovo_y_blocks: int = denovo_y_blocks
        self.denovo_z_blocks: int = denovo_z_blocks
        self.denovo_energy_sets: int = denovo_energy_sets
        self.denovo_partition_upscatter: bool = denovo_partition_upscatter
        self.denovo_quadrature: str = denovo_quadrature
        self.denovo_quad_order: int = denovo_quad_order
        self.denovo_ldfe_order: int = denovo_ldfe_order
        self.denovo_quad_num_azi: int = denovo_quad_num_azi
        self.denovo_quad_num_polar: int = denovo_quad_num_polar
        self.denovo_quad_polar_axis: str = denovo_quad_polar_axis
        self.denovo_quad_file: str = denovo_quad_file
        self.denovo_pn_order: int = denovo_pn_order
        self.denovo_transport_correction: str = denovo_transport_correction
        self.denovo_mc_first_collision: bool = denovo_mc_first_collision
        self.denovo_mc_num_particles: int = denovo_mc_num_particles
        self.denovo_solver: str = denovo_solver
        self.denovo_multigroup_solver: str = denovo_multigroup_solver
        self.denovo_preconditioner: Optional[bool] = denovo_preconditioner
        self.denovo_two_grid: bool = denovo_two_grid
        self.denovo_krylov_space: int = denovo_krylov_space
        self.denovo_max_iterations: int = denovo_max_iterations
        self.denovo_tolerance: float = denovo_tolerance
        self.denovo_upscatter_tolerance: float = denovo_upscatter_tolerance
        self.denovo_upscatter_inner_iterations: int = denovo_upscatter_inner_iterations
        self.denovo_upscatter_inner_tolerance: float = denovo_upscatter_inner_tolerance
        self.denovo_first_group: int = denovo_first_group
        self.denovo_verbose: bool = denovo_verbose

        # output options (mcnp)
        self.mcnp_input_template: Optional[str] = mcnp_input_template
        self.mcnp_mxspln: int = mcnp_mxspln
        self.mcnp_ww_ratio: float = mcnp_ww_ratio
        self.mcnp_sb_type: str = mcnp_sb_type
        self.mcnp_min_sb_samples: float = mcnp_min_sb_samples
        self.mcnp_max_sb_samples: float = mcnp_max_sb_samples
        self.mcnp_target_sb_density: float = mcnp_target_sb_density
        self.mcnp_user_sb_sampling: bool = mcnp_user_sb_sampling
        self.mcnp_num_wgt_samples: float = mcnp_num_wgt_samples
        self.mcnp_ww_collapse_factor: float = mcnp_ww_collapse_factor

        # output options (silo)
        self.silo_response_ids: bool = silo_response_ids
        self.silo_source_ids: bool = silo_source_ids
        self.silo_ww: bool = silo_ww
        self.silo_edit_reactions: Optional[str] = silo_edit_reactions
        self.silo_source_strength: bool = silo_source_strength

    def _section(self, header: str, designators: list[str]) -> str:
        """parser for sections containing input parmeters, which produced
        a string of the input file for advantage.

        Parameters
        ----------
        header : str
            The header text for a given section
        designators : list[str]
            A list of parameter designators

        Returns
        -------
        str
            A string of the section with header and parameters in the input
            format for an advantg input file
        """

        # set up header card and convert to string
        output: str = AdvantgHeader(header).as_string

        parameter_converter = {None: "none", False: "false", True: "true"}

        # driver options
        for designator in designators:
            # dynamically get designators
            parameters = self.__dict__[f"{designator}"]

            # swap our literal types for strings of litterals, e.g. False -> "False"
            if parameters in parameter_converter:
                parameters = parameter_converter[parameters]

            # set up advantg cards and convert them to strings, combinbing into output
            if isinstance(parameters, (list, tuple)):
                output += AdvantgCard(designator, parameters).as_string
            elif isinstance(parameters, dict):
                parameters_parsed = [f"{k} {v}" for k, v in parameters.items()]
                output += AdvantgCard(designator, parameters_parsed).as_string
            else:
                output += AdvantgCard(designator, [parameters]).as_string

        # return output string
        return output

    @property
    def advantg_string(self) -> str:
        """combines each section string to an overall string for the input
        file for advantg input.

        Returns
        -------
        str
            string of the advantg input file
        """
        # list of parameters in each section
        drivers = ["model", "method", "outputs"]

        mcnp = [
            "mcnp_input",
            "mcnp_tallies",
            "mcnp_material_names",
            "mcnp_min_source_samples",
            "mcnp_max_source_samples",
            "mcnp_target_source_density",
            "mcnp_max_point_sources",
            "mcnp_force_point_source",
            "mcnp_min_rays_per_face",
            "mcnp_ray_directions",
            "mcnp_num_rays",
        ]

        fwcadis = [
            "fwcadis_spatial_treatment",
            "fwcadis_response_weighting",
            "fwcadis_min_response",
            "fwcadis_max_response",
        ]

        dx = ["dx_adjoint", "dx_forward"]

        spatial_mesh = [
            "mesh_refinement",
            "mesh_x",
            "mesh_x_ints",
            "mesh_y",
            "mesh_y_ints",
            "mesh_z",
            "mesh_z_ints",
        ]
        if self.mesh_refinement == "uniform":
            spatial_mesh.extend(
                [
                    "mesh_max_width",
                    "mesh_x_max_width",
                    "mesh_y_max_width",
                    "mesh_z_max_width",
                    "mesh_min_width",
                    "mesh_x_min_width",
                    "mesh_y_min_width",
                    "mesh_z_min_width",
                ]
            )

        multigroup_library = ["anisn_library", "anisn_upscatter"]
        if self.anisn_zaid_map is not None:
            multigroup_library.append(self.anisn_zaid_map)

        denovo = [
            "denovo_discretization",
            "denovo_x_blocks",
            "denovo_y_blocks",
            "denovo_z_blocks",
            "denovo_energy_sets",
            "denovo_partition_upscatter",
            "denovo_quadrature",
            "denovo_quad_order",
            "denovo_ldfe_order",
            "denovo_quad_num_azi",
            "denovo_quad_num_polar",
            "denovo_quad_polar_axis",
            "denovo_pn_order",
            "denovo_transport_correction",
            "denovo_solver",
            "denovo_multigroup_solver",
            "denovo_krylov_space",
            "denovo_max_iterations",
            "denovo_tolerance",
            "denovo_upscatter_tolerance",
            "denovo_upscatter_inner_iterations",
            "denovo_upscatter_inner_tolerance",
            "denovo_first_group",
            "denovo_verbose",
        ]

        denovo_adventg_3_0 = [
            "denovo_quad_file",
            "denovo_mc_first_collision",
            "denovo_mc_num_particles",
            "denovo_preconditioner",
            "denovo_two_grid",
        ]

        silo = [
            "silo_response_ids",
            "silo_source_ids",
            "silo_ww",
            "silo_source_strength",
        ]

        if self.silo_edit_reactions:
            silo.append("silo_edit_reactions")

        # add sections to output string
        output = ""

        # add driver options
        output += self._section("driver options", drivers)

        # add mcnp model options
        if self.model == "mcnp":
            output += self._section("model options (mcnp)", mcnp)
        elif self.model == "sword":
            output += self._section("model options (sword)", ["NONE CODED"])

        # add fwcadis options
        if self.method == "fwcadis":
            output += self._section("method options (fwcadis)", fwcadis)

        # add dx options
        output += self._section("dx options", dx)

        # add spatial mesh options
        output += self._section("spatial mesh options", spatial_mesh)

        # add multigroup options
        output += self._section("multigroup library options", multigroup_library)

        # add denovo

        if self.advantg_version == "3.0":
            output += self._section("denovo", denovo + denovo_adventg_3_0)
        else:
            output += self._section("denovo", denovo)

        # add silo
        output += self._section("silo", silo)

        # return combined string of input file
        return output

    def write_advantg(self, output_filename: str | os.PathLike) -> None:
        """Write the ADVANTG input file to a text file.

        Parameters
        ----------
        output_filename : str | os.PathLike
            Path to the output text file
        """

        # write the advantg to a text file
        with open(output_filename, "w", encoding="utf-8") as output_file:
            output_file.write(self.advantg_string)
