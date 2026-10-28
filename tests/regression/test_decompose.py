"""
unittest: regression testing
"""

import os
import copy
import json
import tempfile
import mcio
import unittest
from unittest.mock import patch, mock_open
import cadquery as cq
from dataclasses import dataclass


from unitutils import CustomAssertions, CustomCheckValue, TemporaryWorkingDirectory

from referencebunker import generate_bunker_solids


# run in a temporary working directory, as the decomposer writes files to the working directory
_working_directory = TemporaryWorkingDirectory()


def setUpModule():
    _working_directory.enter()


def tearDownModule():
    _working_directory.exit()


"""
unittest: regression testing
"""

# the directory of the test reference data
REF_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'testdata')


class MCIORegressionTestDecomposeRGB(unittest.TestCase, CustomAssertions, CustomCheckValue):
    def setUp(self):
        # mock-up a materials by rgb json file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_file_path = os.path.join(self.temp_dir.name, "materials.json")
        self.step_file_path = os.path.join(self.temp_dir.name, "my_model.stp")
        self.mcnp_output_path = os.path.join(self.temp_dir.name, "model_mcnp.i")
        self.openmc_output_path = os.path.join(self.temp_dir.name, "model_openmc")
        self.decom_params: mcio.decomposeparams.DecompositionParameters = mcio.decomposeparams.DecompositionParameters()

        # mock-up a material by rgb
        materials_data = {
            "0_0_255": {
                "density": 2.3,
                "composition": {
                    "H": 0.01,
                    "C": 0.001,
                    "O": 0.529107,
                    "Na": 0.016,
                    "Mg": 0.002,
                    "Al": 0.033872,
                    "Si": 0.337021,
                    "K": 0.013,
                    "Ca": 0.044,
                    "Fe": 0.014,
                },
                "number": 1,
                "fraction_type": "wo",
                "name": "Concrete, Ordinary",
                "mcnp": {
                    "sab_cards": [],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
            "255_0_0": {
                "density": 2.266,
                "composition": {"C": 1.0},
                "number": 2,
                "fraction_type": "ao",
                "name": "Carbon, Graphite",
                "mcnp": {
                    "sab_cards": [""],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
            "0_255_0": {
                "density": 2.33,
                "composition": {"Si": 1.0},
                "number": 3,
                "fraction_type": "ao",
                "name": "Silicon",
                "mcnp": {
                    "sab_cards": [],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
            "128_0_255": {
                "density": 7.874,
                "composition": {"Fe": 1.0},
                "number": 4,
                "fraction_type": "ao",
                "name": "Iron",
                "mcnp": {
                    "sab_cards": [],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
            "255_128_0": {
                "density": 8.90,
                "composition": {"Co": 1.0},
                "number": 5,
                "fraction_type": "ao",
                "name": "Cobalt",
                "mcnp": {
                    "sab_cards": [],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
            "0_128_255": {
                "density": 0.997,
                "composition": {"H": 0.666667, "O": 0.333333},
                "number": 6,
                "fraction_type": "ao",
                "name": "Water, Liquid",
                "mcnp": {
                    "sab_cards": ["h-h2o.40t"],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
            "128_255_0": {
                "density": 0.95,
                "composition": {"H": 0.666667, "C": 0.333333},
                "number": 7,
                "fraction_type": "ao",
                "name": "HDPE",
                "mcnp": {
                    "sab_cards": ["h-poly.40t"],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {},
                },
            },
        }
        
        # Write the dynamic string configuration out to the filesystem path
        with open(self.json_file_path, 'w') as f:
            json.dump(materials_data, f, indent=4)

        # mock up a sphere
        bunker_solids = generate_bunker_solids()

        # save sphere to stp
        step_file_writer: mcio.ocp.STEPFileWriter = mcio.ocp.STEPFileWriter(self.decom_params)

        step_file_writer.add_solid(bunker_solids['concrete_walls'], 'bunker', rgb=(0, 0, 255))

        # Sphere 1 - Carbon, Graphite (255_0_0)
        step_file_writer.add_solid(bunker_solids['sphere_1'], 'sphere_1', rgb=(255, 0, 0))

        # Sphere 2 - Silicon (0_255_0)
        step_file_writer.add_solid(bunker_solids['sphere_2'], 'sphere_2', rgb=(0, 255, 0))

        # Sphere 3 - Iron (128_0_255)
        step_file_writer.add_solid(bunker_solids['sphere_3'], 'sphere_3', rgb=(128, 0, 255))

        # Sphere 4 - Cobalt (255_128_0)
        step_file_writer.add_solid(bunker_solids['sphere_4'], 'sphere_4', rgb=(255, 128, 0))

        # Sphere 5 - HDPE (128_255_0)
        step_file_writer.add_solid(bunker_solids['sphere_5'], 'sphere_5', rgb=(128, 255, 0))

        # write step files to temp directory            
        step_file_writer.write(self.step_file_path)

    def tearDown(self):
        # clean up the temporary directory
        self.temp_dir.cleanup()

    def test_run_decomposer(self):
        decomposer = mcio.decompose.Decomposer(
            step_file_paths=[self.step_file_path],
            materials_file_path=self.json_file_path
        )

        try:
            decomposer.run()
        except Exception as e:
            self.fail(f"decomposer.run() failed: {e}")

        decomposer.model.write_mcnp(self.mcnp_output_path)
        decomposer.model.write_openmc(self.openmc_output_path)

        # six material cells, one per solid, plus void cells
        cells = decomposer.model.cells
        material_cells = [cell for cell in cells if cell.material is not None]
        self.assertEqual(len(material_cells), 6)
        self.assertGreater(len(cells), 6)

        # materials assigned by colour
        expected_materials = {
            'bunker': 'Concrete, Ordinary',
            'sphere_1': 'Carbon, Graphite',
            'sphere_2': 'Silicon',
            'sphere_3': 'Iron',
            'sphere_4': 'Cobalt',
            'sphere_5': 'HDPE',
        }
        materials_by_solid = {cell.comment.split('/')[-1]: cell.material.name for cell in material_cells}
        self.assertEqual(materials_by_solid, expected_materials)

        # every surface used by a cell is written in the surface block, and no line is too long
        with open(self.mcnp_output_path, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            self.assertLessEqual(len(line), 80, line)
        start = next(i for i, l in enumerate(lines) if '====== SURFACES' in l)
        end = next(i for i, l in enumerate(lines) if '====== MATERIALS' in l)
        surface_numbers = {int(l.split()[0]) for l in lines[start:end] if l[:1].isdigit()}
        for cell in cells:
            for surface in cell.volume.surfaces:
                self.assertIn(surface.number, surface_numbers)

        # OpenMC model files written
        for file_name in ('geometry.xml', 'materials.xml', 'settings.xml'):
            self.assertTrue(os.path.isfile(os.path.join(self.openmc_output_path, file_name)), file_name)

    def test_cells_have_no_repeated_terms(self):
        # operands of nested Ands are de-duplicated, so no cell repeats a term (e.g. "-12 -12")
        decomposer = mcio.decompose.Decomposer(
            step_file_paths=[self.step_file_path],
            materials_file_path=self.json_file_path
        )
        decomposer.run(cores=1)

        def repeated_terms(boolean):
            if isinstance(boolean, mcio.volume.HalfVolume):
                return 0
            hashes = [sb.__hash_string__() for sb in boolean.sub_booleans if isinstance(sb, mcio.volume.HalfVolume)]
            return len(hashes) - len(set(hashes)) + sum(repeated_terms(sb) for sb in boolean.sub_booleans)

        for cell in decomposer.model.cells:
            boolean = cell.volume.boolean if isinstance(cell.volume, mcio.volume.Volume) else cell.volume
            self.assertEqual(repeated_terms(boolean), 0, cell.comment)


@dataclass
class AnnulusParameters:
    radius_inner: float 
    radius_outer: float
    extrusion_offset: float
    extrusion_length: float
    workplane: str

    def __hash__(self):
        string = ''
        for param, val in self.__dict__.items():
            string += f'{param}_{val:.12e}' if isinstance(val, (float, int)) else f'{param}_{val}'

        return hash(string)


class MCIORegressionTestDecomposeFromOCPSolidsCylinders(unittest.TestCase, CustomAssertions, CustomCheckValue):
    def setUp(self):
        # create two annuli
        annulus_params_1 = AnnulusParameters(
            radius_inner=100,
            radius_outer=150,
            extrusion_offset=-250,
            extrusion_length=375,
            workplane=None
        )

        annulus_params_2 = AnnulusParameters(
            radius_inner=40,
            radius_outer=80,
            extrusion_offset=250,
            extrusion_length=-125,
            workplane=None
        )

        self.params_and_solids = {}
        for workplane in ['XY', 'YZ', 'ZX']:
            for annulus_param in [annulus_params_1, annulus_params_2]:
                params = copy.deepcopy(annulus_param)
                params.workplane = workplane

                solid = (
                    cq.Workplane(params.workplane)
                    .workplane(offset=params.extrusion_offset)
                    .circle(params.radius_outer)
                    .circle(params.radius_inner)
                    .extrude(params.extrusion_length)
                ).val().wrapped

                self.params_and_solids[params] = solid

                # step_file_writer = mcio.ocpfilereadwrite.STEPFileWriter()
                # step_file_writer.add_solid(solid, f'workplane_{workplane}')
                # step_file_writer.write(f'workplane_{workplane}.stp')

                # decomposer = mcio.decompose.Decomposer()
                # decomposer.add_solid(solid, 'solid')
                # decomposer.run(cores=1)
                # decomposer.model.write_mcnp(f'model_workplane_{workplane}.i')

        # define tolerance
        self._tolerance = 1e-6

    def test_add_cylinders_from_cadquery_ocp_annulus(self):
        for params, solid in self.params_and_solids.items():
            decomposer = mcio.decompose.Decomposer()
            decomposer.add_solid(solid, 'solid')
            decomposer.run(cores=1)

            decomposition_solid = decomposer.decomposition_solids[0]

            surface_lookup = {
                'YZ': (mcio.surface.XPlane, mcio.surface.CylinderParallelToXAxis),
                'ZX': (mcio.surface.YPlane, mcio.surface.CylinderParallelToYAxis),
                'XY': (mcio.surface.ZPlane, mcio.surface.CylinderParallelToZAxis)
                }

            # check boolean type in volume
            assert isinstance(decomposition_solid.mcio_cell.volume.boolean, mcio.volume.And)
            assert len(decomposition_solid.mcio_cell.volume.boolean.sub_booleans) == 4

            for half_volume in decomposition_solid.mcio_cell.volume.boolean.sub_booleans:
                assert isinstance(half_volume, mcio.volume.HalfVolume)
                assert isinstance(half_volume.surface, surface_lookup[params.workplane])

                if isinstance(half_volume.surface, mcio.surface.XPlane):
                    # find bottom and top of cylinder
                    x_min = min(params.extrusion_offset, params.extrusion_offset + params.extrusion_length)
                    x_max = max(params.extrusion_offset, params.extrusion_offset + params.extrusion_length)

                    # check x plane parmeters
                    assert self.check_values_within_relative_tol (half_volume.surface.x0, x_min, tol=self._tolerance) or \
                           self.check_values_within_relative_tol(half_volume.surface.x0, x_max, tol=self._tolerance)

                    # check surface directions for planes
                    if self.check_values_within_relative_tol(half_volume.surface.x0, x_min, tol=self._tolerance):
                        assert half_volume.direction == '+'
                    elif self.check_values_within_relative_tol(half_volume.surface.x0, x_max, tol=self._tolerance):
                        assert half_volume.direction == '-'


                if isinstance(half_volume.surface, mcio.surface.YPlane):
                    y_min = min(params.extrusion_offset, params.extrusion_offset + params.extrusion_length)
                    y_max = max(params.extrusion_offset, params.extrusion_offset + params.extrusion_length)

                    assert self.check_values_within_relative_tol(half_volume.surface.y0, y_min, tol=self._tolerance) or \
                            self.check_values_within_relative_tol(half_volume.surface.y0, y_max, tol=self._tolerance)

                    if self.check_values_within_relative_tol(half_volume.surface.y0, y_min, tol=self._tolerance):
                        assert half_volume.direction == '+'
                    elif self.check_values_within_relative_tol(half_volume.surface.y0, y_max, tol=self._tolerance):
                        assert half_volume.direction == '-'

                if isinstance(half_volume.surface, mcio.surface.ZPlane):
                    z_min = min(params.extrusion_offset, params.extrusion_offset + params.extrusion_length)
                    z_max = max(params.extrusion_offset, params.extrusion_offset + params.extrusion_length)

                    assert self.check_values_within_relative_tol (half_volume.surface.z0, params.extrusion_offset, tol=self._tolerance) or \
                           self.check_values_within_relative_tol(half_volume.surface.z0, params.extrusion_offset + params.extrusion_length, tol=self._tolerance)

                    if self.check_values_within_relative_tol(half_volume.surface.z0, z_min, tol=self._tolerance):
                        assert half_volume.direction == '+'
                    elif self.check_values_within_relative_tol(half_volume.surface.z0, z_max, tol=self._tolerance):
                        assert half_volume.direction == '-'


                # if isinstance(half_volume.surface, mcio.surface.CylinderParallelToXAxis(, mcio.surface.CylinderParallelToYAxis, mcio.surface.CylinderParallelToZAxis)):
                #     assert self.check_values_within_relative_tol(half_volume.surface.radius, params.radius_inner, tol=self._tolerance) or \
                #            self.check_values_within_relative_tol(half_volume.surface.radius, params.radius_outer, tol=self._tolerance)


@dataclass
class SphereParameters:
    radius_inner: float
    radius_outer: float
    center_x: float
    center_y: float
    center_z: float

    def __hash__(self):
        string = ""
        for param, val in self.__dict__.items():
            string += (
                f"{param}_{val:.12e}"
                if isinstance(val, (float, int))
                else f"{param}_{val}"
            )
        return hash(string)


class MCIORegressionTestDecomposeFromOCPSolidsSpheres(
    unittest.TestCase, CustomAssertions, CustomCheckValue
):
    def setUp(self):
        # Define base test parameters for spherical shells / hollow spheres
        sphere_params_1 = SphereParameters(
            radius_inner=50.0,
            radius_outer=120.0,
            center_x=10.0,
            center_y=-20.0,
            center_z=30.0,
        )

        sphere_params_2 = SphereParameters(
            radius_inner=25.0,
            radius_outer=75.0,
            center_x=-100.0,
            center_y=200.0,
            center_z=0.0,
        )

        self.params_and_solids = {}
        self.temp_dir = tempfile.TemporaryDirectory()

        for sphere_param in [sphere_params_1, sphere_params_2]:
            params = copy.deepcopy(sphere_param)

            # Construct solid dynamically via CadQuery/OCP without STEP roundtripping
            outer_sphere = cq.Workplane("XY").sphere(params.radius_outer)
            inner_sphere = cq.Workplane("XY").sphere(params.radius_inner)

            solid = (
                outer_sphere.cut(inner_sphere)
                .translate((params.center_x, params.center_y, params.center_z))
                .val()
                .wrapped
            )

            self.params_and_solids[params] = solid

            # STEP export verification for diagnostic trace
            step_file_writer = mcio.ocpfilereadwrite.STEPFileWriter()
            step_file_writer.add_solid(solid, f"sphere_{params.radius_outer}_{params.radius_inner}")
            step_file_writer.write(os.path.join(self.temp_dir.name, f"sphere_{params.radius_outer}_{params.radius_inner}.stp"))

            # Decompose directly from TopoDS_Solid object
            decomposer = mcio.decompose.Decomposer()
            decomposer.add_solid(solid, "solid")
            decomposer.run(cores=1)
            decomposer.model.write_mcnp(
                os.path.join(self.temp_dir.name, f"model_sphere_{params.radius_outer}_{params.radius_inner}.i")
            )

        self._tolerance = 1e-6

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_spheres_from_cadquery_ocp_shell(self):
        for params, solid in self.params_and_solids.items():
            decomposer = mcio.decompose.Decomposer()
            decomposer.add_solid(solid, "solid_1")
            decomposer.run(cores=1)

            decomposition_solid = decomposer.decomposition_solids[0]

            # Verify boolean combination structure
            assert isinstance(
                decomposition_solid.mcio_cell.volume.boolean, mcio.volume.And
            )
            # A spherical shell bounded by inner and outer spheres should consist of 2 half-spaces
            assert (
                len(
                    decomposition_solid.mcio_cell.volume.boolean.sub_booleans
                )
                == 2
            )

            for (
                half_volume
            ) in decomposition_solid.mcio_cell.volume.boolean.sub_booleans:
                assert isinstance(half_volume, mcio.volume.HalfVolume)
                assert isinstance(half_volume.surface, mcio.surface.Sphere)

                # Validate radius matching
                assert self.check_values_within_relative_tol(
                    half_volume.surface.radius,
                    params.radius_inner,
                    tol=self._tolerance,
                ) or self.check_values_within_relative_tol(
                    half_volume.surface.radius,
                    params.radius_outer,
                    tol=self._tolerance,
                )

                # Validate center coordinates
                assert self.check_values_within_relative_tol(
                    half_volume.surface.x0, params.center_x, tol=self._tolerance
                )
                assert self.check_values_within_relative_tol(
                    half_volume.surface.y0, params.center_y, tol=self._tolerance
                )
                assert self.check_values_within_relative_tol(
                    half_volume.surface.z0, params.center_z, tol=self._tolerance
                )
