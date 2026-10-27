
import math
import unittest


from unitutils import CustomAssertions, TemporaryWorkingDirectory


# run in a temporary working directory, as the decomposer writes files to the working directory
_working_directory = TemporaryWorkingDirectory()


def setUpModule():
    _working_directory.enter()


def tearDownModule():
    _working_directory.exit()
from mcio.cell import Cell
from mcio.decomposeparams import DecompositionParameters
from mcio.decompose import Decomposer
from mcio.surface import (
    XPlane,
    YPlane,
    ZPlane,
    Plane,
    ToriParralleltoXAxis,
    Sphere,
    SphereAtOrigin
)
from mcio.geometric import BoundingBox
from mcio.ocpproperties import _volume_of_ocp_solids


class MCIOUnitTestOCPHalfSpace(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self._extent = 50.0
        self._total_volume = (2 * self._extent) ** 3
        self._bounding_box = BoundingBox(
            x_min=-self._extent,
            x_max=self._extent,
            y_min=-self._extent,
            y_max=self._extent,
            z_min=-self._extent,
            z_max=self._extent
            )

        self._expected_planes = (
            XPlane(x0=-self._extent),
            XPlane(x0=self._extent),
            YPlane(y0=-self._extent),
            YPlane(y0=self._extent),
            ZPlane(z0=-self._extent),
            ZPlane(z0=self._extent)
        )

        # det up default decompose parameters along with 
        self._decom_params: DecompositionParameters = DecompositionParameters()
        self._decom_params.bounding_box_multiplier_half_space = 1.0

    def test_mcio_halfspace_x_plane_to_ocp_halfspace(self) -> None:
        # set constants
        expected_volume_pos = self._total_volume / 4.0
        expected_volume_neg = self._total_volume - expected_volume_pos

        # surface
        surf = XPlane(x0=self._extent / 2.0)

        # half_volume
        vol_pos = +surf
        vol_neg = -surf

        # ocp_solid
        ocp_solid_pos = vol_pos.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)
        ocp_solid_neg = vol_neg.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)

        # calculate volume
        calculated_volume_pos = _volume_of_ocp_solids([ocp_solid_pos])
        calculated_volume_neg = _volume_of_ocp_solids([ocp_solid_neg])

        # check expected volume
        self.assertRelativeToleranceAlmostEqual(expected_volume_pos, calculated_volume_pos, 1e-4)
        self.assertRelativeToleranceAlmostEqual(expected_volume_neg, calculated_volume_neg, 1e-4)

        # check inversion
        expected_surfaces_pos = set(
            (
            surf.__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        expected_surfaces_neg = set(
                    (
                    surf.__hash_string__(),
                    XPlane(x0=-self._extent).__hash_string__(),
                    YPlane(y0=-self._extent).__hash_string__(),
                    YPlane(y0=self._extent).__hash_string__(),
                    ZPlane(z0=-self._extent).__hash_string__(),
                    ZPlane(z0=self._extent).__hash_string__()
                    )
                )

        # decompose solid produced
        decomposer_pos = Decomposer()
        decomposer_pos.add_solid(ocp_solid_pos, 'ocp_solid_pos')
        decomposer_pos.run(cores=1)

        decomposer_neg = Decomposer()
        decomposer_neg.add_solid(ocp_solid_neg, 'ocp_solid_neg')
        decomposer_neg.run(cores=1)

        # get surfaces produced
        cell_pos: Cell = decomposer_pos.model.cells[0]
        cell_neg: Cell = decomposer_neg.model.cells[0]

        for surface in cell_pos.surfaces:
            assert surface.__hash_string__() in expected_surfaces_pos

        for surface in cell_neg.surfaces:
            assert surface.__hash_string__() in expected_surfaces_neg

    def test_mcio_halfspace_y_plane_to_ocp_halfspace(self) -> None:
        # set constants
        expected_volume_pos = self._total_volume / 4.0
        expected_volume_neg = self._total_volume - expected_volume_pos

        # surface
        surf = YPlane(y0=self._extent / 2.0)

        # half_volume
        vol_pos = +surf
        vol_neg = -surf

        # ocp_solid
        ocp_solid_pos = vol_pos.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)
        ocp_solid_neg = vol_neg.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)

        # calculate volume
        calculated_volume_pos = _volume_of_ocp_solids([ocp_solid_pos])
        calculated_volume_neg = _volume_of_ocp_solids([ocp_solid_neg])

        # check expected volume
        self.assertRelativeToleranceAlmostEqual(expected_volume_pos, calculated_volume_pos, 1e-4)
        self.assertRelativeToleranceAlmostEqual(expected_volume_neg, calculated_volume_neg, 1e-4)

        # check inversion
        expected_surfaces_pos = set(
            (
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            surf.__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        expected_surfaces_neg = set(
            (
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            surf.__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        # decompose solid produced
        decomposer_pos = Decomposer()
        decomposer_pos.add_solid(ocp_solid_pos, 'ocp_solid_pos')
        decomposer_pos.run(cores=1)

        decomposer_neg = Decomposer()
        decomposer_neg.add_solid(ocp_solid_neg, 'ocp_solid_neg')
        decomposer_neg.run(cores=1)

        # get surfaces produced
        cell_pos: Cell = decomposer_pos.model.cells[0]
        cell_neg: Cell = decomposer_neg.model.cells[0]

        for surface in cell_pos.surfaces:
            assert surface.__hash_string__() in expected_surfaces_pos

        for surface in cell_neg.surfaces:
            assert surface.__hash_string__() in expected_surfaces_neg

    def test_mcio_halfspace_z_plane_to_ocp_halfspace(self) -> None:
        # set constants
        expected_volume_pos = self._total_volume / 4.0
        expected_volume_neg = self._total_volume - expected_volume_pos

        # surface
        surf = ZPlane(z0=self._extent / 2.0)

        # half_volume
        vol_pos = +surf
        vol_neg = -surf

        # ocp_solid
        ocp_solid_pos = vol_pos.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)
        ocp_solid_neg = vol_neg.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)

        # calculate volume
        calculated_volume_pos = _volume_of_ocp_solids([ocp_solid_pos])
        calculated_volume_neg = _volume_of_ocp_solids([ocp_solid_neg])

        # check expected volume
        self.assertRelativeToleranceAlmostEqual(expected_volume_pos, calculated_volume_pos, 1e-4)
        self.assertRelativeToleranceAlmostEqual(expected_volume_neg, calculated_volume_neg, 1e-4)

        # check inversion
        expected_surfaces_pos = set(
            (
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            surf.__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        expected_surfaces_neg = set(
            (
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            surf.__hash_string__()
            )
        )

        # decompose solid produced
        decomposer_pos = Decomposer()
        decomposer_pos.add_solid(ocp_solid_pos, 'ocp_solid_pos')
        decomposer_pos.run(cores=1)

        decomposer_neg = Decomposer()
        decomposer_neg.add_solid(ocp_solid_neg, 'ocp_solid_neg')
        decomposer_neg.run(cores=1)

        # get surfaces produced
        cell_pos: Cell = decomposer_pos.model.cells[0]
        cell_neg: Cell = decomposer_neg.model.cells[0]

        for surface in cell_pos.surfaces:
            assert surface.__hash_string__() in expected_surfaces_pos

        for surface in cell_neg.surfaces:
            assert surface.__hash_string__() in expected_surfaces_neg

    def test_mcio_halfspace_plane_to_ocp_halfspace(self) -> None:
        # set constants
        a, b, c, d = 1, 1, 0, 10

        norm = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        a_norm = a / norm
        b_norm = b / norm
        c_norm = c / norm
        d_norm = d / norm

        expected_volume_pos = ((math.sqrt(2 * self._extent ** 2) - math.sqrt(2 * (d/2) ** 2)) ** 2) * 2 * self._extent

        print(expected_volume_pos)
        expected_volume_neg = self._total_volume - expected_volume_pos

        # surface
        surf = Plane(a=a_norm, b=b_norm, c=c_norm, d=d_norm)

        # half_volume
        vol_pos = +surf
        vol_neg = -surf

        # ocp_solid
        ocp_solid_pos = vol_pos.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)
        ocp_solid_neg = vol_neg.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)

        # calculate volume
        calculated_volume_pos = _volume_of_ocp_solids([ocp_solid_pos])
        calculated_volume_neg = _volume_of_ocp_solids([ocp_solid_neg])

        # check expected volume
        self.assertRelativeToleranceAlmostEqual(expected_volume_pos, calculated_volume_pos, 1e-4)
        self.assertRelativeToleranceAlmostEqual(expected_volume_neg, calculated_volume_neg, 1e-4)

        # check inversion
        expected_surfaces_pos = set(
            (
            surf.__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        expected_surfaces_neg = set(
            (
            surf.__hash_string__(),
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__(),
            )
        )

        # decompose solid produced
        decomposer_pos = Decomposer()
        decomposer_pos.add_solid(ocp_solid_pos, 'ocp_solid_pos')
        decomposer_pos.run(cores=1)

        decomposer_neg = Decomposer()
        decomposer_neg.add_solid(ocp_solid_neg, 'ocp_solid_neg')
        decomposer_neg.run(cores=1)

        # get surfaces produced
        cell_pos: Cell = decomposer_pos.model.cells[0]
        cell_neg: Cell = decomposer_neg.model.cells[0]

        for surface in cell_pos.surfaces:
            if isinstance(surface, (XPlane, YPlane, ZPlane)):
                msg = f'surfaces not found in look up : {surface.__hash_string__()}\n -> ' 
                msg += '\n -> '.join(expected_surfaces_pos)
                assert surface.__hash_string__() in expected_surfaces_pos, msg
            elif isinstance(surface, Plane):
                self.assertRelativeToleranceAlmostEqual(surface.a, a_norm, tol=1e-4)
                self.assertRelativeToleranceAlmostEqual(surface.b, b_norm, tol=1e-4)
                self.assertRelativeToleranceAlmostEqual(surface.c, c_norm, tol=1e-4)
                self.assertRelativeToleranceAlmostEqual(surface.d, d_norm, tol=1e-4)

        for surface in cell_neg.surfaces:
            if isinstance(surface, (XPlane, YPlane, ZPlane)):
                msg = f'surfaces not found in look up : {surface.__hash_string__()}\n -> ' 
                msg += '\n -> '.join(expected_surfaces_neg)
                assert surface.__hash_string__() in expected_surfaces_neg, msg
            elif isinstance(surface, Plane):
                self.assertRelativeToleranceAlmostEqual(surface.a, a_norm, tol=1e-4)
                self.assertRelativeToleranceAlmostEqual(surface.b, b_norm, tol=1e-4)
                self.assertRelativeToleranceAlmostEqual(surface.c, c_norm, tol=1e-4)
                self.assertRelativeToleranceAlmostEqual(surface.d, d_norm, tol=1e-4)

    def test_mcio_halfspace_sphere_to_ocp_halfspace(self) -> None:
        # set constants
        radius = 25.0
        expected_volume_neg = (4.0/3.0) * math.pi * radius ** 3
        expected_volume_pos = self._total_volume - expected_volume_neg

        # surface
        surf = Sphere(radius=radius, x0=10, y0=5, z0=-2)

        # half_volume
        vol_pos = +surf
        vol_neg = -surf

        # ocp_solid
        ocp_solid_pos = vol_pos.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)
        ocp_solid_neg = vol_neg.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)

        # calculate volume
        calculated_volume_pos = _volume_of_ocp_solids([ocp_solid_pos])
        calculated_volume_neg = _volume_of_ocp_solids([ocp_solid_neg])

        # check expected volume
        self.assertRelativeToleranceAlmostEqual(expected_volume_pos, calculated_volume_pos, 1e-4)
        self.assertRelativeToleranceAlmostEqual(expected_volume_neg, calculated_volume_neg, 1e-4)

        # check inversion
        expected_surfaces_pos = set(
            (
            surf.__hash_string__(),
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        expected_surfaces_neg = surf.__hash_string__()

        # decompose solid produced
        decomposer_pos = Decomposer()
        decomposer_pos.add_solid(ocp_solid_pos, 'ocp_solid_pos')
        decomposer_pos.run(cores=1)

        decomposer_neg = Decomposer()
        decomposer_neg.add_solid(ocp_solid_neg, 'ocp_solid_neg')
        decomposer_neg.run(cores=1)

        # get surfaces produced
        cell_pos: Cell = decomposer_pos.model.cells[0]
        cell_neg: Cell = decomposer_neg.model.cells[0]

        for surface in cell_pos.surfaces:
            self.assertvalueInIterable(surface.__hash_string__(), expected_surfaces_pos)

        for surface in cell_neg.surfaces:
            msg = f'surface {surface.__hash_string__()} not equal to {expected_surfaces_neg}'
            assert surface.__hash_string__() == expected_surfaces_neg, msg

    def test_mcio_halfspace_sphere_at_origin_to_ocp_halfspace(self) -> None:
        # set constants
        radius = 25.0
        expected_volume_neg = (4.0/3.0) * math.pi * radius ** 3
        expected_volume_pos = self._total_volume - expected_volume_neg

        # surface
        surf = SphereAtOrigin(radius=radius)

        # half_volume
        vol_pos = +surf
        vol_neg = -surf

        # ocp_solid
        ocp_solid_pos = vol_pos.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)
        ocp_solid_neg = vol_neg.ocp_solid(bounding_box=self._bounding_box, decom_params=self._decom_params)

        # calculate volume
        calculated_volume_pos = _volume_of_ocp_solids([ocp_solid_pos])
        calculated_volume_neg = _volume_of_ocp_solids([ocp_solid_neg])

        # check expected volume
        self.assertRelativeToleranceAlmostEqual(expected_volume_pos, calculated_volume_pos, 1e-4)
        self.assertRelativeToleranceAlmostEqual(expected_volume_neg, calculated_volume_neg, 1e-4)

        # check inversion
        expected_surfaces_pos = set(
            (
            surf.__hash_string__(),
            XPlane(x0=-self._extent).__hash_string__(),
            XPlane(x0=self._extent).__hash_string__(),
            YPlane(y0=-self._extent).__hash_string__(),
            YPlane(y0=self._extent).__hash_string__(),
            ZPlane(z0=-self._extent).__hash_string__(),
            ZPlane(z0=self._extent).__hash_string__()
            )
        )

        expected_surfaces_neg = surf.__hash_string__()

        # decompose solid produced
        decomposer_pos = Decomposer()
        decomposer_pos.add_solid(ocp_solid_pos, 'ocp_solid_pos')
        decomposer_pos.run(cores=1)

        decomposer_neg = Decomposer()
        decomposer_neg.add_solid(ocp_solid_neg, 'ocp_solid_neg')
        decomposer_neg.run(cores=1)

        # get surfaces produced
        cell_pos: Cell = decomposer_pos.model.cells[0]
        cell_neg: Cell = decomposer_neg.model.cells[0]

        for surface in cell_pos.surfaces:
            self.assertvalueInIterable(surface.__hash_string__(), expected_surfaces_pos)

        for surface in cell_neg.surfaces:
            msg = f'surface {surface.__hash_string__()} not equal to {expected_surfaces_neg}'
            assert surface.__hash_string__() == expected_surfaces_neg, msg
