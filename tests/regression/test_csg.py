import unittest
import mcio


from unitutils import (
    CustomAssertions,
    CustomParameterChecks
)


class MCIORegressionTestCSGsurfaces(unittest.TestCase, CustomAssertions, CustomParameterChecks):
    """Test suite for MCIO CSG Surface implementations."""

    def setUp(self) -> None:
        self._tolerance = 1e-4

    def generate_surface_and_assert_parameters(self, 
                                                 surface_type: type[mcio.surface.Surface], 
                                                 parameters: dict,
                                                 parameters_openmc: dict = None,
                                                 openmc_equivalent: bool = True) -> None:
        """Helper method to instantiate a surface and assert instance/OpenMC parameters.

        Parameters
        ----------
        surface_type : type[mcio.surface.Surface]
            Class type of the surface to instantiate
        parameters : dict
            Dictionary of parameter names and values for MCIO instance
        parameters_openmc : dict, optional
            Dictionary of expected parameter names and values for OpenMC instance.
            If None, `parameters` dictionary is used.
        openmc_equivalent : bool, optional
            Whether an OpenMC surface equivalent exists for validation, by default True
        """
        # create instance of surface_type
        surf = surface_type(**parameters)

        # assert correct type created
        self.assertTrue(isinstance(surf, surface_type))

        # assert parameters are correct in instance of surface
        self.assertParametersInClassInstance(surf, parameters, self._tolerance)

        # access openmc surface and assert parameters in surface are correct if an equivalent exists
        if openmc_equivalent:
            expected_openmc = parameters_openmc if parameters_openmc is not None else parameters
            self.assertParametersInClassInstance(surf.openmc_surface, expected_openmc, self._tolerance)

    def test_plane(self) -> None:
        parameters = {
            'a': 1.0,
            'b': 0.5,
            'c': 0.75,
            'd': 4.6
        }
        self.generate_surface_and_assert_parameters(mcio.surface.Plane, parameters)

    def test_xplane(self) -> None:
        parameters = {'x0': 11.918}
        self.generate_surface_and_assert_parameters(mcio.surface.XPlane, parameters)

    def test_yplane(self) -> None:
        parameters = {'y0': -5.25}
        self.generate_surface_and_assert_parameters(mcio.surface.YPlane, parameters)

    def test_zplane(self) -> None:
        parameters = {'z0': 3.1415}
        self.generate_surface_and_assert_parameters(mcio.surface.ZPlane, parameters)

    def test_sphere(self) -> None:
        parameters = {
            'x0': 1.0,
            'y0': 2.0,
            'z0': 3.0,
            'radius': 10.5
        }
        parameters_openmc = {
            'x0': 1.0,
            'y0': 2.0,
            'z0': 3.0,
            'r': 10.5
        }
        self.generate_surface_and_assert_parameters(
            mcio.surface.Sphere, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_at_origin(self) -> None:
        parameters = {'radius': 8.0}
        parameters_openmc = {'r': 8.0}
        self.generate_surface_and_assert_parameters(
            mcio.surface.SphereAtOrigin, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_on_xaxis(self) -> None:
        parameters = {'x0': 4.0, 'radius': 12.0}
        parameters_openmc = {'x0': 4.0, 'r': 12.0}
        self.generate_surface_and_assert_parameters(
            mcio.surface.SphereOnXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_on_yaxis(self) -> None:
        parameters = {'y0': -2.5, 'radius': 7.2}
        parameters_openmc = {'y0': -2.5, 'r': 7.2}
        self.generate_surface_and_assert_parameters(
            mcio.surface.SphereOnYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_on_zaxis(self) -> None:
        parameters = {'z0': 9.1, 'radius': 15.0}
        parameters_openmc = {'z0': 9.1, 'r': 15.0}
        self.generate_surface_and_assert_parameters(
            mcio.surface.SphereOnZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_parallel_to_xaxis(self) -> None:
        parameters = {'y0': 2.0, 'z0': 3.0, 'radius': 5.0}
        parameters_openmc = {'y0': 2.0, 'z0': 3.0, 'r': 5.0}
        self.generate_surface_and_assert_parameters(
            mcio.surface.CylinderParallelToXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_parallel_to_yaxis(self) -> None:
        parameters = {'x0': 1.0, 'z0': 4.0, 'radius': 6.0}
        parameters_openmc = {'x0': 1.0, 'z0': 4.0, 'r': 6.0}
        self.generate_surface_and_assert_parameters(
            mcio.surface.CylinderParallelToYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_parallel_to_zaxis(self) -> None:
        parameters = {'x0': -3.0, 'y0': 2.5, 'radius': 4.5}
        parameters_openmc = {'x0': -3.0, 'y0': 2.5, 'r': 4.5}
        self.generate_surface_and_assert_parameters(
            mcio.surface.CylinderParallelToZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_on_xaxis(self) -> None:
        parameters = {'radius': 10.0}
        parameters_openmc = {'r': 10.0}
        self.generate_surface_and_assert_parameters(
            mcio.surface.CylinderOnXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_on_yaxis(self) -> None:
        parameters = {'radius': 12.5}
        parameters_openmc = {'r': 12.5}
        self.generate_surface_and_assert_parameters(
            mcio.surface.CylinderOnYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_on_zaxis(self) -> None:
        parameters = {'radius': 3.2}
        parameters_openmc = {'r': 3.2}
        self.generate_surface_and_assert_parameters(
            mcio.surface.CylinderOnZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_parallel_to_xaxis(self) -> None:
        parameters = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 't_sq': 0.25, 'sheet': 0}
        parameters_openmc = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 'r2': 0.25}
        self.generate_surface_and_assert_parameters(
            mcio.surface.ConeParallelToXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_parallel_to_yaxis(self) -> None:
        parameters = {'x0': 0.5, 'y0': 1.5, 'z0': 2.5, 't_sq': 0.16, 'sheet': 0}
        parameters_openmc = {'x0': 0.5, 'y0': 1.5, 'z0': 2.5, 'r2': 0.16}
        self.generate_surface_and_assert_parameters(
            mcio.surface.ConeParallelToYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_parallel_to_zaxis(self) -> None:
        parameters = {'x0': -1.0, 'y0': 0.0, 'z0': 5.0, 't_sq': 0.36, 'sheet': 0}
        parameters_openmc = {'x0': -1.0, 'y0': 0.0, 'z0': 5.0, 'r2': 0.36}
        self.generate_surface_and_assert_parameters(
            mcio.surface.ConeParallelToZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_on_xaxis(self) -> None:
        parameters = {'x0': 2.0, 't_sq': 0.5, 'sheet': 0}
        parameters_openmc = {'x0': 2.0, 'y0': 0.0, 'z0': 0.0, 'r2': 0.5}
        self.generate_surface_and_assert_parameters(
            mcio.surface.ConeOnXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_on_yaxis(self) -> None:
        parameters = {'y0': -3.0, 't_sq': 0.2, 'sheet': 0}
        parameters_openmc = {'x0': 0.0, 'y0': -3.0, 'z0': 0.0, 'r2': 0.2}
        self.generate_surface_and_assert_parameters(
            mcio.surface.ConeOnYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_on_zaxis(self) -> None:
        parameters = {'z0': 4.0, 't_sq': 0.1, 'sheet': 0}
        parameters_openmc = {'x0': 0.0, 'y0': 0.0, 'z0': 4.0, 'r2': 0.1}
        self.generate_surface_and_assert_parameters(
            mcio.surface.ConeOnZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_ellipsoid(self) -> None:
        parameters = {
            'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 0.5, 'e': 0.0, 
            'f': 0.0, 'g': 1.0, 'x0': 0.0, 'y0': 0.0, 'z0': 0.0
        }
        self.generate_surface_and_assert_parameters(
            mcio.surface.Ellipsoid, 
            parameters, 
            openmc_equivalent=False
        )

    def test_general_quadratic(self) -> None:
        parameters = {
            'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 0.0, 'e': 0.0, 
            'f': 0.0, 'g': 0.0, 'h': 0.0, 'j': 0.0, 'k': -1.0
        }
        self.generate_surface_and_assert_parameters(mcio.surface.GeneralQuadratic, parameters)

    def test_tori_parallel_to_xaxis(self) -> None:
        parameters = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 'a': 10.0, 'b': 2.0, 'c': 2.0}
        self.generate_surface_and_assert_parameters(mcio.surface.ToriParralleltoXAxis, parameters)

    def test_tori_parallel_to_yaxis(self) -> None:
        parameters = {'x0': 0.0, 'y0': 1.0, 'z0': 2.0, 'a': 8.0, 'b': 1.5, 'c': 1.5}
        self.generate_surface_and_assert_parameters(mcio.surface.ToriParralleltoYAxis, parameters)

    def test_tori_parallel_to_zaxis(self) -> None:
        parameters = {'x0': -2.0, 'y0': 0.0, 'z0': 1.0, 'a': 12.0, 'b': 3.0, 'c': 3.0}
        self.generate_surface_and_assert_parameters(mcio.surface.ToriParralleltoZAxis, parameters)


class MCIORegressionTestCSGHalfSpace(unittest.TestCase, CustomAssertions, CustomParameterChecks):
    """Test suite for HalfVolume spaces generated from CSG Surface instances."""

    def setUp(self) -> None:
        self._tolerance = 1e-4

    def generate_halfspace_and_assert_parameters(self, 
                                                 surface_type: type[mcio.surface.Surface], 
                                                 parameters: dict,
                                                 parameters_openmc: dict = None,
                                                 openmc_equivalent: bool = True) -> None:
        """Helper method to instantiate half-spaces and assert properties including OpenMC regions.

        Parameters
        ----------
        surface_type : type[mcio.surface.Surface]
            Class type of the surface to instantiate
        parameters : dict
            Dictionary of parameter names and values for MCIO instance
        parameters_openmc : dict, optional
            Dictionary of expected parameter names and values for OpenMC instance.
            If None, `parameters` dictionary is used.
        openmc_equivalent : bool, optional
            Whether an OpenMC surface equivalent exists for validation, by default True
        """
        surf = surface_type(**parameters)

        half_space_pos: mcio.volume.HalfVolume = +surf
        half_space_neg: mcio.volume.HalfVolume = -surf

        self.assertEqual(half_space_pos.direction, '+')
        self.assertEqual(half_space_neg.direction, '-')

        self.assertTrue(isinstance(half_space_pos.surface, surface_type))
        self.assertTrue(isinstance(half_space_neg.surface, surface_type))

        # Check surface parameters
        self.assertParametersInClassInstance(half_space_pos.surface, parameters, self._tolerance)
        self.assertParametersInClassInstance(half_space_neg.surface, parameters, self._tolerance)

        # Check OpenMC region properties on half-spaces if an equivalent exists
        if openmc_equivalent:
            openmc_region_pos = half_space_pos.openmc_region
            openmc_region_neg = half_space_neg.openmc_region

            self.assertIsNotNone(openmc_region_pos)
            self.assertIsNotNone(openmc_region_neg)

            # Check underlying surface in OpenMC region
            expected_openmc = parameters_openmc if parameters_openmc is not None else parameters
            self.assertParametersInClassInstance(half_space_pos.surface.openmc_surface, expected_openmc, self._tolerance)
            self.assertParametersInClassInstance(half_space_neg.surface.openmc_surface, expected_openmc, self._tolerance)

    def test_plane_halfspace(self) -> None:
        parameters = {'a': 1.0, 'b': 0.5, 'c': 0.75, 'd': 4.6}
        self.generate_halfspace_and_assert_parameters(mcio.surface.Plane, parameters)

    def test_xplane_halfspace(self) -> None:
        parameters = {'x0': 11.918}
        self.generate_halfspace_and_assert_parameters(mcio.surface.XPlane, parameters)

    def test_yplane_halfspace(self) -> None:
        parameters = {'y0': -5.25}
        self.generate_halfspace_and_assert_parameters(mcio.surface.YPlane, parameters)

    def test_zplane_halfspace(self) -> None:
        parameters = {'z0': 3.1415}
        self.generate_halfspace_and_assert_parameters(mcio.surface.ZPlane, parameters)

    def test_sphere_halfspace(self) -> None:
        parameters = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 'radius': 10.5}
        parameters_openmc = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 'r': 10.5}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.Sphere, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_at_origin_halfspace(self) -> None:
        parameters = {'radius': 8.0}
        parameters_openmc = {'r': 8.0}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.SphereAtOrigin, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_on_xaxis_halfspace(self) -> None:
        parameters = {'x0': 4.0, 'radius': 12.0}
        parameters_openmc = {'x0': 4.0, 'r': 12.0}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.SphereOnXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_on_yaxis_halfspace(self) -> None:
        parameters = {'y0': -2.5, 'radius': 7.2}
        parameters_openmc = {'y0': -2.5, 'r': 7.2}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.SphereOnYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_sphere_on_zaxis_halfspace(self) -> None:
        parameters = {'z0': 9.1, 'radius': 15.0}
        parameters_openmc = {'z0': 9.1, 'r': 15.0}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.SphereOnZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_parallel_to_xaxis_halfspace(self) -> None:
        parameters = {'y0': 2.0, 'z0': 3.0, 'radius': 5.0}
        parameters_openmc = {'y0': 2.0, 'z0': 3.0, 'r': 5.0}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.CylinderParallelToXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_parallel_to_yaxis_halfspace(self) -> None:
        parameters = {'x0': 1.0, 'z0': 4.0, 'radius': 6.0}
        parameters_openmc = {'x0': 1.0, 'z0': 4.0, 'r': 6.0}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.CylinderParallelToYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_parallel_to_zaxis_halfspace(self) -> None:
        parameters = {'x0': -3.0, 'y0': 2.5, 'radius': 4.5}
        parameters_openmc = {'x0': -3.0, 'y0': 2.5, 'r': 4.5}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.CylinderParallelToZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_on_xaxis_halfspace(self) -> None:
        parameters = {'radius': 10.0}
        parameters_openmc = {'r': 10.0}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.CylinderOnXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_on_yaxis_halfspace(self) -> None:
        parameters = {'radius': 12.5}
        parameters_openmc = {'r': 12.5}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.CylinderOnYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cylinder_on_zaxis_halfspace(self) -> None:
        parameters = {'radius': 3.2}
        parameters_openmc = {'r': 3.2}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.CylinderOnZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_parallel_to_xaxis_halfspace(self) -> None:
        parameters = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 't_sq': 0.25, 'sheet': 0}
        parameters_openmc = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 'r2': 0.25}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.ConeParallelToXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_parallel_to_yaxis_halfspace(self) -> None:
        parameters = {'x0': 0.5, 'y0': 1.5, 'z0': 2.5, 't_sq': 0.16, 'sheet': 0}
        parameters_openmc = {'x0': 0.5, 'y0': 1.5, 'z0': 2.5, 'r2': 0.16}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.ConeParallelToYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_parallel_to_zaxis_halfspace(self) -> None:
        parameters = {'x0': -1.0, 'y0': 0.0, 'z0': 5.0, 't_sq': 0.36, 'sheet': 0}
        parameters_openmc = {'x0': -1.0, 'y0': 0.0, 'z0': 5.0, 'r2': 0.36}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.ConeParallelToZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_on_xaxis_halfspace(self) -> None:
        parameters = {'x0': 2.0, 't_sq': 0.5, 'sheet': 0}
        parameters_openmc = {'x0': 2.0, 'y0': 0.0, 'z0': 0.0, 'r2': 0.5}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.ConeOnXAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_on_yaxis_halfspace(self) -> None:
        parameters = {'y0': -3.0, 't_sq': 0.2, 'sheet': 0}
        parameters_openmc = {'x0': 0.0, 'y0': -3.0, 'z0': 0.0, 'r2': 0.2}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.ConeOnYAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_cone_on_zaxis_halfspace(self) -> None:
        parameters = {'z0': 4.0, 't_sq': 0.1, 'sheet': 0}
        parameters_openmc = {'x0': 0.0, 'y0': 0.0, 'z0': 4.0, 'r2': 0.1}
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.ConeOnZAxis, 
            parameters, 
            parameters_openmc=parameters_openmc
        )

    def test_ellipsoid_halfspace(self) -> None:
        parameters = {
            'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 0.5, 'e': 0.0, 
            'f': 0.0, 'g': 1.0, 'x0': 0.0, 'y0': 0.0, 'z0': 0.0
        }
        self.generate_halfspace_and_assert_parameters(
            mcio.surface.Ellipsoid, 
            parameters, 
            openmc_equivalent=False
        )

    def test_general_quadratic_halfspace(self) -> None:
        parameters = {
            'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 0.0, 'e': 0.0, 
            'f': 0.0, 'g': 0.0, 'h': 0.0, 'j': 0.0, 'k': -1.0
        }
        self.generate_halfspace_and_assert_parameters(mcio.surface.GeneralQuadratic, parameters)

    def test_tori_parallel_to_xaxis_halfspace(self) -> None:
        parameters = {'x0': 1.0, 'y0': 2.0, 'z0': 3.0, 'a': 10.0, 'b': 2.0, 'c': 2.0}
        self.generate_halfspace_and_assert_parameters(mcio.surface.ToriParralleltoXAxis, parameters)

    def test_tori_parallel_to_yaxis_halfspace(self) -> None:
        parameters = {'x0': 0.0, 'y0': 1.0, 'z0': 2.0, 'a': 8.0, 'b': 1.5, 'c': 1.5}
        self.generate_halfspace_and_assert_parameters(mcio.surface.ToriParralleltoYAxis, parameters)

    def test_tori_parallel_to_zaxis_halfspace(self) -> None:
        parameters = {'x0': -2.0, 'y0': 0.0, 'z0': 1.0, 'a': 12.0, 'b': 3.0, 'c': 3.0}
        self.generate_halfspace_and_assert_parameters(mcio.surface.ToriParralleltoZAxis, parameters)
