import unittest
import math
import tempfile
import os
import json

from mcio.material import (
    Material,
    _materials_by_colour_from_json,
    _materials_by_label_from_json,
    _materials_from_json,
    _material_lookup,
    _normalise_label
)
from mcio.data import (
    ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION,
    ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION
)
from mcio.decomposeparams import DecompositionParameters


from unitutils import CustomAssertions


class CustomAssertions:
    def assertRelativeToleranceAlmostEqual(self, a, b, tol):
        msg = f'{a} and {b} are not within relative tolerance {tol}'
        assert math.isclose(a, b, rel_tol=tol), msg

    def assertListsRelativeToleranceAlmostEqual(self, list_a, list_b, tol):
        for a, b in zip(list_a, list_b):
            msg = f'{a} and {b} are not within relative tolerance {tol}'
            assert math.isclose(a, b, rel_tol=tol), msg


class MCIOUnitTestMaterial(unittest.TestCase, CustomAssertions):
    def setUp(self):
        # Base density for water tests
        self.density_water = 0.9718

        # 1. Basic water by element atom fraction
        self.composition_water_by_element = {'H': 2.0, 'O': 1.0}
        self.mat_water_element = Material(
            density=self.density_water,
            composition=self.composition_water_by_element,
            fraction_type='ao'
        )

        # 2. Water explicitly defined using specific isotopes
        self.composition_water_by_isotope = {'H1': 2.0, 'O16': 1.0}
        self.mat_water_isotope = Material(
            density=self.density_water,
            composition=self.composition_water_by_isotope,
            fraction_type='ao'
        )

        # 3. Water with repeated elements and mixed isotopes
        # H x2, O x2, plus explicit isotopes H1, H2, O16, O17
        self.composition_water_merged = {
            'H': 2.0,
            'O': 2.0,
            'H1': 1.0,
            'H2': 0.5,
            'O16': 1.0,
            'O17': 0.1
        }

        self.mat_water_merged = Material(
            density=self.density_water,
            composition=self.composition_water_merged,
            fraction_type='ao'
        )

        self.requited_tol = 1e-6

    def test_combine_isotopes_elemental_water(self):
        """Test splitting elemental H and O into natural isotopic abundances."""
        combined = self.mat_water_element._combine_isotopes()

        # Check H isotopes match NIST natural abundances scaled by H count (2.0)
        expected_h1 = 2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['H']['H1']
        expected_h2 = 2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['H']['H2']
        self.assertRelativeToleranceAlmostEqual(combined['H1'], expected_h1, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['H2'], expected_h2, self.requited_tol)

        # Check O isotopes match NIST natural abundances scaled by O count (1.0)
        expected_o16 = 1.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O16']
        expected_o17 = 1.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O17']
        expected_o18 = 1.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O18']
        self.assertRelativeToleranceAlmostEqual(combined['O16'], expected_o16, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['O17'], expected_o17, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['O18'], expected_o18, self.requited_tol)

    def test_combine_isotopes_explicit_water(self):
        """Test material defined explicitly with isotope fractions."""
        combined = self.mat_water_isotope._combine_isotopes()

        self.assertEqual(combined['H1'], 2.0)
        self.assertEqual(combined['O16'], 1.0)
        self.assertNotIn('H2', combined)
        self.assertNotIn('O17', combined)

    def test_combine_isotopes_merged_repeated_definitions(self):
        """Test that elemental fractions and repeated isotope definitions merge correctly."""
        combined = self.mat_water_merged._combine_isotopes()

        # Expected totals:
        # H1 = (H x 2.0 * nat_abundance(H1)) + explicit H1 (1.0)
        expected_h1 = (2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['H']['H1']) + 1.0
        # H2 = (H x 2.0 * nat_abundance(H2)) + explicit H2 (0.5)
        expected_h2 = (2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['H']['H2']) + 0.5

        # O16 = (O x 2.0 * nat_abundance(O16)) + explicit O16 (1.0)
        expected_o16 = (2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O16']) + 1.0
        # O17 = (O x 2.0 * nat_abundance(O17)) + explicit O17 (0.1)
        expected_o17 = (2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O17']) + 0.1
        # O18 = (O x 2.0 * nat_abundance(O18))
        expected_o18 = 2.0 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O18']

        self.assertRelativeToleranceAlmostEqual(combined['H1'], expected_h1, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['H2'], expected_h2, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['O16'], expected_o16, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['O17'], expected_o17, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(combined['O18'], expected_o18, self.requited_tol)


class MCIOUnitTestMaterialFromJSONRGB(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_file_path = os.path.join(self.temp_dir.name, "materials.json")
        self.step_file_path = os.path.join(self.temp_dir.name, "my_model.stp")
        self.decom_params: DecompositionParameters = DecompositionParameters()

        # Mock material data by RGB with variable digit lengths
        self.materials_data = {
            "000_255_000": {
                "density": 0.997,
                "composition": {
                    "H": 0.666667,
                    "O": 0.333333
                },
                "number": 5,
                "fraction_type": "ao",
                "name": "Water, Liquid",
                "mcnp": {
                    "sab_cards": ["h-h2o.40t"],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {}
                }
            },
            "5_128_27": {
                "density": 1e-2,
                "composition": {
                    "H": 1.2,
                    "H1": 5E-1,
                    "O": 0.6,
                    "O17": 0.02,
                    "Cu": 0.25
                },
                "number": 100.0,
                "fraction_type": "wo",
                "name": "copper and hydrogen test material",
                "mcnp": {
                    "sab_cards": ["h-h2o.40t"],
                    "libs": {},
                    "library": ".32c",
                    "library_by_isotope": {'O17': '.80c'}
                }
            }
        }

        # Write out to temporary JSON file
        with open(self.json_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.materials_data, f, indent=4)

        # define required tolerance
        self.requited_tol = 1e-6

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_materials_by_colour_from_json_formatting(self):
        """Test that keys are zero-padded to 3-digit RGB strings."""
        materials = _materials_by_colour_from_json(self.json_file_path)

        self.assertIn("000_255_000", materials)
        # "5_128_27" should be reformatted to "005_128_027"
        self.assertIn("005_128_027", materials)
        self.assertNotIn("5_128_27", materials)

    def test_materials_by_colour_composition(self):
        """Test raw composition matching and correct isotopic combination."""
        materials = _materials_by_colour_from_json(self.json_file_path)

        # 1. Test Water (atom fraction / 'ao')
        water_mat = materials["000_255_000"]
        self.assertEqual(water_mat.composition, self.materials_data["000_255_000"]["composition"])

        water_isotopes = water_mat._combine_isotopes()
        expected_h1_ao = 0.666667 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['H']['H1']
        expected_o16_ao = 0.333333 * ISOTOPIC_COMPOSITION_BY_ATOM_FRACTION['O']['O16']
        self.assertRelativeToleranceAlmostEqual(water_isotopes['H1'], expected_h1_ao, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(water_isotopes['O16'], expected_o16_ao, self.requited_tol)

        # 2. Test Copper/Hydrogen material (weight fraction / 'wo')
        cu_mat = materials["005_128_027"]
        self.assertEqual(cu_mat.composition, self.materials_data["5_128_27"]["composition"])

        cu_isotopes = cu_mat._combine_isotopes()
        # H1 weight fraction = (H x 1.2 * wo_nat_abundance(H1)) + explicit H1 (0.5)
        expected_h1_wo = (1.2 * ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION['H']['H1']) + 0.5
        # O17 weight fraction = (O x 0.6 * wo_nat_abundance(O17)) + explicit O17 (0.02)
        expected_o17_wo = (0.6 * ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION['O']['O17']) + 0.02
        # Cu63 weight fraction = Cu x 0.25 * wo_nat_abundance(Cu63)
        expected_cu63_wo = 0.25 * ISOTOPIC_COMPOSITION_BY_WEIGHT_FRACTION['Cu']['Cu63']

        self.assertRelativeToleranceAlmostEqual(cu_isotopes['H1'], expected_h1_wo, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(cu_isotopes['O17'], expected_o17_wo, self.requited_tol)
        self.assertRelativeToleranceAlmostEqual(cu_isotopes['Cu63'], expected_cu63_wo, self.requited_tol)

    def test_mcnp_parameters(self):
        """Test parsing of MCNPMaterialParameters including library, library_by_isotope, and sab_cards."""
        materials = _materials_by_colour_from_json(self.json_file_path)

        # Water MCNP options
        water_mcnp = materials["000_255_000"].mcnp
        self.assertEqual(water_mcnp.library, ".32c")
        self.assertEqual(water_mcnp.sab_cards, ["h-h2o.40t"])
        self.assertEqual(water_mcnp.library_by_isotope, {})

        # Copper/Hydrogen MCNP options
        cu_mcnp = materials["005_128_027"].mcnp
        self.assertEqual(cu_mcnp.library, ".32c")
        self.assertEqual(cu_mcnp.sab_cards, ["h-h2o.40t"])
        self.assertEqual(cu_mcnp.library_by_isotope, {'O17': '.80c'})

class MCIOUnitTestMaterialFromJSONLabel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.json_file_path = os.path.join(self.temp_dir.name, "materials.json")
        self.decom_params: DecompositionParameters = DecompositionParameters()

        water = {
            "density": 0.997,
            "composition": {"H": 2.0, "O": 1.0},
            "number": 1,
            "fraction_type": "ao",
            "name": "Water, Liquid"
        }
        steel = {
            "density": 7.93,
            "composition": {"Fe": 0.7, "Cr": 0.2, "Ni": 0.1},
            "number": 2,
            "fraction_type": "wo",
            "name": "steel"
        }

        # labels written with and without underscores
        self.materials_data = {"_m1_": water, "m2": steel}

        with open(self.json_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.materials_data, f, indent=4)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, data):
        path = os.path.join(self.temp_dir.name, "other.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return path

    def test_water(self):
        method, materials = _materials_from_json(self.json_file_path)
        self.assertEqual(method, 'label')
        water = materials["m1"]
        self.assertEqual(water.name, "Water, Liquid")
        self.assertEqual(water.number, 1)
        self.assertAlmostEqual(water.density, 0.997)

    def test_labels_have_underscores_removed(self):
        method, materials = _materials_from_json(self.json_file_path)
        self.assertEqual(sorted(materials), ["m1", "m2"])
        self.assertEqual(sorted(_materials_by_label_from_json(self.json_file_path)), ["m1", "m2"])

    def test_rgb_keys_detected(self):
        path = self._write({"1_2_3": self.materials_data["m2"], "255_0_0": self.materials_data["_m1_"]})
        method, materials = _materials_from_json(path)
        self.assertEqual(method, 'rgb')
        self.assertEqual(sorted(materials), ["001_002_003", "255_000_000"])

    def test_mixed_keys_raise(self):
        path = self._write({"1_2_3": self.materials_data["m2"], "m1": self.materials_data["_m1_"]})
        with self.assertRaises(KeyError):
            _materials_from_json(path)

    def test_repeated_rgb_keys_raise(self):
        path = self._write({"1_2_3": self.materials_data["m2"], "001_002_003": self.materials_data["_m1_"]})
        with self.assertRaises(KeyError):
            _materials_from_json(path)

    def test_lookup_by_label(self):
        _, materials = _materials_from_json(self.json_file_path)
        material = _material_lookup('label', 0, 0, 0, ['assembly', 'tank_m1_', 'solid'], materials, self.decom_params)
        self.assertIs(material, materials["m1"])
        material = _material_lookup('label', 0, 0, 0, ['pipe_m2_'], materials, self.decom_params)
        self.assertIs(material, materials["m2"])

    def test_lookup_label_repeated_in_tree(self):
        # the same label on a container and its solid is one label
        _, materials = _materials_from_json(self.json_file_path)
        material = _material_lookup('label', 0, 0, 0, ['container_m2_', 'solid_m2_'], materials, self.decom_params)
        self.assertIs(material, materials["m2"])

    def test_lookup_two_labels_raise(self):
        _, materials = _materials_from_json(self.json_file_path)
        with self.assertRaises(Exception):
            _material_lookup('label', 0, 0, 0, ['container_m1_', 'solid_m2_'], materials, self.decom_params)

    def test_lookup_missing_label_raises(self):
        _, materials = _materials_from_json(self.json_file_path)
        with self.assertRaises(ValueError):
            _material_lookup('label', 0, 0, 0, ['solid_m3_'], materials, self.decom_params)

    def test_lookup_void(self):
        _, materials = _materials_from_json(self.json_file_path)
        self.assertIsNone(_material_lookup('label', 0, 0, 0, ['air_void_'], materials, self.decom_params))
        self.assertIsNone(_material_lookup('label', 0, 0, 0, ['gap_m0_'], materials, self.decom_params))
        # void labels win over material labels
        self.assertIsNone(_material_lookup('label', 0, 0, 0, ['air_void_', 'solid_m1_'], materials, self.decom_params))

    def test_lookup_no_label_implicit_void(self):
        _, materials = _materials_from_json(self.json_file_path)
        self.decom_params.model_implicit_void_for_no_mateirals = True
        self.assertIsNone(_material_lookup('label', 0, 0, 0, ['assembly', 'solid'], materials, self.decom_params))

    def test_lookup_no_label_without_implicit_void_raises(self):
        _, materials = _materials_from_json(self.json_file_path)
        self.decom_params.model_implicit_void_for_no_mateirals = False
        with self.assertRaises(Exception):
            _material_lookup('label', 0, 0, 0, ['assembly', 'solid'], materials, self.decom_params)

    def test_numeric_label_key(self):
        path = self._write({"1": self.materials_data["_m1_"]})
        method, materials = _materials_from_json(path)
        self.assertEqual(method, 'label')
        material = _material_lookup('label', 0, 0, 0, ['tank_m1_'], materials, self.decom_params)
        self.assertEqual(material.number, 1)

    def test_numeric_label_key_by_label_loader(self):
        path = self._write({"2": self.materials_data["m2"], "_m1_": self.materials_data["_m1_"]})
        self.assertEqual(sorted(_materials_by_label_from_json(path)), ["m1", "m2"])

    def test_normalise_label(self):
        for label in ("_m3_", "m3", "3", "_3_"):
            self.assertEqual(_normalise_label(label), "m3")

    def test_lookup_by_rgb(self):
        path = self._write({"5_128_27": self.materials_data["m2"]})
        _, materials = _materials_from_json(path)
        material = _material_lookup('rgb', 5, 128, 27, ['solid'], materials, self.decom_params)
        self.assertEqual(material.name, "steel")
        with self.assertRaises(Exception):
            _material_lookup('rgb', 1, 2, 3, ['solid'], materials, self.decom_params)

    def test_lookup_unknown_method_raises(self):
        _, materials = _materials_from_json(self.json_file_path)
        with self.assertRaises(ValueError):
            _material_lookup('name', 0, 0, 0, ['solid_m1_'], materials, self.decom_params)
