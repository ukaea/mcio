import unittest
from pathlib import Path
import tempfile
import mcio
import xml.etree.ElementTree as ET


from unitutils import CustomAssertions


class MCIOUnitTestTally(unittest.TestCase, CustomAssertions):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)

        surf_sphere = mcio.surface.Sphere(radius=50)
        surf_sphere.openmc_surface.id = 1
        surf_sphere_g = mcio.surface.Sphere(radius=100)

        vol_sph = -surf_sphere
        vol_v = +surf_sphere & -surf_sphere_g
        vol_g = +surf_sphere_g

        cell_sph = mcio.cell.Cell(volume=vol_sph)
        cell_sph.openmc_cell.id = 1
        cell_v = mcio.cell.Cell(volume=vol_v)
        cell_v.openmc_cell.id = 2
        cell_g = mcio.cell.Cell(volume=vol_g, graveyard=True)

        tally_flux = mcio.tally.VolumeFluenceTally(cells=[cell_sph, cell_v])
        tally_surfflux = mcio.tally.SurfaceFluenceTally(surfaces=[surf_sphere])
        tally_current = mcio.tally.SurfaceCurrent(surfaces=[surf_sphere])
        tally_dose = mcio.tally.VolumeFluenceTally(cells=[cell_sph,cell_v], 
                                                dose_conversion_factors=[[5, 10, 20], [0.15, 0.30, 0.45]],
                                                comment='Dose Tally',
                                                energies=[5.,10.,20.])
        tally_mesh = mcio.tally.FluenceMeshTally(x_min=-50, x_max=50, y_min=-40, y_max=40, z_min=-30, z_max=30,
                                                x_ints=10, y_ints=8, z_ints=6)

        tally_surfflux.mcnp.tally_divisor = 1
        tally_flux.mcnp.multiplier = 2
        tally_mesh.mcnp.dose_interpolation = 'linear-linear'
        tally_mesh.openmc_params.dose_interpolation = 'histogram'

        sph_model = mcio.model.Model(cells=[cell_sph, cell_v, cell_g], tallies=[tally_flux, tally_surfflux, tally_current, 
                                    tally_dose, tally_mesh])
        sph_model.mcnp.modes = ['neutron']

        sph_model.write_mcnp(self.temp_dir / 'sph_model.mcnp')
        sph_model.write_openmc(self.temp_dir)

    def tearDown(self):
        # clean up the temporary directory
        self.temp_dir_obj.cleanup()

    def get_section(self, filename, start_marker, end_marker):
        with open(filename) as f:
            lines = f.read().splitlines()

        start = next(
            i for i, line in enumerate(lines)
            if start_marker in line
        )

        end = next(
            i for i, line in enumerate(lines[start + 1:], start + 1)
            if end_marker in line
        )

        return "".join(lines[start:end])

    def test_mcnp_tally(self):
        reference_file = Path(__file__).parent / 'sph_model' / 'sph_model.mcnp'
        generated_file = self.temp_dir / 'sph_model.mcnp'

        generated = self.get_section(generated_file, "C ====== TALLIES", "C ====== RUN PARAMETERS")
        reference = self.get_section(reference_file, "C ====== TALLIES", "C ====== RUN PARAMETERS")

        self.assertEqual(generated, reference)

    def build_id_map(self, root1, root2, tag_name):
        """Generic builder to map auto-generated IDs between two XML trees for a given tag name."""
        mapping = {}
        items1 = root1.findall(tag_name)
        items2 = root2.findall(tag_name)

        for i1, i2 in zip(items1, items2):
            id1 = i1.attrib.get("id")
            id2 = i2.attrib.get("id")
            if id1 and id2:
                mapping[id1] = id2
        return mapping

    def build_global_id_map(self, root1, root2):
        """Builds maps for all elements containing IDs across the entire tree."""
        mapping = {}
        for elem1, elem2 in zip(root1.iter(), root2.iter()):
            id1 = elem1.attrib.get("id")
            id2 = elem2.attrib.get("id")
            if id1 and id2:
                mapping[id2] = id1  # Maps generated ID back to reference ID for text matching
        return mapping

    def compare_elements(self, e1, e2, global_map):
        self.assertEqual(e1.tag, e2.tag)

        attrs1 = dict(e1.attrib)
        attrs2 = dict(e2.attrib)

        # Ignore IDs in attributes comparison since they can be auto-incremented differently
        attrs1.pop("id", None)
        attrs2.pop("id", None)

        self.assertEqual(attrs1, attrs2)

        text1 = (e1.text or "").strip()
        text2 = (e2.text or "").strip()

        # If the text consists of space-separated IDs (e.g., cell lists, filter lists, surface lists)
        if text1 and text2:
            parts1 = text1.split()
            parts2 = text2.split()
            if len(parts1) == len(parts2):
                # Try translating text tokens if they match known ID mappings
                translated_parts2 = [global_map.get(p2, p2) for p2 in parts2]
                if parts1 == translated_parts2:
                    text2 = text1

        self.assertEqual(text1, text2)

        children1 = list(e1)
        children2 = list(e2)

        self.assertEqual(len(children1), len(children2))

        for c1, c2 in zip(children1, children2):
            self.compare_elements(c1, c2, global_map)

    def compare_xml(self, file1, file2):
        root1 = ET.parse(file1).getroot()
        root2 = ET.parse(file2).getroot()

        global_map = self.build_global_id_map(root1, root2)
        self.compare_elements(root1, root2, global_map)

    def test_openmc_tally(self):
        reference_file = Path(__file__).parent / 'sph_model' / 'tallies.xml'
        generated_file = self.temp_dir / 'tallies.xml'

        self.compare_xml(reference_file, generated_file)
