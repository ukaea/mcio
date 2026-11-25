"""unittest: the OCP version compatibility layer (mcio.ocpcompat)."""

import unittest

from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Face, TopoDS_Solid

from mcio.ocpcompat import (
    TDF_LabelSequence,
    TopTools_IndexedMapOfShape,
    XCAFPrs_IndexedDataMapOfShapeStyle,
    add_missing_static_aliases,
    bnd_box_limits,
    resolve_static,
    topods_cast,
)


class _OnlySuffixed:
    @staticmethod
    def Thing_s(x):
        return ("suffixed", x)


class _OnlyPlain:
    @staticmethod
    def Thing(x):
        return ("plain", x)


class _Both:
    @staticmethod
    def Thing(x):
        return ("plain", x)

    @staticmethod
    def Thing_s(x):
        return ("suffixed", x)


class MCIOUnitTestResolveStatic(unittest.TestCase):
    """The lookup works whichever spelling the bindings use."""

    def test_suffixed_only(self):
        self.assertEqual(resolve_static(_OnlySuffixed, "Thing")(1), ("suffixed", 1))

    def test_plain_only(self):
        self.assertEqual(resolve_static(_OnlyPlain, "Thing")(1), ("plain", 1))

    def test_plain_preferred_when_both_exist(self):
        self.assertEqual(resolve_static(_Both, "Thing")(1), ("plain", 1))

    def test_missing_raises(self):
        with self.assertRaises(AttributeError):
            resolve_static(_OnlyPlain, "Other")


class MCIOUnitTestStaticAliases(unittest.TestCase):
    """A dropped ``_s`` suffix is added back so existing call sites keep working."""

    def test_alias_added_when_suffix_missing(self):
        class Dropped:
            @staticmethod
            def Thing(x):
                return x * 2

        added = add_missing_static_aliases(Dropped, ["Thing_s"])
        self.assertEqual(added, ["Thing_s"])
        self.assertEqual(Dropped.Thing_s(3), 6)

    def test_existing_suffixed_method_is_left_alone(self):
        original = _Both.Thing_s
        self.assertEqual(add_missing_static_aliases(_Both, ["Thing_s"]), [])
        self.assertIs(_Both.Thing_s, original)

    def test_unknown_name_is_ignored(self):
        self.assertEqual(add_missing_static_aliases(_OnlyPlain, ["Other_s"]), [])


class MCIOUnitTestTopoDSCasts(unittest.TestCase):
    """The shape casts return the specific type with the installed OCP."""

    def setUp(self):
        self.box = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()

    def test_cast_to_solid(self):
        explorer = TopExp_Explorer(self.box, TopAbs_SOLID)
        self.assertIsInstance(topods_cast.Solid(explorer.Current()), TopoDS_Solid)

    def test_cast_to_face(self):
        explorer = TopExp_Explorer(self.box, TopAbs_FACE)
        self.assertIsInstance(topods_cast.Face(explorer.Current()), TopoDS_Face)

    def test_wrong_type_raises(self):
        explorer = TopExp_Explorer(self.box, TopAbs_FACE)
        with self.assertRaises(Exception):
            topods_cast.Solid(explorer.Current())

    def test_all_casts_exist(self):
        for name in (
            "Compound",
            "CompSolid",
            "Solid",
            "Shell",
            "Face",
            "Wire",
            "Edge",
            "Vertex",
        ):
            self.assertTrue(callable(getattr(topods_cast, name)), name)


class MCIOUnitTestBoundingBoxLimits(unittest.TestCase):
    def test_limits_of_a_box(self):
        shape = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
        box = Bnd_Box()
        box.SetGap(0.0)
        BRepBndLib.Add_s(shape, box, True)
        x_min, y_min, z_min, x_max, y_max, z_max = bnd_box_limits(box)
        self.assertAlmostEqual(x_max - x_min, 1.0, places=4)
        self.assertAlmostEqual(y_max - y_min, 2.0, places=4)
        self.assertAlmostEqual(z_max - z_min, 3.0, places=4)


class MCIOUnitTestCollections(unittest.TestCase):
    """The collection classes are usable under their old names."""

    def test_label_sequence(self):
        sequence = TDF_LabelSequence()
        self.assertEqual(sequence.Length(), 0)

    def test_indexed_map_of_shape(self):
        shape_map = TopTools_IndexedMapOfShape()
        shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
        self.assertEqual(shape_map.Add(shape), 1)
        self.assertEqual(shape_map.Extent(), 1)

    def test_indexed_data_map_of_shape_style(self):
        self.assertEqual(XCAFPrs_IndexedDataMapOfShapeStyle().Extent(), 0)


if __name__ == "__main__":
    unittest.main()
