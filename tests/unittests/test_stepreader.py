"""
unittest: colours and names of solids read from STEP files by STEPFileReader.

STEP files are written with OCCT's XCAF tools (as CAD exporters do) covering
the ways generic colours are attached: part colour, instance colours overriding
the part colour, face colours (ignored by the reader) and colours inherited
from a sub-assembly.
"""

import os
import tempfile
import unittest

from OCP.STEPCAFControl import STEPCAFControl_Writer, STEPCAFControl_Controller
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf, XCAFDoc_ColorGen
from OCP.Quantity import Quantity_Color, Quantity_TOC_sRGB
from OCP.TDataStd import TDataStd_Name
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Trsf, gp_Vec, gp_Pnt
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE

from mcio.decomposeparams import DecompositionParameters
from mcio.ocpfilereadwrite import STEPFileReader, STEPFileWriter


def new_doc():
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf")); app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    return doc, XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()), XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

def col(rgb): return Quantity_Color(rgb[0]/255, rgb[1]/255, rgb[2]/255, Quantity_TOC_sRGB)
def box(x): return BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), 10, 10, 10).Shape()
def name(lab, n): TDataStd_Name.Set_s(lab, TCollection_ExtendedString(n))
def loc(x): t = gp_Trsf(); t.SetTranslation(gp_Vec(x, 0, 0)); return TopLoc_Location(t)
def write(doc, path):
    # write sub-shape names whatever the test order: the reader crashes (in OCP) on unnamed sub-shapes
    STEPCAFControl_Controller.Init_s(); Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 1)
    w = STEPCAFControl_Writer(); w.SetColorMode(True); w.SetNameMode(True); w.Transfer(doc, STEPControl_AsIs); w.Write(path)


def write_cases(directory):
    """Write the colour test cases, returning {file path: expected colours}."""
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        expected = {}
        # A: colour on part (prototype), assembly instance without colour
        doc, st, ct = new_doc(); root = st.NewShape(); name(root, "root")
        p = st.AddShape(box(0), False); name(p, "part_A"); ct.SetColor(p, col((200, 30, 30)), XCAFDoc_ColorGen)
        i = st.AddComponent(root, p, TopLoc_Location()); name(i, "inst_A")
        st.UpdateAssemblies(); write(doc, "A_part_colour.stp"); expected["A_part_colour.stp"] = [(200, 30, 30)]

        # B: same part used twice, instance colours override (one red, one blue), prototype green
        doc, st, ct = new_doc(); root = st.NewShape(); name(root, "root")
        p = st.AddShape(box(0), False); name(p, "part_B"); ct.SetColor(p, col((0, 255, 0)), XCAFDoc_ColorGen)
        i1 = st.AddComponent(root, p, loc(0)); name(i1, "inst_red"); ct.SetColor(i1, col((255, 0, 0)), XCAFDoc_ColorGen)
        i2 = st.AddComponent(root, p, loc(50)); name(i2, "inst_blue"); ct.SetColor(i2, col((0, 0, 255)), XCAFDoc_ColorGen)
        st.UpdateAssemblies(); write(doc, "B_instance_override.stp"); expected["B_instance_override.stp"] = [(255, 0, 0), (0, 0, 255)]

        # C: generic colour on part
        doc, st, ct = new_doc(); root = st.NewShape(); name(root, "root")
        p = st.AddShape(box(0), False); name(p, "part_C"); ct.SetColor(p, col((10, 120, 250)), XCAFDoc_ColorGen)
        i = st.AddComponent(root, p, TopLoc_Location()); name(i, "inst_C")
        st.UpdateAssemblies(); write(doc, "C_generic_colour.stp"); expected["C_generic_colour.stp"] = [(10, 120, 250)]

        # D: colour on faces only, which is ignored (only generic colours are read)
        doc, st, ct = new_doc(); root = st.NewShape(); name(root, "root")
        b = box(0); p = st.AddShape(b, False); name(p, "part_D")
        e = TopExp_Explorer(b, TopAbs_FACE)
        while e.More():
            fl = st.AddSubShape(p, e.Current()); ct.SetColor(fl, col((90, 180, 60)), XCAFDoc_ColorSurf); e.Next()
        i = st.AddComponent(root, p, TopLoc_Location()); name(i, "inst_D")
        st.UpdateAssemblies(); write(doc, "D_face_colours.stp"); expected["D_face_colours.stp"] = [(-1, -1, -1)]

        # G: nested sub-assembly, colour set on the sub-assembly instance only
        doc, st, ct = new_doc(); root = st.NewShape(); name(root, "root")
        sub = st.NewShape(); name(sub, "subassembly")
        p = st.AddShape(box(0), False); name(p, "part_G")
        st.AddComponent(sub, p, TopLoc_Location())
        i = st.AddComponent(root, sub, TopLoc_Location()); name(i, "inst_sub"); ct.SetColor(i, col((128, 64, 32)), XCAFDoc_ColorGen)
        st.UpdateAssemblies(); write(doc, "G_subassembly_colour.stp"); expected["G_subassembly_colour.stp"] = [(128, 64, 32)]


        return {os.path.join(directory, f): [tuple(c) for c in cols] for f, cols in expected.items()}
    finally:
        os.chdir(cwd)


class MCIOUnitTestSTEPReaderColours(unittest.TestCase):
    """Colours read from STEP files must match the colours in the file."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.cases = write_cases(cls.temp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def _colours(self, path):
        reader = STEPFileReader(path, DecompositionParameters())
        return sorted((s.colour.red, s.colour.green, s.colour.blue) for s in reader.solids)

    def _check(self, name):
        path = os.path.join(self.temp_dir.name, name)
        self.assertEqual(self._colours(path), sorted(self.cases[path]))

    def test_part_colour(self):
        self._check("A_part_colour.stp")

    def test_instance_colours_override_part_colour(self):
        self._check("B_instance_override.stp")

    def test_generic_colour(self):
        self._check("C_generic_colour.stp")

    def test_face_colours_are_ignored(self):
        self._check("D_face_colours.stp")

    def test_colour_inherited_from_subassembly(self):
        self._check("G_subassembly_colour.stp")

    def test_instance_names_are_distinct(self):
        reader = STEPFileReader(os.path.join(self.temp_dir.name, "B_instance_override.stp"), DecompositionParameters())
        names = sorted(s.structure[-1] for s in reader.solids)
        self.assertEqual(names, ["inst_blue", "inst_red"])

    def test_writer_reader_round_trip_all_values(self):
        path = os.path.join(self.temp_dir.name, "round_trip.stp")
        writer = STEPFileWriter(DecompositionParameters())
        for v in range(256):
            box = BRepPrimAPI_MakeBox(gp_Pnt(v * 20.0, 0, 0), 10, 10, 10).Shape()
            writer.add_solid(box, f"s{v}", rgb=(v, (v * 7) % 256, 255 - v))
        writer.write(path)
        reader = STEPFileReader(path, DecompositionParameters())
        self.assertEqual(len(reader.solids), 256)
        for s in reader.solids:
            v = int(s.structure[-1][1:])
            self.assertEqual((s.colour.red, s.colour.green, s.colour.blue), (v, (v * 7) % 256, 255 - v))
