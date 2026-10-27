"""Reading and writing of CAD files using OpenCASCADE (OCP).

Contains helpers for creating XCAF documents, setting OCCT ``Interface_Static``
parameters, exploring/down-casting OCP shapes, loading STEP files (with names,
colours and assembly structure) via :class:`STEPFileReader`, writing STEP
assemblies via :class:`STEPFileWriter`, and exporting STL meshes.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.notebook import tqdm as tqdm_notebook

# import first to allow logging of optional modules 
from mcio import logger

try:
    import cadquery as cq
except:
    pass

try:
    from jupyter_cadquery import show
except:
    pass

try:
    import casadi
    import OCP
    from OCP.BRep import (
        BRep_Tool,
        BRep_Builder
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_Copy,
        BRepBuilderAPI_Transform
    )
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.Geom import (
        Geom_RectangularTrimmedSurface
    )
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.Quantity import (
        Quantity_Color,
        Quantity_TOC_RGB,
        Quantity_ColorRGBA,
        Quantity_TOC_sRGB
    )
    from OCP.STEPCAFControl import (
        STEPCAFControl_Reader,
        STEPCAFControl_Writer,
        STEPCAFControl_Controller
    )
    from OCP.STEPControl import (
        STEPControl_AsIs,
    )
    from OCP.StlAPI import StlAPI_Writer
    from OCP.TCollection import (
        TCollection_ExtendedString
    )
    from OCP.TCollection import (
        TCollection_ExtendedString,
        TCollection_AsciiString
    )
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TDF import (
        TDF_Label,
        TDF_LabelSequence
    )
    from OCP.TopAbs import (
        TopAbs_FACE,
        TopAbs_COMPOUND,
        TopAbs_SOLID,
        TopAbs_COMPSOLID,
        TopAbs_SHELL,
        TopAbs_EDGE,
        TopAbs_VERTEX,
        TopAbs_WIRE,
        TopAbs_SHAPE
    )
    from OCP.TopExp import (
        TopExp_Explorer
    )
    from OCP.TopoDS import (
        TopoDS,
        TopoDS_Face,
        TopoDS_Solid,
        TopoDS_Compound,
        TopoDS_Shape
    )
    from OCP.TopLoc import (
        TopLoc_Location
    )
    from OCP.XCAFApp import (
        XCAFApp_Application
    )
    from OCP.XCAFDoc import (
        XCAFDoc_DocumentTool,
        XCAFDoc_ColorGen,
        XCAFDoc_ColorSurf
    )
    from OCP.XCAFPrs import (
        XCAFPrs,
        XCAFPrs_DocumentExplorer,
        XCAFPrs_DocumentExplorerFlags_None,
        XCAFPrs_IndexedDataMapOfShapeStyle,
        XCAFPrs_Style
    )

    TOP_ABS_SHAPES: tuple[TopAbs_ShapeEnum, ...] = (TopAbs_COMPOUND,
                        TopAbs_COMPSOLID,
                        TopAbs_SOLID,
                        TopAbs_SHELL,
                        TopAbs_FACE,
                        TopAbs_WIRE,
                        TopAbs_EDGE,
                        TopAbs_VERTEX,
                        TopAbs_SHAPE)

    TOPODS_CAST_METHODS: dict[TopAbs_ShapeEnum, Callable[[TopoDS_Shape], Any]] = {TopAbs_COMPOUND: TopoDS.Compound_s,
                                 TopAbs_COMPSOLID: TopoDS.CompSolid_s,
                                 TopAbs_SOLID: TopoDS.Solid_s,
                                 TopAbs_SHELL: TopoDS.Shell_s,
                                 TopAbs_FACE: TopoDS.Face_s,
                                 TopAbs_WIRE: TopoDS.Wire_s,
                                 TopAbs_EDGE: TopoDS.Edge_s,
                                 TopAbs_VERTEX: TopoDS.Vertex_s,
                                 TopAbs_SHAPE: lambda shape: shape}

except Exception as e:
    logger.warning(f'failed to load OCP')
    logger.warning(f'{e}')

if TYPE_CHECKING:
    from OCP.Geom import Geom_Surface
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.XCAFDoc import XCAFDoc_ColorTool, XCAFDoc_ShapeTool

from .colour import ColourRGB
from .decomposeparams import DecompositionParameters
from .material import Material
from .profile import function_timer
from .utils import (
    check_folder_exists_for_file_path,
    in_notebook
)

def _create_ocp_compound(shapes_to_add: list[TopoDS_Shape]=[]) -> TopoDS_Compound:
    """Create an OCP compound containing the given shapes.

    Parameters
    ----------
    shapes_to_add : list[TopoDS_Shape], optional
        Shapes to add to the compound, by default []

    Returns
    -------
    TopoDS_Compound
        Compound containing all of the shapes.
    """
    # set up builder
    builder: BRep_Builder = BRep_Builder()

    # make a compound
    compound: TopoDS_Compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    
    # add shapes to compound
    for shape in shapes_to_add:
        builder.Add(compound, shape)

    # return the compound
    return compound

def create_ocp_document(document_name: str, decom_params: DecompositionParameters) -> TDocStd_Document:
    """Create an OCP ``TDocStd_Document`` XCAF document.

    The document length unit is set from ``decom_params.doc_unit_in_metres`` and
    the document is initialised with the ``doc_xcaf_formatting`` storage format.

    Parameters
    ----------
    document_name : str
        The document name.
    decom_params : DecompositionParameters
        The decomposition parameters passed.

    Returns
    -------
    TDocStd_Document
        The document created.
    """

    # get the applicatoin
    app: "XCAFApp_Application" = XCAFApp_Application.GetApplication_s()

    # create a document
    doc: "TDocStd_Document" = TDocStd_Document(TCollection_ExtendedString(document_name))

    # set the units of the document
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, decom_params.doc_unit_in_metres)

    # initialise new document
    app.NewDocument(TCollection_ExtendedString(decom_params.doc_xcaf_formatting), doc)

    # return the document
    return doc

def _set_ocp_interface_static(parameter_label: str, parameter: int | float | str) -> None:
    """Set an OCCT ``Interface_Static`` parameter if not already set.

    The setter/getter (integer, real or string) is chosen from the Python type
    of ``parameter``.

    Parameters
    ----------
    parameter_label : str
        Name of the static parameter, e.g. ``'read.step.nonmanifold'``.
    parameter : int | float | str
        Value to assign.

    Raises
    ------
    Exception
        If ``parameter`` is not an int, float or str.
    ValueError
        If OCCT refuses to set the parameter.
    """
    # get corresponding function to set parameter type
    if isinstance(parameter, int):
        param_get_function = Interface_Static.IVal_s
        param_set_function = Interface_Static.SetIVal_s
    elif isinstance(parameter, float):
        param_get_function = Interface_Static.RVal_s
        param_set_function = Interface_Static.SetRVal_s
    elif isinstance(parameter, str):
        param_get_function = Interface_Static.CVal_s
        param_set_function = Interface_Static.SetCVal_s
    else:
        msg = f'required type for InterfaceStatic parameter is int, float or str, not type {type(parameter)}'
        raise Exception(msg)
    
    # check if parameter already set, else set its value
    if param_get_function(parameter_label) == parameter:
        pass
    elif param_set_function(parameter_label, parameter):
        msg = f'[OK] parameter_label {parameter_label} set to {parameter}'
        logger.info(msg)
    else:
        msg = f'[ERROR] unable to set {parameter_label} to {parameter}'
        logger.fatal(msg)
        raise ValueError(msg)

def _get_ocp_interface_static(parameter_label: str, parameter_type: type[int] | type[float] | type[str]) -> int | float | str:
    """Get the value of an OCCT ``Interface_Static`` parameter.

    Parameters
    ----------
    parameter_label : str
        Name of the static parameter.
    parameter_type : type[int] | type[float] | type[str]
        Python type of the parameter (``int``, ``float`` or ``str``), selecting
        the OCCT getter used.

    Returns
    -------
    int | float | str
        The current value of the parameter.

    Raises
    ------
    Exception
        If ``parameter_type`` is not int, float or str.
    """
    # get corresponding function to set parameter type
    if parameter_type == int:
        param_get_function = Interface_Static.IVal_s
    elif parameter_type == float:
        param_get_function = Interface_Static.RVal_s
    elif parameter_type == str:
        param_get_function = Interface_Static.CVal_s
    else:
        msg = f'required type for InterfaceStatic parameter is int, float or str, not type {type(parameter_type)}'
        raise Exception(msg)
    
    return param_get_function(parameter_label)

def extract_ocp_shape_type_from_shape(shape: TopoDS_Shape, shape_type: TopAbs_ShapeEnum, cast_from_shape: bool=True) -> list[TopoDS_Shape]:
    """Extract all sub-shapes of a given type from a shape.

    Parameters
    ----------
    shape : TopoDS_Shape
        Shape to explore.
    shape_type : TopAbs_ShapeEnum
        Type of sub-shape to extract, e.g. ``TopAbs_FACE`` or ``TopAbs_SOLID``.
    cast_from_shape : bool, optional
        If True, down-cast each result to its specific TopoDS type (e.g.
        ``TopoDS_Face``), by default True

    Returns
    -------
    list[TopoDS_Shape]
        Sub-shapes found by ``TopExp_Explorer``, in exploration order.

    Raises
    ------
    Exception
        If ``shape_type`` is not a supported ``TopAbs`` shape type.
    """
    # check if shape type in accepted list    
    if shape_type in TOP_ABS_SHAPES:
        # extract shape type from shape and return in a list
        explorer: TopExp_Explorer = TopExp_Explorer(shape, shape_type)
        results: list[TopoDS_Shape] = []
        while explorer.More():
            results.append(explorer.Current())
            explorer.Next()

        if not cast_from_shape:
            return results
        else:
            results_cast = [TOPODS_CAST_METHODS[shape_type](r) for r in results]
            return results_cast
    else:
        msg = f'shape type is not in TOPODS_CAST_METHODS - {shape_type}'
        raise Exception(msg)

# def save_ocp_doc_to_step(doc, output_path, decom_params):
#     # initialise controller
#     STEPCAFControl_Controller.Init_s()

#     # set output document setting
#     _set_ocp_interface_static('write.stepcaf.subshapes.name', 1)
#     _set_ocp_interface_static('write.step.schema', 'AP242DIS')
#     # _set_ocp_interface_static('write.step.schema', 'AP214IS')
#     # _set_ocp_interface_static('write.step.color.mode', 1)

#     # write files 
#     writer = STEPCAFControl_Writer()
#     XCAFDoc_DocumentTool.SetLengthUnit_s(doc, decom_params.DOC_UNIT_IN_METRES)
#     writer.SetColorMode(True)
#     writer.SetNameMode(True)
#     writer.SetLayerMode(True)

#     print(writer.GetColorMode(), 'colour mode')

#     # transfer document to writer
#     if not writer.Transfer(doc, STEPControl_AsIs):
#         msg = f'[ERROR] - transfer of document to writer has failed...'
#         raise Exception(msg)

#     # write file
#     if not writer.Write(output_path) == IFSelect_RetDone:
#         msg = f'[ERROR] - save of document to step file failed: {output_path}'
#         raise Exception(msg)

# @check_folder_exists_for_file_path("output_path")
# def save_ocp_shapes_to_step(
#         shapes: list["TopoDS_Shape"],
#         output_path: str, 
#         decom_params: DecompositionParameters, 
#         enumerate_files=False, 
#         compound_labels: dict["TopoDS_Shape": str]={}) -> os.PathLike:

#     # check output folder exists
#     # dirs, file = os.path.split(output_path)
#     # if dirs and not os.path.isdir(dirs):
#     #     os.makedirs(dirs, exist_ok=True)
#     #     logger.info(f'making dir {dir}')

#     # enumerate files so that file is not overwritten
#     if enumerate_files:
#         folder, filename = os.path.split(output_path)
#         n_files = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
#         output_path = os.path.join(folder, f'{n_files+1:0>6}_{filename}')
    
#     # save to file
#     doc = create_ocp_document('output_shape', decom_params)
#     shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
#     for shape in shapes:
#         label = shape_tool.AddShape(shape)
#         if shape in compound_labels:
#             TDataStd_Name.Set_s(label, TCollection_ExtendedString(compound_labels[shape]))

#     save_ocp_doc_to_step(doc, output_path, decom_params)

#     return output_path

def save_ocp_shapes_to_stl(
        shapes: list[TopoDS_Shape],
        output_path: str,
        linear_deflection: float = 0.1,
        angular_deflection: float = 0.1,
        relative: bool = True,
        enumerate_files: bool = False) -> None:
    """Mesh shapes and save them to an STL file.

    Multiple shapes are combined into a single compound before meshing with
    ``BRepMesh_IncrementalMesh``.

    Parameters
    ----------
    shapes : list[TopoDS_Shape]
        Shapes to export.
    output_path : str
        Path of the STL file to write; missing folders are created.
    linear_deflection : float, optional
        Linear deflection used for meshing, by default 0.1
    angular_deflection : float, optional
        Angular deflection used for meshing (radians), by default 0.1
    relative : bool, optional
        Whether the linear deflection is relative to edge size, by default True
    enumerate_files : bool, optional
        If True, prefix the filename with a 6-digit counter based on the number
        of files in the output folder, by default False
    """

    # Ensure output directory exists
    dirs, file = os.path.split(output_path)
    if dirs and not os.path.isdir(dirs):
        os.makedirs(dirs, exist_ok=True)

    # Optional file enumeration
    if enumerate_files:
        folder, filename = os.path.split(output_path)
        n_files = len([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ])
        output_path = os.path.join(folder, f'{n_files+1:0>6}_{filename}')

    # If multiple shapes → export as a compound
    if len(shapes) == 1:
        shape: TopoDS_Shape = shapes[0]
    else:
        from OCP.BRep import BRep_Builder
        
        from OCP.TopoDS import TopoDS_Compound

        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for s in shapes:
            builder.Add(compound, s)
        shape = compound

    # Mesh the shape before STL export
    mesh = BRepMesh_IncrementalMesh(
        shape,
        linear_deflection,
        relative,
        angular_deflection,
        True  # parallel meshing
    )
    mesh.Perform()

    # Write STL
    writer = StlAPI_Writer()
    # writer.SetASCIIMode(False)  # binary STL (smaller, faster)
    writer.Write(shape, output_path)

    logger.into(f"Saved STL to: {output_path}")


@function_timer
def load_ocp_step_to_doc(input_paths: list[str], decom_params: DecompositionParameters, document_label: str='input_step_data') -> TDocStd_Document:
    """Load one or more STEP files into a single XCAF document.

    Names, colours, layers and properties are read. Progress is shown with a
    tqdm bar in notebooks or a rich progress bar otherwise.

    Parameters
    ----------
    input_paths : list[str]
        Paths of the STEP files to load.
    decom_params : DecompositionParameters
        Decomposition parameters used to create the document.
    document_label : str, optional
        Name given to the created document, by default 'input_step_data'

    Returns
    -------
    TDocStd_Document
        Document containing the transferred contents of all STEP files.

    Raises
    ------
    Exception
        If a STEP file fails to read or to transfer to the document.
    """
    # output to logger
    logger.info(f'[LOADING] - loading step file from {len(input_paths)} file' + ('s' if len(input_paths) > 1 else ''))

    # initialise step caf controller
    STEPCAFControl_Controller.Init_s()

    # set input parameters 
    # ensure that sub shape names are loaded correctly
    _set_ocp_interface_static('read.stepcaf.subshapes.name', 1)
    # _set_ocp_interface_static('read.step.product.context', 1)
    # _set_ocp_interface_static('read.step.product.mode', 1)
    _set_ocp_interface_static('read.surfacecurve.mode', 0)

    # _set_ocp_interface_static('read.step.shape.relationship', 1)
    # _set_ocp_interface_static('read.step.shape.repr_level', 1)
    # _set_ocp_interface_static('read.step.nonmanifold', 1)

    # _set_ocp_interface_static('read.step.nonmanifold', 0)
    
    # set precision and surface curve modes - reduces chance of introduced periodic surfaces
    # _set_ocp_interface_static('read.step.schema', 3)

    # print('step reader shema: ')
    # print(_get_ocp_interface_static('read.step.schema', int))

    # exit()

    # _set_ocp_interface_static('')

    # 0 = use both 3D and 2D curves as-is
    # 1 = rebuild 3D curves from 2D pcurves
    # 2 = rebuild 2D pcurves from 3D curves
    # 3 = strong prefernce for 3D curves

    # _set_ocp_interface_static('read.precision.mode', 1)
    # _set_ocp_interface_static('read.step.assembly.level', 1)
    # # _set_ocp_interface_static('read.stepcaf.product.mode', 1)
    # _set_ocp_interface_static('read.step.shape.relationship', 1)
    # _set_ocp_interface_static('read.step.all.shapes', 1)
    # _set_ocp_interface_static('read.step.resource.name', 1)
    # # _set_ocp_interface_static('read.step.shape.repr_level', 1)

    # Interface_Static.SetIVal_s("read.step.assembly.level", 1)

    # create_document
    doc = create_ocp_document(document_label, decom_params)

    desc_stage = '[Loading STEP files]'
    is_notebook_env = in_notebook()

    if is_notebook_env:
        desc_tqdm = f'{desc_stage: <30}'
        pbar = tqdm_notebook(total=len(input_paths), desc=desc_tqdm, unit="file")

        def update_progress() -> None:
            """Advance the notebook progress bar by one file."""
            pbar.update(1)

        def close_progress() -> None:
            """Close the notebook progress bar."""
            pbar.close()

    else:
        desc = f'[cyan]{desc_stage: <30}[/cyan]'
        progress = Progress(
            TextColumn(desc),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        progress.start()
        task_id = progress.add_task("loading", total=len(input_paths))

        def update_progress() -> None:
            """Advance the rich progress bar by one file."""
            progress.advance(task_id)

        def close_progress() -> None:
            """Stop the rich progress bar."""
            progress.stop()


    # iterate over all step files for input
    for i, step_file_path in enumerate(input_paths, start=1):
        # log each file loading
        logger.info(f'[LOADING] - loading step file {i} {step_file_path}')

        # create a reader for each step file
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        reader.SetLayerMode(True)
        # reader.SetPropsMode(True)

        # shape_from_step_file = ocp_load_step_file(step_file_path)
        if (reader.ReadFile(step_file_path) == IFSelect_RetDone):
            msg = f'[OK] STEP file loaded : {step_file_path}'
            logger.info(msg)
        else:
            msg = f'[ERROR] STEP file failed to load : {step_file_path}'
            logger.fatal(msg)
            raise Exception(msg)

        # transfer to document
        if reader.Transfer(doc):
            msg = f'[OK] transfered to document : {step_file_path}'
            logger.info(msg)
        else:
            msg = f'[ERROR] failed to transfer to document : {step_file_path}'
            logger.fatal(msg)
            raise Exception(msg)

        # advance progress bar following loading of each step file to doc
        update_progress()

    close_progress()

    return doc


class CADFileWriter:
    """Placeholder base class for CAD file writers (currently empty)."""

    def __init__(self) -> None:
        """Initialise the CADFileWriter."""
        pass


class STEPSolid:
    """A solid loaded from a STEP file together with its metadata.

    Parameters
    ----------
    solid : TopoDS_Solid
        The OCP solid (may be a compsolid down-cast from the STEP shape).
    colour_rgb : ColourRGB
        Colour of the solid; components are -1,-1,-1 if no colour was defined.
    structure : list[str]
        Names of the assembly tree nodes leading to the solid.
    """

    def __init__(self, solid: TopoDS_Solid, colour_rgb: ColourRGB, structure: list[str]) -> None:
        """Initialise the STEPSolid."""
        self.solid: TopoDS_Solid = solid
        self.colour: ColourRGB = colour_rgb
        self.structure: list[str] = structure

class STEPFileReader:
    def __init__(self, file_path: os.PathLike, decom_params: DecompositionParameters=None) -> None:
        # decomposition parameters
        self.file_path: os.PathLike = file_path
        self.decom_params: list[DecompositionParameters] = decom_params if decom_params else DecompositionParameters()
        
        # set up document, shape / colour tools and xcaf explorer
        self.doc = self._transfer_from_step()
        self.colour_tool = XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main())
        self.shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        
        # set up XCAF document explorer
        self.doc_exp = XCAFPrs_DocumentExplorer(self.doc, XCAFPrs_DocumentExplorerFlags_None)
        self.solids: list[STEPSolid] = self._extract_solids()

    def _transfer_from_step(self):
        # initialise step caf controller
        STEPCAFControl_Controller.Init_s()

        # set input parameters 
        # ensure that sub shape names are loaded correctly
        _set_ocp_interface_static('read.stepcaf.subshapes.name', 1)
        _set_ocp_interface_static('read.surfacecurve.mode', 0)

        # create_document
        doc = create_ocp_document("input_step_data", self.decom_params)
        
        # log each file loading
        logger.info(f'[LOADING] - loading step file {self.file_path}')

        # create a reader for each step file
        reader = STEPCAFControl_Reader()
        reader.SetNameMode(True)
        reader.SetColorMode(True)
        reader.SetLayerMode(True)
        # reader.SetPropsMode(True)

        # shape_from_step_file = ocp_load_step_file(step_file_path)
        if (reader.ReadFile(self.file_path) == IFSelect_RetDone):
            msg = f'[OK] STEP file loaded : {self.file_path}'
            logger.info(msg)
        else:
            msg = f'[ERROR] STEP file failed to load : {self.file_path}'
            logger.fatal(msg)
            raise Exception(msg)

        # transfer to document
        if reader.Transfer(doc):
            msg = f'[OK] transfered to document : {self.file_path}'
            logger.info(msg)
        else:
            msg = f'[ERROR] failed to transfer to document : {self.file_path}'
            logger.fatal(msg)
            raise Exception(msg)

        # return the document
        return doc

    def _get_colour(self, ref_label, label, colour_tool):
        colour = Quantity_ColorRGBA()

        # instance (label) colours are checked before part (ref_label) colours, so that a colour set on
        # an instance overrides the colour of the part it uses
        if colour_tool.GetColor_s(label, XCAFDoc_ColorGen, colour):
            rgb = colour.GetRGB()
            red, green, blue = rgb.Values(Quantity_TOC_sRGB)
            red_rounded, green_rounded, blue_rounded =  round(red*255), round(green*255), round(blue*255)
        elif colour_tool.GetColor_s(ref_label, XCAFDoc_ColorGen, colour):
            rgb = colour.GetRGB()
            red, green, blue = rgb.Values(Quantity_TOC_sRGB)
            red_rounded, green_rounded, blue_rounded =  round(red*255), round(green*255), round(blue*255)
        elif colour_tool.GetColor_s(label, XCAFDoc_ColorSurf, colour):
            rgb = colour.GetRGB()
            red, green, blue = rgb.Values(Quantity_TOC_sRGB)
            red_rounded, green_rounded, blue_rounded =  round(red*255), round(green*255), round(blue*255)
        elif colour_tool.GetColor_s(ref_label, XCAFDoc_ColorSurf, colour):
            rgb = colour.GetRGB()
            red, green, blue = rgb.Values(Quantity_TOC_sRGB)
            red_rounded, green_rounded, blue_rounded =  round(red*255), round(green*255), round(blue*255)
        else:
            red_rounded, green_rounded, blue_rounded =  -1, -1, -1

        return ColourRGB(red=red_rounded, green=green_rounded, blue=blue_rounded)

    def _extract_solids(self) -> list[STEPSolid]:
        # store solids and meta data
        solids = []

        # store the current structure tree for a given node / sub-shape
        structure = []

        # iterate over all nodes loaded into document
        while self.doc_exp.More():
            # get node, shape and shape type
            node = self.doc_exp.Current()
            node_shape = self.shape_tool.GetShape_s(node.Label)
            node_shape.Location(node.Location)
            node_type = node_shape.ShapeType()
            node_depth = self.doc_exp.CurrentDepth() + 1

            # set up string to print location in structure tree
            node_name_with_depth = TCollection_ExtendedString((self.doc_exp.CurrentDepth()+2), '-')

            # get the node name
            node_name = TDataStd_Name()

            # if (node.RefLabel.FindAttribute(TDataStd_Name.GetID_s(), node_name) or node.Label.FindAttribute(TDataStd_Name.GetID_s(), node_name)):
            if node.RefLabel.FindAttribute(TDataStd_Name.GetID_s(), node_name) | node.Label.FindAttribute(TDataStd_Name.GetID_s(), node_name):
                node_name_with_depth = node_name_with_depth + node_name.Get() + TCollection_ExtendedString(f' [{node_type.name}]')

            # conver the node name to a python string
            node_name_as_string = TCollection_AsciiString(node_name.Get()).ToCString()

            # add to / update structure list of sub-shapes parent objects and collect parent transforms
            if node_depth > len(structure):
                structure.append(node_name_as_string)
            elif node_depth == len(structure):
                structure[-1] = node_name_as_string
            else:
                structure = structure[:node_depth-1]
                structure.append(node_name_as_string)

            # checks if node is assembly
            if node.IsAssembly:
                logger.debug('/'.join(structure) + ' [Assembly]')

            # case if node is a solid
            elif node_type == TopAbs_SOLID or node_type == TopAbs_COMPSOLID:
                # get colour of solid or compsolid
                colour = self._get_colour(node.RefLabel, node.Label, self.colour_tool)

                # if the solid and its instance have no colour, use the colour inherited from the
                # assemblies above it (node.Style holds the inherited colour, never face colours)
                if colour.red == -1:
                    colour = _colour_rgb_from_style(node.Style) or colour
                # colour = _colour_rgb_from_style(node.Style) or ColourRGB(red=-1, green=-1, blue=-1)
                
                logger.debug('/'.join(structure) + f' [{colour.red:< 4}{colour.green:< 4}{colour.blue:< 4}] [{node_type}]')

                solid = STEPSolid(
                    solid=downcast_ocp_shape(node_shape),
                    colour_rgb=colour,
                    structure=[s for s in structure],
                )

                solids.append(solid)

            elif node_type == TopAbs_COMPOUND:
                # loop over sub shapes of current node
                sub_shape_labels = TDF_LabelSequence()
                
                # collect sub shapes which are solids or compsolids
                sub_shapes_solids_or_compsolids = []
                sub_shapes_non_solids_or_compsolids = []

                # get sub shape labels
                self.shape_tool.GetSubShapes_s(node.RefLabel, sub_shape_labels)
                    
                for sub_shape_label in sub_shape_labels:
                    # set up string to store sub shape locations and names
                    sub_shape_name_with_depth = TCollection_ExtendedString((self.doc_exp.CurrentDepth()+3), '-')
                    
                    # get the sub shape
                    sub_shape = self.shape_tool.GetShape_s(sub_shape_label)
                    sub_shape_type = sub_shape.ShapeType()

                    # check if sub shape type is a solid or compsolid, otherwise ignore
                    if sub_shape_type not in [TopAbs_SOLID, TopAbs_COMPSOLID]:
                        sub_shapes_non_solids_or_compsolids.append(sub_shape)
                    else:
                        sub_shapes_solids_or_compsolids.append(sub_shape)

                    # print(sub_shape.ShapeType())

                    # check if sub shape is a solid
                    if sub_shape.ShapeType() == TopAbs_SOLID or sub_shape.ShapeType() == TopAbs_COMPSOLID:
                        # get the node name
                        sub_shape_name = TDataStd_Name()
                        if (sub_shape_label.FindAttribute(TDataStd_Name.GetID_s(), sub_shape_name)):
                            sub_shape_name_with_depth = sub_shape_name_with_depth + sub_shape_name.Get() + TCollection_ExtendedString(f' [{sub_shape.ShapeType().name}]')

                        # get the sub structure
                        sub_structure = [*structure, sub_shape_name.Get().ToExtString()]

                        # get the shapes colour if it has one
                        colour = self._get_colour(sub_shape_label, sub_shape_label, self.colour_tool)

                        # log solid extraction
                        logger.debug('/'.join(sub_structure) + f'[{colour.red:< 4}{colour.green:< 4}{colour.blue:< 4}] [{node_type} / sub-shape]')

                        # apply node transform to the sub solids
                        transform_to_apply = node.Location.Transformation()
                        sub_shape_transformed = BRepBuilderAPI_Transform(sub_shape, transform_to_apply, True).Shape()

                        solid = STEPSolid(
                                solid=downcast_ocp_shape(sub_shape_transformed),
                                colour_rgb=colour,
                                structure=[s for s in sub_structure],
                            )

                        solids.append(solid)
                
            self.doc_exp.Next()

        return solids



def _colour_rgb_from_quantity(colour: Quantity_Color) -> ColourRGB:
    """Convert an OCP colour to a 0 - 255 sRGB :class:`ColourRGB`.

    OCCT stores colours in linear RGB and converts STEP file values as sRGB, so
    the sRGB values are the ones written in the file.

    Parameters
    ----------
    colour : Quantity_Color
        OCP colour.

    Returns
    -------
    ColourRGB
        Colour with components rounded to integers in 0 - 255.
    """
    red, green, blue = colour.Values(Quantity_TOC_sRGB)
    return ColourRGB(red=round(red * 255), green=round(green * 255), blue=round(blue * 255))


def _colour_rgb_from_style(style: XCAFPrs_Style) -> ColourRGB | None:
    """Return the surface colour of an XCAF style, if set.

    Parameters
    ----------
    style : XCAFPrs_Style
        Style to read.

    Returns
    -------
    ColourRGB or None
        Surface colour, or None if the style has no surface colour.
    """
    if style.IsSetColorSurf():
        return _colour_rgb_from_quantity(style.GetColorSurf())
    return None


def _common_face_colour(solid: TopoDS_Shape, sub_styles: XCAFPrs_IndexedDataMapOfShapeStyle) -> ColourRGB | None:
    """Return the colour shared by the coloured faces of a solid.

    Parameters
    ----------
    solid : TopoDS_Shape
        Solid (in the prototype's frame, as in ``sub_styles``).
    sub_styles : XCAFPrs_IndexedDataMapOfShapeStyle
        Styles of sub-shapes from ``XCAFPrs.CollectStyleSettings``.

    Returns
    -------
    ColourRGB or None
        The colour if at least one face is coloured and all coloured faces
        share it; None otherwise (a warning is logged for mixed colours).
    """
    colours: set[tuple[int, int, int]] = set()
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        if sub_styles.Contains(face):
            colour = _colour_rgb_from_style(sub_styles.FindFromKey(face))
            if colour is not None:
                colours.add((colour.red, colour.green, colour.blue))
        explorer.Next()

    if len(colours) == 1:
        red, green, blue = colours.pop()
        return ColourRGB(red=red, green=green, blue=blue)
    if len(colours) > 1:
        logger.warning(f'solid has faces of {len(colours)} different colours; no face colour used')
    return None


def _solids_in_shape(shape: TopoDS_Shape) -> list[TopoDS_Shape]:
    """Return the compsolids of a shape and the solids not inside a compsolid.

    Parameters
    ----------
    shape : TopoDS_Shape
        Shape to search (e.g. a compound).

    Returns
    -------
    list[TopoDS_Shape]
        Compsolids followed by free solids, each with the location it has within
        ``shape``.
    """
    found: list[TopoDS_Shape] = []
    explorer = TopExp_Explorer(shape, TopAbs_COMPSOLID)
    while explorer.More():
        found.append(explorer.Current())
        explorer.Next()
    explorer = TopExp_Explorer(shape, TopAbs_SOLID, TopAbs_COMPSOLID)
    while explorer.More():
        found.append(explorer.Current())
        explorer.Next()
    return found


def _label_name(label: TDF_Label) -> str:
    """Return the name attribute of a label, or an empty string.

    Parameters
    ----------
    label : TDF_Label
        Document label.

    Returns
    -------
    str
        Name of the label, or '' if it has none.
    """
    # check the attribute exists first: FindAttribute can crash (in OCP) on a
    # label without a name attribute
    if label.IsNull() or not label.IsAttribute(TDataStd_Name.GetID_s()):
        return ''
    name = TDataStd_Name()
    label.FindAttribute(TDataStd_Name.GetID_s(), name)
    return TCollection_AsciiString(name.Get()).ToCString()


def _shape_key(shape: TopoDS_Shape) -> tuple[int, int]:
    """Return a key identifying a shape and its location (orientation ignored).

    Parameters
    ----------
    shape : TopoDS_Shape
        Shape.

    Returns
    -------
    tuple[int, int]
        Hash of the underlying shape and of its location.
    """
    return (hash(shape.TShape()), hash(shape.Location()))


def _sub_shape_names(shape_tool: XCAFDoc_ShapeTool, label: TDF_Label) -> dict[tuple[int, int], str]:
    """Return the names of the labelled sub-shapes of a shape label.

    Parameters
    ----------
    shape_tool : XCAFDoc_ShapeTool
        Document shape tool.
    label : TDF_Label
        Label of the (prototype) shape.

    Returns
    -------
    dict[tuple[int, int], str]
        Sub-shape key (see :func:`_shape_key`) to name, for named sub-shapes.
    """
    names: dict[tuple[int, int], str] = {}
    sub_shape_labels = TDF_LabelSequence()
    shape_tool.GetSubShapes_s(label, sub_shape_labels)
    for sub_shape_label in sub_shape_labels:
        name = _label_name(sub_shape_label)
        if name:
            names[_shape_key(shape_tool.GetShape_s(sub_shape_label))] = name
    return names


def downcast_ocp_shape(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Down-cast a general ``TopoDS_Shape`` to its specific type.

    For example a shape of type ``TopAbs_SOLID`` is returned as a
    ``TopoDS_Solid``, ``TopAbs_FACE`` as a ``TopoDS_Face``, etc.

    Parameters
    ----------
    shape : TopoDS_Shape
        Shape to down-cast.

    Returns
    -------
    TopoDS_Shape
        The shape cast to its specific TopoDS subclass.
    """

    casting_method_loopuk: dict[TopAbs_ShapeEnum, Callable[[TopoDS_Shape], TopoDS_Shape]] = {
        TopAbs_COMPOUND: TopoDS.Compound_s,
        TopAbs_COMPSOLID: TopoDS.CompSolid_s,
        TopAbs_SOLID: TopoDS.Solid_s,
        TopAbs_SHELL: TopoDS.Shell_s,
        TopAbs_FACE: TopoDS.Face_s,
        TopAbs_WIRE: TopoDS.Wire_s,
        TopAbs_EDGE: TopoDS.Edge_s,
        TopAbs_VERTEX: TopoDS.Vertex_s
    }

    return casting_method_loopuk[shape.ShapeType()](shape)

class STEPFileWriter:
    """Build an XCAF assembly of named, coloured solids and write it to STEP.

    Parameters
    ----------
    decom_params : DecompositionParameters, optional
        Decomposition parameters (document units/format); defaults are used if
        None, by default None
    """

    def __init__(self, decom_params: DecompositionParameters | None = None) -> None:
        """Initialise the STEPFileWriter and its empty root assembly."""
        # decomposition parameters
        self.decom_params: DecompositionParameters = decom_params if decom_params else DecompositionParameters()

        # set up document, shape and colours tool
        self._doc: TDocStd_Document = create_ocp_document('output_shape', self.decom_params)
        self._shape_tool: XCAFDoc_ShapeTool = XCAFDoc_DocumentTool.ShapeTool_s(self._doc.Main())
        self._colour_tool: XCAFDoc_ColorTool = XCAFDoc_DocumentTool.ColorTool_s(self._doc.Main())

        # store all shapes and their labels, if duplicate, don't generate new label or colours will be messed up
        self._shape_labels: dict[TopoDS_Shape, TDF_Label] = {}

        # set up a builder nad root compound
        self._builder: BRep_Builder = BRep_Builder()
        root_compound: TopoDS_Compound = TopoDS_Compound()
        self._builder.MakeCompound(root_compound)

        # add root compound to document at top level
        self._root_label: TDF_Label = self._shape_tool.NewShape()

    def _add_shape_to_assembly(self, shape: TopoDS_Shape, label: str, rgb: list[int] | None = None) -> None:
        """Add a shape to the document as a named component of the root assembly.

        Parameters
        ----------
        shape : TopoDS_Shape
            Shape to add.
        label : str
            Name given to both the part and its instance in the assembly.
        rgb : list[int], optional
            RGB colour (0 - 255); currently unused, by default None
        """
        # add shape to document, and get label of shape added.
        part_label = self._shape_tool.AddShape(shape, False, False)

        # add string to shape added
        TDataStd_Name.Set_s(part_label, TCollection_ExtendedString(label))

        # add the componet to the root assembly
        identity_loc = TopLoc_Location()
        instance_label = self._shape_tool.AddComponent(self._root_label, part_label, identity_loc)
        TDataStd_Name.Set_s(instance_label, TCollection_ExtendedString(label))

    def add_solids(self, solids: list[TopoDS_Solid], label: str, rgb: list[int] | None = None) -> None:
        """Add several solids as a single named compound in the root assembly.

        Solids already added to the writer are copied so that they can be
        coloured independently.

        Parameters
        ----------
        solids : list[TopoDS_Solid]
            Solids to add.
        label : str
            Name of the compound component.
        rgb : list[int], optional
            RGB colour (0 - 255, sRGB) applied to each solid, by default None
        """
        # add solids to new component
        compound: TopoDS_Compound = TopoDS_Compound()
        self._builder.MakeCompound(compound)

        if rgb:
            red, green, blue = rgb
            quantity_colour = Quantity_Color(red / 255.0, green / 255.0, blue / 255.0, Quantity_TOC_sRGB)

        independent_solid: list[TopoDS_Solid] = []
        for solid in solids:
            # if solid already added to shape tool, create a copy to allow independent colour assignment
            if solid in self._shape_labels:
                # create copy of solid
                solid = BRepBuilderAPI_Copy(solid).Shape()
                solid_label = self._shape_tool.AddShape(solid)
                self._shape_labels[solid] = solid_label
            else:
                solid_label = self._shape_tool.AddShape(solid)
                self._shape_labels[solid] = solid_label

            # apply colour to solid
            if rgb:
                self._colour_tool.SetColor(solid_label, quantity_colour, XCAFDoc_ColorGen)

            # add solid to compound
            self._builder.Add(compound, solid)

        # add compound to root assembly
        self._add_shape_to_assembly(compound, label, rgb)

    def add_solid(self, solid: TopoDS_Solid, label: str, rgb: list[int] | None = None) -> None:
        """Add a single named solid to the root assembly.

        A solid already added to the writer is copied first.

        Parameters
        ----------
        solid : TopoDS_Solid
            Solid to add.
        label : str
            Name of the component.
        rgb : list[int], optional
            RGB colour (0 - 255, sRGB) applied to the solid, by default None
        """
        # case where solid already added, create a copy
        if solid in self._shape_labels:
            solid = BRepBuilderAPI_Copy(solid).Shape()

        # if new solid, add to shape tool and get label, store in lookup
        solid_label = self._shape_tool.AddShape(solid, False, False)
        self._shape_labels[solid] = solid_label

        # apply colour
        if rgb:
            red, green, blue = rgb
            quantity_colour = Quantity_Color(red / 255.0, green / 255.0, blue / 255.0, Quantity_TOC_sRGB)
            self._colour_tool.SetColor(solid_label, quantity_colour, XCAFDoc_ColorGen)

        # add solid to root assembly
        self._add_shape_to_assembly(solid, label, rgb)

    @check_folder_exists_for_file_path("output_path")
    def write(self, output_path: os.PathLike, enumerate_files: bool = False) -> str:
        """Write the assembly to a STEP file (AP214IS schema).

        The root assembly is named after the output filename.

        Parameters
        ----------
        output_path : os.PathLike
            Path of the STEP file; missing folders are created.
        enumerate_files : bool, optional
            If True, prefix the filename with a 6-digit counter based on the
            number of existing files in the folder ending with the same name,
            by default False

        Returns
        -------
        str
            The path actually written to.

        Raises
        ------
        Exception
            If transfer of the document to the writer or the write fails.
        """
        dirs, file = os.path.split(output_path)

        if dirs and not os.path.isdir(dirs):
            os.makedirs(dirs, exist_ok=True)
            logger.info(f'making dir {dirs}')

        if enumerate_files:
            dirs = dirs if dirs else '.'
            n_files = len([
                f for f in os.listdir(dirs)
                if os.path.isfile(os.path.join(dirs, f)) and (len(f) >= len(file)) and (f[-len(file):] == file)
            ])
            output_path = os.path.join(dirs, f'{n_files+1:0>6}_{file}')

        # get the root assembly to match the output filename
        filename_base = os.path.splitext(os.path.basename(output_path))[0]
        TDataStd_Name.Set_s(self._root_label, TCollection_ExtendedString(filename_base))

        # rebuild assembly geometry compounds from component sub-labels, important, don't remove this!
        self._shape_tool.UpdateAssemblies()

        # initalise controller and set output parameters
        STEPCAFControl_Controller.Init_s()
        Interface_Static.SetIVal_s('write.stepcaf.subshapes.name', 1)
        Interface_Static.SetCVal_s('write.step.schema', 'AP214IS')

        # set up writer and set modes
        writer: STEPCAFControl_Writer = STEPCAFControl_Writer()
        XCAFDoc_DocumentTool.SetLengthUnit_s(self._doc, self.decom_params.doc_unit_in_metres)
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.SetLayerMode(True)

        # transfer document to writer
        if not writer.Transfer(self._doc, STEPControl_AsIs):
            raise Exception('[ERROR] - transfer of document to writer has failed...')

        # write step file
        if writer.Write(output_path) != IFSelect_RetDone:
            raise Exception(f'[ERROR] - save of document to step file failed: {output_path}')

        return output_path

def _get_ocp_surface_from_ocp_face(face: TopoDS_Face, remove_trimmed_surface: bool=True) -> Geom_Surface:
    """Get the underlying geometric surface of a face.

    Parameters
    ----------
    face : TopoDS_Face
        Face whose surface is required.
    remove_trimmed_surface : bool, optional
        If True and the surface is a ``Geom_RectangularTrimmedSurface``, return
        its basis surface instead, by default True

    Returns
    -------
    Geom_Surface
        The face's surface.
    """
    # extract surface from face
    surface: Geom_Surface = BRep_Tool.Surface_s(TopoDS.Face_s(face))

    # check if it's Geom_RectangularTrimmedSurface
    if remove_trimmed_surface and isinstance(surface, Geom_RectangularTrimmedSurface):
        surface = BRep_Tool.Surface_s(TopoDS.Face_s(face)).BasisSurface()

    # return surface
    return surface
