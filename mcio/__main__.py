"""Command line parser: methods exposed which can be
called from the command line.

Run as ``python -m mcio <command> ...``; sub-commands are ``decompose``,
``lost_particles``, ``analysis`` and ``stochastic_mcnp``. Heavy imports (OCP,
rich, mcio submodules) are deferred until after argument parsing, inside the
``__main__`` block, and the command functions rely on those module globals.
"""

from __future__ import annotations

from typing import Any

import argparse
import os



def run_decompose(args: argparse.Namespace) -> None:
    """Run the CAD to CSG decomposer and write the output models.

    Writes MCNP (and, if available, OpenMC) models, the decomposition
    parameters, debugging STEP files for slow or heavily split solids, the void
    octree (if requested) and an ADVANTG compatible MCNP model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of the ``decompose`` sub-command.
    """

    # set debugging level
    if args.debug:
        logger.setLevel(LOGGER_LEVELS.DEBUG)

    # set up default decompose parameters
    decom_params = DecompositionParameters()

    # write decom params file
    if args.write_decompose_parameters:
        decom_params.write(args.write_decompose_parameters)
        exit()

    # read decompose params from json file
    if args.read_decompose_parameters:
        decom_params = DecompositionParameters.read(args.read_decompose_parameters)

    decomposer = mcio.decompose.Decomposer(
        step_file_paths=args.input,
        materials_file_path=args.materials if args.materials else None,
        decompose_parameters=decom_params,
        surface_number_min=args.surface_number_minimum,
        cell_number_min=args.cell_number_minimum
    )

    decomposer.run(cores=args.cores, debug=args.debug)

    # apply tallies from json if provided
    if args.tallies:
        decomposer.tallies_from_json(args.tallies)

    # write output files for mcnp
    decomposer.model.mcnp._prdmp_intervals = 100
    decomposer.model.mcnp._void = True
    decomposer.model.mcnp.xsdir = (
        r"/lustre/restricted_software/mcnp/xs/xsdir_mcnp6.2_all"
    )
    decomposer.model.write_mcnp(args.output_mcnp)

    # save decompose params
    decomposer.decom_params.write(os.path.join("output", "decompose_parameters.json"))

    # write output files for openmc if available
    try:
        import openmc
    except ModuleNotFoundError:
        pass

    # check if openmc loaded, if so, write openmc output file
    if importlib.util.find_spec("openmc") is not None:
        decomposer.model.write_openmc(args.output_openmc)

    # debugging
    for i, decomposition_solid in enumerate(decomposer.decomposition_solids, start=1):
        n_split_solids = len(decomposition_solid.split_solids)
        if n_split_solids > decomposer.decom_params.debug_large_number_of_split_solids:
            step_file_writer = STEPFileWriter(decomposer.decom_params)
            step_file_writer.add_solids([decomposition_solid.original_solid], "original_solid")
            step_file_writer.add_solids(
                [
                    split_solid.split_solid
                    for split_solid in decomposition_solid.split_solids
                ],
                "split_solids",
            )
            output_path = step_file_writer.write(
                os.path.join(
                    decomposer.decom_params.debug_large_number_of_split_solids_directory,
                    f"solid_{i}_split_solids_{n_split_solids}.stp",
                )
            )
            logger.info(
                f"solid_{i} produced {n_split_solids} split solids - see {output_path}"
            )

        if (
            decomposition_solid.profile_time_decomposition
            > decomposer.decom_params.debug_long_conversion_time
        ):
            step_file_writer = STEPFileWriter(decomposer.decom_params)
            step_file_writer.add_solids(
                [decomposition_solid.original_solid], "original_solid"
            )
            step_file_writer.add_solids(
                [
                    split_solid.split_solid
                    for split_solid in decomposition_solid.split_solids
                ],
                "split_solids",
            )
            output_path = step_file_writer.write(
                os.path.join(
                    decomposer.decom_params.debug_long_conversion_time_directory,
                    f"solid_{i}_time_{decomposition_solid.profile_time_decomposition:.2e}.stp",
                )
            )
            logger.info(
                f"solid_{i} produced {n_split_solids} split solids - see {output_path}"
            )

        output_path = decomposition_solid.save(os.path.join(decomposer.decom_params.debug_save_solid, f'decomposed_solid_{i}.step'))

    # save void octree
    if args.save_voids:
        all_voids = decomposer._void_octree.voids
        start_index = len(decomposer.decomposition_solids) + 1

        desc_stage = '[Saving void octree]'
        desc = f'[cyan]{desc_stage: <30}[/cyan]'
        progress = Progress(
            TextColumn(desc),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )

        with progress:
            task_id = progress.add_task("saving_voids", total=len(all_voids))

            for i, void_octree in enumerate(all_voids, start=start_index):
                void_octree.save(os.path.join("analysis", "voids", f"void_octree_{i}.stp"))
                progress.advance(task_id)

    # write alternative mcnp output for advantg removing mising isotopes
    if args.output_mcnp_advantg:
        ADVANTG_MISSING_ISOTOPES = [
            "C13",
            "O18",
            "Pt190",
            "Pt192",
            "Pt194",
            "Pt195",
            "Pt196",
            "Pt196",
            "Pt198",
            "Tl203",
            "Tl205",
        ]

        for cell in decomposer.model.cells:
            if cell.material:
                cell.material.strip_isotopes(ADVANTG_MISSING_ISOTOPES)

        decomposer.model.mcnp.xsdir = (
            "/lustre/restricted_software/mcnp/xs/xsdir_mcnp5_all"
        )
        decomposer.model.write_mcnp(args.output_mcnp_advantg)


def lost_particle(args: argparse.Namespace) -> None:
    """Convert lost particles from an MCNP output file to CAD.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of the ``lost_particles`` sub-command.
    """

    lost_particles_from_mcnp_run(
        input_file_path=args.input,
        output_file_path=args.output,
        max_particles=args.max_particles,
        vector_length=args.vector_length,
        radius=args.radius,
    )


def run_analysis(input_file_paths: Any) -> None:
    """Print a per-body analysis of the input CAD.

    Parameters
    ----------
    input_file_paths : Any
        STEP file paths passed to ``Decomposer``. Note the command dispatch
        passes the parsed ``argparse.Namespace`` here.
    """
    decomposer = Decomposer(input_file_paths)
    print(decomposer.analysis.per_body)


def stochastic_mcnp(args: argparse.Namespace) -> None:
    """Plot MCNP stochastic volume check results (not yet implemented).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments of the ``stochastic_mcnp`` sub-command.
    """
    pass


def stochastic_openmc(args: argparse.Namespace) -> None:
    """Plot OpenMC stochastic volume check results (not yet implemented).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    pass


def main() -> argparse.Namespace | None:
    """Build the command line parser and parse the arguments.

    Help is printed when no command, or the ``help`` command, is given.

    Returns
    -------
    argparse.Namespace or None
        Parsed arguments, or None if help was printed.
    """
    parser = argparse.ArgumentParser(description="MCIO command-line interface")

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # help - allow sub parser printing of parameters
    help_parser = subparsers.add_parser("help", help="Show help for a command")
    help_parser.add_argument("topic", nargs="?")

    # sub_parser - decompose
    decomp_help_msg = 'run cad to csg decomposer, e.g. ' +\
                      'python -m mcio decompose -i model.stp model_2.stp -n 12'
    decomp = subparsers.add_parser("decompose", help=decomp_help_msg)
    decomp.add_argument("--input", "-i", nargs="+", help="list of step files")
    decomp.add_argument(
        "--output_mcnp",
        "-om",
        default=os.path.join("output", "model_mcnp.i"),
        help="output folder path for mcnp model",
    )
    decomp.add_argument(
        "--output_openmc",
        "-oo",
        default=os.path.join("output", "model_openmc"),
        help="output folder path for openmc model",
    )
    decomp.add_argument(
        "--output_mcnp_advantg",
        "-oa",
        default=os.path.join('output', 'model_advantg.i'),
        help="generate model without missing advantg isotopes",
    )
    decomp.add_argument(
        "--cores",
        "-c",
        default=20,
        type=int,
        help="number of cores to apply multiprocess",
    )
    decomp.add_argument(
        "--materials",
        "-m",
        default=None,
        type=str,
        help="assign materials from json file, either rgb or _Mn_ labels",
    )
    decomp.add_argument(
        "--write_decompose_parameters",
        "-wdp",
        nargs="?",
        type=str,
        default=None,
        const="decompose_parameters.json",
    )
    decomp.add_argument(
        "--read_decompose_parameters",
        "-rdp",
        nargs="?",
        type=str,
        default=None,
        const="decompose_parameters.json",
    )
    decomp.add_argument("--write_formatting", "-wf")
    decomp.add_argument(
        "--read_formatting",
        "-rf",
        nargs="?",
        type=str,
        default=None,
        const="mcnp_formatting.json",
    )
    decomp.add_argument(
        "--tallies", "-t", type=str, default=None, help="json tally file"
    )
    decomp.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="no debugging will allow fatal exceptions to be raised and " +\
             "halt program if a problematic solid is found",
    )
    decomp.add_argument(
        "--save_voids",
        "-sv",
        action="store_true",
        help="save void octree with interseting solids to step files " +\
             "(warning, can produce large volumes of files)",
    )
    decomp.add_argument(
        "--surface_number_minimum",
        '-snm',
        default=None,
        type=int,
        help='minimum surface number for mcnp surface decks'
    )
    decomp.add_argument(
            "--cell_number_minimum",
            '-cnm',
            default=None,
            type=int,
            help='minimum cell number for mcnp surface decks'
        )


    # sub_parser - lost_particles
    lp_help_msg = "lost particles to cad, e.g. python -m mcio " +\
                  "lost_particles -i run1.o -o lost_particles.stp"
    lp = subparsers.add_parser("lost_particles", help=lp_help_msg)
    lp.add_argument("--input", "-i", required=True, help="path to mcnp output")
    lp.add_argument(
        "--output",
        "-o",
        required=False,
        default="lost_particles.stp",
        help="output step file path",
    )
    lp.add_argument(
        "--vector_length",
        "-vl",
        type=float,
        default=50.0,
        help="length of vector lines",
    )
    lp.add_argument(
        "--max_particles",
        "-m",
        type=int,
        default=100,
        help="number of lost particles to extract",
    )
    lp.add_argument(
        "--radius",
        "-r",
        type=float,
        default=2.0,
        help="radius of sphere representing point particle is lost",
    )

    # sub_parser - analysis
    analysis_help_msg = (
        "analysis of input cad, e.g. python -m mcio analysis -i m1.stp m2.stp"
    )
    analysis = subparsers.add_parser("analysis", help=analysis_help_msg)
    analysis.add_argument(
        "--input", "-i", nargs="+", required=True, help="input step file"
    )

    # sub parser - MCNP stochastic volume checks
    stochastic_mcnp_vol_msg = (
        "plotting for stochastic volume check, e.g. --model model_1.i"
    )
    stochastic_mcnp_vol = subparsers.add_parser(
        "stochastic_mcnp", help=stochastic_mcnp_vol_msg
    )
    stochastic_mcnp_vol.add_argument(
        "--mctal",
        "--m",
        type=str,
        required=True,
        help="MCTAL output file from MCNP stochastic volume run",
    )
    stochastic_mcnp_vol.add_argument(
        "--volumes", "--v", type=str, required=False, help="volumes json file"
    )

    # Parse args
    args = parser.parse_args()

    # handle help for sub-parsers
    if (args.command == "help") or isinstance(args.command, type(None)):
        if isinstance(args.command, type(None)):
            parser.print_help()
        elif args.command == "help" and (args.topic in subparsers.choices):
            subparsers.choices[args.topic].print_help()
        else:
            parser.print_help()
        return

    return args



if __name__ == "__main__":
    # 
    args = main()

    # store functions for command line methods
    METHODS = {
        "lost_particles": lost_particle,
        "analysis": run_analysis,
        "decompose": run_decompose,
        "stochastic_mcnp": stochastic_mcnp,
    }

    # imports after parse of command line args, to avoid imoport of ocp as slow
    import importlib
    import importlib.util
    import json
    from multiprocessing import Pool
    from rich.progress import (
        Progress,
        ProgressColumn,
        TextColumn,
        MofNCompleteColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    import mcio
    from . import LOGGER_LEVELS
    from .diagnostic import lost_particles_from_mcnp_run
    from .decompose import Decomposer
    from .decomposeparams import DecompositionParameters
    from .ocp import STEPFileWriter
    from .decomposemultiprocess import _run_decomposer_solid
    from mcio import logger
    from .parallellogging import (
        start_multiprocess_logging,
        restore_main_process_logging,
        configure_worker_logger,
    )
    from .utils import in_notebook

    if args.command in METHODS:
        METHODS[args.command](args)
