"""
unittest: command line parser - sub-command arguments, defaults and help.
"""

import contextlib
import io
import os
import unittest
from unittest import mock

from mcio.__main__ import main


def parse(*argv):
    """Run the parser on ``argv``, returning (args, printed output)."""
    output = io.StringIO()
    with mock.patch("sys.argv", ["mcio", *argv]), contextlib.redirect_stdout(output):
        args = main()
    return args, output.getvalue()


class MCIOUnitTestCLI(unittest.TestCase):
    def test_decompose_defaults(self):
        args, _ = parse("decompose", "-i", "model.stp")
        self.assertEqual(args.command, "decompose")
        self.assertEqual(args.input, ["model.stp"])
        self.assertEqual(args.cores, 20)
        self.assertEqual(args.output_mcnp, os.path.join("output", "model_mcnp.i"))
        self.assertEqual(args.output_openmc, os.path.join("output", "model_openmc"))
        self.assertIsNone(args.materials)
        self.assertIsNone(args.tallies)
        self.assertIsNone(args.xsdir)
        self.assertIsNone(args.xsdir_advantg)
        self.assertFalse(args.debug)

    def test_decompose_options(self):
        args, _ = parse(
            "decompose",
            "-i",
            "a.stp",
            "b.stp",
            "-c",
            "8",
            "-m",
            "mats.json",
            "-t",
            "tallies.json",
            "--xsdir",
            "/data/xsdir",
            "-d",
        )
        self.assertEqual(args.input, ["a.stp", "b.stp"])
        self.assertEqual(args.cores, 8)
        self.assertEqual(args.materials, "mats.json")
        self.assertEqual(args.tallies, "tallies.json")
        self.assertEqual(args.xsdir, "/data/xsdir")
        self.assertTrue(args.debug)

    def test_decompose_parameter_files_default_name_when_flag_given_alone(self):
        args, _ = parse("decompose", "-i", "a.stp", "-rdp")
        self.assertEqual(args.read_decompose_parameters, "decompose_parameters.json")
        args, _ = parse("decompose", "-i", "a.stp", "-rdp", "mine.json")
        self.assertEqual(args.read_decompose_parameters, "mine.json")

    def test_lost_particles(self):
        args, _ = parse(
            "lost_particles", "-i", "run.o", "-m", "10", "-r", "0.5", "-vl", "200"
        )
        self.assertEqual(args.command, "lost_particles")
        self.assertEqual(args.input, "run.o")
        self.assertEqual(args.output, "lost_particles.stp")
        self.assertEqual(args.max_particles, 10)
        self.assertEqual(args.radius, 0.5)
        self.assertEqual(args.vector_length, 200.0)

    def test_analysis(self):
        args, _ = parse("analysis", "-i", "a.stp")
        self.assertEqual(args.command, "analysis")
        self.assertEqual(args.input, ["a.stp"])

    def test_no_command_prints_help(self):
        args, output = parse()
        self.assertIsNone(args)
        self.assertIn("decompose", output)

    def test_help_for_sub_command(self):
        args, output = parse("help", "decompose")
        self.assertIsNone(args)
        self.assertIn("--xsdir", output)

    def test_help_example_uses_valid_options(self):
        # the example in the decompose help text must itself parse
        _, output = parse("help")
        self.assertIn("-c 12", output)
        args, _ = parse("decompose", "-i", "model.stp", "model_2.stp", "-c", "12")
        self.assertEqual(args.cores, 12)

    def _run_decompose_with_mock_decomposer(self, *argv):
        """Run ``run_decompose`` with a mocked decomposer, returning its solids."""
        import mcio.__main__ as cli

        solids = [
            mock.Mock(split_solids=[], profile_time_decomposition=0.0) for _ in range(3)
        ]
        decomposer = mock.Mock(decomposition_solids=solids)
        decomposer.decom_params.debug_large_number_of_split_solids = 50
        decomposer.decom_params.debug_long_conversion_time = 20.0
        decomposer.decom_params.debug_save_solid = "analysis/solids"
        fake_mcio = mock.Mock()
        fake_mcio.decompose.Decomposer.return_value = decomposer
        fake_importlib = mock.Mock()
        fake_importlib.util.find_spec.return_value = None  # no openmc output

        # run_decompose uses module globals the __main__ block normally imports
        with mock.patch.multiple(
            cli,
            create=True,
            mcio=fake_mcio,
            importlib=fake_importlib,
            logger=mock.Mock(),
            LOGGER_LEVELS=mock.Mock(),
            DecompositionParameters=mock.Mock(),
            STEPFileWriter=mock.Mock(),
        ):
            args, _ = parse("decompose", "-i", "a.stp", *argv)
            cli.run_decompose(args)
        return solids

    def test_decompose_saves_solids_only_when_debugging(self):
        for solid in self._run_decompose_with_mock_decomposer():
            solid.save.assert_not_called()

        solids = self._run_decompose_with_mock_decomposer("--debug")
        for i, solid in enumerate(solids, start=1):
            solid.save.assert_called_once_with(
                os.path.join("analysis/solids", f"decomposed_solid_{i}.step")
            )

    def test_unknown_option_exits(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse("decompose", "-i", "a.stp", "-n", "12")


if __name__ == "__main__":
    unittest.main()
