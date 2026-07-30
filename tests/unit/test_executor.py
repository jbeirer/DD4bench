"""Unit tests for k4bench.runner.executor.

_build_command is a pure function (no subprocess, no filesystem), so we
test it directly.  run_ddsim itself requires a live ddsim binary and
belongs in the integration suite, except for the subprocess-mocked
regression tests in TestVerboseReturncode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from k4bench.runner.executor import _build_command, run_ddsim

XML = Path("/geo/ALLEGRO_o1_v03.xml")
OUTPUT = Path("/tmp/out.edm4hep.root")
SETUP = Path("/opt/k4geo/bin/thisk4geo.sh")


class TestBuildCommandManagedArgs:
    """The three executor-owned flags are always present and correctly set."""

    def _cmd(self, **kwargs) -> str:
        defaults = dict(
            xml_path=XML, n_events=5, output_file=OUTPUT,
            setup_script=None, extra_args=[], plugin_available=False,
        )
        return _build_command(**{**defaults, **kwargs})

    def test_compact_file_present(self):
        assert f"--compactFile={XML}" in self._cmd()

    def test_number_of_events_present(self):
        assert "--numberOfEvents=5" in self._cmd()

    def test_output_file_present(self):
        assert f"--outputFile={OUTPUT}" in self._cmd()

    def test_time_wrapper_present(self):
        assert "/usr/bin/time -v ddsim" in self._cmd()

    def test_n_events_reflects_argument(self):
        assert "--numberOfEvents=99" in self._cmd(n_events=99)


class TestBuildCommandSetupScript:
    """setup_script is optional; when supplied it is sourced first."""

    def test_no_setup_script_omits_source_line(self):
        cmd = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=[], plugin_available=False,
        )
        assert "source" not in cmd

    def test_setup_script_sourced_before_ddsim(self):
        cmd = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=SETUP, extra_args=[], plugin_available=False,
        )
        assert f"source {SETUP}" in cmd
        # source line must precede the ddsim invocation
        assert cmd.index("source") < cmd.index("ddsim")


class TestBuildCommandExtraArgs:
    """Caller-supplied extra_args are passed through and shell-quoted."""

    def _cmd(self, extra_args: list[str]) -> str:
        return _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=extra_args, plugin_available=False,
        )

    def test_single_flag_present(self):
        cmd = self._cmd(["--runType=batch"])
        assert "--runType=batch" in cmd

    def test_multiple_flags_all_present(self):
        cmd = self._cmd(["--runType=batch", "--enableGun"])
        assert "--runType=batch" in cmd
        assert "--enableGun" in cmd

    def test_key_value_pair_present(self):
        cmd = self._cmd(["--gun.particle", "e-"])
        assert "--gun.particle" in cmd
        assert "e-" in cmd

    def test_value_with_spaces_is_quoted(self):
        # shlex.quote wraps values containing spaces in single quotes.
        cmd = self._cmd(["--someArg", "value with spaces"])
        assert "'value with spaces'" in cmd

    def test_empty_extra_args_does_not_add_noise(self):
        cmd_empty = self._cmd([])
        cmd_none = _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=[], plugin_available=False,
        )
        assert cmd_empty == cmd_none

    def test_extra_args_come_after_managed_args(self):
        cmd = self._cmd(["--enableGun"])
        managed_end = max(
            cmd.index("--compactFile"),
            cmd.index("--numberOfEvents"),
            cmd.index("--outputFile"),
        )
        assert cmd.index("--enableGun") > managed_end


class TestBuildCommandPluginAvailability:
    """plugin_available controls whether the timing action is injected."""

    def _cmd(self, *, plugin_available: bool, extra_args: list[str] | None = None) -> str:
        return _build_command(
            xml_path=XML, n_events=2, output_file=OUTPUT,
            setup_script=None, extra_args=extra_args or [],
            plugin_available=plugin_available,
        )

    def test_timing_action_injected_when_plugin_available(self):
        cmd = self._cmd(plugin_available=True)
        assert "k4BenchTimingAction" in cmd
        assert "--action.event" in cmd

    def test_timing_action_absent_when_plugin_unavailable(self):
        cmd = self._cmd(plugin_available=False)
        assert "k4BenchTimingAction" not in cmd

    def test_timing_action_not_duplicated_when_already_in_extra_args(self):
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--action.event", "k4BenchTimingAction"],
        )
        assert cmd.count("k4BenchTimingAction") == 1

    def test_region_actions_injected_when_plugin_available(self):
        cmd = self._cmd(plugin_available=True)
        assert "--action.step" in cmd
        assert "k4BenchRegionTimingAction" in cmd
        assert "--action.track" in cmd
        assert "k4BenchRegionTrackingAction" in cmd
        assert "k4BenchRegionEventAction" in cmd

    def test_region_actions_absent_when_plugin_unavailable(self):
        cmd = self._cmd(plugin_available=False)
        assert "k4BenchRegion" not in cmd

    def test_region_actions_not_duplicated_when_already_in_extra_args(self):
        cmd = self._cmd(
            plugin_available=True,
            extra_args=[
                "--action.step",  "k4BenchRegionTimingAction",
                "--action.track", "k4BenchRegionTrackingAction",
                "--action.event", "k4BenchRegionEventAction",
            ],
        )
        assert cmd.count("k4BenchRegionTimingAction") == 1
        assert cmd.count("k4BenchRegionTrackingAction") == 1
        assert cmd.count("k4BenchRegionEventAction") == 1

    def test_timing_action_not_duplicated_when_supplied_with_an_equals_sign(self):
        # argparse accepts --flag=value as readily as --flag value, so a caller
        # spelling it that way has registered the action just as surely.
        # Injecting a second copy would run the same DDG4 action twice.
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--action.event=k4BenchTimingAction"],
        )
        assert cmd.count("k4BenchTimingAction") == 1

    def test_region_actions_not_duplicated_when_supplied_with_an_equals_sign(self):
        cmd = self._cmd(
            plugin_available=True,
            extra_args=[
                "--action.step=k4BenchRegionTimingAction",
                "--action.track=k4BenchRegionTrackingAction",
                "--action.event=k4BenchRegionEventAction",
            ],
        )
        assert cmd.count("k4BenchRegionTimingAction") == 1
        assert cmd.count("k4BenchRegionTrackingAction") == 1
        assert cmd.count("k4BenchRegionEventAction") == 1

    def test_equals_form_of_a_non_action_flag_does_not_suppress_injection(self):
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--somearg=k4BenchRegionTimingAction"],
        )
        assert cmd.count("k4BenchRegionTimingAction") == 2

    def test_action_name_in_non_action_flag_does_not_suppress_injection(self):
        # Action name appearing after a non --action.* flag must not count
        # as the action being registered (old substring match would suppress it).
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--somearg", "k4BenchRegionTimingAction"],
        )
        assert cmd.count("k4BenchRegionTimingAction") == 2

    def test_missing_region_actions_injected_when_only_step_is_present(self):
        # If the user supplies only the step action, track and event should
        # still be injected rather than silently left out.
        cmd = self._cmd(
            plugin_available=True,
            extra_args=["--action.step", "k4BenchRegionTimingAction"],
        )
        assert cmd.count("k4BenchRegionTimingAction") == 1
        assert "k4BenchRegionTrackingAction" in cmd
        assert "k4BenchRegionEventAction" in cmd


class TestVerboseReturncode:
    """Regression: returncode must not be None after either streaming path."""

    def _make_proc(self):
        """Simulate a Popen where returncode is None until wait() is called."""
        mock_proc = MagicMock()
        mock_proc.returncode = None  # real Popen starts here
        mock_proc.stdout = iter([])  # no output lines

        def set_returncode():
            mock_proc.returncode = 0

        mock_proc.wait.side_effect = set_returncode
        return mock_proc

    def _run(self, tmp_path: Path, verbose: bool):
        mock_proc = self._make_proc()
        with patch("k4bench.runner.executor.subprocess.Popen", return_value=mock_proc):
            return run_ddsim(
                xml_path=XML,
                label="test",
                n_events=2,
                output_file=tmp_path / "out.root",
                log_dir=tmp_path / "logs",
                verbose=verbose,
            )

    def test_verbose_returncode_is_not_none(self, tmp_path):
        """Verbose path must call proc.wait() so returncode is populated."""
        result = self._run(tmp_path, verbose=True)
        assert result.returncode is not None

    def test_nonverbose_returncode_is_not_none(self, tmp_path):
        """Non-verbose path (communicate) must also populate returncode."""
        result = self._run(tmp_path, verbose=False)
        assert result.returncode is not None