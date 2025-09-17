"""Tests for CLI module."""

import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from src.sim.cli.cli import bertrand_main, cournot_main, main


class TestCournotMain:
    """Test the cournot_main function."""

    def test_cournot_main_success(self) -> None:
        """Test successful Cournot simulation."""
        test_args = [
            "cournot",
            "--a",
            "100.0",
            "--b",
            "1.0",
            "--costs",
            "10,20",
            "--q",
            "15,25",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("src.sim.cli.cli.cournot_simulation") as mock_sim:
                # Mock simulation result
                mock_result = Mock()
                mock_result.price = 60.0
                mock_result.quantities = [15.0, 25.0]
                mock_result.profits = [750.0, 1000.0]
                mock_sim.return_value = mock_result

                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    cournot_main()

                    output = mock_stdout.getvalue()
                    assert "P=60.0" in output
                    assert "q_0=15.0, π_0=750.0" in output
                    assert "q_1=25.0, π_1=1000.0" in output

    def test_cournot_main_parsing_error(self) -> None:
        """Test Cournot main with parsing error."""
        test_args = [
            "cournot",
            "--a",
            "100.0",
            "--b",
            "1.0",
            "--costs",
            "invalid",
            "--q",
            "15,25",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("sys.stderr", new_callable=StringIO):
                with pytest.raises(SystemExit):
                    cournot_main()

    def test_cournot_main_simulation_error(self) -> None:
        """Test Cournot main with simulation error."""
        test_args = [
            "cournot",
            "--a",
            "100.0",
            "--b",
            "1.0",
            "--costs",
            "10,20",
            "--q",
            "15,25",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("src.sim.cli.cli.cournot_simulation") as mock_sim:
                mock_sim.side_effect = ValueError("Simulation error")

                with patch("sys.stderr", new_callable=StringIO):
                    with pytest.raises(SystemExit):
                        cournot_main()

    def test_cournot_main_unexpected_error(self) -> None:
        """Test Cournot main with unexpected error."""
        test_args = [
            "cournot",
            "--a",
            "100.0",
            "--b",
            "1.0",
            "--costs",
            "10,20",
            "--q",
            "15,25",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("src.sim.cli.cli.cournot_simulation") as mock_sim:
                mock_sim.side_effect = RuntimeError("Unexpected error")

                with patch("sys.stderr", new_callable=StringIO):
                    with pytest.raises(SystemExit):
                        cournot_main()

    def test_cournot_main_missing_required_args(self) -> None:
        """Test Cournot main with missing required arguments."""
        test_args = ["cournot", "--a", "100.0"]  # Missing --b, --costs, --q

        with patch.object(sys, "argv", test_args):
            with patch("sys.stderr", new_callable=StringIO):
                with pytest.raises(SystemExit):
                    cournot_main()


class TestBertrandMain:
    """Test the bertrand_main function."""

    def test_bertrand_main_success(self) -> None:
        """Test successful Bertrand simulation."""
        test_args = [
            "bertrand",
            "--alpha",
            "120.0",
            "--beta",
            "1.2",
            "--costs",
            "20,25",
            "--p",
            "22,24",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("src.sim.cli.cli.bertrand_simulation") as mock_sim:
                # Mock simulation result
                mock_result = Mock()
                mock_result.total_demand = 50.0
                mock_result.prices = [22.0, 24.0]
                mock_result.quantities = [25.0, 25.0]
                mock_result.profits = [50.0, -25.0]
                mock_sim.return_value = mock_result

                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    bertrand_main()

                    output = mock_stdout.getvalue()
                    assert "Q=50.0" in output
                    assert "p_0=22.0, q_0=25.0, π_0=50.0" in output
                    assert "p_1=24.0, q_1=25.0, π_1=-25.0" in output

    def test_bertrand_main_parsing_error(self) -> None:
        """Test Bertrand main with parsing error."""
        test_args = [
            "bertrand",
            "--alpha",
            "120.0",
            "--beta",
            "1.2",
            "--costs",
            "invalid",
            "--p",
            "22,24",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("sys.stderr", new_callable=StringIO):
                with pytest.raises(SystemExit):
                    bertrand_main()

    def test_bertrand_main_simulation_error(self) -> None:
        """Test Bertrand main with simulation error."""
        test_args = [
            "bertrand",
            "--alpha",
            "120.0",
            "--beta",
            "1.2",
            "--costs",
            "20,25",
            "--p",
            "22,24",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("src.sim.cli.cli.bertrand_simulation") as mock_sim:
                mock_sim.side_effect = ValueError("Simulation error")

                with patch("sys.stderr", new_callable=StringIO):
                    with pytest.raises(SystemExit):
                        bertrand_main()

    def test_bertrand_main_unexpected_error(self) -> None:
        """Test Bertrand main with unexpected error."""
        test_args = [
            "bertrand",
            "--alpha",
            "120.0",
            "--beta",
            "1.2",
            "--costs",
            "20,25",
            "--p",
            "22,24",
        ]

        with patch.object(sys, "argv", test_args):
            with patch("src.sim.cli.cli.bertrand_simulation") as mock_sim:
                mock_sim.side_effect = RuntimeError("Unexpected error")

                with patch("sys.stderr", new_callable=StringIO):
                    with pytest.raises(SystemExit):
                        bertrand_main()

    def test_bertrand_main_missing_required_args(self) -> None:
        """Test Bertrand main with missing required arguments."""
        test_args = ["bertrand", "--alpha", "120.0"]  # Missing --beta, --costs, --p

        with patch.object(sys, "argv", test_args):
            with patch("sys.stderr", new_callable=StringIO):
                with pytest.raises(SystemExit):
                    bertrand_main()


class TestMain:
    """Test the main function."""

    def test_main_calls_cournot_main(self) -> None:
        """Test that main calls cournot_main."""
        with patch("src.sim.cli.cli.cournot_main") as mock_cournot_main:
            main()
            mock_cournot_main.assert_called_once()

    def test_main_module_execution(self) -> None:
        """Test that main is called when module is executed."""
        # This test verifies the if __name__ == "__main__" block
        # We can't easily test this directly, but we can verify the function exists
        assert callable(main)


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    def test_cournot_help_message(self) -> None:
        """Test that Cournot CLI shows help message."""
        test_args = ["cournot", "--help"]

        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit):
                    cournot_main()

                help_output = mock_stdout.getvalue()
                assert "Run a one-round Cournot oligopoly simulation" in help_output
                assert "--a" in help_output
                assert "--b" in help_output
                assert "--costs" in help_output
                assert "--q" in help_output

    def test_bertrand_help_message(self) -> None:
        """Test that Bertrand CLI shows help message."""
        test_args = ["bertrand", "--help"]

        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit):
                    bertrand_main()

                help_output = mock_stdout.getvalue()
                assert "Run a one-round Bertrand oligopoly simulation" in help_output
                assert "--alpha" in help_output
                assert "--beta" in help_output
                assert "--costs" in help_output
                assert "--p" in help_output

    def test_cournot_examples_in_help(self) -> None:
        """Test that Cournot help includes examples."""
        test_args = ["cournot", "--help"]

        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit):
                    cournot_main()

                help_output = mock_stdout.getvalue()
                assert "cournot --a 100 --b 1 --costs 10,20 --q 10,20" in help_output

    def test_bertrand_examples_in_help(self) -> None:
        """Test that Bertrand help includes examples."""
        test_args = ["bertrand", "--help"]

        with patch.object(sys, "argv", test_args):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit):
                    bertrand_main()

                help_output = mock_stdout.getvalue()
                assert (
                    "bertrand --alpha 120 --beta 1.2 --costs 20,20,25 --p 22,21,24"
                    in help_output
                )
