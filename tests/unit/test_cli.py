"""Tests for Cournot CLI functionality.

This module tests the command-line interface for the Cournot simulation,
ensuring that the CLI properly parses arguments and produces expected output.
"""

import subprocess
import sys
from io import StringIO
from unittest.mock import patch
import pytest

from src.sim.cli import main


class TestCournotCLI:
    """Test cases for Cournot CLI functionality."""
    
    def test_cli_basic_functionality(self) -> None:
        """Test CLI with basic arguments produces expected output."""
        # Capture stdout
        with patch('sys.argv', ['cournot', '--a', '100', '--b', '1', '--costs', '10,20', '--q', '10,20']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                
                # Check that output contains expected values
                assert "P=70" in output
                assert "q_0=10" in output
                assert "q_1=20" in output
                assert "π_0=600" in output
                assert "π_1=1000" in output
                
    def test_cli_single_firm(self) -> None:
        """Test CLI with single firm."""
        with patch('sys.argv', ['cournot', '--a', '50', '--b', '0.5', '--costs', '15', '--q', '20']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                
                # P = 50 - 0.5 * 20 = 40
                # π = (40 - 15) * 20 = 500
                assert "P=40" in output
                assert "q_0=20" in output
                assert "π_0=500" in output
                
    def test_cli_multiple_firms(self) -> None:
        """Test CLI with multiple firms."""
        with patch('sys.argv', ['cournot', '--a', '60', '--b', '0.5', '--costs', '5,10,15', '--q', '20,15,10']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                
                # P = 60 - 0.5 * (20 + 15 + 10) = 37.5
                assert "P=37.5" in output
                assert "q_0=20" in output
                assert "q_1=15" in output
                assert "q_2=10" in output
                
    def test_cli_error_handling_negative_quantities(self) -> None:
        """Test CLI error handling for negative quantities."""
        with patch('sys.argv', ['cournot', '--a', '100', '--b', '1', '--costs', '10', '--q', '-5']):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                error_output = mock_stderr.getvalue()
                assert "Quantity q_0 = -5.0 must be non-negative" in error_output
                
    def test_cli_error_handling_invalid_costs(self) -> None:
        """Test CLI error handling for invalid costs format."""
        with patch('sys.argv', ['cournot', '--a', '100', '--b', '1', '--costs', '10,abc', '--q', '10,20']):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                error_output = mock_stderr.getvalue()
                assert "Invalid costs format" in error_output
                
    def test_cli_error_handling_invalid_quantities(self) -> None:
        """Test CLI error handling for invalid quantities format."""
        with patch('sys.argv', ['cournot', '--a', '100', '--b', '1', '--costs', '10,20', '--q', '10,xyz']):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                error_output = mock_stderr.getvalue()
                assert "Invalid quantities format" in error_output
                
    def test_cli_mismatched_lengths(self) -> None:
        """Test CLI error handling for mismatched costs and quantities lengths."""
        with patch('sys.argv', ['cournot', '--a', '100', '--b', '1', '--costs', '10,20', '--q', '10,20,30']):
            with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                with pytest.raises(SystemExit):
                    main()
                error_output = mock_stderr.getvalue()
                assert "Costs list length (2) must match quantities list length (3)" in error_output
                
    def test_cli_zero_price_scenario(self) -> None:
        """Test CLI with scenario that results in zero price."""
        with patch('sys.argv', ['cournot', '--a', '100', '--b', '1', '--costs', '10,20', '--q', '50,50']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                
                # P = 0, π1 = -500, π2 = -1000
                assert "P=0" in output
                assert "π_0=-500" in output
                assert "π_1=-1000" in output
                
    def test_cli_fractional_values(self) -> None:
        """Test CLI with fractional values."""
        with patch('sys.argv', ['cournot', '--a', '75.5', '--b', '1.25', '--costs', '12.5', '--q', '30.2']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                
                # P = 75.5 - 1.25 * 30.2 = 37.75
                # π = (37.75 - 12.5) * 30.2 = 762.55
                assert "P=37.75" in output
                assert "π_0=762.55" in output
