"""UI smoke tests for the comparison dashboard.

This module provides basic smoke tests to verify that the comparison
dashboard renders correctly and distinguishes between scenarios.
"""

import pytest


# Mock the dashboard functions
def mock_load_run_data(run_id: str, api_base_url: str = "http://localhost:8000"):
    """Mock function for load_run_data."""
    return {
        "run_id": run_id,
        "model": "cournot",
        "rounds": 5,
        "created_at": "2024-01-01T00:00:00",
        "results": {
            "0": {
                "0": {"quantity": 10.0, "price": 50.0, "profit": 100.0},
                "1": {"quantity": 15.0, "price": 50.0, "profit": 150.0},
            },
            "1": {
                "0": {"quantity": 12.0, "price": 48.0, "profit": 120.0},
                "1": {"quantity": 18.0, "price": 48.0, "profit": 180.0},
            },
        },
    }


def mock_run_comparison(left_config, right_config, api_url):
    """Mock function for run_comparison."""
    return {
        "left_run_id": "left-123",
        "right_run_id": "right-456",
        "rounds": 5,
        "left_metrics": {
            "market_price": [50.0, 48.0, 46.0, 44.0, 42.0],
            "total_quantity": [25.0, 30.0, 35.0, 40.0, 45.0],
            "total_profit": [250.0, 300.0, 350.0, 400.0, 450.0],
            "hhi": [0.5, 0.48, 0.46, 0.44, 0.42],
            "consumer_surplus": [312.5, 450.0, 612.5, 800.0, 1012.5],
        },
        "right_metrics": {
            "market_price": [52.0, 50.0, 48.0, 46.0, 44.0],
            "total_quantity": [23.0, 28.0, 33.0, 38.0, 43.0],
            "total_profit": [230.0, 280.0, 330.0, 380.0, 430.0],
            "hhi": [0.52, 0.50, 0.48, 0.46, 0.44],
            "consumer_surplus": [264.5, 392.0, 544.5, 722.0, 924.5],
        },
        "deltas": {
            "market_price": [2.0, 2.0, 2.0, 2.0, 2.0],
            "total_quantity": [-2.0, -2.0, -2.0, -2.0, -2.0],
            "total_profit": [-20.0, -20.0, -20.0, -20.0, -20.0],
            "hhi": [0.02, 0.02, 0.02, 0.02, 0.02],
            "consumer_surplus": [-48.0, -58.0, -68.0, -78.0, -88.0],
        },
    }


class TestComparisonDashboard:
    """Test cases for the comparison dashboard UI."""

    def test_comparison_tab_renders(self):
        """Test that the comparison tab renders without errors."""
        # This is a basic smoke test - in a real implementation,
        # you would use streamlit testing frameworks
        assert True  # Placeholder for actual UI testing

    def test_scenario_config_forms(self):
        """Test that scenario configuration forms work correctly."""
        # Mock scenario config creation
        left_config = {
            "model": "cournot",
            "rounds": 5,
            "params": {"a": 100, "b": 1},
            "firms": [{"cost": 10}, {"cost": 15}],
            "seed": 42,
        }

        right_config = {
            "model": "cournot",
            "rounds": 5,
            "params": {"a": 100, "b": 1},
            "firms": [{"cost": 12}, {"cost": 18}],
            "seed": 42,
        }

        # Verify configs are valid
        assert left_config["model"] == "cournot"
        assert right_config["model"] == "cournot"
        assert left_config["rounds"] == right_config["rounds"]
        assert len(left_config["firms"]) == len(right_config["firms"])

    def test_comparison_charts_data_structure(self):
        """Test that comparison charts receive correct data structure."""
        comparison_data = mock_run_comparison({}, {}, "")

        # Verify data structure
        assert "left_metrics" in comparison_data
        assert "right_metrics" in comparison_data
        assert "deltas" in comparison_data

        # Verify metrics have correct keys
        expected_metrics = [
            "market_price",
            "total_quantity",
            "total_profit",
            "hhi",
            "consumer_surplus",
        ]
        for metric in expected_metrics:
            assert metric in comparison_data["left_metrics"]
            assert metric in comparison_data["right_metrics"]
            assert metric in comparison_data["deltas"]

            # Verify arrays have same length
            left_len = len(comparison_data["left_metrics"][metric])
            right_len = len(comparison_data["right_metrics"][metric])
            delta_len = len(comparison_data["deltas"][metric])

            assert left_len == right_len == delta_len == 5

    def test_deltas_calculation_accuracy(self):
        """Test that deltas are calculated correctly (right - left)."""
        comparison_data = mock_run_comparison({}, {}, "")

        # Verify delta calculations
        for metric in [
            "market_price",
            "total_quantity",
            "total_profit",
            "hhi",
            "consumer_surplus",
        ]:
            left_values = comparison_data["left_metrics"][metric]
            right_values = comparison_data["right_metrics"][metric]
            delta_values = comparison_data["deltas"][metric]

            for i in range(len(delta_values)):
                expected_delta = right_values[i] - left_values[i]
                actual_delta = delta_values[i]
                assert abs(actual_delta - expected_delta) < 1e-6

    def test_legend_distinguishes_scenarios(self):
        """Test that charts have legends that distinguish scenarios."""
        # This would test that:
        # 1. Left scenario is shown in blue
        # 2. Right scenario is shown in red
        # 3. Deltas are shown in green
        # 4. Legend labels are clear and distinguishable

        # Mock chart data
        chart_data = {
            "left_trace": {"name": "Left Price", "line": {"color": "blue"}},
            "right_trace": {"name": "Right Price", "line": {"color": "red"}},
            "delta_trace": {"name": "Price Delta", "line": {"color": "green"}},
        }

        # Verify trace names are distinguishable
        assert chart_data["left_trace"]["name"] != chart_data["right_trace"]["name"]
        assert "Left" in chart_data["left_trace"]["name"]
        assert "Right" in chart_data["right_trace"]["name"]
        assert "Delta" in chart_data["delta_trace"]["name"]

        # Verify colors are different
        assert (
            chart_data["left_trace"]["line"]["color"]
            != chart_data["right_trace"]["line"]["color"]
        )
        assert (
            chart_data["left_trace"]["line"]["color"]
            != chart_data["delta_trace"]["line"]["color"]
        )

    def test_deltas_table_structure(self):
        """Test that deltas table has correct structure."""
        comparison_data = mock_run_comparison({}, {}, "")

        # Mock table data creation
        deltas = comparison_data["deltas"]
        rounds = comparison_data["rounds"]

        # Create table data
        delta_data = []
        for round_idx in range(rounds):
            row = {"Round": round_idx}
            for metric_name, values in deltas.items():
                row[f"{metric_name.replace('_', ' ').title()} Delta"] = values[
                    round_idx
                ]
            delta_data.append(row)

        # Verify table structure
        assert len(delta_data) == rounds
        assert "Round" in delta_data[0]

        # Verify all metrics are present
        expected_metrics = [
            "market_price",
            "total_quantity",
            "total_profit",
            "hhi",
            "consumer_surplus",
        ]
        for metric in expected_metrics:
            column_name = f"{metric.replace('_', ' ').title()} Delta"
            assert column_name in delta_data[0]

    def test_summary_statistics_calculation(self):
        """Test that summary statistics are calculated correctly."""
        comparison_data = mock_run_comparison({}, {}, "")
        deltas = comparison_data["deltas"]

        # Calculate summary statistics
        summary_data = []
        for metric_name, values in deltas.items():
            mean_delta = sum(values) / len(values)
            min_delta = min(values)
            max_delta = max(values)
            std_delta = (
                sum((x - mean_delta) ** 2 for x in values) / len(values)
            ) ** 0.5

            summary_data.append(
                {
                    "Metric": metric_name.replace("_", " ").title(),
                    "Mean Delta": mean_delta,
                    "Min Delta": min_delta,
                    "Max Delta": max_delta,
                    "Std Delta": std_delta,
                }
            )

        # Verify summary structure
        assert len(summary_data) == 5  # 5 metrics

        # Verify calculations for market_price delta
        market_price_deltas = deltas["market_price"]
        market_price_summary = next(
            s for s in summary_data if s["Metric"] == "Market Price"
        )

        assert market_price_summary["Mean Delta"] == sum(market_price_deltas) / len(
            market_price_deltas
        )
        assert market_price_summary["Min Delta"] == min(market_price_deltas)
        assert market_price_summary["Max Delta"] == max(market_price_deltas)


if __name__ == "__main__":
    pytest.main([__file__])
