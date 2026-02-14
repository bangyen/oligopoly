"""Visualization system for oligopoly experiment results.

This module provides tools for generating Plotly charts from experiment CSV results,
focusing on profits, HHI, and collusion dynamics.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px


class ExperimentVisualizer:
    """Generates automated plots from experiment result CSVs."""

    def __init__(self, output_dir: str = "artifacts/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self, csv_path: str) -> List[str]:
        """Generate a standard suite of plots from a result CSV."""
        df = pd.read_csv(csv_path)
        plot_paths = []

        # 1. Profit Comparison by Strategy
        plot_paths.append(self.plot_profit_by_strategy(df))

        # 2. HHI vs Average Price (Market Power)
        plot_paths.append(self.plot_market_power(df))

        # 3. Cartel Duration Distribution (if applicable)
        if "cartel_duration" in df.columns and df["cartel_duration"].sum() > 0:
            plot_paths.append(self.plot_cartel_dynamics(df))

        return [p for p in plot_paths if p is not None]

    def plot_profit_by_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """Plot mean profits grouped by strategy types across experiments."""
        # Reshape data to long format for strategies if multiple firms
        # This is complex because we have firm_n_strategy and firm_n_profit
        # Let's simplify and plot mean profit per configuration
        fig = px.box(
            df,
            x="config_id",
            y="mean_profit_per_firm",
            color="model",
            title="Profit Distribution by Experiment Configuration",
            labels={
                "mean_profit_per_firm": "Mean Profit per Firm",
                "config_id": "Experiment ID",
            },
        )

        output_path = self.output_dir / "profit_distribution.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def plot_market_power(self, df: pd.DataFrame) -> Optional[str]:
        """Scatter plot of HHI vs Average Price."""
        fig = px.scatter(
            df,
            x="avg_hhi",
            y="avg_price",
            color="config_id",
            size="mean_profit_per_firm",
            hover_data=["model", "num_firms"],
            title="Market Power: HHI vs Average Price",
            labels={
                "avg_hhi": "Market Concentration (HHI)",
                "avg_price": "Average Market Price",
            },
        )

        output_path = self.output_dir / "market_power_scatter.html"
        fig.write_html(str(output_path))
        return str(output_path)

    def plot_cartel_dynamics(self, df: pd.DataFrame) -> Optional[str]:
        """Plot defection rates and cartel duration."""
        if "total_defections" not in df.columns:
            return None

        fig = px.bar(
            df.groupby("config_id")["total_defections"].mean().reset_index(),
            x="config_id",
            y="total_defections",
            title="Average Defections per Configuration",
            labels={
                "total_defections": "Mean Defections",
                "config_id": "Experiment ID",
            },
        )

        output_path = self.output_dir / "cartel_dynamics.html"
        fig.write_html(str(output_path))
        return str(output_path)
