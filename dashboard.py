"""Streamlit dashboard for oligopoly simulation visualization.

This module provides a minimal web interface for exploring simulation results,
including price, quantity, profit, HHI, and consumer surplus over time.
"""

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


def load_run_data(
    run_id: str, api_base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Load simulation run data from the API.

    Args:
        run_id: Unique identifier for the simulation run
        api_base_url: Base URL for the API

    Returns:
        Dictionary containing run results and metadata

    Raises:
        requests.RequestException: If API request fails
        ValueError: If run_id is not found
    """
    try:
        response = requests.get(f"{api_base_url}/runs/{run_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load run data: {e}")
        raise


def calculate_metrics_for_run(run_data: Dict[str, Any]) -> pd.DataFrame:
    """Calculate HHI and consumer surplus metrics for each round.

    Args:
        run_data: Dictionary containing run results and metadata

    Returns:
        DataFrame with metrics for each round
    """
    from sim.metrics import (
        calculate_round_metrics_bertrand,
        calculate_round_metrics_cournot,
    )

    rounds_data = run_data["results"]
    model = run_data["model"]

    metrics_data = []

    for round_idx, round_data in rounds_data.items():
        round_idx = int(round_idx)

        # Extract firm data
        firms_data = list(round_data.values())
        if not firms_data:
            continue

        # Get quantities, prices, and profits
        quantities = [firm["quantity"] for firm in firms_data]
        prices = [firm["price"] for firm in firms_data]
        profits = [firm["profit"] for firm in firms_data]

        # Calculate market metrics
        if model == "cournot":
            market_price = (
                prices[0] if prices else 0.0
            )  # All firms have same price in Cournot
            # Use default demand parameters - in real implementation, these should be stored
            demand_a = 100.0  # This should come from the simulation parameters
            hhi, cs = calculate_round_metrics_cournot(
                quantities, market_price, demand_a
            )
        else:  # bertrand
            total_demand = sum(quantities)
            demand_alpha = 100.0  # This should come from the simulation parameters
            hhi, cs = calculate_round_metrics_bertrand(
                prices, quantities, total_demand, demand_alpha
            )

        metrics_data.append(
            {
                "round": round_idx,
                "market_price": (
                    market_price
                    if model == "cournot"
                    else min(prices) if prices else 0.0
                ),
                "total_quantity": sum(quantities),
                "total_profit": sum(profits),
                "hhi": hhi,
                "consumer_surplus": cs,
                "num_firms": len(firms_data),
            }
        )

    return pd.DataFrame(metrics_data)


def create_metrics_charts(df: pd.DataFrame) -> None:
    """Create interactive charts for market metrics.

    Args:
        df: DataFrame containing metrics data
    """
    if df.empty:
        st.warning("No data available for visualization")
        return

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Market Price",
            "Total Quantity",
            "Total Profit",
            "HHI",
            "Consumer Surplus",
            "Number of Firms",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Price chart
    fig.add_trace(
        go.Scatter(
            x=df["round"], y=df["market_price"], mode="lines+markers", name="Price"
        ),
        row=1,
        col=1,
    )

    # Quantity chart
    fig.add_trace(
        go.Scatter(
            x=df["round"], y=df["total_quantity"], mode="lines+markers", name="Quantity"
        ),
        row=1,
        col=2,
    )

    # Profit chart
    fig.add_trace(
        go.Scatter(
            x=df["round"], y=df["total_profit"], mode="lines+markers", name="Profit"
        ),
        row=2,
        col=1,
    )

    # HHI chart
    fig.add_trace(
        go.Scatter(x=df["round"], y=df["hhi"], mode="lines+markers", name="HHI"),
        row=2,
        col=2,
    )

    # Consumer Surplus chart
    fig.add_trace(
        go.Scatter(
            x=df["round"], y=df["consumer_surplus"], mode="lines+markers", name="CS"
        ),
        row=3,
        col=1,
    )

    # Number of firms chart
    fig.add_trace(
        go.Scatter(
            x=df["round"], y=df["num_firms"], mode="lines+markers", name="Firms"
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        height=900, showlegend=False, title_text="Market Metrics Over Time"
    )
    fig.update_xaxes(title_text="Round")

    st.plotly_chart(fig, use_container_width=True)


def create_firm_breakdown_chart(run_data: Dict[str, Any]) -> None:
    """Create charts showing individual firm performance.

    Args:
        run_data: Dictionary containing run results and metadata
    """
    rounds_data = run_data["results"]

    # Prepare data for firm-level analysis
    firm_data = []
    for round_idx, round_data in rounds_data.items():
        round_idx = int(round_idx)
        for firm_id, firm_info in round_data.items():
            firm_data.append(
                {
                    "round": round_idx,
                    "firm_id": int(firm_id),
                    "quantity": firm_info["quantity"],
                    "price": firm_info["price"],
                    "profit": firm_info["profit"],
                }
            )

    if not firm_data:
        st.warning("No firm data available")
        return

    df_firms = pd.DataFrame(firm_data)

    # Create firm performance charts
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Firm Quantities",
            "Firm Prices",
            "Firm Profits",
            "Market Share",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Get unique firms
    firms = sorted(df_firms["firm_id"].unique())

    for firm_id in firms:
        firm_df = df_firms[df_firms["firm_id"] == firm_id]

        # Quantities
        fig.add_trace(
            go.Scatter(
                x=firm_df["round"],
                y=firm_df["quantity"],
                mode="lines+markers",
                name=f"Firm {firm_id}",
            ),
            row=1,
            col=1,
        )

        # Prices
        fig.add_trace(
            go.Scatter(
                x=firm_df["round"],
                y=firm_df["price"],
                mode="lines+markers",
                name=f"Firm {firm_id}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Profits
        fig.add_trace(
            go.Scatter(
                x=firm_df["round"],
                y=firm_df["profit"],
                mode="lines+markers",
                name=f"Firm {firm_id}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Calculate market shares for each round
    market_share_data = []
    for round_idx, round_data in rounds_data.items():
        round_idx = int(round_idx)
        quantities = [firm["quantity"] for firm in round_data.values()]
        total_qty = sum(quantities)

        for firm_id, firm_info in round_data.items():
            firm_id = int(firm_id)
            market_share = firm_info["quantity"] / total_qty if total_qty > 0 else 0
            market_share_data.append(
                {"round": round_idx, "firm_id": firm_id, "market_share": market_share}
            )

    df_shares = pd.DataFrame(market_share_data)
    for firm_id in firms:
        firm_df = df_shares[df_shares["firm_id"] == firm_id]
        fig.add_trace(
            go.Scatter(
                x=firm_df["round"],
                y=firm_df["market_share"],
                mode="lines+markers",
                name=f"Firm {firm_id}",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=600, title_text="Individual Firm Performance")
    fig.update_xaxes(title_text="Round")

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Oligopoly Simulation Dashboard", page_icon="📊", layout="wide"
    )

    st.title("📊 Oligopoly Simulation Dashboard")
    st.markdown("Visualize market dynamics, concentration, and consumer welfare")

    # Sidebar for run selection
    st.sidebar.header("Run Selection")

    # For demo purposes, we'll use a hardcoded run_id
    # In a real implementation, you'd fetch available runs from the API
    run_id = st.sidebar.text_input(
        "Run ID", value="demo-run-123", help="Enter the simulation run ID to visualize"
    )

    api_url = st.sidebar.text_input(
        "API URL", value="http://localhost:8000", help="Base URL for the oligopoly API"
    )

    if st.sidebar.button("Load Run Data"):
        try:
            # Load run data
            with st.spinner("Loading run data..."):
                run_data = load_run_data(run_id, api_url)

            st.success(f"Loaded run {run_id}")

            # Display run metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", run_data.get("model", "Unknown"))
            with col2:
                st.metric("Rounds", run_data.get("rounds", 0))
            with col3:
                st.metric("Created", run_data.get("created_at", "Unknown")[:10])

            # Calculate and display metrics
            with st.spinner("Calculating metrics..."):
                metrics_df = calculate_metrics_for_run(run_data)

            if not metrics_df.empty:
                # Display summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Avg HHI", f"{metrics_df['hhi'].mean():.3f}")
                with col2:
                    st.metric(
                        "Avg Consumer Surplus",
                        f"{metrics_df['consumer_surplus'].mean():.1f}",
                    )
                with col3:
                    st.metric("Avg Price", f"{metrics_df['market_price'].mean():.2f}")
                with col4:
                    st.metric(
                        "Avg Quantity", f"{metrics_df['total_quantity'].mean():.1f}"
                    )

                # Create charts
                st.subheader("📊 Market Metrics Over Time")
                create_metrics_charts(metrics_df)

                st.subheader("🏢 Individual Firm Performance")
                create_firm_breakdown_chart(run_data)

                # Display raw data
                st.subheader("📋 Raw Data")
                st.dataframe(metrics_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading run data: {e}")

    # Add some demo information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        """
    This dashboard visualizes oligopoly simulation results including:

    - **HHI**: Herfindahl-Hirschman Index for market concentration
    - **Consumer Surplus**: Welfare measure for consumers
    - **Price & Quantity**: Market dynamics over time
    - **Firm Performance**: Individual firm strategies and outcomes
    """
    )

    # Demo data section
    st.markdown("---")
    st.subheader("🎯 Demo Data")
    st.markdown(
        """
    For demonstration purposes, here's what the metrics mean:

    - **HHI = 0.5**: Two firms with equal market share (50% each)
    - **HHI = 1.0**: Monopoly (single firm has 100% market share)
    - **Consumer Surplus**: Higher values indicate better consumer welfare
    """
    )

    # Example calculations
    st.subheader("🧮 Example Calculations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**HHI Examples:**")
        st.code(
            """
        # Two equal firms: [0.5, 0.5]
        HHI = 0.5² + 0.5² = 0.25 + 0.25 = 0.5

        # Monopoly: [1.0, 0.0]
        HHI = 1.0² + 0.0² = 1.0 + 0.0 = 1.0
        """
        )

    with col2:
        st.markdown("**Consumer Surplus:**")
        st.code(
            """
        # Linear demand: P = a - b*Q
        # CS = 0.5 * (a - P_market) * Q_market

        # Example: a=100, P=70, Q=30
        CS = 0.5 * (100 - 70) * 30 = 450
        """
        )


if __name__ == "__main__":
    main()
