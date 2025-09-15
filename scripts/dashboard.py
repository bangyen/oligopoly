"""Streamlit dashboard for oligopoly simulation visualization.

This module provides a minimal web interface for exploring simulation results,
including price, quantity, profit, HHI, and consumer surplus over time.
"""

from typing import Any, Dict, Optional, cast

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
        return cast(Dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load run data: {e}")
        raise


def load_events_data(
    run_id: str, api_base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Load events data for a simulation run.

    Args:
        run_id: Unique identifier for the simulation run
        api_base_url: Base URL for the API

    Returns:
        Dictionary containing events data

    Raises:
        requests.RequestException: If API request fails
    """
    try:
        response = requests.get(f"{api_base_url}/runs/{run_id}/events")
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load events data: {e}")
        raise


def load_replay_data(
    run_id: str, api_base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Load replay data for a simulation run.

    Args:
        run_id: Unique identifier for the simulation run
        api_base_url: Base URL for the API

    Returns:
        Dictionary containing replay frames

    Raises:
        requests.RequestException: If API request fails
    """
    try:
        response = requests.get(f"{api_base_url}/runs/{run_id}/replay")
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load replay data: {e}")
        raise


def display_event_feed(events_data: Dict[str, Any]) -> None:
    """Display a scrolling event feed.

    Args:
        events_data: Dictionary containing events data
    """
    st.subheader("Event Feed")

    events = events_data.get("events", [])
    total_events = events_data.get("total_events", 0)

    if total_events == 0:
        st.info("No events recorded for this simulation.")
        return

    st.write(f"**Total Events:** {total_events}")

    # Create a scrolling container for events
    with st.container():
        for event in events:
            with st.expander(
                f"Round {int(event['round_idx']) + 1}: {event['description']}",
                expanded=False,
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Type:** {event['event_type']}")
                    st.write(f"**Description:** {event['description']}")
                    if event.get("firm_id") is not None:
                        try:
                            st.write(f"**Firm:** {int(event['firm_id']) + 1}")
                        except Exception:
                            st.write(f"**Firm:** {event['firm_id']}")
                    st.write(f"**Timestamp:** {event['created_at']}")

                with col2:
                    # Display event icon if available
                    event_data = event.get("event_data", {})
                    icon = event_data.get("icon", "📝")
                    st.write(f"**{icon}**")

                    # Display category if available
                    category = event_data.get("category", "other")
                    st.write(f"**Category:** {category}")


def display_replay_controls(replay_data: Dict[str, Any]) -> None:
    """Display replay controls and frame-by-frame playback.

    Args:
        replay_data: Dictionary containing replay frames
    """
    st.subheader("Simulation Replay")

    frames = replay_data.get("frames", [])
    total_frames = replay_data.get("total_frames", 0)
    frames_with_events = replay_data.get("frames_with_events", 0)
    event_rounds = replay_data.get("event_rounds", [])

    if total_frames == 0:
        st.info("No replay data available for this simulation.")
        return

    st.write(f"**Total Frames:** {total_frames}")
    st.write(f"**Frames with Events:** {frames_with_events}")

    if event_rounds:
        st.write(f"**Event Rounds:** {', '.join(map(str, event_rounds))}")

    # Initialize session state
    if "replay_frame" not in st.session_state:
        # Internal zero-based index of the selected frame
        st.session_state.replay_frame = 0
    if "replay_playing" not in st.session_state:
        st.session_state.replay_playing = False

    # Frame selector
    # One-based slider for user display; convert to zero-based internally
    displayed_frame = st.slider(
        "Select Frame",
        min_value=1,
        max_value=max(1, total_frames),
        value=st.session_state.replay_frame + 1,
        key="frame_slider",
    )

    # Update session state when slider changes
    if displayed_frame - 1 != st.session_state.replay_frame:
        st.session_state.replay_frame = displayed_frame - 1

    # Display current frame (convert back to zero-based index)
    if st.session_state.replay_frame < len(frames):
        frame = frames[st.session_state.replay_frame]
        display_replay_frame(frame)


def display_replay_frame(frame: Dict[str, Any]) -> None:
    """Display a single replay frame.

    Args:
        frame: Dictionary containing frame data
    """
    st.subheader(f"Frame {int(frame['round_idx']) + 1}")

    # Frame metrics (2x3 grid)
    row1 = st.columns(3)
    with row1[0]:
        st.metric("Market Price", f"${frame['market_price']:.2f}")
    with row1[1]:
        st.metric("Total Quantity", f"{frame['total_quantity']:.2f}")
    with row1[2]:
        st.metric("Total Profit", f"${frame['total_profit']:.2f}")

    row2 = st.columns(3)
    with row2[0]:
        st.metric("HHI", f"{frame['hhi']:.3f}")
    with row2[1]:
        st.metric("Consumer Surplus", f"${frame['consumer_surplus']:.2f}")
    with row2[2]:
        firm_data_preview = frame.get("firm_data", {})
        num_firms = len(firm_data_preview) if isinstance(firm_data_preview, dict) else 0
        if num_firms > 0:
            st.metric("Firms", f"{num_firms}")

    # Firm data
    st.subheader("Firm Actions")
    firm_data = frame.get("firm_data", {})

    if firm_data:
        firm_df = pd.DataFrame.from_dict(firm_data, orient="index")
        firm_df.columns = ["Action", "Price", "Quantity", "Profit"]
        st.dataframe(firm_df, hide_index=True)

    # Events in this frame
    events = frame.get("events", [])
    if events:
        st.subheader("Events in This Frame")
        for event in events:
            st.write(f"• {event['description']}")

    # Annotations
    annotations = frame.get("annotations", [])
    if annotations:
        st.subheader("Annotations")
        for annotation in annotations:
            st.write(f"• {annotation}")


def calculate_metrics_for_run(run_data: Dict[str, Any]) -> pd.DataFrame:
    """Calculate HHI and consumer surplus metrics for each round.

    Args:
        run_data: Dictionary containing run results and metadata

    Returns:
        DataFrame with metrics for each round
    """
    from sim.models.metrics import (
        calculate_round_metrics_bertrand,
        calculate_round_metrics_cournot,
    )

    rounds_data = run_data["rounds_data"]
    firms_data = run_data["firms_data"]
    model = run_data["model"]

    metrics_data = []

    for round_idx, round_data in enumerate(rounds_data):
        # Get quantities, prices, and profits for this round
        quantities = []
        prices = []
        profits = []

        for firm_data in firms_data:
            if round_idx < len(firm_data["quantities"]):
                quantities.append(firm_data["quantities"][round_idx])
                profits.append(firm_data["profits"][round_idx])
                if model == "cournot":
                    # In Cournot, all firms have the same price
                    prices.append(round_data["price"])
                else:  # bertrand
                    # In Bertrand, actions are prices
                    prices.append(firm_data["actions"][round_idx])

        if not quantities:
            continue

        # Calculate market metrics
        if model == "cournot":
            market_price = round_data["price"]
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
                "round": round_idx + 1,
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

    st.plotly_chart(fig, width="stretch")


def create_firm_breakdown_chart(run_data: Dict[str, Any]) -> None:
    """Create charts showing individual firm performance.

    Args:
        run_data: Dictionary containing run results and metadata
    """
    rounds_data = run_data["rounds_data"]
    firms_data = run_data["firms_data"]
    model = run_data["model"]

    # Prepare data for firm-level analysis
    firm_data = []
    for round_idx, round_data in enumerate(rounds_data):
        for firm_idx, firm_info in enumerate(firms_data):
            if round_idx < len(firm_info["quantities"]):
                firm_data.append(
                    {
                        "round": round_idx + 1,
                        "firm_id": firm_idx,
                        "quantity": firm_info["quantities"][round_idx],
                        "price": (
                            round_data["price"]
                            if model == "cournot"
                            else firm_info["actions"][round_idx]
                        ),
                        "profit": firm_info["profits"][round_idx],
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
                name=f"Firm {firm_id + 1}",
                legendgroup=f"firm-{firm_id}",
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
                name=f"Firm {firm_id + 1}",
                showlegend=False,
                legendgroup=f"firm-{firm_id}",
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
                name=f"Firm {firm_id + 1}",
                showlegend=False,
                legendgroup=f"firm-{firm_id}",
            ),
            row=2,
            col=1,
        )

    # Calculate market shares for each round
    market_share_data = []
    for round_idx, round_data in enumerate(rounds_data):
        quantities = []
        for firm_info in firms_data:
            if round_idx < len(firm_info["quantities"]):
                quantities.append(firm_info["quantities"][round_idx])

        total_qty = sum(quantities)

        for firm_idx, firm_info in enumerate(firms_data):
            if round_idx < len(firm_info["quantities"]):
                market_share = (
                    firm_info["quantities"][round_idx] / total_qty
                    if total_qty > 0
                    else 0
                )
                market_share_data.append(
                    {
                        "round": round_idx + 1,
                        "firm_id": firm_idx,
                        "market_share": market_share,
                    }
                )

    df_shares = pd.DataFrame(market_share_data)
    for firm_id in firms:
        firm_df = df_shares[df_shares["firm_id"] == firm_id]
        fig.add_trace(
            go.Scatter(
                x=firm_df["round"],
                y=firm_df["market_share"],
                mode="lines+markers",
                name=f"Firm {firm_id + 1}",
                showlegend=False,
                legendgroup=f"firm-{firm_id}",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=600, title_text="Individual Firm Performance")
    fig.update_xaxes(title_text="Round")

    st.plotly_chart(fig, width="stretch")


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Oligopoly Simulation Dashboard", page_icon="📊", layout="wide"
    )

    st.title("📊 Oligopoly Simulation Dashboard")
    st.markdown("Visualize market dynamics, concentration, and consumer welfare")

    # Global API configuration
    st.sidebar.header("API Configuration")
    api_url = st.sidebar.text_input(
        "API URL",
        value="http://localhost:8000",
        help="Base URL for the oligopoly API",
        key="global_api_url",
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Single Run", "⚖️ Comparison", "🔥 Heatmaps"])

    with tab1:
        single_run_tab(api_url)

    with tab2:
        comparison_tab(api_url)

    with tab3:
        heatmap_tab(api_url)


def single_run_tab(api_url: str) -> None:
    """Single run visualization tab."""
    # Sidebar for run selection
    st.sidebar.header("Run Selection")

    # For demo purposes, we'll use a hardcoded run_id
    # In a real implementation, you'd fetch available runs from the API
    run_id = st.sidebar.text_input(
        "Run ID",
        value="demo-run-123",
        help="Enter the simulation run ID to visualize",
        key="single_run_id",
    )

    # Check if we have data loaded or if user clicked load button
    if st.sidebar.button("Load Run Data") or st.session_state.get(
        "run_data_loaded", False
    ):
        # Get run data from session state or load it
        if (
            st.session_state.get("run_data_loaded", False)
            and st.session_state.get("run_id") == run_id
        ):
            # Use cached data
            run_data = st.session_state.run_data
            st.success(f"Loaded run {run_id} (cached)")
        else:
            # Load new data
            try:
                with st.spinner("Loading run data..."):
                    run_data = load_run_data(run_id, api_url)

                st.success(f"Loaded run {run_id}")

                # Set session state to indicate data is loaded and store the data
                st.session_state.run_data_loaded = True
                st.session_state.run_data = run_data
                st.session_state.run_id = run_id

            except Exception as e:
                st.error(f"Error loading run data: {e}")
                return

        # Display run metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            model = run_data.get("model", "Unknown")
            st.metric("Model", model.title())
        with col2:
            st.metric("Rounds", run_data.get("rounds", 0))
        with col3:
            st.metric("Created", run_data.get("created_at", "Unknown")[:10])

        # Calculate and display metrics
        with st.spinner("Calculating metrics..."):
            metrics_df = calculate_metrics_for_run(run_data)

        if not metrics_df.empty:
            # Display summary statistics
            st.subheader("Summary Statistics")
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
                st.metric("Avg Quantity", f"{metrics_df['total_quantity'].mean():.1f}")

            # Create charts
            st.subheader("Market Metrics Over Time")
            create_metrics_charts(metrics_df)

            st.subheader("Individual Firm Performance")
            create_firm_breakdown_chart(run_data)

            # Display raw data
            st.subheader("Raw Data")
            # Format column names for better display
            display_df = metrics_df.copy()
            display_df.columns = [
                "Round",
                "Market Price",
                "Total Quantity",
                "Total Profit",
                "HHI",
                "Consumer Surplus",
                "Number of Firms",
            ]
            st.dataframe(display_df, width="stretch", hide_index=True)

            # Load and display events
            try:
                with st.spinner("Loading events..."):
                    events_data = load_events_data(run_id, api_url)
                display_event_feed(events_data)
            except Exception as e:
                st.warning(f"Could not load events: {e}")

            # Load and display replay
            try:
                with st.spinner("Loading replay data..."):
                    replay_data = load_replay_data(run_id, api_url)
                display_replay_controls(replay_data)
            except Exception as e:
                st.warning(f"Could not load replay data: {e}")

    # Add some demo information in sidebar
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

    # Demo information section (only show when no data is loaded)
    if not st.session_state.get("run_data_loaded", False):
        st.subheader("Demo Data")
        st.markdown(
            """
        For demonstration purposes, here's what the metrics mean:

        - **HHI = 0.5**: Two firms with equal market share (50% each)
        - **HHI = 1.0**: Monopoly (single firm has 100% market share)
        - **Consumer Surplus**: Higher values indicate better consumer welfare
        """
        )

        # Example calculations
        st.subheader("Example Calculations")

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


def comparison_tab(api_url: str) -> None:
    """Comparison visualization tab."""
    st.header("⚖️ Scenario Comparison")
    st.markdown("Compare two simulation scenarios side by side")

    # Configuration sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Left Scenario")
        left_config = create_scenario_config_form("left")

    with col2:
        st.subheader("Right Scenario")
        right_config = create_scenario_config_form("right")

    # Run comparison button
    if st.button("Run Comparison", type="primary"):
        if left_config and right_config:
            try:
                with st.spinner("Running comparison..."):
                    comparison_data = run_comparison(left_config, right_config, api_url)

                if comparison_data:
                    st.success("Comparison completed successfully!")

                    # Display comparison results
                    display_comparison_results(comparison_data)

            except Exception as e:
                st.error(f"Error running comparison: {e}")
        else:
            st.error("Please configure both scenarios before running comparison")


def heatmap_tab(api_url: str) -> None:
    """Heatmap visualization tab."""
    st.header("🔥 Strategy/Action Space Heatmaps")
    st.markdown(
        "Visualize profit surfaces and market share surfaces for different firm actions"
    )

    # Model selection
    model = st.selectbox(
        "Competition Model",
        ["Cournot", "Bertrand"],
        help="Choose between Cournot (quantity) or Bertrand (price) competition",
    )
    # Convert to lowercase for processing
    model = model.lower()

    # Firm configuration
    st.subheader("Firm Configuration")
    num_firms = st.slider("Number of Firms", min_value=2, max_value=10, value=3)

    # Dynamic firm cost inputs
    costs = []
    for i in range(num_firms):
        cost = st.number_input(
            f"Firm {i} Cost",
            min_value=0.0,
            value=10.0 + i * 2.0,
            step=0.1,
            key=f"cost_{i}",
        )
        costs.append(cost)

    # Heatmap configuration
    st.subheader("Heatmap Configuration")

    col1, col2 = st.columns(2)
    with col1:
        firm_i = st.selectbox(
            "Firm I (Surface Computed For)",
            range(num_firms),
            help="Firm to compute profit surface for",
            format_func=lambda i: str(i + 1),
        )
    with col2:
        firm_j = st.selectbox(
            "Firm J (Second Firm)",
            [i for i in range(num_firms) if i != firm_i],
            help="Second firm in the heatmap",
            format_func=lambda i: str(i + 1),
        )

    # Grid configuration
    grid_size = st.slider(
        "Grid Size",
        min_value=5,
        max_value=30,
        value=15,
        help="Number of grid points per dimension (higher = more detailed but slower)",
    )

    # Action range
    if model == "cournot":
        action_label = "Quantity"
        default_min, default_max = 0.0, 50.0
    else:  # bertrand
        action_label = "Price"
        default_min, default_max = 0.0, 100.0

    col1, col2 = st.columns(2)
    with col1:
        min_action = st.number_input(
            f"Min {action_label}",
            min_value=0.0,
            value=default_min,
            step=1.0,
            key="min_action",
        )
    with col2:
        max_action = st.number_input(
            f"Max {action_label}",
            min_value=min_action + 1.0,
            value=default_max,
            step=1.0,
            key="max_action",
        )

    # Other firms' actions
    st.subheader("🔧 Other Firms' Actions")
    other_actions = []
    other_firms = [i for i in range(num_firms) if i != firm_i and i != firm_j]

    for i, firm_idx in enumerate(other_firms):
        action = st.number_input(
            f"Firm {firm_idx} {action_label}",
            min_value=0.0,
            value=20.0,
            step=1.0,
            key=f"other_action_{i}",
        )
        other_actions.append(action)

    # Market parameters
    st.subheader("Market Parameters")

    if model == "cournot":
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input(
                "Demand Intercept (a)", min_value=1.0, value=100.0, step=1.0
            )
        with col2:
            b = st.number_input("Demand Slope (b)", min_value=0.1, value=1.0, step=0.1)
        params = {"a": a, "b": b}
    else:  # bertrand
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.number_input(
                "Demand Intercept (α)", min_value=1.0, value=100.0, step=1.0
            )
        with col2:
            beta = st.number_input(
                "Demand Slope (β)", min_value=0.1, value=1.0, step=0.1
            )
        params = {"alpha": alpha, "beta": beta}

    # Generate heatmap button
    if st.button("Generate Heatmap", type="primary"):
        try:
            with st.spinner("Computing heatmap..."):
                heatmap_data = compute_heatmap(
                    model=model,
                    firm_i=firm_i,
                    firm_j=firm_j,
                    grid_size=grid_size,
                    action_range=(min_action, max_action),
                    other_actions=other_actions,
                    params=params,
                    costs=costs,
                    api_url=api_url,
                )

            if heatmap_data:
                st.success(
                    f"Heatmap computed in {heatmap_data['computation_time_ms']:.1f}ms"
                )
                display_heatmap(heatmap_data, model)

        except Exception as e:
            st.error(f"Error computing heatmap: {e}")


def compute_heatmap(
    model: str,
    firm_i: int,
    firm_j: int,
    grid_size: int,
    action_range: tuple,
    other_actions: list,
    params: dict,
    costs: list,
    api_url: str = "http://localhost:8000",
) -> Optional[Dict[str, Any]]:
    """Compute heatmap data using the API.

    Args:
        model: Competition model ("cournot" or "bertrand")
        firm_i: Index of firm to compute surface for
        firm_j: Index of second firm
        grid_size: Number of grid points per dimension
        action_range: Tuple of (min, max) action values
        other_actions: List of fixed actions for other firms
        params: Market parameters
        costs: List of firm costs
        api_url: Base URL for the API

    Returns:
        Dictionary containing heatmap data or None if failed
    """
    try:
        # Prepare request data
        request_data = {
            "model": model,
            "firm_i": firm_i,
            "firm_j": firm_j,
            "grid_size": grid_size,
            "action_range": action_range,
            "other_actions": other_actions,
            "params": params,
            "firms": [{"cost": cost} for cost in costs],
        }

        # Make API request
        response = requests.post(f"{api_url}/heatmap", json=request_data)
        response.raise_for_status()

        return response.json()  # type: ignore

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def display_heatmap(heatmap_data: Dict[str, Any], model: str) -> None:
    """Display heatmap visualization.

    Args:
        heatmap_data: Dictionary containing heatmap data from API
        model: Competition model type
    """
    # Extract data
    profit_surface = heatmap_data["profit_surface"]
    market_share_surface = heatmap_data.get("market_share_surface")
    action_i_grid = heatmap_data["action_i_grid"]
    action_j_grid = heatmap_data["action_j_grid"]
    firm_i = heatmap_data["firm_i"]
    firm_j = heatmap_data["firm_j"]

    # Determine action labels
    action_label = "Quantity" if model == "cournot" else "Price"

    # Create profit heatmap
    st.subheader(f"Profit Surface for Firm {firm_i + 1}")

    fig_profit = go.Figure(
        data=go.Heatmap(
            z=profit_surface,
            x=action_j_grid,
            y=action_i_grid,
            colorscale="Viridis",
            colorbar=dict(title="Profit"),
        )
    )

    fig_profit.update_layout(
        title=f"Profit Surface: Firm {firm_i + 1} vs Firm {firm_j + 1}",
        xaxis_title=f"Firm {firm_j + 1} {action_label}",
        yaxis_title=f"Firm {firm_i + 1} {action_label}",
        width=600,
        height=500,
    )

    st.plotly_chart(fig_profit, width="stretch")

    # Create market share heatmap for Bertrand
    if model == "bertrand" and market_share_surface:
        st.subheader(f"Market Share Surface for Firm {firm_i + 1}")

        fig_market_share = go.Figure(
            data=go.Heatmap(
                z=market_share_surface,
                x=action_j_grid,
                y=action_i_grid,
                colorscale="Blues",
                colorbar=dict(title="Market Share"),
            )
        )

        fig_market_share.update_layout(
            title=f"Market Share Surface: Firm {firm_i + 1} vs Firm {firm_j + 1}",
            xaxis_title=f"Firm {firm_j + 1} {action_label}",
            yaxis_title=f"Firm {firm_i + 1} {action_label}",
            width=600,
            height=500,
        )

        st.plotly_chart(fig_market_share, width="stretch")

    # Display summary statistics
    st.subheader("Summary Statistics")

    import numpy as np

    profit_array = np.array(profit_surface)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Profit", f"{profit_array.max():.2f}")
    with col2:
        st.metric("Min Profit", f"{profit_array.min():.2f}")
    with col3:
        st.metric("Mean Profit", f"{profit_array.mean():.2f}")
    with col4:
        st.metric("Std Profit", f"{profit_array.std():.2f}")

    # Display raw data
    st.subheader("Raw Data")

    # Create DataFrame for profit surface
    import pandas as pd

    df_profit = pd.DataFrame(
        profit_surface,
        index=[round(x, 1) for x in action_i_grid],
        columns=[round(x, 1) for x in action_j_grid],
    )

    # Axis titles for the raw data table
    st.caption(
        f"Rows: Firm {firm_i + 1} {action_label} -- Columns: Firm {firm_j + 1} {action_label}"
    )
    # Keep index visible for heatmap raw data to indicate Firm i actions
    st.dataframe(df_profit, width="stretch", hide_index=False)


def create_scenario_config_form(side: str) -> Dict[str, Any]:
    """Create a configuration form for a scenario.

    Args:
        side: "left" or "right" to distinguish the scenario

    Returns:
        Dictionary containing the scenario configuration
    """
    config: Dict[str, Any] = {}

    # Model selection
    model = st.selectbox("Model", ["Cournot", "Bertrand"], key=f"model_{side}")
    # Convert to lowercase for processing
    model = model.lower()
    config["model"] = model

    # Rounds
    rounds = st.number_input(
        "Rounds", min_value=1, max_value=1000, value=10, key=f"rounds_{side}"
    )
    config["rounds"] = rounds

    # Add spacing
    st.markdown("")

    # Parameters
    st.markdown("**Parameters**")
    if model == "cournot":
        a = st.number_input(
            "Demand Intercept (a)",
            min_value=1.0,
            value=100.0,
            key=f"a_{side}",
        )
        b = st.number_input(
            "Demand Slope (b)", min_value=0.1, value=1.0, key=f"b_{side}"
        )
        config["params"] = {"a": a, "b": b}
    else:  # bertrand
        alpha = st.number_input(
            "Demand Alpha", min_value=1.0, value=100.0, key=f"alpha_{side}"
        )
        beta = st.number_input(
            "Demand Beta", min_value=0.1, value=1.0, key=f"beta_{side}"
        )
        config["params"] = {"alpha": alpha, "beta": beta}

    # Add spacing
    st.markdown("")

    # Firms
    st.markdown("**Firms**")
    num_firms = st.number_input(
        "Number of Firms",
        min_value=1,
        max_value=10,
        value=2,
        key=f"num_firms_{side}",
    )

    firms = []
    for i in range(num_firms):
        cost = st.number_input(
            f"Firm {i+1} Cost",
            min_value=0.1,
            value=10.0 + i * 5.0,
            key=f"firm_{i}_{side}",
        )
        firms.append({"cost": cost})

    config["firms"] = firms

    # Add spacing
    st.markdown("")

    # Seed
    default_seed = 42 if side == "left" else 123
    seed = st.number_input(
        "Random Seed", min_value=0, value=default_seed, key=f"seed_{side}"
    )
    config["seed"] = seed

    return config


def run_comparison(
    left_config: Dict[str, Any], right_config: Dict[str, Any], api_url: str
) -> Optional[Dict[str, Any]]:
    """Run comparison between two scenarios.

    Args:
        left_config: Left scenario configuration
        right_config: Right scenario configuration
        api_url: API base URL

    Returns:
        Comparison results or None if failed
    """
    try:
        # Prepare comparison request
        comparison_request = {
            "left_config": left_config,
            "right_config": right_config,
        }

        # Run comparison
        response = requests.post(f"{api_url}/compare", json=comparison_request)
        response.raise_for_status()

        comparison_response = response.json()
        left_run_id = comparison_response["left_run_id"]
        right_run_id = comparison_response["right_run_id"]

        # Get comparison results
        results_response = requests.get(
            f"{api_url}/compare/{left_run_id}/{right_run_id}"
        )
        results_response.raise_for_status()

        return cast(Dict[str, Any], results_response.json())

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def display_comparison_results(comparison_data: Dict[str, Any]) -> None:
    """Display comparison results with charts and deltas table.

    Args:
        comparison_data: Comparison results from API
    """
    # Display run IDs
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Left Run ID", comparison_data["left_run_id"])
    with col2:
        st.metric("Right Run ID", comparison_data["right_run_id"])

    # Create comparison charts
    st.subheader("Comparison Charts")
    create_comparison_charts(comparison_data)

    # Display deltas table
    st.subheader("Deltas Table (Right - Left)")
    create_deltas_table(comparison_data)


def create_comparison_charts(comparison_data: Dict[str, Any]) -> None:
    """Create comparison charts showing both scenarios.

    Args:
        comparison_data: Comparison results from API
    """
    left_metrics = comparison_data["left_metrics"]
    right_metrics = comparison_data["right_metrics"]
    rounds = comparison_data["rounds"]

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
            "Deltas Overview",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    rounds_list = [i + 1 for i in range(rounds)]

    # First metric chart (Market Price)
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=left_metrics["market_price"],
            mode="lines+markers",
            name="Left",
            legendgroup="left",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=right_metrics["market_price"],
            mode="lines+markers",
            name="Right",
            legendgroup="right",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    # Quantity chart
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=left_metrics["total_quantity"],
            mode="lines+markers",
            name="Left",
            legendgroup="left",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=right_metrics["total_quantity"],
            mode="lines+markers",
            name="Right",
            legendgroup="right",
            line=dict(color="red"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Profit chart
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=left_metrics["total_profit"],
            mode="lines+markers",
            name="Left",
            legendgroup="left",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=right_metrics["total_profit"],
            mode="lines+markers",
            name="Right",
            legendgroup="right",
            line=dict(color="red"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # HHI chart
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=left_metrics["hhi"],
            mode="lines+markers",
            name="Left",
            legendgroup="left",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=right_metrics["hhi"],
            mode="lines+markers",
            name="Right",
            legendgroup="right",
            line=dict(color="red"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Consumer Surplus chart
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=left_metrics["consumer_surplus"],
            mode="lines+markers",
            name="Left",
            legendgroup="left",
            line=dict(color="blue"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=right_metrics["consumer_surplus"],
            mode="lines+markers",
            name="Right",
            legendgroup="right",
            line=dict(color="red"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Deltas overview chart
    deltas = comparison_data["deltas"]
    fig.add_trace(
        go.Scatter(
            x=rounds_list,
            y=deltas["market_price"],
            mode="lines+markers",
            name="Price Delta",
            line=dict(color="green"),
            showlegend=False,
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        height=900,
        title_text="Scenario Comparison",
        showlegend=True,
    )
    fig.update_xaxes(title_text="Round")

    st.plotly_chart(fig, width="stretch")


def create_deltas_table(comparison_data: Dict[str, Any]) -> None:
    """Create a table showing deltas for each metric.

    Args:
        comparison_data: Comparison results from API
    """
    deltas = comparison_data["deltas"]
    rounds = comparison_data["rounds"]

    # Create DataFrame for deltas
    delta_data = []
    for round_idx in range(rounds):
        row = {"Round": round_idx + 1}
        for metric_name, values in deltas.items():
            # Special handling for HHI to keep it all caps
            if metric_name == "hhi":
                display_name = "HHI Delta"
            else:
                display_name = f"{metric_name.replace('_', ' ').title()} Delta"
            row[display_name] = values[round_idx]
        delta_data.append(row)

    delta_df = pd.DataFrame(delta_data)
    st.dataframe(delta_df, width="stretch", hide_index=True)

    # Summary statistics
    st.subheader("Delta Summary Statistics")
    summary_data = []
    for metric_name, values in deltas.items():
        # Special handling for HHI to keep it all caps
        if metric_name == "hhi":
            display_name = "HHI"
        else:
            display_name = metric_name.replace("_", " ").title()
        summary_data.append(
            {
                "Metric": display_name,
                "Mean Delta": f"{sum(values) / len(values):.4f}",
                "Min Delta": f"{min(values):.4f}",
                "Max Delta": f"{max(values):.4f}",
                "STD Delta": f"{(sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5:.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
