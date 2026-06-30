"""
=============================================================================
Visualization Engine — Professional Financial Charts
=============================================================================
Module  : src/visualization/charts.py
Purpose : Production-quality Plotly visualizations for Hull-White model
          output: yield curves, rate path fans, vol surfaces, calibration
          diagnostics, Greeks profiles, and risk dashboards.
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ── Color Palette (Finance-Grade Dark Theme) ──────────────────────────────────
COLORS = {
    "primary":   "#00D4FF",   # Cyan
    "secondary": "#FF6B6B",   # Coral red
    "tertiary":  "#FFE66D",   # Amber
    "success":   "#4CAF50",   # Green
    "warning":   "#FF9800",   # Orange
    "bg_dark":   "#0D1117",   # Deep black
    "bg_mid":    "#161B22",   # Dark gray
    "bg_card":   "#21262D",   # Card background
    "grid":      "#30363D",   # Grid lines
    "text":      "#E6EDF3",   # Light text
    "muted":     "#8B949E",   # Muted text
}

# Gradient palette for multi-line charts (10 colors)
PATH_COLORS = [
    "#00D4FF", "#FF6B6B", "#FFE66D", "#4CAF50", "#AB47BC",
    "#26C6DA", "#FFA726", "#66BB6A", "#EF5350", "#7E57C2"
]

# ── Chart Template ────────────────────────────────────────────────────────────
DARK_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", color=COLORS["text"]),
        paper_bgcolor=COLORS["bg_dark"],
        plot_bgcolor=COLORS["bg_mid"],
        xaxis=dict(
            gridcolor=COLORS["grid"], gridwidth=0.5,
            zerolinecolor=COLORS["grid"], zerolinewidth=1,
            linecolor=COLORS["grid"],
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"], gridwidth=0.5,
            zerolinecolor=COLORS["grid"], zerolinewidth=1,
            linecolor=COLORS["grid"],
        ),
        legend=dict(
            bgcolor=COLORS["bg_card"],
            bordercolor=COLORS["grid"],
            borderwidth=1,
        ),
        colorway=PATH_COLORS,
    )
)


def _apply_dark_theme(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    """Apply consistent dark finance theme to a Plotly figure."""
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS["text"], family="Inter"),
            x=0.02, y=0.97,
        ),
        height=height,
        margin=dict(l=60, r=40, t=70, b=50),
        hovermode="x unified",
    )
    return fig


# =============================================================================
# 1. YIELD CURVE VISUALIZATION
# =============================================================================

def plot_yield_curve(
    curve_builder,
    model: Optional[object] = None,
    r0: Optional[float] = None,
    title: str = "US Treasury Zero-Coupon Yield Curve"
) -> go.Figure:
    """
    Plot market and model-implied yield curves side-by-side.

    Parameters
    ----------
    curve_builder : YieldCurveBuilder
        Bootstrapped yield curve.
    model : HullWhiteModel, optional
        If provided, overlays the HW model curve.
    r0 : float, optional
        Short rate for model curve computation.
    """
    T_grid = np.linspace(0.08, 30, 300)

    # Market (bootstrapped) curves
    zero_rates = np.array([curve_builder.zero_rate(T) * 100 for T in T_grid])
    fwd_rates = np.array([curve_builder.forward_rate(T) * 100 for T in T_grid])
    discount_factors = np.array([curve_builder.discount_factor(T) for T in T_grid])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Zero Rates", "Instantaneous Forward Rates", "Discount Factors"],
        horizontal_spacing=0.08
    )

    # Zero rates
    fig.add_trace(go.Scatter(
        x=T_grid, y=zero_rates,
        name="Market Zero Rate",
        line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="%{x:.2f}Y: %{y:.3f}%<extra>Zero Rate</extra>",
    ), row=1, col=1)

    # Pillar points
    pillar_data = curve_builder.get_pillar_data()
    fig.add_trace(go.Scatter(
        x=pillar_data["Maturity (Y)"],
        y=pillar_data["Zero Rate (%)"],
        name="Pillar Points",
        mode="markers",
        marker=dict(color=COLORS["tertiary"], size=8, symbol="diamond"),
        hovertemplate="T=%{x:.2f}Y: %{y:.3f}%<extra>Pillar</extra>",
    ), row=1, col=1)

    # Model curve if provided
    if model is not None and r0 is not None:
        hw_yields = np.array([model.zcb_yield(r0, 0, T) * 100 for T in T_grid])
        fig.add_trace(go.Scatter(
            x=T_grid, y=hw_yields,
            name="HW Model Curve",
            line=dict(color=COLORS["secondary"], width=2, dash="dash"),
            hovertemplate="%{x:.2f}Y: %{y:.3f}%<extra>HW Model</extra>",
        ), row=1, col=1)

    # Forward rates
    fig.add_trace(go.Scatter(
        x=T_grid, y=fwd_rates,
        name="Forward Rate",
        line=dict(color=COLORS["success"], width=2.5),
        showlegend=False,
        hovertemplate="%{x:.2f}Y: %{y:.3f}%<extra>Fwd Rate</extra>",
    ), row=1, col=2)

    # Discount factors
    fig.add_trace(go.Scatter(
        x=T_grid, y=discount_factors,
        name="Discount Factor",
        line=dict(color=COLORS["tertiary"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255, 230, 109, 0.1)",
        showlegend=False,
        hovertemplate="%{x:.2f}Y: %{y:.6f}<extra>Discount Factor</extra>",
    ), row=1, col=3)

    # Axis labels
    for col, ylabel in [(1, "Rate (%)"), (2, "Rate (%)"), (3, "P(0,T)")]:
        fig.update_yaxes(title_text=ylabel, row=1, col=col)
        fig.update_xaxes(title_text="Maturity (Years)", row=1, col=col)

    fig = _apply_dark_theme(fig, title, height=480)
    fig.update_layout(legend=dict(x=0.02, y=0.85, xanchor="left"))
    return fig


# =============================================================================
# 2. RATE PATH FAN CHART
# =============================================================================

def plot_rate_path_fan(
    rate_paths: np.ndarray,
    time_grid: np.ndarray,
    n_display_paths: int = 50,
    r0: Optional[float] = None,
    title: str = "Hull-White Short Rate Simulation — Path Fan Chart"
) -> go.Figure:
    """
    Fan chart of Monte Carlo short-rate paths with percentile bands.

    Parameters
    ----------
    rate_paths : np.ndarray
        Shape (n_paths, n_time_steps). Rates in decimal.
    time_grid : np.ndarray
        Time grid in years.
    n_display_paths : int
        Number of individual paths to overlay.
    """
    fig = go.Figure()

    rates_pct = rate_paths * 100  # Convert to percentage

    # Percentile bands (confidence fan)
    percentiles = [(5, 95), (15, 85), (25, 75), (35, 65)]
    alphas = [0.08, 0.12, 0.18, 0.25]

    for (p_lo, p_hi), alpha in zip(percentiles, alphas):
        lo = np.percentile(rates_pct, p_lo, axis=0)
        hi = np.percentile(rates_pct, p_hi, axis=0)

        fig.add_trace(go.Scatter(
            x=np.concatenate([time_grid, time_grid[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=f"rgba(0, 212, 255, {alpha})",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"P{p_lo}-P{p_hi}",
            showlegend=True,
            hoverinfo="skip",
        ))

    # Individual paths (subset)
    n_show = min(n_display_paths, len(rate_paths))
    step = max(1, len(rate_paths) // n_show)
    selected = rates_pct[::step][:n_show]

    for i, path in enumerate(selected):
        fig.add_trace(go.Scatter(
            x=time_grid, y=path,
            mode="lines",
            line=dict(width=0.4, color=f"rgba(0, 212, 255, 0.25)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Median path
    median = np.median(rates_pct, axis=0)
    fig.add_trace(go.Scatter(
        x=time_grid, y=median,
        name="Median Path",
        line=dict(color=COLORS["tertiary"], width=2.5),
        hovertemplate="t=%{x:.2f}Y, Median: %{y:.3f}%<extra></extra>",
    ))

    # Mean path
    mean_path = rates_pct.mean(axis=0)
    fig.add_trace(go.Scatter(
        x=time_grid, y=mean_path,
        name="Mean Path",
        line=dict(color=COLORS["secondary"], width=2, dash="dot"),
        hovertemplate="t=%{x:.2f}Y, Mean: %{y:.3f}%<extra></extra>",
    ))

    # Initial rate marker
    if r0 is not None:
        fig.add_hline(
            y=r0 * 100, line_dash="dash",
            line_color=COLORS["muted"], line_width=1,
            annotation_text=f"r₀ = {r0*100:.2f}%",
            annotation_position="right",
        )

    fig.update_xaxes(title_text="Time (Years)")
    fig.update_yaxes(title_text="Short Rate (%)")
    fig = _apply_dark_theme(fig, title, height=520)
    return fig


# =============================================================================
# 3. VOLATILITY SURFACE
# =============================================================================

def plot_swaption_vol_surface(
    vol_surface: pd.DataFrame,
    model_vols: Optional[pd.DataFrame] = None,
    title: str = "Swaption Implied Volatility Surface (Normal, bps)"
) -> go.Figure:
    """
    3D surface plot of swaption implied vols with optional model overlay.

    Parameters
    ----------
    vol_surface : pd.DataFrame
        Market vols in bps. Index = expiries, columns = tenors.
    model_vols : pd.DataFrame, optional
        Model vols for comparison.
    """
    if model_vols is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Market Vol Surface", "Model Vol Surface"],
            specs=[[{"type": "surface"}, {"type": "surface"}]],
            horizontal_spacing=0.05
        )
        surfaces = [(vol_surface, 1), (model_vols, 2)]
    else:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "surface"}]]
        )
        surfaces = [(vol_surface, 1)]

    for df, col in surfaces:
        # Parse numeric axes
        x = [float(t.replace("Y", "")) for t in df.columns]
        y = [float(e.replace("Y", "")) for e in df.index]

        fig.add_trace(go.Surface(
            x=x, y=y, z=df.values,
            colorscale=[
                [0.0, "#0D3B66"],
                [0.2, "#1565C0"],
                [0.4, "#00BCD4"],
                [0.6, "#4CAF50"],
                [0.8, "#FF9800"],
                [1.0, "#F44336"],
            ],
            opacity=0.90,
            showscale=True,
            colorbar=dict(
                title="Vol (bps)",
                titlefont=dict(color=COLORS["text"]),
                tickfont=dict(color=COLORS["text"]),
                x=1.05 if col == 2 else 1.0,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True)
            ),
            hovertemplate="Tenor: %{x}Y<br>Expiry: %{y}Y<br>Vol: %{z:.1f} bps<extra></extra>",
        ), row=1, col=col)

        scene_key = "scene" if col == 1 else "scene2"
        fig.update_layout(**{scene_key: dict(
            xaxis=dict(title="Swap Tenor (Y)", color=COLORS["text"]),
            yaxis=dict(title="Option Expiry (Y)", color=COLORS["text"]),
            zaxis=dict(title="Normal Vol (bps)", color=COLORS["text"]),
            bgcolor=COLORS["bg_mid"],
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
        )})

    fig = _apply_dark_theme(fig, title, height=560)
    return fig


# =============================================================================
# 4. CALIBRATION DIAGNOSTICS
# =============================================================================

def plot_calibration_diagnostics(
    calibration_df: pd.DataFrame,
    loss_history: Optional[np.ndarray] = None,
    title: str = "Hull-White Calibration Diagnostics"
) -> go.Figure:
    """
    Multi-panel calibration quality chart.

    Panels:
    - Market vs. Model vols (scatter)
    - Error heatmap by expiry/tenor
    - Optimization loss history
    """
    n_panels = 3 if loss_history is not None else 2
    fig = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=[
            "Market vs. Model Vol (bps)",
            "Calibration Error Heatmap (bps)",
            "Optimization Loss History"
        ][:n_panels],
        horizontal_spacing=0.10
    )

    # Panel 1: Scatter plot
    market_vols = calibration_df["Market Vol (bps)"]
    model_vols = calibration_df["Model Vol (bps)"]
    errors = calibration_df["Error (bps)"]

    # Color by error magnitude
    abs_errors = errors.abs()
    max_err = abs_errors.max()

    fig.add_trace(go.Scatter(
        x=market_vols, y=model_vols,
        mode="markers",
        marker=dict(
            size=10,
            color=abs_errors,
            colorscale=[[0, COLORS["success"]], [0.5, COLORS["warning"]], [1, COLORS["secondary"]]],
            colorbar=dict(
                title="Error (bps)",
                x=-0.15,
                titlefont=dict(color=COLORS["text"]),
                tickfont=dict(color=COLORS["text"]),
            ),
            showscale=True,
        ),
        text=[f"Exp:{row['Expiry (Y)']}Y Tenor:{row['Tenor (Y)']}Y" for _, row in calibration_df.iterrows()],
        hovertemplate="%{text}<br>Mkt: %{x:.1f} | Model: %{y:.1f} bps<extra></extra>",
        name="Instruments",
    ), row=1, col=1)

    # 45-degree line (perfect calibration)
    vol_range = [market_vols.min() * 0.95, market_vols.max() * 1.05]
    fig.add_trace(go.Scatter(
        x=vol_range, y=vol_range,
        mode="lines",
        line=dict(color=COLORS["muted"], dash="dash", width=1),
        name="Perfect Fit",
        showlegend=False,
    ), row=1, col=1)

    # Panel 2: Error bar chart
    instrument_labels = [
        f"{row['Expiry (Y)']}Yx{row['Tenor (Y)']}Y"
        for _, row in calibration_df.iterrows()
    ]
    fig.add_trace(go.Bar(
        x=instrument_labels,
        y=calibration_df["Error (bps)"],
        name="Error (bps)",
        marker=dict(
            color=calibration_df["Error (bps)"],
            colorscale=[[0, COLORS["secondary"]], [0.5, COLORS["bg_mid"]], [1, COLORS["success"]]],
            cmid=0,
        ),
        hovertemplate="%{x}: %{y:.2f} bps<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line_color=COLORS["muted"], line_dash="dash", line_width=1, row=1, col=2)

    # Panel 3: Loss history
    if loss_history is not None and len(loss_history) > 0:
        clean_history = loss_history[loss_history < 1e8]  # Remove outliers
        fig.add_trace(go.Scatter(
            x=np.arange(len(clean_history)),
            y=clean_history,
            mode="lines",
            line=dict(color=COLORS["primary"], width=1.5),
            name="Loss",
            showlegend=False,
            hovertemplate="Iter %{x}: loss=%{y:.6e}<extra></extra>",
        ), row=1, col=3)
        fig.update_yaxes(type="log", row=1, col=3)
        fig.update_xaxes(title_text="Iteration", row=1, col=3)
        fig.update_yaxes(title_text="Loss (log scale)", row=1, col=3)

    fig.update_xaxes(title_text="Market Vol (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Model Vol (bps)", row=1, col=1)
    fig.update_xaxes(title_text="Instrument", row=1, col=2, tickangle=-45)
    fig.update_yaxes(title_text="Error (bps)", row=1, col=2)

    fig = _apply_dark_theme(fig, title, height=500)
    return fig


# =============================================================================
# 5. DERIVATIVE PRICING DASHBOARD
# =============================================================================

def plot_derivative_prices(
    results: Dict,
    title: str = "Interest Rate Derivative Pricing — Hull-White Model"
) -> go.Figure:
    """
    Bar chart comparing prices of various IR derivatives.

    Parameters
    ----------
    results : dict
        {'Cap 5Y @4%': 0.021, 'Floor 5Y @4%': 0.019, ...}
    """
    instruments = list(results.keys())
    prices = [results[k] * 10_000 for k in instruments]  # Convert to bps

    colors = [COLORS["primary"] if p >= 0 else COLORS["secondary"] for p in prices]

    fig = go.Figure(go.Bar(
        x=instruments,
        y=prices,
        marker=dict(
            color=colors,
            line=dict(color=COLORS["bg_dark"], width=1),
            opacity=0.85,
        ),
        text=[f"{p:.1f}" for p in prices],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
        hovertemplate="%{x}<br>Price: %{y:.2f} bps<extra></extra>",
        name="Price (bps)",
    ))

    fig.update_xaxes(title_text="Instrument", tickangle=-30)
    fig.update_yaxes(title_text="Price (bps of notional)")
    fig = _apply_dark_theme(fig, title, height=480)
    return fig


# =============================================================================
# 6. RATE DISTRIBUTION EVOLUTION
# =============================================================================

def plot_rate_distributions(
    rate_paths: np.ndarray,
    time_grid: np.ndarray,
    time_points: Optional[List[float]] = None,
    title: str = "Short Rate Distribution Evolution"
) -> go.Figure:
    """
    Violin plots showing the distribution of r(t) at selected time points.

    Parameters
    ----------
    rate_paths : np.ndarray
        Shape (n_paths, n_time_steps).
    time_grid : np.ndarray
        Time points in years.
    time_points : list, optional
        Times at which to plot distributions. Default: [0.5, 1, 2, 5, 10].
    """
    time_points = time_points or [0.5, 1, 2, 5, 10]
    time_points = [t for t in time_points if t <= time_grid[-1]]

    fig = go.Figure()

    for i, t in enumerate(time_points):
        idx = np.searchsorted(time_grid, t)
        r_t = rate_paths[:, idx] * 100  # To percentage

        color = PATH_COLORS[i % len(PATH_COLORS)]
        rgba = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.7)"

        fig.add_trace(go.Violin(
            x=[f"t = {t:.1f}Y"] * len(r_t),
            y=r_t,
            name=f"t = {t:.1f}Y",
            box_visible=True,
            meanline_visible=True,
            fillcolor=rgba,
            line_color=color,
            opacity=0.8,
            points=False,
            hovertemplate=f"t={t:.1f}Y<br>Rate: %{{y:.3f}}%<extra></extra>",
        ))

    fig.update_xaxes(title_text="Time Horizon")
    fig.update_yaxes(title_text="Short Rate (%)")
    fig = _apply_dark_theme(fig, title, height=500)
    return fig


# =============================================================================
# 7. GREEKS PROFILE
# =============================================================================

def plot_greeks_profile(
    model,
    r0: float,
    T_grid: Optional[np.ndarray] = None,
    strike: float = 0.04,
    title: str = "Cap Greeks Profile — Delta & Vega vs. Maturity"
) -> go.Figure:
    """
    Plot Delta, DV01 and Vega of caps across a range of maturities.
    """
    from ..analytics.risk_metrics import RiskMetrics

    T_grid = T_grid if T_grid is not None else np.linspace(0.5, 10, 30)
    metrics = RiskMetrics(model, r0)

    deltas, dv01s, vega_approx = [], [], []

    for T in T_grid:
        def cap_pricer(r, _T=T, _K=strike):
            return model.cap_price(r, 0.0, _T, _K)

        delta_val = metrics.delta(cap_pricer, h=1e-4)
        dv01_val = metrics.dv01(cap_pricer)
        deltas.append(delta_val)
        dv01s.append(dv01_val)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Delta (∂Cap/∂r₀)", "DV01 (per 1bp shift)"],
        horizontal_spacing=0.10
    )

    fig.add_trace(go.Scatter(
        x=T_grid, y=deltas,
        name="Delta",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(0, 212, 255, 0.1)",
        hovertemplate="T=%{x:.2f}Y, Δ=%{y:.5f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=T_grid, y=dv01s,
        name="DV01",
        line=dict(color=COLORS["secondary"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(255, 107, 107, 0.1)",
        hovertemplate="T=%{x:.2f}Y, DV01=%{y:.5f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_color=COLORS["muted"], line_width=1, line_dash="dash")

    fig.update_xaxes(title_text="Cap Maturity (Years)", row=1, col=1)
    fig.update_xaxes(title_text="Cap Maturity (Years)", row=1, col=2)
    fig.update_yaxes(title_text="Delta", row=1, col=1)
    fig.update_yaxes(title_text="DV01 ($)", row=1, col=2)

    fig = _apply_dark_theme(fig, title, height=450)
    return fig


# =============================================================================
# 8. SCENARIO ANALYSIS WATERFALL
# =============================================================================

def plot_scenario_waterfall(
    scenario_df: pd.DataFrame,
    title: str = "Scenario Analysis — P&L Impact of Rate Shocks"
) -> go.Figure:
    """
    Waterfall/bar chart of P&L impact under BCBS rate shock scenarios.
    """
    scenarios = scenario_df["Scenario"]
    pnl_bps = scenario_df["P&L (bps)"]

    colors = [
        COLORS["success"] if v >= 0 else COLORS["secondary"]
        for v in pnl_bps
    ]

    fig = go.Figure(go.Bar(
        x=scenarios,
        y=pnl_bps,
        marker=dict(
            color=colors,
            line=dict(width=0.5, color=COLORS["bg_dark"]),
            opacity=0.85,
        ),
        text=[f"{v:+.1f}" for v in pnl_bps],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="Scenario: %{x}<br>P&L: %{y:.2f} bps<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["muted"], line_width=1.5)
    fig.update_xaxes(title_text="Rate Shock Scenario")
    fig.update_yaxes(title_text="P&L Impact (bps of notional)")
    fig = _apply_dark_theme(fig, title, height=460)
    return fig


# =============================================================================
# 9. PARAMETER LOSS SURFACE
# =============================================================================

def plot_loss_surface(
    loss_df: pd.DataFrame,
    optimal_a: Optional[float] = None,
    optimal_sigma: Optional[float] = None,
    title: str = "Calibration Loss Surface — (a, σ) Parameter Space"
) -> go.Figure:
    """
    3D loss surface plot for HW parameter space visualization.

    Parameters
    ----------
    loss_df : pd.DataFrame
        Columns: 'a', 'sigma', 'loss'. Output of calibrator.sensitivity_analysis().
    """
    pivot = loss_df.pivot(index="a", columns="sigma", values="loss")
    a_vals = pivot.index.values
    sig_vals = pivot.columns.values
    Z = np.log1p(pivot.values)  # Log-scale for better visual range

    fig = go.Figure(go.Surface(
        x=sig_vals * 100,
        y=a_vals,
        z=Z,
        colorscale=[
            [0.0, "#00D4FF"],
            [0.3, "#4CAF50"],
            [0.6, "#FF9800"],
            [1.0, "#F44336"],
        ],
        opacity=0.9,
        showscale=True,
        colorbar=dict(
            title="log(1 + Loss)",
            titlefont=dict(color=COLORS["text"]),
            tickfont=dict(color=COLORS["text"]),
        ),
        hovertemplate="σ=%{x:.2f}%, a=%{y:.3f}<br>log Loss=%{z:.4f}<extra></extra>",
    ))

    # Mark optimal point
    if optimal_a is not None and optimal_sigma is not None:
        opt_loss = loss_df[
            (loss_df["a"].sub(optimal_a).abs() < 0.01) &
            (loss_df["sigma"].sub(optimal_sigma).abs() < 0.001)
        ]["loss"]
        opt_z = float(np.log1p(opt_loss.iloc[0])) if len(opt_loss) > 0 else 0

        fig.add_trace(go.Scatter3d(
            x=[optimal_sigma * 100],
            y=[optimal_a],
            z=[opt_z],
            mode="markers",
            marker=dict(size=12, color=COLORS["tertiary"], symbol="diamond"),
            name="Optimal (a, σ)",
            hovertemplate=f"Optimal: a={optimal_a:.4f}, σ={optimal_sigma*100:.3f}%<extra></extra>",
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="σ (%)", color=COLORS["text"]),
            yaxis=dict(title="a (mean reversion)", color=COLORS["text"]),
            zaxis=dict(title="log(1 + Loss)", color=COLORS["text"]),
            bgcolor=COLORS["bg_mid"],
            camera=dict(eye=dict(x=2.0, y=-1.5, z=1.2)),
        )
    )
    fig = _apply_dark_theme(fig, title, height=580)
    return fig


# =============================================================================
# 10. EXPOSURE PROFILE (EPE/ENE)
# =============================================================================

def plot_exposure_profile(
    exposure_df: pd.DataFrame,
    title: str = "Credit Exposure Profile — EPE & PFE 95%"
) -> go.Figure:
    """
    Plot Expected Positive Exposure and Potential Future Exposure profiles.
    Used in CVA calculation.
    """
    fig = go.Figure()

    # PFE 95% band
    fig.add_trace(go.Scatter(
        x=exposure_df["Time (Y)"].tolist() + exposure_df["Time (Y)"].tolist()[::-1],
        y=exposure_df["PFE 95%"].tolist() + [0] * len(exposure_df),
        fill="toself",
        fillcolor="rgba(255, 107, 107, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="PFE 95%",
        hoverinfo="skip",
    ))

    # EPE line
    fig.add_trace(go.Scatter(
        x=exposure_df["Time (Y)"],
        y=exposure_df["EPE"],
        name="EPE (Expected Positive Exposure)",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)",
        hovertemplate="t=%{x:.2f}Y, EPE=%{y:.4f}<extra></extra>",
    ))

    # ENE line
    fig.add_trace(go.Scatter(
        x=exposure_df["Time (Y)"],
        y=exposure_df["ENE"].abs(),
        name="ENE (Expected Negative Exposure)",
        line=dict(color=COLORS["secondary"], width=2, dash="dash"),
        hovertemplate="t=%{x:.2f}Y, |ENE|=%{y:.4f}<extra></extra>",
    ))

    # PFE 95%
    fig.add_trace(go.Scatter(
        x=exposure_df["Time (Y)"],
        y=exposure_df["PFE 95%"],
        name="PFE 95%",
        line=dict(color=COLORS["warning"], width=1.5, dash="dot"),
        hovertemplate="t=%{x:.2f}Y, PFE=%{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
    fig.update_xaxes(title_text="Time (Years)")
    fig.update_yaxes(title_text="Exposure (fraction of notional)")
    fig = _apply_dark_theme(fig, title, height=480)
    return fig
