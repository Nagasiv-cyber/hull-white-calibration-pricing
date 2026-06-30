"""
=============================================================================
Risk Analytics — Greeks, DV01, Duration, VaR, CVaR
=============================================================================
Module  : src/analytics/risk_metrics.py
Purpose : Compute interest rate risk measures for Hull-White derivative
          portfolios: Delta, Vega, DV01, Modified Duration, Convexity,
          Value-at-Risk (VaR), Conditional VaR (CVaR/Expected Shortfall).
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================

Risk Measures Implemented:
--------------------------
1. DV01 (Dollar Value of a Basis Point):
   DV01 = -(∂V/∂r) × 0.0001
   Computed via finite difference bump-and-reprice.

2. Modified Duration:
   D_mod = -1/P · ∂P/∂y

3. Convexity:
   C = 1/P · ∂²P/∂y²

4. Delta (rate sensitivity):
   Δ = ∂V/∂r₀ (sensitivity to initial short rate)

5. Vega (vol sensitivity):
   ν = ∂V/∂σ (sensitivity to HW vol parameter)

6. VaR / CVaR (from Monte Carlo):
   VaR_α = -inf{x : P(V < x) ≤ 1-α}
   CVaR_α = E[V | V < VaR_α]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Callable
import logging
import warnings

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Computes interest rate risk measures for Hull-White instruments.

    Parameters
    ----------
    model : HullWhiteModel
        Calibrated Hull-White model.
    r0 : float
        Current short rate.
    """

    def __init__(self, model, r0: float):
        self.model = model
        self.r0 = r0

    # ─────────────────────────────────────────────────────────────────────────
    # Finite Difference Greeks
    # ─────────────────────────────────────────────────────────────────────────

    def dv01(
        self,
        pricer: Callable[[float], float],
        bump_bps: float = 1.0
    ) -> float:
        """
        Compute DV01: price change per 1 basis point parallel shift.

        DV01 = [V(r + 1bp) - V(r - 1bp)] / 2

        Uses central finite difference for O(h²) accuracy.

        Parameters
        ----------
        pricer : callable
            Function f(r0) → price.
        bump_bps : float
            Bump size in basis points.

        Returns
        -------
        float
            DV01 in same units as pricer output × notional.
        """
        h = bump_bps / 10_000.0
        V_up = pricer(self.r0 + h)
        V_dn = pricer(self.r0 - h)
        return (V_up - V_dn) / 2.0

    def delta(
        self,
        pricer: Callable[[float], float],
        h: float = 1e-4
    ) -> float:
        """
        Delta: ∂V/∂r₀ via central finite difference.

        Parameters
        ----------
        pricer : callable
            f(r0) → price.
        h : float
            Finite difference step size.

        Returns
        -------
        float
            First derivative of price with respect to short rate.
        """
        V_up = pricer(self.r0 + h)
        V_dn = pricer(self.r0 - h)
        return (V_up - V_dn) / (2.0 * h)

    def gamma(
        self,
        pricer: Callable[[float], float],
        h: float = 1e-4
    ) -> float:
        """
        Gamma: ∂²V/∂r₀² via central finite difference.

        Returns
        -------
        float
            Second derivative (convexity-like measure).
        """
        V_0 = pricer(self.r0)
        V_up = pricer(self.r0 + h)
        V_dn = pricer(self.r0 - h)
        return (V_up - 2.0 * V_0 + V_dn) / (h**2)

    def vega(
        self,
        model_factory: Callable[[float], object],
        pricer_from_model: Callable[[object], float],
        h: float = 1e-4
    ) -> float:
        """
        Vega: ∂V/∂σ (sensitivity to Hull-White volatility parameter).

        Parameters
        ----------
        model_factory : callable
            f(sigma) → HullWhiteModel (builds new model with perturbed sigma).
        pricer_from_model : callable
            f(model) → price.
        h : float
            Perturbation step.

        Returns
        -------
        float
            Price change per unit change in σ.
        """
        model_up = model_factory(self.model.sigma + h)
        model_dn = model_factory(self.model.sigma - h)
        V_up = pricer_from_model(model_up)
        V_dn = pricer_from_model(model_dn)
        return (V_up - V_dn) / (2.0 * h)

    def mean_reversion_sensitivity(
        self,
        model_factory: Callable[[float], object],
        pricer_from_model: Callable[[object], float],
        h: float = 1e-4
    ) -> float:
        """
        Sensitivity to mean reversion speed 'a': ∂V/∂a.
        """
        model_up = model_factory(self.model.a + h)
        model_dn = model_factory(self.model.a - h)
        V_up = pricer_from_model(model_up)
        V_dn = pricer_from_model(model_dn)
        return (V_up - V_dn) / (2.0 * h)

    # ─────────────────────────────────────────────────────────────────────────
    # Bond Duration & Convexity
    # ─────────────────────────────────────────────────────────────────────────

    def modified_duration(
        self,
        T: float,
        h_bps: float = 1.0
    ) -> float:
        """
        Modified Duration of a HW zero-coupon bond.

        D_mod = -1/P · dP/dy ≈ -[P(r+h) - P(r-h)] / [2h·P(r)]

        Parameters
        ----------
        T : float
            Bond maturity.
        h_bps : float
            Rate bump in bps.

        Returns
        -------
        float
            Modified duration in years.
        """
        h = h_bps / 10_000.0
        P_0 = self.model.zcb_price(self.r0, 0, T)
        P_up = self.model.zcb_price(self.r0 + h, 0, T)
        P_dn = self.model.zcb_price(self.r0 - h, 0, T)

        if P_0 < 1e-12:
            return 0.0

        return -(P_up - P_dn) / (2.0 * h * P_0)

    def convexity(
        self,
        T: float,
        h_bps: float = 1.0
    ) -> float:
        """
        Convexity of a HW zero-coupon bond.

        C = 1/P · d²P/dy² ≈ [P(r+h) - 2P(r) + P(r-h)] / [h²·P(r)]

        Returns
        -------
        float
            Convexity in years².
        """
        h = h_bps / 10_000.0
        P_0 = self.model.zcb_price(self.r0, 0, T)
        P_up = self.model.zcb_price(self.r0 + h, 0, T)
        P_dn = self.model.zcb_price(self.r0 - h, 0, T)

        if P_0 < 1e-12:
            return 0.0

        return (P_up - 2.0 * P_0 + P_dn) / (h**2 * P_0)

    def portfolio_dv01(
        self,
        positions: List[Dict]
    ) -> pd.DataFrame:
        """
        Compute DV01 for a portfolio of IR instruments.

        Parameters
        ----------
        positions : list of dict
            Each dict: {'type': 'cap'|'floor'|'zcb'|'swaption',
                        'params': {...},  # pricing params
                        'notional': float}

        Returns
        -------
        pd.DataFrame
            DV01, delta, duration for each position and portfolio total.
        """
        records = []
        h = 1e-4  # 1 bp

        for pos in positions:
            pos_type = pos["type"]
            params = pos.get("params", {})
            notional = pos.get("notional", 1_000_000.0)

            try:
                if pos_type == "zcb":
                    T = params["T"]
                    def pricer_zcb(r, _T=T):
                        return notional * self.model.zcb_price(r, 0, _T)
                    V0 = pricer_zcb(self.r0)
                    dv01_val = self.dv01(pricer_zcb)
                    dur = self.modified_duration(T)
                    convex = self.convexity(T)

                elif pos_type == "cap":
                    T = params["T"]
                    K = params["K"]
                    tenor = params.get("tenor", 0.25)
                    def pricer_cap(r, _T=T, _K=K, _ten=tenor):
                        return notional * self.model.cap_price(r, 0, _T, _K, _ten)
                    V0 = pricer_cap(self.r0)
                    dv01_val = self.dv01(pricer_cap)
                    dur = -dv01_val / (V0 * h) if V0 > 1e-8 else 0
                    convex = self.gamma(pricer_cap) * (V0 if V0 > 1e-8 else 1)

                elif pos_type == "floor":
                    T = params["T"]
                    K = params["K"]
                    tenor = params.get("tenor", 0.25)
                    def pricer_floor(r, _T=T, _K=K, _ten=tenor):
                        return notional * self.model.floor_price(r, 0, _T, _K, _ten)
                    V0 = pricer_floor(self.r0)
                    dv01_val = self.dv01(pricer_floor)
                    dur = -dv01_val / (V0 * h) if V0 > 1e-8 else 0
                    convex = 0.0

                else:
                    V0, dv01_val, dur, convex = 0, 0, 0, 0

                records.append({
                    "Type": pos_type.upper(),
                    "Notional": f"${notional:,.0f}",
                    "MtM Value": round(V0, 6),
                    "DV01 ($)": round(dv01_val, 4),
                    "Duration (Y)": round(dur, 4),
                    "Convexity (Y²)": round(convex, 4),
                })

            except Exception as e:
                logger.warning(f"Failed to price position {pos_type}: {e}")
                records.append({"Type": pos_type.upper(), "Error": str(e)})

        df = pd.DataFrame(records)

        # Portfolio total
        if "DV01 ($)" in df.columns:
            total_dv01 = df["DV01 ($)"].sum()
            total_mtm = df["MtM Value"].sum()
            total_row = pd.DataFrame([{
                "Type": "PORTFOLIO TOTAL",
                "MtM Value": round(total_mtm, 6),
                "DV01 ($)": round(total_dv01, 4),
            }])
            df = pd.concat([df, total_row], ignore_index=True)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Monte Carlo Risk Measures
    # ─────────────────────────────────────────────────────────────────────────

    def var_cvar(
        self,
        pnl_distribution: np.ndarray,
        confidence: float = 0.99
    ) -> Dict[str, float]:
        """
        Compute VaR and CVaR (Expected Shortfall) from P&L distribution.

        VaR_α = -Q_{1-α}(P&L)
        CVaR_α = -E[P&L | P&L ≤ -VaR_α]

        Parameters
        ----------
        pnl_distribution : np.ndarray
            Array of simulated P&L values (positive = profit).
        confidence : float
            Confidence level (e.g., 0.99 for 99% VaR).

        Returns
        -------
        dict
            VaR, CVaR, Sharpe-like ratio, max_drawdown, mean, std.
        """
        pnl = np.asarray(pnl_distribution)

        var_threshold = np.percentile(pnl, (1 - confidence) * 100)
        var = -var_threshold  # Convention: VaR is a positive loss number

        tail_losses = pnl[pnl <= var_threshold]
        cvar = -float(np.mean(tail_losses)) if len(tail_losses) > 0 else var

        return {
            "VaR": round(var, 6),
            "CVaR": round(cvar, 6),
            "Mean P&L": round(float(np.mean(pnl)), 6),
            "Std P&L": round(float(np.std(pnl)), 6),
            "Min P&L": round(float(np.min(pnl)), 6),
            "Max P&L": round(float(np.max(pnl)), 6),
            "VaR/Std": round(var / max(np.std(pnl), 1e-10), 4),
            "Confidence": confidence,
        }

    def scenario_analysis(
        self,
        pricer: Callable[[float], float],
        rate_shifts_bps: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute price impact across a grid of rate scenarios.

        Standard regulatory scenarios per BCBS IRRBB (2016):
        ±200bp, ±100bp, ±50bp, ±25bp, ±10bp, flat.

        Parameters
        ----------
        pricer : callable
            f(r0) → price.
        rate_shifts_bps : list, optional
            Rate shifts in bps. Default: BCBS standard scenarios.

        Returns
        -------
        pd.DataFrame
            Scenario prices and P&L impact.
        """
        if rate_shifts_bps is None:
            # BCBS IRRBB standard shock scenarios
            rate_shifts_bps = [-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200]

        base_price = pricer(self.r0)
        records = []

        for shift in rate_shifts_bps:
            r_shocked = self.r0 + shift / 10_000.0
            shocked_price = pricer(r_shocked)
            pnl = shocked_price - base_price
            pnl_bps = pnl * 10_000

            records.append({
                "Scenario": f"{'+' if shift >= 0 else ''}{shift} bps",
                "Rate (%)": round((self.r0 + shift / 10_000) * 100, 3),
                "Price": round(shocked_price, 6),
                "P&L": round(pnl, 6),
                "P&L (bps)": round(pnl_bps, 2),
            })

        return pd.DataFrame(records)

    def interest_rate_var(
        self,
        rate_paths: np.ndarray,
        pricer: Callable[[float], float],
        horizon_days: int = 1,
        confidence: float = 0.99
    ) -> Dict:
        """
        Historical simulation VaR using simulated rate paths.

        For each simulated short rate at horizon, reprice the instrument
        and compute the P&L distribution.

        Parameters
        ----------
        rate_paths : np.ndarray
            MC rate paths, shape (n_paths, n_time_steps).
        pricer : callable
            f(r0) → price.
        horizon_days : int
            Risk horizon in days (1D VaR, 10D VaR, etc.).
        confidence : float
            VaR confidence level.

        Returns
        -------
        dict
            VaR, CVaR, and distribution statistics.
        """
        # Get rate distribution at horizon
        steps_per_day = rate_paths.shape[1] // int(self._T_simulated_days(rate_paths))
        horizon_idx = min(horizon_days * steps_per_day, rate_paths.shape[1] - 1)

        r_horizon = rate_paths[:, horizon_idx]
        base_price = pricer(self.r0)

        # Reprice at each horizon rate
        horizon_prices = np.array([pricer(r) for r in r_horizon])
        pnl = horizon_prices - base_price

        return self.var_cvar(pnl, confidence)

    @staticmethod
    def _T_simulated_days(rate_paths: np.ndarray) -> float:
        """Estimate simulation horizon from path shape."""
        return 252.0  # Default assumption: 1 year = 252 steps
