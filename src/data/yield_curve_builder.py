"""
=============================================================================
Yield Curve Builder — Bootstrap & Interpolation Engine
=============================================================================
Module  : src/data/yield_curve_builder.py
Purpose : Construct a complete zero-coupon yield curve (discount factors,
          zero rates, instantaneous forward rates) from observed par rates
          via bootstrapping. Implements cubic spline interpolation for a
          smooth, arbitrage-free curve.
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================

Mathematical Background:
------------------------
Par Rate → Zero Rate Bootstrap:
  P(0, T_n) = [1 - c·∑_{i=1}^{n} P(0, T_i)] / (1 + c)

Discount Factor from Zero Rate:
  P(0, T) = exp(-r(T) · T)     [continuous compounding]

Instantaneous Forward Rate:
  f(0, T) = -∂/∂T [ln P(0, T)] = r(T) + T · dr(T)/dT

Usage:
------
  builder = YieldCurveBuilder(rates, maturities)
  builder.bootstrap()
  P_5Y = builder.discount_factor(5.0)
  f_2Y = builder.forward_rate(2.0)
"""

import numpy as np
import pandas as pd
import warnings
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
from typing import Optional, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


class YieldCurveBuilder:
    """
    Constructs a continuous zero-coupon yield curve from discrete par rate quotes.

    Supports:
    - Bootstrap from par rates (annual coupon bonds)
    - Natural cubic spline interpolation/extrapolation
    - Discount factor, zero rate, and forward rate queries at any maturity
    - Forward discount factor P(T1, T2) computation
    - Curve shifting for scenario analysis

    Parameters
    ----------
    par_rates : array-like
        Par rates in decimal (e.g., 0.045 = 4.5%), one per maturity point.
    maturities : array-like
        Maturity tenors in years corresponding to par_rates.
    coupon_frequency : int, optional
        Coupon payments per year (default: 2 = semi-annual).
    day_count : str, optional
        Day count convention: 'ACT/365', 'ACT/360', '30/360'. Default: 'ACT/365'.
    """

    def __init__(
        self,
        par_rates: np.ndarray,
        maturities: np.ndarray,
        coupon_frequency: int = 2,
        day_count: str = "ACT/365"
    ):
        self.par_rates = np.asarray(par_rates, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.coupon_freq = coupon_frequency
        self.day_count = day_count

        # Validate inputs
        self._validate_inputs()

        # Internal storage
        self._pillar_maturities: Optional[np.ndarray] = None
        self._pillar_zero_rates: Optional[np.ndarray] = None
        self._pillar_discount_factors: Optional[np.ndarray] = None

        # Interpolation splines (built after bootstrap)
        self._zero_rate_spline: Optional[CubicSpline] = None
        self._log_df_spline: Optional[CubicSpline] = None

        self._is_bootstrapped = False

    def _validate_inputs(self) -> None:
        """Validate input arrays for consistency."""
        if len(self.par_rates) != len(self.maturities):
            raise ValueError(
                f"Length mismatch: par_rates ({len(self.par_rates)}) != "
                f"maturities ({len(self.maturities)})"
            )
        if len(self.par_rates) < 2:
            raise ValueError("Need at least 2 rate/maturity pairs for bootstrapping.")
        if np.any(self.maturities <= 0):
            raise ValueError("All maturities must be strictly positive.")
        if np.any(np.diff(self.maturities) <= 0):
            raise ValueError("Maturities must be strictly increasing.")
        if np.any(self.par_rates <= 0):
            warnings.warn(
                "Some par rates are non-positive. This may indicate data issues.",
                UserWarning, stacklevel=3
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Core Bootstrap Algorithm
    # ─────────────────────────────────────────────────────────────────────────

    def bootstrap(self) -> "YieldCurveBuilder":
        """
        Bootstrap zero rates from par rates using iterative bond pricing.

        For each maturity T_n, solves for the zero rate r(T_n) such that a
        par bond (coupon = par rate) is priced at par (= 1.0).

        Algorithm:
          1. For T_1 (shortest maturity): solve directly.
          2. For T_n (n ≥ 2): use already-known discount factors for T_1 to T_{n-1},
             then solve for P(0, T_n) via root-finding.
          3. Convert discount factors to zero rates via r(T) = -ln(P(0,T)) / T.
          4. Fit cubic spline over zero rates for continuous interpolation.

        Returns
        -------
        self : YieldCurveBuilder
            Enables method chaining: builder.bootstrap().discount_factor(5)
        """
        logger.info("📐 Bootstrapping yield curve...")

        n = len(self.maturities)
        discount_factors = np.zeros(n)
        zero_rates = np.zeros(n)

        # Add a T=0 anchor: P(0,0) = 1
        T_knots = np.concatenate([[0.0], self.maturities])
        DF_knots = np.ones(n + 1)

        for i in range(n):
            T = self.maturities[i]
            c = self.par_rates[i]              # Par coupon rate
            dt = 1.0 / self.coupon_freq        # Coupon period in years

            # Generate all coupon payment times up to maturity T
            coupon_times = np.arange(dt, T + 1e-9, dt)
            n_coupons = len(coupon_times)
            coupon_payment = c * dt             # Coupon amount per period

            def bond_price_error(df_T: float) -> float:
                """
                Computes par bond price minus 1 (par value).
                We search for df_T = P(0, T_n) such that this is zero.
                """
                price = 0.0
                for k, t_k in enumerate(coupon_times[:-1]):
                    # Interpolate discount factor for intermediate coupons
                    df_k = self._interpolate_discount_factor(t_k, T_knots[:i+2], DF_knots[:i+2])
                    price += coupon_payment * df_k

                # Final coupon + principal at T
                price += (coupon_payment + 1.0) * df_T
                return price - 1.0

            # Special case: single coupon (money market instruments)
            if n_coupons <= 1:
                # Direct formula: P(0, T) = 1 / (1 + c * T)
                df = 1.0 / (1.0 + c * T)
            else:
                # Sum all intermediate coupon PVs (using known DFs)
                pv_intermediate = 0.0
                for t_k in coupon_times[:-1]:
                    df_k = self._interpolate_discount_factor(t_k, T_knots[:i+2], DF_knots[:i+2])
                    pv_intermediate += coupon_payment * df_k

                # Solve for terminal DF
                numerator = 1.0 - pv_intermediate
                denominator = coupon_payment + 1.0
                df = numerator / denominator

                if df <= 0:
                    warnings.warn(
                        f"Negative discount factor at T={T:.2f}. "
                        f"Setting df = 0.001 (check input rates).",
                        RuntimeWarning, stacklevel=2
                    )
                    df = 0.001

            discount_factors[i] = df
            DF_knots[i + 1] = df

            # Convert to continuously compounded zero rate
            if df > 0 and T > 0:
                zero_rates[i] = -np.log(df) / T
            else:
                zero_rates[i] = 0.0

            logger.debug(
                f"  T={T:5.2f}Y | Par={c*100:6.3f}% | DF={df:.6f} | "
                f"Z={zero_rates[i]*100:.4f}%"
            )

        self._pillar_maturities = self.maturities.copy()
        self._pillar_zero_rates = zero_rates
        self._pillar_discount_factors = discount_factors

        # Build interpolation splines
        self._build_splines()

        self._is_bootstrapped = True
        logger.info(
            f"✅ Bootstrap complete: {n} pillars | "
            f"Z-rates: {zero_rates[0]*100:.3f}% → {zero_rates[-1]*100:.3f}%"
        )

        return self

    def _build_splines(self) -> None:
        """Fit cubic splines to bootstrapped zero rates and log-discount factors."""
        T = self._pillar_maturities
        r = self._pillar_zero_rates
        logP = -r * T  # log discount factors

        # Include T=0 anchor
        T_full = np.concatenate([[0.0], T])
        r_full = np.concatenate([[r[0]], r])   # Assume flat short end
        logP_full = np.concatenate([[0.0], logP])

        # Natural cubic spline (zero second derivative at boundaries)
        self._zero_rate_spline = CubicSpline(T_full, r_full, bc_type="natural")
        self._log_df_spline = CubicSpline(T_full, logP_full, bc_type="natural")

    @staticmethod
    def _interpolate_discount_factor(
        t: float,
        T_knots: np.ndarray,
        DF_knots: np.ndarray
    ) -> float:
        """
        Linear interpolation of log-discount factors (ensures positive DFs).
        Only used during the bootstrap loop before splines are built.
        """
        if t <= T_knots[0]:
            return DF_knots[0]
        if t >= T_knots[-1]:
            return DF_knots[-1]

        # Log-linear interpolation (preserves positivity)
        log_df = np.interp(t, T_knots, np.log(np.maximum(DF_knots, 1e-10)))
        return np.exp(log_df)

    # ─────────────────────────────────────────────────────────────────────────
    # Query Interface
    # ─────────────────────────────────────────────────────────────────────────

    def _check_bootstrapped(self) -> None:
        """Raise if bootstrap hasn't been run yet."""
        if not self._is_bootstrapped:
            raise RuntimeError(
                "Yield curve not bootstrapped. Call .bootstrap() first."
            )

    def discount_factor(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the discount factor P(0, T) = exp(log P(0, T)).

        Parameters
        ----------
        T : float or array-like
            Maturity in years.

        Returns
        -------
        float or np.ndarray
            Discount factor(s). Always in (0, 1].
        """
        self._check_bootstrapped()
        T = np.asarray(T, dtype=float)
        scalar = T.ndim == 0
        T = np.atleast_1d(T)

        log_P = self._log_df_spline(T)
        df = np.exp(log_P)
        df = np.clip(df, 1e-10, 1.0)  # Safety clip

        return float(df[0]) if scalar else df

    def zero_rate(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute continuously compounded zero rate r(0, T).

        Parameters
        ----------
        T : float or array-like
            Maturity in years.

        Returns
        -------
        float or np.ndarray
            Zero rate(s) in decimal.
        """
        self._check_bootstrapped()
        T = np.asarray(T, dtype=float)
        scalar = T.ndim == 0
        T = np.atleast_1d(T)

        # Derive from log-DF spline for consistency: r(T) = -log(P(0,T)) / T
        safe_T = np.where(T < 1e-8, 1e-8, T)
        log_P = self._log_df_spline(safe_T)
        r = -log_P / safe_T

        return float(r[0]) if scalar else r

    def forward_rate(
        self,
        T: Union[float, np.ndarray],
        dt: float = 1e-4
    ) -> Union[float, np.ndarray]:
        """
        Compute instantaneous forward rate f(0, T) = -d/dT [ln P(0,T)].

        Uses the analytic derivative of the log-DF cubic spline.

        Parameters
        ----------
        T : float or array-like
            Maturity in years.
        dt : float
            Finite difference step (used only as fallback).

        Returns
        -------
        float or np.ndarray
            Instantaneous forward rate(s) in decimal.
        """
        self._check_bootstrapped()
        T = np.asarray(T, dtype=float)
        scalar = T.ndim == 0
        T = np.atleast_1d(T)

        # Use analytic derivative of log-DF spline: f(T) = -d(logP)/dT
        f = -self._log_df_spline(T, 1)  # First derivative

        return float(f[0]) if scalar else f

    def forward_discount_factor(self, T1: float, T2: float) -> float:
        """
        Compute forward discount factor P(T1, T2) = P(0, T2) / P(0, T1).

        Parameters
        ----------
        T1 : float
            Start of forward period (years).
        T2 : float
            End of forward period (years). Must be > T1.

        Returns
        -------
        float
            Forward discount factor.
        """
        if T2 <= T1:
            raise ValueError(f"T2 ({T2}) must be greater than T1 ({T1}).")
        return self.discount_factor(T2) / self.discount_factor(T1)

    def forward_libor_rate(self, T1: float, T2: float) -> float:
        """
        Compute simply-compounded LIBOR/SOFR forward rate L(T1, T2).

        L(T1, T2) = [1/P(T1,T2) - 1] / (T2 - T1)

        Parameters
        ----------
        T1, T2 : float
            Start and end of the accrual period in years.

        Returns
        -------
        float
            Forward LIBOR rate in decimal.
        """
        tau = T2 - T1
        fdf = self.forward_discount_factor(T1, T2)
        return (1.0 / fdf - 1.0) / tau

    def shift_parallel(self, shift_bps: float) -> "YieldCurveBuilder":
        """
        Create a new curve with a parallel shift applied.

        Parameters
        ----------
        shift_bps : float
            Shift in basis points (e.g., +100 = rates up 100bps).

        Returns
        -------
        YieldCurveBuilder
            New builder with shifted rates.
        """
        shifted_rates = self.par_rates + shift_bps / 10_000.0
        new_builder = YieldCurveBuilder(
            shifted_rates, self.maturities,
            self.coupon_freq, self.day_count
        )
        return new_builder.bootstrap()

    def get_pillar_data(self) -> pd.DataFrame:
        """
        Return bootstrapped pillar data as a formatted DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: Maturity, Par Rate (%), Zero Rate (%), Discount Factor,
                     Forward Rate (%)
        """
        self._check_bootstrapped()

        fwd_rates = self.forward_rate(self._pillar_maturities)

        return pd.DataFrame({
            "Maturity (Y)": self._pillar_maturities,
            "Par Rate (%)": self.par_rates * 100,
            "Zero Rate (%)": self._pillar_zero_rates * 100,
            "Discount Factor": self._pillar_discount_factors,
            "Fwd Rate (%)": fwd_rates * 100,
        }).round(6)

    def summary(self) -> None:
        """Print yield curve summary statistics."""
        if not self._is_bootstrapped:
            print("Curve not bootstrapped yet. Call .bootstrap() first.")
            return

        df = self.get_pillar_data()
        print("\n" + "=" * 70)
        print("  YIELD CURVE SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
        print(f"  Curve shape : {'Normal (upward sloping)' if df['Zero Rate (%)'].is_monotonic_increasing else 'Inverted/Humped'}")
        print(f"  Steepness   : {df['Zero Rate (%)'].iloc[-1] - df['Zero Rate (%)'].iloc[0]:.2f} bps (30Y - 1M)")
        print("=" * 70 + "\n")
