"""
=============================================================================
Monte Carlo Simulation Engine — Hull-White Short Rate Paths
=============================================================================
Module  : src/models/monte_carlo.py
Purpose : Simulates thousands of short-rate paths under the Hull-White model
          using Euler-Maruyama and exact simulation schemes. Computes path-
          dependent quantities: bond prices, discount factors, exposure profiles.
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================

Discretization Schemes:
-----------------------
1. Euler-Maruyama (standard):
   r(t+dt) ≈ r(t) + [θ(t) - a·r(t)]·dt + σ·√dt·Z
   where Z ~ N(0,1)

2. Exact Simulation (no discretization error):
   r(T) | r(t) ~ Normal(μ(t,T), v²(t,T))
   where:
     μ(t,T) = r(t)·e^{-a(T-t)} + α(T) - α(t)·e^{-a(T-t)}
     v²(t,T) = σ²/(2a) · (1 - e^{-2a(T-t)})
     α(t) = f_M(0,t) + σ²/(2a²)·(1-e^{-at})²

3. Antithetic Variates (variance reduction):
   For each Z, also simulate with -Z, halving MC variance.

4. Quasi-Monte Carlo (Sobol sequences):
   Low-discrepancy sequences for faster convergence.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import warnings
import logging
from tqdm import tqdm
import time

from .hull_white import HullWhiteModel

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for Hull-White short-rate paths.

    Generates risk-neutral short-rate path simulations and computes
    path-dependent statistics for derivative pricing and risk analysis.

    Parameters
    ----------
    model : HullWhiteModel
        Calibrated Hull-White model instance.
    n_paths : int
        Number of Monte Carlo paths (default: 10,000).
    n_steps : int
        Time steps per year (default: 252 for daily).
    scheme : str
        Discretization scheme: 'euler', 'exact', or 'milstein'.
    seed : int, optional
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates for variance reduction (default: True).

    Examples
    --------
    >>> mc = MonteCarloEngine(model, n_paths=10000, n_steps=252)
    >>> paths = mc.simulate(T=10.0)
    >>> bond_price = mc.price_zcb(T=5.0)
    """

    def __init__(
        self,
        model: HullWhiteModel,
        n_paths: int = 10_000,
        n_steps: int = 252,
        scheme: str = "exact",
        seed: Optional[int] = 42,
        antithetic: bool = True
    ):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.scheme = scheme.lower()
        self.seed = seed
        self.antithetic = antithetic

        # Storage for simulated paths
        self._rate_paths: Optional[np.ndarray] = None   # shape: (n_paths, n_time)
        self._time_grid: Optional[np.ndarray] = None
        self._T_simulated: Optional[float] = None

        # RNG initialization
        self._rng = np.random.default_rng(seed)

        logger.info(
            f"🎲 MC Engine initialized: {n_paths:,} paths, "
            f"{n_steps} steps/yr, scheme={scheme}, "
            f"antithetic={antithetic}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        T: float,
        r0: Optional[float] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Simulate Hull-White short-rate paths over [0, T].

        Parameters
        ----------
        T : float
            Simulation horizon in years.
        r0 : float, optional
            Initial short rate. Defaults to model curve's short rate.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        rate_paths : np.ndarray, shape (n_paths, n_time_steps + 1)
            Matrix of simulated short-rate paths.
            rows = paths, columns = time points.
        """
        if r0 is None:
            r0 = self.model.curve.forward_rate(1e-4)

        total_steps = int(T * self.n_steps)
        dt = T / total_steps
        time_grid = np.linspace(0, T, total_steps + 1)

        # If antithetic, we generate n_paths/2 base paths and mirror
        if self.antithetic:
            n_base = self.n_paths // 2
        else:
            n_base = self.n_paths

        start = time.time()

        if self.scheme == "exact":
            paths = self._simulate_exact(r0, time_grid, n_base, show_progress)
        elif self.scheme == "euler":
            paths = self._simulate_euler(r0, time_grid, n_base, dt, show_progress)
        elif self.scheme == "milstein":
            paths = self._simulate_milstein(r0, time_grid, n_base, dt, show_progress)
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}. Use 'exact', 'euler', or 'milstein'.")

        # Apply antithetic variates
        if self.antithetic:
            mean_paths = paths.mean(axis=1, keepdims=True)
            anti_paths = 2 * mean_paths - paths   # Mirror around mean
            paths = np.vstack([paths, anti_paths])
            # Truncate/pad to exact n_paths
            paths = paths[:self.n_paths]

        self._rate_paths = paths
        self._time_grid = time_grid
        self._T_simulated = T

        elapsed = time.time() - start
        logger.info(
            f"✅ Simulation complete: {self.n_paths:,} paths × {total_steps:,} steps "
            f"in {elapsed:.1f}s "
            f"(r0={r0*100:.2f}%, T={T}Y)"
        )

        return paths

    def _simulate_exact(
        self,
        r0: float,
        time_grid: np.ndarray,
        n_paths: int,
        show_progress: bool
    ) -> np.ndarray:
        """
        Exact simulation of Hull-White paths.

        Since HW is a Gaussian model (affine), r(t) conditional on r(s)
        is exactly normally distributed. No discretization error.

        At each step t → t+dt:
          r(t+dt) | r(t) ~ N(μ, v²)
          μ = r(t)·e^{-a·dt} + [α(t+dt) - α(t)·e^{-a·dt}]
          v² = σ²/(2a)·(1 - e^{-2a·dt})

        where α(t) = f(0,t) + σ²/(2a²)·(1-e^{-at})²
        """
        n_steps = len(time_grid) - 1
        dt = time_grid[1] - time_grid[0]
        a = self.model.a
        sigma = self.model.sigma

        # Precompute step-level constants
        exp_a_dt = np.exp(-a * dt)

        # Exact variance per step
        if abs(a) < 1e-8:
            step_var = sigma**2 * dt
        else:
            step_var = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * dt))
        step_std = np.sqrt(step_var)

        # Precompute alpha function: α(t) = f(0,t) + σ²/(2a²)(1-e^{-at})²
        def alpha(t: float) -> float:
            f_t = self.model.curve.forward_rate(t) if t > 1e-6 else self.model.curve.forward_rate(1e-4)
            if abs(a) < 1e-8:
                return f_t + 0.5 * sigma**2 * t**2
            return f_t + (sigma**2 / (2 * a**2)) * (1 - np.exp(-a * t))**2

        # Cache alpha values for each time step (expensive with cubic spline)
        alpha_vals = np.array([alpha(t) for t in time_grid])

        # Initialize path matrix
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        # Generate all standard normal increments at once (vectorized)
        # Shape: (n_paths, n_steps)
        Z = self._rng.standard_normal((n_paths, n_steps))

        iterator = tqdm(range(n_steps), desc="  Simulating paths", leave=False) \
            if show_progress else range(n_steps)

        for step in iterator:
            t_cur = time_grid[step]
            t_next = time_grid[step + 1]

            # Drift: mean reversion + θ(t) adjustment
            alpha_cur = alpha_vals[step]
            alpha_next = alpha_vals[step + 1]

            # Exact mean: μ = r(t)·e^{-a·dt} + [α(t+dt) - α(t)·e^{-a·dt}]
            drift_term = alpha_next - alpha_cur * exp_a_dt
            paths[:, step + 1] = (
                paths[:, step] * exp_a_dt
                + drift_term
                + step_std * Z[:, step]
            )

        return paths

    def _simulate_euler(
        self,
        r0: float,
        time_grid: np.ndarray,
        n_paths: int,
        dt: float,
        show_progress: bool
    ) -> np.ndarray:
        """
        Euler-Maruyama discretization of Hull-White SDE.

        r(t+dt) = r(t) + [θ(t) - a·r(t)]·dt + σ·√dt·Z

        θ(t) = f_M(0,t) + (σ²/2a²)(1-e^{-2at}) + a·f_M(0,t)
               ≈ ∂f_M/∂t(0,t) + a·f_M(0,t) + σ²/a·(1-e^{-at})·e^{-at}
        """
        n_steps = len(time_grid) - 1
        a = self.model.a
        sigma = self.model.sigma
        sqrt_dt = np.sqrt(dt)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0

        Z = self._rng.standard_normal((n_paths, n_steps))

        iterator = tqdm(range(n_steps), desc="  Simulating paths (Euler)", leave=False) \
            if show_progress else range(n_steps)

        for step in iterator:
            t = time_grid[step]
            f_t = self.model.curve.forward_rate(t) if t > 1e-6 else self.model.curve.forward_rate(1e-4)

            # Approximate θ(t) using market forward rate derivative
            eps = 1e-4
            f_t_eps = self.model.curve.forward_rate(t + eps)
            df_dt = (f_t_eps - f_t) / eps

            if abs(a) < 1e-8:
                theta_t = df_dt + sigma**2 * t
            else:
                theta_t = df_dt + a * f_t + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))

            drift = (theta_t - a * paths[:, step]) * dt
            diffusion = sigma * sqrt_dt * Z[:, step]
            paths[:, step + 1] = paths[:, step] + drift + diffusion

        return paths

    def _simulate_milstein(
        self,
        r0: float,
        time_grid: np.ndarray,
        n_paths: int,
        dt: float,
        show_progress: bool
    ) -> np.ndarray:
        """
        Milstein scheme (same as Euler for Hull-White since diffusion has no r-dependence).

        For Hull-White: diffusion = σ (constant), so Milstein = Euler.
        Provided for completeness and code organization.
        """
        logger.debug("Milstein = Euler for Hull-White (constant diffusion). Using Euler.")
        return self._simulate_euler(r0, time_grid, n_paths, dt, show_progress)

    # ─────────────────────────────────────────────────────────────────────────
    # Pricing via Monte Carlo
    # ─────────────────────────────────────────────────────────────────────────

    def _check_simulated(self) -> None:
        """Raise if simulation hasn't been run."""
        if self._rate_paths is None:
            raise RuntimeError("No paths simulated. Call .simulate(T) first.")

    def price_zcb(self, T: float) -> Dict:
        """
        Price a zero-coupon bond maturing at T via Monte Carlo.

        P(0, T) ≈ E^Q[exp(-∫_0^T r(t) dt)]
        Approximated by: mean over paths of exp(-∑ r(t_i)·dt)

        Parameters
        ----------
        T : float
            Bond maturity in years. Must be ≤ simulation horizon.

        Returns
        -------
        dict with 'price', 'std_error', 'ci_lower', 'ci_upper', 'analytic_price'
        """
        self._check_simulated()

        if T > self._T_simulated:
            raise ValueError(f"T={T} exceeds simulation horizon {self._T_simulated}.")

        # Find time index closest to T
        T_idx = np.searchsorted(self._time_grid, T)
        dt = self._time_grid[1] - self._time_grid[0]

        # Compute path-wise discount factors: exp(-∑ r_i · dt)
        integral_r = np.sum(self._rate_paths[:, :T_idx], axis=1) * dt
        discount_factors = np.exp(-integral_r)

        mc_price = float(np.mean(discount_factors))
        mc_std = float(np.std(discount_factors) / np.sqrt(self.n_paths))
        ci_95 = 1.96 * mc_std

        # Analytic price for comparison
        r0 = float(self._rate_paths[:, 0].mean())
        analytic_price = self.model.zcb_price(r0, 0.0, T)

        return {
            "price": mc_price,
            "std_error": mc_std,
            "ci_lower": mc_price - ci_95,
            "ci_upper": mc_price + ci_95,
            "analytic_price": analytic_price,
            "pricing_error_bps": abs(mc_price - analytic_price) * 10_000,
            "maturity": T,
        }

    def price_cap(
        self,
        K: float,
        T: float,
        tenor: float = 0.25,
        notional: float = 1.0
    ) -> Dict:
        """
        Price an interest rate cap via Monte Carlo simulation.

        At each reset date T_i, the caplet payoff is:
          max(L(T_i, T_{i+1}) - K, 0) × τ × notional

        The LIBOR rate L(T_i, T_{i+1}) is approximated from the
        simulated short rate using: L ≈ r(T_i) (proxy).

        Parameters
        ----------
        K : float
            Cap strike rate in decimal.
        T : float
            Cap maturity.
        tenor : float
            Caplet reset period.
        notional : float
            Notional amount.

        Returns
        -------
        dict with MC price, analytic price, and std error.
        """
        self._check_simulated()

        reset_times = np.arange(tenor, T + 1e-9, tenor)
        total_cap_pv = np.zeros(self.n_paths)

        dt = self._time_grid[1] - self._time_grid[0]

        for T_start in reset_times[:-1]:
            T_end = T_start + tenor
            idx_start = np.searchsorted(self._time_grid, T_start)
            idx_end = np.searchsorted(self._time_grid, T_end)

            # LIBOR approximation using path average over period
            r_period_avg = np.mean(
                self._rate_paths[:, idx_start:idx_end], axis=1
            )

            # Discount to time 0
            integral_r = np.sum(self._rate_paths[:, :idx_end], axis=1) * dt
            df_paths = np.exp(-integral_r)

            # Caplet payoff
            payoff = np.maximum(r_period_avg - K, 0) * tenor * notional
            total_cap_pv += payoff * df_paths

        mc_price = float(np.mean(total_cap_pv))
        mc_std = float(np.std(total_cap_pv) / np.sqrt(self.n_paths))

        # Analytic comparison
        r0 = float(self._rate_paths[0, 0])
        analytic = self.model.cap_price(r0, 0.0, T, K, tenor, notional)

        return {
            "price": mc_price,
            "analytic_price": analytic,
            "std_error": mc_std,
            "ci_lower": mc_price - 1.96 * mc_std,
            "ci_upper": mc_price + 1.96 * mc_std,
            "pricing_error_bps": abs(mc_price - analytic) * 10_000,
        }

    def compute_exposure_profile(
        self,
        derivative_type: str = "swap",
        K: float = 0.04,
        T: float = 10.0,
        n_monitoring_dates: int = 40
    ) -> pd.DataFrame:
        """
        Compute the Expected Positive Exposure (EPE) profile for a derivative.

        EPE(t) = E^Q[max(V(t), 0)]

        This is the key input to CVA (Credit Valuation Adjustment) calculations,
        which are mandatory under Basel III/FRTB for OTC derivatives.

        Parameters
        ----------
        derivative_type : str
            'swap', 'cap', or 'floor'.
        K : float
            Strike/fixed rate.
        T : float
            Derivative maturity.
        n_monitoring_dates : int
            Number of exposure monitoring dates.

        Returns
        -------
        pd.DataFrame
            Columns: time, EPE (expected positive exposure), ENE (negative),
                     PFE_95 (95th percentile future exposure).
        """
        self._check_simulated()

        monitoring_times = np.linspace(0, T, n_monitoring_dates + 1)
        records = []

        for t_mon in monitoring_times:
            idx = np.searchsorted(self._time_grid, t_mon)
            r_t = self._rate_paths[:, idx]

            # Residual value of derivative at each monitoring time
            if derivative_type == "cap":
                residual_T = T - t_mon
                tenor_thresh = 0.25
                if residual_T < tenor_thresh:
                    values = np.zeros(self.n_paths)
                else:
                    values = np.array([
                        self.model.cap_price(r, t_mon, T, K)
                        for r in r_t[::max(1, self.n_paths // 500)]  # Subsample for speed
                    ])
                    # Expand to full paths
                    if len(values) < self.n_paths:
                        values = np.interp(
                            np.arange(self.n_paths),
                            np.arange(0, self.n_paths, max(1, self.n_paths // 500))[:len(values)],
                            values
                        )
            elif derivative_type == "swap":
                # Simplified: PV = annuity × (swap_rate - K)
                swap_rates = np.array([
                    self.model.model_par_rate(r, T, 0.5) for r in r_t[::max(1, self.n_paths // 500)]
                ])
                if len(swap_rates) < self.n_paths:
                    swap_rates = np.interp(
                        np.arange(self.n_paths),
                        np.arange(0, self.n_paths, max(1, self.n_paths // 500))[:len(swap_rates)],
                        swap_rates
                    )
                annuity_approx = sum(
                    0.5 * self.model.zcb_price(float(r_t.mean()), t_mon, T_i)
                    for T_i in np.arange(t_mon + 0.5, T + 1e-9, 0.5)
                )
                values = annuity_approx * (swap_rates - K)
            else:
                values = np.zeros(self.n_paths)

            epe = float(np.mean(np.maximum(values, 0)))
            ene = float(np.mean(np.minimum(values, 0)))
            pfe_95 = float(np.percentile(np.maximum(values, 0), 95))

            records.append({
                "Time (Y)": round(t_mon, 4),
                "EPE": epe,
                "ENE": ene,
                "PFE 95%": pfe_95,
                "Mean MtM": float(np.mean(values)),
            })

        return pd.DataFrame(records)

    def get_statistics(self) -> pd.DataFrame:
        """
        Compute descriptive statistics across all simulated short-rate paths.

        Returns
        -------
        pd.DataFrame
            Mean, std, percentiles of the rate distribution at each time step.
        """
        self._check_simulated()

        percentiles = [5, 25, 50, 75, 95]
        records = []

        # Sample at regular intervals to reduce output size
        sample_idx = np.linspace(0, len(self._time_grid) - 1, 100, dtype=int)

        for idx in sample_idx:
            t = self._time_grid[idx]
            r_t = self._rate_paths[:, idx]
            record = {
                "Time (Y)": round(t, 4),
                "Mean Rate (%)": round(r_t.mean() * 100, 4),
                "Std Rate (%)": round(r_t.std() * 100, 4),
                "Min Rate (%)": round(r_t.min() * 100, 4),
                "Max Rate (%)": round(r_t.max() * 100, 4),
            }
            for p in percentiles:
                record[f"P{p} (%)"] = round(np.percentile(r_t, p) * 100, 4)
            records.append(record)

        return pd.DataFrame(records)

    def compute_term_structure_paths(
        self,
        bond_maturities: List[float],
        n_sample_paths: int = 20
    ) -> Dict[float, np.ndarray]:
        """
        Compute the evolution of the yield curve along sampled paths.

        For each sampled path, compute the zero yield at each monitoring time
        for each bond maturity using the HW ZCB formula.

        Parameters
        ----------
        bond_maturities : list of float
            Bond maturities to track.
        n_sample_paths : int
            Number of paths to use (subset for performance).

        Returns
        -------
        dict
            {maturity: array of shape (n_sample_paths, n_time_steps)}
        """
        self._check_simulated()

        selected_paths = self._rate_paths[:n_sample_paths, :]
        n_times = len(self._time_grid)

        yield_paths = {m: np.zeros((n_sample_paths, n_times)) for m in bond_maturities}

        for path_idx in range(n_sample_paths):
            for time_idx, t in enumerate(self._time_grid):
                r_t = selected_paths[path_idx, time_idx]
                for m in bond_maturities:
                    T_bond = t + m
                    y = self.model.zcb_yield(r_t, t, T_bond)
                    yield_paths[m][path_idx, time_idx] = y

        return yield_paths
