"""
=============================================================================
Hull-White Model Calibration Engine
=============================================================================
Module  : src/models/calibration.py
Purpose : Calibrates Hull-White parameters (a, σ) to market data using
          numerical optimization. Supports calibration to:
          1. Swaption implied volatility surface
          2. Cap/floor implied vol surface
          3. Bond price time series
          4. Direct yield curve fitting
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================

Calibration Methodology:
------------------------
The Hull-White model exactly fits any initial yield curve (via θ(t)).
The two free parameters (a, σ) are calibrated to fit:
  - Market swaption prices (or implied vols)
  - Market cap prices (or implied vols)

Objective Function:
  min_{a,σ} ∑_i w_i · (model_vol_i - market_vol_i)²

Constraints:
  - a > 0 (mean reversion must be positive)
  - σ > 0 (volatility must be positive)
  - 0.001 ≤ a ≤ 1.0 (practical bounds)
  - 0.0001 ≤ σ ≤ 0.20 (practical bounds)

Optimization:
  - Primary: L-BFGS-B with analytical gradient (fast, robust)
  - Fallback: Nelder-Mead (derivative-free, slower but more robust)
  - Global: Differential Evolution (for multi-modal landscapes)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Optional, Dict, List, Tuple, Callable
import warnings
import logging
from tqdm import tqdm
import time

from .hull_white import HullWhiteModel

logger = logging.getLogger(__name__)

# ── Default parameter bounds ──────────────────────────────────────────────────
PARAM_BOUNDS = {
    "a": (1e-4, 1.0),       # Mean reversion: 0.01% to 100% annualized
    "sigma": (1e-5, 0.20),  # Volatility: 0.001% to 20% annualized
}


class HullWhiteCalibrator:
    """
    Calibrates Hull-White one-factor model to market instrument prices/vols.

    Supports multiple calibration targets and optimization methods.

    Parameters
    ----------
    yield_curve : YieldCurveBuilder
        Initial zero-coupon yield curve (used for θ(t) fitting).
    r0 : float
        Current short rate (0-date value). Typically f(0,0) from the curve.
    calibration_target : str
        What to calibrate to: 'swaption_vols', 'cap_prices', or 'bond_prices'.

    Examples
    --------
    >>> calibrator = HullWhiteCalibrator(yield_curve=curve, r0=0.04)
    >>> calibrator.set_swaption_targets(vol_surface_df, expiries, tenors)
    >>> result = calibrator.calibrate(method='L-BFGS-B')
    >>> print(f"Calibrated: a={result.a:.4f}, σ={result.sigma:.4f}")
    """

    def __init__(
        self,
        yield_curve,
        r0: Optional[float] = None,
        calibration_target: str = "swaption_vols"
    ):
        self.curve = yield_curve
        self.r0 = r0 if r0 is not None else yield_curve.forward_rate(1e-4)
        self.calibration_target = calibration_target

        # Calibration data (set via setter methods)
        self._market_expiries: Optional[np.ndarray] = None
        self._market_tenors: Optional[np.ndarray] = None
        self._market_vols: Optional[np.ndarray] = None
        self._market_prices: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

        # Calibration results
        self._calibration_result: Optional[Dict] = None
        self._calibrated_model: Optional[HullWhiteModel] = None

        # Optimization history
        self._loss_history: List[float] = []
        self._param_history: List[Tuple[float, float]] = []

        logger.info(
            f"🔧 HW Calibrator initialized: target='{calibration_target}', "
            f"r0={self.r0*100:.3f}%"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Target Setting
    # ─────────────────────────────────────────────────────────────────────────

    def set_swaption_targets(
        self,
        vol_surface: pd.DataFrame,
        weights: Optional[np.ndarray] = None
    ) -> "HullWhiteCalibrator":
        """
        Set swaption implied vol surface as calibration target.

        Parameters
        ----------
        vol_surface : pd.DataFrame
            Normal implied vols in bps. Index = option expiry labels (e.g., '1Y'),
            Columns = swap tenor labels (e.g., '5Y').
        weights : np.ndarray, optional
            Relative weights for each vol point. Default: uniform.

        Returns
        -------
        self : enables method chaining
        """
        self._vol_surface_df = vol_surface.copy()

        # Parse expiry and tenor labels (e.g., '1Y' → 1.0, '6M' → 0.5)
        def parse_tenor(s: str) -> float:
            s = str(s).strip()
            if s.endswith("M"):
                return float(s[:-1]) / 12.0
            elif s.endswith("Y"):
                return float(s[:-1])
            else:
                return float(s)

        expiries = np.array([parse_tenor(e) for e in vol_surface.index])
        tenors = np.array([parse_tenor(t) for t in vol_surface.columns])

        # Flatten to 1D arrays for optimization
        self._market_expiries = []
        self._market_tenors = []
        self._market_vols = []

        for i, exp in enumerate(expiries):
            for j, ten in enumerate(tenors):
                vol_bps = vol_surface.iloc[i, j]
                if not np.isnan(vol_bps) and vol_bps > 0:
                    self._market_expiries.append(exp)
                    self._market_tenors.append(ten)
                    self._market_vols.append(vol_bps / 10_000.0)  # bps → decimal

        self._market_expiries = np.array(self._market_expiries)
        self._market_tenors = np.array(self._market_tenors)
        self._market_vols = np.array(self._market_vols)

        n = len(self._market_vols)
        self._weights = weights if weights is not None else np.ones(n) / n

        logger.info(
            f"📐 Swaption targets set: {n} vol points from "
            f"{len(expiries)} expiries × {len(tenors)} tenors"
        )
        return self

    def set_cap_targets(
        self,
        cap_prices: np.ndarray,
        maturities: np.ndarray,
        strikes: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> "HullWhiteCalibrator":
        """
        Set cap prices as calibration target.

        Parameters
        ----------
        cap_prices : np.ndarray
            Market cap prices (normalized, e.g., 0.005 = 50 bps upfront per notional).
        maturities : np.ndarray
            Cap maturities in years.
        strikes : np.ndarray
            Cap strike rates in decimal.
        weights : np.ndarray, optional
            Relative weights. Default: uniform.
        """
        self._market_prices = np.asarray(cap_prices)
        self._market_expiries = np.asarray(maturities)
        self._cap_strikes = np.asarray(strikes)
        n = len(cap_prices)
        self._weights = weights if weights is not None else np.ones(n) / n
        self.calibration_target = "cap_prices"
        logger.info(f"📋 Cap price targets set: {n} caps")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Objective Functions
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_model_swaption_normal_vol(
        self,
        model: HullWhiteModel,
        T_option: float,
        T_swap: float,
        swap_tenor: float = 0.5
    ) -> float:
        """
        Compute normal implied vol of a payer swaption from the HW model.

        Uses the analytical HW swaption formula, then inverts Bachelier's
        normal model formula to extract implied normal vol.

        σ_normal_impl = swaption_price / (A · Φ(h) · T^0.5)
        where A is the annuity and h is the moneyness.
        """
        # Par swap rate (at-the-money)
        K = model.model_par_rate(self.r0, T_swap, swap_tenor)

        # HW swaption price (ATM)
        swaption_pv = model.swaption_price(
            r=self.r0,
            t=0.0,
            T_option=T_option,
            T_swap=T_swap,
            K=K,
            swap_tenor=swap_tenor,
            payer_receiver="payer"
        )

        # Compute annuity A(0, T_swap)
        pay_times = np.arange(swap_tenor, T_swap + 1e-9, swap_tenor)
        annuity = sum(swap_tenor * model.zcb_price(self.r0, 0, T) for T in pay_times)

        if annuity < 1e-12 or T_option < 1e-6:
            return 0.0

        # Bachelier ATM vol: σ_N = price / (annuity * sqrt(T/2π))
        from scipy.stats import norm as _norm
        sigma_N = swaption_pv / (annuity * np.sqrt(T_option / (2 * np.pi)))

        return sigma_N

    def _swaption_vol_objective(self, params: np.ndarray) -> float:
        """
        Weighted RMSE loss between model and market normal swaption vols.

        Parameters
        ----------
        params : np.ndarray
            [log(a), log(sigma)] in log-space for unconstrained optimization.
        """
        try:
            # Exponentiate to enforce positivity constraints
            a = np.exp(params[0])
            sigma = np.exp(params[1])

            # Clip to practical bounds
            a = np.clip(a, PARAM_BOUNDS["a"][0], PARAM_BOUNDS["a"][1])
            sigma = np.clip(sigma, PARAM_BOUNDS["sigma"][0], PARAM_BOUNDS["sigma"][1])

            model = HullWhiteModel(a, sigma, self.curve)

            total_loss = 0.0
            for i, (exp, ten, mkt_vol, w) in enumerate(zip(
                self._market_expiries,
                self._market_tenors,
                self._market_vols,
                self._weights
            )):
                T_swap = exp + ten
                model_vol = self._compute_model_swaption_normal_vol(model, exp, T_swap)

                # Relative error weighted by liquidity (weight)
                error = (model_vol - mkt_vol) ** 2
                total_loss += w * error

            # Track history
            self._loss_history.append(total_loss)
            self._param_history.append((a, sigma))

            return total_loss

        except Exception as e:
            logger.debug(f"Objective evaluation failed: {e}")
            return 1e10  # Return large penalty on failure

    def _cap_price_objective(self, params: np.ndarray) -> float:
        """
        Weighted RMSE loss between model and market cap prices.
        """
        try:
            a = np.exp(params[0])
            sigma = np.exp(params[1])
            a = np.clip(a, *PARAM_BOUNDS["a"])
            sigma = np.clip(sigma, *PARAM_BOUNDS["sigma"])

            model = HullWhiteModel(a, sigma, self.curve)

            total_loss = 0.0
            for i, (T, K, mkt_price, w) in enumerate(zip(
                self._market_expiries,
                self._cap_strikes,
                self._market_prices,
                self._weights
            )):
                model_price = model.cap_price(self.r0, 0.0, T, K)
                error = (model_price - mkt_price) ** 2
                total_loss += w * error

            self._loss_history.append(total_loss)
            self._param_history.append((a, sigma))
            return total_loss

        except Exception as e:
            return 1e10

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Routine
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(
        self,
        a0: float = 0.10,
        sigma0: float = 0.015,
        method: str = "L-BFGS-B",
        n_restarts: int = 5,
        verbose: bool = True
    ) -> HullWhiteModel:
        """
        Calibrate Hull-White parameters to market data.

        Uses log-space parameterization [log(a), log(σ)] to enforce positivity.
        Runs multiple restarts and returns the best result.

        Parameters
        ----------
        a0 : float
            Initial guess for mean reversion speed.
        sigma0 : float
            Initial guess for volatility.
        method : str
            Optimization method: 'L-BFGS-B', 'Nelder-Mead', or 'differential_evolution'.
        n_restarts : int
            Number of random restarts for robustness.
        verbose : bool
            Print calibration progress.

        Returns
        -------
        HullWhiteModel
            Calibrated model with optimal (a, σ).
        """
        if self._market_vols is None and self._market_prices is None:
            raise RuntimeError(
                "No calibration targets set. Call set_swaption_targets() or "
                "set_cap_targets() first."
            )

        # Select objective function
        if self.calibration_target == "swaption_vols":
            objective = self._swaption_vol_objective
        else:
            objective = self._cap_price_objective

        if verbose:
            print("\n" + "=" * 65)
            print("  🔧 Hull-White Calibration Engine")
            print("=" * 65)
            print(f"  Method   : {method}")
            print(f"  Restarts : {n_restarts}")
            print(f"  Target   : {self.calibration_target}")
            print(f"  Points   : {len(self._market_vols or self._market_prices)}")
            print(f"  Initial  : a={a0:.4f}, σ={sigma0:.4f}")
            print("=" * 65)

        # Log-space initial parameters
        x0_log = np.array([np.log(a0), np.log(sigma0)])

        best_loss = np.inf
        best_params = x0_log.copy()

        start_time = time.time()

        # ── Multi-start optimization ──────────────────────────────────────────
        for i in range(n_restarts):
            if i == 0:
                x0_i = x0_log.copy()
            else:
                # Random restart in log-space
                np.random.seed(42 + i)
                x0_i = np.array([
                    np.random.uniform(np.log(0.01), np.log(0.5)),
                    np.random.uniform(np.log(0.001), np.log(0.05)),
                ])

            if verbose:
                print(
                    f"  Restart {i+1}/{n_restarts}: "
                    f"a={np.exp(x0_i[0]):.4f}, σ={np.exp(x0_i[1]):.6f} ",
                    end=""
                )

            try:
                if method in ["L-BFGS-B", "Nelder-Mead", "SLSQP"]:
                    result = minimize(
                        objective,
                        x0_i,
                        method=method,
                        options={
                            "maxiter": 5000,
                            "ftol": 1e-12,
                            "gtol": 1e-8,
                        }
                    )
                    loss = result.fun
                    params = result.x
                elif method == "differential_evolution":
                    bounds_log = [
                        (np.log(PARAM_BOUNDS["a"][0]), np.log(PARAM_BOUNDS["a"][1])),
                        (np.log(PARAM_BOUNDS["sigma"][0]), np.log(PARAM_BOUNDS["sigma"][1])),
                    ]
                    result = differential_evolution(
                        objective, bounds_log,
                        seed=42, maxiter=2000, tol=1e-9,
                        popsize=15, mutation=(0.5, 1.5), recombination=0.7
                    )
                    loss = result.fun
                    params = result.x
                    i = n_restarts  # DE is global, skip restarts
                else:
                    raise ValueError(f"Unknown method: {method}")

                if verbose:
                    print(f"→ Loss={loss:.6e}")

                if loss < best_loss:
                    best_loss = loss
                    best_params = params.copy()

            except Exception as e:
                if verbose:
                    print(f"→ FAILED ({e})")
                continue

        # ── Extract optimal parameters ────────────────────────────────────────
        a_opt = float(np.clip(np.exp(best_params[0]), *PARAM_BOUNDS["a"]))
        sigma_opt = float(np.clip(np.exp(best_params[1]), *PARAM_BOUNDS["sigma"]))

        elapsed = time.time() - start_time

        # ── Compute calibration quality metrics ───────────────────────────────
        self._calibrated_model = HullWhiteModel(a_opt, sigma_opt, self.curve)
        metrics = self._compute_calibration_metrics(self._calibrated_model)

        # Store results
        self._calibration_result = {
            "a": a_opt,
            "sigma": sigma_opt,
            "final_loss": best_loss,
            "rmse_bps": metrics["rmse_bps"],
            "max_error_bps": metrics["max_error_bps"],
            "elapsed_seconds": elapsed,
            "n_evaluations": len(self._loss_history),
            "method": method,
        }

        if verbose:
            print("\n" + "─" * 65)
            print("  ✅ CALIBRATION COMPLETE")
            print("─" * 65)
            print(f"  a (mean reversion) : {a_opt:.6f} ({1/a_opt:.1f}Y half-life)")
            print(f"  σ (volatility)     : {sigma_opt:.6f} ({sigma_opt*100:.3f}%)")
            print(f"  RMSE               : {metrics['rmse_bps']:.2f} bps")
            print(f"  Max Error          : {metrics['max_error_bps']:.2f} bps")
            print(f"  Loss               : {best_loss:.6e}")
            print(f"  Time               : {elapsed:.1f}s")
            print("─" * 65 + "\n")

        logger.info(
            f"✅ Calibrated: a={a_opt:.6f}, σ={sigma_opt:.6f}, "
            f"RMSE={metrics['rmse_bps']:.2f} bps"
        )

        return self._calibrated_model

    def _compute_calibration_metrics(self, model: HullWhiteModel) -> Dict:
        """Compute in-sample calibration quality metrics."""
        if self._market_vols is not None:
            model_vols = []
            for exp, ten in zip(self._market_expiries, self._market_tenors):
                T_swap = exp + ten
                mv = self._compute_model_swaption_normal_vol(model, exp, T_swap)
                model_vols.append(mv)

            model_vols = np.array(model_vols)
            errors_bps = (model_vols - self._market_vols) * 10_000

        else:
            model_prices = []
            for T, K in zip(self._market_expiries, self._cap_strikes):
                mp = model.cap_price(self.r0, 0.0, T, K)
                model_prices.append(mp)
            model_prices = np.array(model_prices)
            errors_bps = (model_prices - self._market_prices) * 10_000

        return {
            "rmse_bps": float(np.sqrt(np.mean(errors_bps**2))),
            "max_error_bps": float(np.max(np.abs(errors_bps))),
            "mean_error_bps": float(np.mean(errors_bps)),
        }

    def get_calibration_report(self) -> pd.DataFrame:
        """
        Generate a detailed calibration diagnostic report.

        Returns
        -------
        pd.DataFrame
            Per-instrument model vs. market comparison.
        """
        if self._calibrated_model is None:
            raise RuntimeError("Must calibrate first. Call .calibrate().")

        model = self._calibrated_model
        records = []

        if self._market_vols is not None:
            for exp, ten, mkt_vol in zip(
                self._market_expiries, self._market_tenors, self._market_vols
            ):
                T_swap = exp + ten
                model_vol = self._compute_model_swaption_normal_vol(model, exp, T_swap)
                error_bps = (model_vol - mkt_vol) * 10_000

                records.append({
                    "Expiry (Y)": round(exp, 2),
                    "Tenor (Y)": round(ten, 2),
                    "Market Vol (bps)": round(mkt_vol * 10_000, 2),
                    "Model Vol (bps)": round(model_vol * 10_000, 2),
                    "Error (bps)": round(error_bps, 2),
                    "Rel. Error (%)": round(100 * abs(error_bps) / max(mkt_vol * 10_000, 1), 2),
                })

        df = pd.DataFrame(records)
        return df

    @property
    def calibrated_model(self) -> Optional[HullWhiteModel]:
        """The calibrated HullWhiteModel, or None if not yet calibrated."""
        return self._calibrated_model

    @property
    def calibration_result(self) -> Optional[Dict]:
        """Summary dict of calibration result."""
        return self._calibration_result

    def plot_loss_history(self):
        """Return loss history for plotting."""
        return np.array(self._loss_history)

    def sensitivity_analysis(
        self,
        a_grid: Optional[np.ndarray] = None,
        sigma_grid: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute loss surface over a grid of (a, σ) values.

        Useful for visualizing the calibration landscape and diagnosing
        parameter identifiability.

        Parameters
        ----------
        a_grid : np.ndarray, optional
            Grid of 'a' values to evaluate.
        sigma_grid : np.ndarray, optional
            Grid of 'σ' values to evaluate.

        Returns
        -------
        pd.DataFrame
            Loss values indexed by (a, σ) pairs.
        """
        a_grid = a_grid if a_grid is not None else np.linspace(0.01, 0.5, 20)
        sigma_grid = sigma_grid if sigma_grid is not None else np.linspace(0.002, 0.05, 20)

        records = []
        total = len(a_grid) * len(sigma_grid)

        print(f"Computing loss surface ({total} points)...")
        for a_val in a_grid:
            for s_val in sigma_grid:
                params = np.array([np.log(a_val), np.log(s_val)])
                if self.calibration_target == "swaption_vols":
                    loss = self._swaption_vol_objective(params)
                else:
                    loss = self._cap_price_objective(params)
                records.append({"a": a_val, "sigma": s_val, "loss": loss})

        df = pd.DataFrame(records)
        return df
