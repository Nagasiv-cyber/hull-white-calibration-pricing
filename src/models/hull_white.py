"""
=============================================================================
Hull-White One-Factor Short-Rate Model — Core Implementation
=============================================================================
Module  : src/models/hull_white.py
Purpose : Implements the Hull-White (1990) one-factor short-rate model with
          full analytical pricing formulas for zero-coupon bonds, bond
          options, caps, floors, and swaptions.
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================

Model Specification (Hull-White 1990):
---------------------------------------
  dr(t) = [θ(t) - a·r(t)] dt + σ dW(t)

  where:
    r(t)   = instantaneous short rate at time t
    a      = mean reversion speed (a > 0)
    σ      = short rate volatility (σ > 0)
    θ(t)   = time-dependent drift, chosen to exactly fit the initial term
             structure (this is the key calibration condition)
    dW(t)  = Brownian motion increment under risk-neutral measure Q

Key Analytical Results:
-----------------------
  1. θ(t) = f_M(0,t) + (σ²/2a²)(1 - e^{-2at})  + a·f_M(0,t)
     where f_M(0,t) is the market instantaneous forward rate.

  2. Zero-Coupon Bond Price:
     P^{HW}(t,T) = A(t,T) · exp(-B(t,T) · r(t))

     B(t,T) = (1 - e^{-a(T-t)}) / a
     A(t,T) = P_M(0,T)/P_M(0,t) · exp(B(t,T)·f_M(0,t) - σ²/(4a)·(1-e^{-2at})·B(t,T)²)

  3. Bond Option Price (analytical Black-like formula)
  4. Cap/Floor Pricing via portfolio of bond options (caplets/floorlets)
  5. Swaption via Jamshidian decomposition

References:
-----------
  - Hull & White (1990), "Pricing Interest-Rate Derivative Securities",
    The Review of Financial Studies, 3(4), 573-592.
  - Brigo & Mercurio (2006), "Interest Rate Models — Theory and Practice",
    Springer Finance.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional, Union, Tuple, Callable
import warnings
import logging

logger = logging.getLogger(__name__)


class HullWhiteModel:
    """
    Hull-White One-Factor Short-Rate Model.

    Provides analytical pricing of:
    - Zero-coupon bonds (ZCB)
    - Coupon-bearing bonds
    - Caplets and floorlets
    - Caps and floors
    - Swaptions (via Jamshidian decomposition)
    - Bond call/put options

    Parameters
    ----------
    a : float
        Mean reversion speed. Typical range: [0.01, 0.30].
    sigma : float
        Short rate volatility. Typical range: [0.005, 0.03].
    yield_curve : YieldCurveBuilder
        Bootstrapped initial yield curve object used to compute P_M(0,T),
        f_M(0,T), and r_M(0).

    Examples
    --------
    >>> model = HullWhiteModel(a=0.10, sigma=0.015, yield_curve=curve)
    >>> P = model.zcb_price(r0=0.03, t=0, T=5.0)
    >>> cap_price = model.cap_price(r0=0.03, K=0.04, T=5.0, tenor=0.5)
    """

    def __init__(
        self,
        a: float,
        sigma: float,
        yield_curve  # YieldCurveBuilder instance
    ):
        # Parameter validation
        if a <= 0:
            raise ValueError(f"Mean reversion 'a' must be positive. Got: {a}")
        if sigma <= 0:
            raise ValueError(f"Volatility 'sigma' must be positive. Got: {sigma}")

        self.a = float(a)
        self.sigma = float(sigma)
        self.curve = yield_curve

        logger.info(
            f"🏛️  Hull-White model initialized: a={a:.6f}, σ={sigma:.6f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core Analytical Functions
    # ─────────────────────────────────────────────────────────────────────────

    def B(self, t: float, T: float) -> float:
        """
        Compute B(t, T) coefficient in the affine ZCB formula.

        B(t, T) = (1 - e^{-a(T-t)}) / a

        This represents the sensitivity of the bond price to the short rate.
        As a → 0, B(t,T) → T - t (Vasicek limit).

        Parameters
        ----------
        t : float
            Current time (years).
        T : float
            Bond maturity (years). Must be ≥ t.

        Returns
        -------
        float
            B coefficient.
        """
        tau = T - t
        if tau < 1e-10:
            return 0.0
        if abs(self.a) < 1e-8:
            # Limiting case a → 0: B → tau (Ho-Lee model)
            return tau
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def A(self, t: float, T: float) -> float:
        """
        Compute A(t, T) coefficient in the affine ZCB formula.

        A(t, T) = P_M(0,T)/P_M(0,t) · exp[B(t,T)·f_M(0,t) - (σ²/4a)(1-e^{-2at})·B(t,T)²]

        This ensures the model exactly fits the initial yield curve (HW calibration).

        Parameters
        ----------
        t : float
            Current time (years).
        T : float
            Bond maturity (years). Must be ≥ t.

        Returns
        -------
        float
            A coefficient (positive).
        """
        tau = T - t
        if tau < 1e-10:
            return 1.0

        # Market discount factors and forward rates
        P_M_T = self.curve.discount_factor(T)
        P_M_t = self.curve.discount_factor(t) if t > 1e-10 else 1.0
        f_M_t = self.curve.forward_rate(t) if t > 1e-10 else self.curve.forward_rate(1e-4)

        B_tT = self.B(t, T)

        # Variance term: (σ²/4a)(1 - e^{-2at}) · B(t,T)²
        if t < 1e-10:
            variance_term = 0.0
        else:
            if abs(self.a) < 1e-8:
                variance_term = 0.5 * self.sigma**2 * t * B_tT**2
            else:
                variance_term = (self.sigma**2 / (4.0 * self.a)) * (
                    1.0 - np.exp(-2.0 * self.a * t)
                ) * B_tT**2

        log_A = (
            np.log(P_M_T / P_M_t)
            + B_tT * f_M_t
            - variance_term
        )

        return np.exp(log_A)

    def zcb_price(
        self,
        r: float,
        t: float,
        T: float
    ) -> float:
        """
        Price of a zero-coupon bond at time t with short rate r(t) = r.

        P^{HW}(t, T) = A(t, T) · exp(-B(t, T) · r)

        Parameters
        ----------
        r : float
            Current short rate at time t (decimal).
        t : float
            Current time (years).
        T : float
            Bond maturity (years).

        Returns
        -------
        float
            ZCB price (between 0 and 1 for standard rates).
        """
        if T <= t:
            return 1.0 if abs(T - t) < 1e-10 else 0.0

        A_tT = self.A(t, T)
        B_tT = self.B(t, T)

        price = A_tT * np.exp(-B_tT * r)
        return max(price, 0.0)

    def zcb_yield(self, r: float, t: float, T: float) -> float:
        """
        Continuously compounded zero yield implied by HW ZCB price.

        y(t, T) = -ln(P(t,T)) / (T-t)
        """
        P = self.zcb_price(r, t, T)
        tau = T - t
        if tau < 1e-10 or P <= 0:
            return r
        return -np.log(P) / tau

    def coupon_bond_price(
        self,
        r: float,
        t: float,
        coupon_times: np.ndarray,
        coupon_amounts: np.ndarray
    ) -> float:
        """
        Price of a coupon-bearing bond as a sum of ZCB prices.

        P_coupon(t, r) = ∑_i c_i · P(t, T_i)

        Parameters
        ----------
        r : float
            Current short rate.
        t : float
            Current time.
        coupon_times : np.ndarray
            Cash flow payment times T_1, T_2, ..., T_n (years).
        coupon_amounts : np.ndarray
            Cash flow amounts (coupon + principal at maturity).

        Returns
        -------
        float
            Coupon bond price.
        """
        if len(coupon_times) != len(coupon_amounts):
            raise ValueError("coupon_times and coupon_amounts must have the same length.")

        price = sum(
            amt * self.zcb_price(r, t, T)
            for T, amt in zip(coupon_times, coupon_amounts)
            if T > t
        )
        return price

    # ─────────────────────────────────────────────────────────────────────────
    # Bond Option Pricing
    # ─────────────────────────────────────────────────────────────────────────

    def bond_option_price(
        self,
        r: float,
        t: float,
        T_option: float,
        T_bond: float,
        K: float,
        option_type: str = "call"
    ) -> float:
        """
        Analytical price of a European option on a zero-coupon bond.

        Uses Jamshidian's (1989) closed-form formula for affine models.

        Price_call = P(t,T_B)·N(d₊) - K·P(t,T_O)·N(d₋)
        Price_put  = K·P(t,T_O)·N(-d₋) - P(t,T_B)·N(-d₊)

        where:
          d₊ = [ln(P(t,T_B) / (K·P(t,T_O))) + σ_P²/2] / σ_P
          d₋ = d₊ - σ_P
          σ_P = σ · B(T_O, T_B) · sqrt[(1 - e^{-2a(T_O-t)}) / (2a)]

        Parameters
        ----------
        r : float
            Current short rate.
        t : float
            Current time.
        T_option : float
            Option expiry (years).
        T_bond : float
            Underlying bond maturity (years). Must be > T_option.
        K : float
            Strike price (as fraction of par, e.g., 0.85).
        option_type : str
            'call' or 'put'.

        Returns
        -------
        float
            Option price.
        """
        if T_bond <= T_option:
            raise ValueError("Bond maturity T_bond must exceed option expiry T_option.")

        # Bond and discount prices
        P_t_TO = self.zcb_price(r, t, T_option)
        P_t_TB = self.zcb_price(r, t, T_bond)

        if P_t_TO <= 0 or P_t_TB <= 0:
            return 0.0

        # Option volatility σ_P
        tau_option = T_option - t
        B_TO_TB = self.B(T_option, T_bond)

        if tau_option < 1e-10:
            # Option is expiring
            intrinsic = max(P_t_TB - K * P_t_TO, 0.0)
            return intrinsic

        if abs(self.a) < 1e-8:
            sigma_P = self.sigma * B_TO_TB * np.sqrt(tau_option)
        else:
            variance = (self.sigma**2 / (2.0 * self.a)) * (
                1.0 - np.exp(-2.0 * self.a * tau_option)
            )
            sigma_P = B_TO_TB * np.sqrt(variance)

        if sigma_P < 1e-10:
            # Near-zero vol: use intrinsic value
            intrinsic = max(P_t_TB - K * P_t_TO, 0.0)
            return intrinsic

        # Black-like formula
        d_plus = (np.log(P_t_TB / (K * P_t_TO)) + 0.5 * sigma_P**2) / sigma_P
        d_minus = d_plus - sigma_P

        if option_type.lower() == "call":
            price = P_t_TB * norm.cdf(d_plus) - K * P_t_TO * norm.cdf(d_minus)
        elif option_type.lower() == "put":
            price = K * P_t_TO * norm.cdf(-d_minus) - P_t_TB * norm.cdf(-d_plus)
        else:
            raise ValueError(f"option_type must be 'call' or 'put'. Got: {option_type}")

        return max(price, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Cap / Floor Pricing
    # ─────────────────────────────────────────────────────────────────────────

    def caplet_price(
        self,
        r: float,
        t: float,
        T_start: float,
        T_end: float,
        K: float,
        notional: float = 1.0
    ) -> float:
        """
        Price of a single caplet using the ZCB option formula.

        A caplet pays: N · τ · max(L(T_start, T_end) - K, 0)  at T_end

        This is equivalent to: N · (1 + τK) · put on ZCB with:
          - Expiry:  T_start
          - Maturity: T_end
          - Strike:   1 / (1 + τ·K)

        Parameters
        ----------
        r : float
            Current short rate.
        t : float
            Current time.
        T_start : float
            Caplet reset date (years).
        T_end : float
            Caplet payment date (years).
        K : float
            Cap strike rate (decimal).
        notional : float
            Notional amount.

        Returns
        -------
        float
            Caplet price.
        """
        tau = T_end - T_start

        # Bond strike equivalent
        K_bond = 1.0 / (1.0 + tau * K)
        factor = notional * (1.0 + tau * K)

        # Caplet = factor × put on ZCB
        put_price = self.bond_option_price(
            r, t, T_start, T_end, K_bond, option_type="put"
        )
        return factor * put_price

    def floorlet_price(
        self,
        r: float,
        t: float,
        T_start: float,
        T_end: float,
        K: float,
        notional: float = 1.0
    ) -> float:
        """
        Price of a single floorlet (call on ZCB equivalent).

        Parameters
        ----------
        Same as caplet_price.

        Returns
        -------
        float
            Floorlet price.
        """
        tau = T_end - T_start
        K_bond = 1.0 / (1.0 + tau * K)
        factor = notional * (1.0 + tau * K)

        call_price = self.bond_option_price(
            r, t, T_start, T_end, K_bond, option_type="call"
        )
        return factor * call_price

    def cap_price(
        self,
        r: float,
        t: float,
        T_cap: float,
        K: float,
        tenor: float = 0.25,
        notional: float = 1.0
    ) -> float:
        """
        Price of an interest rate cap as a portfolio of caplets.

        A cap with expiry T_cap, strike K, reset frequency δ consists of
        caplets on LIBOR over [T_i, T_{i+1}] for T_i = t+δ, t+2δ, ..., T_cap.

        Parameters
        ----------
        r : float
            Current short rate.
        t : float
            Current time (typically 0).
        T_cap : float
            Cap expiry in years.
        K : float
            Cap strike rate (decimal).
        tenor : float
            Caplet reset period (0.25 = quarterly, 0.5 = semi-annual).
        notional : float
            Notional amount.

        Returns
        -------
        float
            Cap price.
        """
        reset_times = np.arange(t + tenor, T_cap + 1e-9, tenor)

        if len(reset_times) == 0:
            return 0.0

        cap = 0.0
        for T_start in reset_times[:-1] if len(reset_times) > 1 else []:
            T_end = T_start + tenor
            cap += self.caplet_price(r, t, T_start, T_end, K, notional)

        # Include final caplet
        if len(reset_times) >= 1:
            T_start = reset_times[-1] if len(reset_times) == 1 else reset_times[-2]
            T_end = T_start + tenor
            cap += self.caplet_price(r, t, T_start, T_end, K, notional)

        return cap

    def floor_price(
        self,
        r: float,
        t: float,
        T_floor: float,
        K: float,
        tenor: float = 0.25,
        notional: float = 1.0
    ) -> float:
        """
        Price of an interest rate floor as a portfolio of floorlets.

        Parameters
        ----------
        Same as cap_price.

        Returns
        -------
        float
            Floor price.
        """
        reset_times = np.arange(t + tenor, T_floor + 1e-9, tenor)

        if len(reset_times) == 0:
            return 0.0

        floor = 0.0
        for i in range(len(reset_times) - 1):
            T_start = reset_times[i]
            T_end = reset_times[i + 1]
            floor += self.floorlet_price(r, t, T_start, T_end, K, notional)

        return floor

    # ─────────────────────────────────────────────────────────────────────────
    # Swaption Pricing — Jamshidian Decomposition
    # ─────────────────────────────────────────────────────────────────────────

    def swaption_price(
        self,
        r: float,
        t: float,
        T_option: float,
        T_swap: float,
        K: float,
        swap_tenor: float = 0.5,
        notional: float = 1.0,
        payer_receiver: str = "payer"
    ) -> float:
        """
        Price of a European swaption via Jamshidian (1989) decomposition.

        A payer swaption gives the right to enter a pay-fixed, receive-float swap.
        Under HW, the swap is a portfolio of ZCBs, and the swaption can be
        decomposed into a portfolio of bond options.

        Jamshidian's key insight: Since the coupon bond price is monotonically
        decreasing in r, there exists a critical rate r* such that the bond's
        value equals par. Each bond option in the decomposition is struck at
        the ZCB price corresponding to r*.

        Parameters
        ----------
        r : float
            Current short rate.
        t : float
            Current time.
        T_option : float
            Swaption expiry (years from now).
        T_swap : float
            Swap maturity (years from now). Must be > T_option.
        K : float
            Fixed rate on the underlying swap (decimal).
        swap_tenor : float
            Fixed leg payment frequency (0.5 = semi-annual).
        notional : float
            Notional.
        payer_receiver : str
            'payer' (long cap-like) or 'receiver' (long floor-like).

        Returns
        -------
        float
            Swaption price.
        """
        if T_swap <= T_option:
            raise ValueError("Swap maturity must exceed swaption expiry.")

        # Build swap cash flow schedule from T_option to T_swap
        pay_times = np.arange(
            T_option + swap_tenor,
            T_swap + 1e-9,
            swap_tenor
        )
        if len(pay_times) == 0:
            logger.warning("No payment dates generated for swaption. Check parameters.")
            return 0.0

        coupon = K * swap_tenor   # Fixed coupon per period
        # Cash flows: coupon at each payment date, plus notional at T_swap
        cash_flows = np.full(len(pay_times), coupon)
        cash_flows[-1] += 1.0    # Principal repayment

        # ── Jamshidian: Find critical rate r* ────────────────────────────────
        def swap_bond_parity(r_star: float) -> float:
            """Coupon bond value at T_option equals 1 (par) at r_star."""
            bond_val = sum(
                cf * self.zcb_price(r_star, T_option, T)
                for T, cf in zip(pay_times, cash_flows)
            )
            return bond_val - 1.0

        try:
            r_star = brentq(swap_bond_parity, -0.5, 2.0, xtol=1e-8, maxiter=200)
        except ValueError:
            # Fallback: ATM approximation
            r_star = self.curve.zero_rate(T_option)
            logger.debug(
                f"Brentq failed for swaption r*; using ATM fallback: {r_star:.4f}"
            )

        # ── Strike ZCB prices at r* ───────────────────────────────────────────
        # Jamshidian: K_i = P^HW(T_option, T_i | r*)
        K_i = np.array([
            self.zcb_price(r_star, T_option, T)
            for T in pay_times
        ])

        # ── Sum of bond options ───────────────────────────────────────────────
        swaption_value = 0.0
        for i, (T_i, cf_i, k_i) in enumerate(zip(pay_times, cash_flows, K_i)):
            if payer_receiver.lower() == "payer":
                # Payer swaption = portfolio of put options on ZCBs
                opt = self.bond_option_price(r, t, T_option, T_i, k_i, "put")
            else:
                # Receiver swaption = portfolio of call options on ZCBs
                opt = self.bond_option_price(r, t, T_option, T_i, k_i, "call")

            swaption_value += cf_i * opt

        return notional * swaption_value

    # ─────────────────────────────────────────────────────────────────────────
    # Utility & Calibration Support
    # ─────────────────────────────────────────────────────────────────────────

    def conditional_mean(self, r0: float, t: float, T: float) -> float:
        """
        E^Q[r(T) | r(t) = r0] under risk-neutral measure.

        E[r(T)|r(t)] = r(t)·e^{-a(T-t)} + α(T) - α(t)·e^{-a(T-t)}

        Approximated using the initial forward rate structure.
        """
        tau = T - t
        f_T = self.curve.forward_rate(T)
        f_t = self.curve.forward_rate(t) if t > 1e-10 else self.curve.forward_rate(1e-4)

        mean = (
            r0 * np.exp(-self.a * tau)
            + f_T
            - f_t * np.exp(-self.a * tau)
            + (self.sigma**2 / (2 * self.a**2)) * (1 - np.exp(-self.a * tau))**2
        )
        return mean

    def conditional_variance(self, t: float, T: float) -> float:
        """
        Var^Q[r(T) | r(t)] under risk-neutral measure.

        Var[r(T)|r(t)] = σ² / (2a) · (1 - e^{-2a(T-t)})
        """
        tau = T - t
        if abs(self.a) < 1e-8:
            return self.sigma**2 * tau
        return (self.sigma**2 / (2.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * tau))

    def model_par_rate(
        self,
        r: float,
        T: float,
        tenor: float = 0.5
    ) -> float:
        """
        Compute the model-implied par swap rate for a swap of maturity T.

        S(0, T) = [P(0, t_0) - P(0, T)] / [τ · ∑ P(0, T_i)]

        Parameters
        ----------
        r : float
            Current short rate.
        T : float
            Swap maturity (years).
        tenor : float
            Fixed leg payment frequency.

        Returns
        -------
        float
            Par swap rate in decimal.
        """
        pay_times = np.arange(tenor, T + 1e-9, tenor)
        if len(pay_times) == 0:
            return r

        annuity = sum(tenor * self.zcb_price(r, 0.0, T_i) for T_i in pay_times)
        P_T = self.zcb_price(r, 0.0, T)
        P_0 = 1.0  # P(0,0) = 1

        if annuity < 1e-10:
            return r

        return (P_0 - P_T) / annuity

    def __repr__(self) -> str:
        return (
            f"HullWhiteModel(a={self.a:.6f}, σ={self.sigma:.6f})"
        )
