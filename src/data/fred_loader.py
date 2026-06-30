"""
=============================================================================
FRED API & Market Data Loader
=============================================================================
Module  : src/data/fred_loader.py
Purpose : Download US Treasury yield curve data from FRED API and Yahoo
          Finance. Provides synthetic fallback when API keys are unavailable.
Author  : Senior Quantitative Developer
Version : 1.0.0
=============================================================================

FRED Series Used:
-----------------
  DGS1MO   → 1-Month Treasury Constant Maturity Rate
  DGS3MO   → 3-Month Treasury Constant Maturity Rate
  DGS6MO   → 6-Month Treasury Constant Maturity Rate
  DGS1      → 1-Year Treasury Constant Maturity Rate
  DGS2      → 2-Year Treasury Constant Maturity Rate
  DGS3      → 3-Year Treasury Constant Maturity Rate
  DGS5      → 5-Year Treasury Constant Maturity Rate
  DGS7      → 7-Year Treasury Constant Maturity Rate
  DGS10     → 10-Year Treasury Constant Maturity Rate
  DGS20     → 20-Year Treasury Constant Maturity Rate
  DGS30     → 30-Year Treasury Constant Maturity Rate

Usage:
------
  loader = FREDDataLoader(api_key="your_fred_api_key")
  yield_data = loader.get_treasury_yield_curve()
  rates, maturities = loader.get_latest_yield_curve()
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    warnings.warn(
        "fredapi not installed. Install with: pip install fredapi\n"
        "Falling back to synthetic yield curve data.",
        ImportWarning,
        stacklevel=2
    )

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── FRED Series Mapping ───────────────────────────────────────────────────────
FRED_SERIES: Dict[str, float] = {
    "DGS1MO":  1/12,   # 1-month maturity in years
    "DGS3MO":  0.25,
    "DGS6MO":  0.50,
    "DGS1":    1.00,
    "DGS2":    2.00,
    "DGS3":    3.00,
    "DGS5":    5.00,
    "DGS7":    7.00,
    "DGS10":  10.00,
    "DGS20":  20.00,
    "DGS30":  30.00,
}


class FREDDataLoader:
    """
    Downloads and manages US Treasury yield curve data from the FRED API.

    Parameters
    ----------
    api_key : str, optional
        FRED API key. If None, attempts to read from FRED_API_KEY env variable.
        Falls back to synthetic data if key is unavailable.
    cache_dir : str, optional
        Directory to cache downloaded data. Default: './data/raw'

    Examples
    --------
    >>> loader = FREDDataLoader(api_key="abcdef1234567890")
    >>> rates, maturities = loader.get_latest_yield_curve()
    >>> print(maturities, rates)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "./data/raw"
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.cache_dir = cache_dir
        self._fred_client: Optional[object] = None
        self._cache: Dict[str, pd.DataFrame] = {}

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize FRED client
        self._init_fred_client()

    def _init_fred_client(self) -> None:
        """Initialize FRED API client with error handling."""
        if not FRED_AVAILABLE:
            logger.warning("fredapi package not available. Using synthetic data.")
            return

        if not self.api_key:
            logger.warning(
                "No FRED API key provided. Get a free key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "Using synthetic yield curve data as fallback."
            )
            return

        try:
            self._fred_client = Fred(api_key=self.api_key)
            # Quick validation: try to fetch a single observation
            test = self._fred_client.get_series("DGS10", limit=1)
            if test is not None and len(test) > 0:
                logger.info("✅ FRED API connection established successfully.")
            else:
                logger.warning("FRED API returned empty data. Check API key validity.")
                self._fred_client = None
        except Exception as e:
            logger.warning(
                f"FRED API initialization failed: {e}\n"
                "Using synthetic yield curve data as fallback."
            )
            self._fred_client = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public Methods
    # ─────────────────────────────────────────────────────────────────────────

    def get_treasury_yield_curve(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve historical US Treasury yield curve data.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str, optional
            End date. Defaults to today.
        use_cache : bool
            Whether to use cached data if available.

        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and maturity tenor columns.
            Rates are in percentage (e.g., 4.5 = 4.5%).
        """
        end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        cache_key = f"treasury_{start_date}_{end_date}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")

        # ── Try disk cache first ──────────────────────────────────────────────
        if use_cache and os.path.exists(cache_file):
            logger.info(f"📂 Loading yield curve from cache: {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        # ── Download from FRED ────────────────────────────────────────────────
        if self._fred_client is not None:
            logger.info(f"📡 Downloading yield curve from FRED ({start_date} → {end_date})...")
            df = self._download_from_fred(start_date, end_date)
        else:
            logger.info("🔄 Generating synthetic yield curve data...")
            df = self._generate_synthetic_curve(start_date, end_date)

        # ── Cache to disk ─────────────────────────────────────────────────────
        df.to_csv(cache_file)
        logger.info(f"💾 Yield curve data cached: {cache_file} ({len(df)} rows)")

        return df

    def get_latest_yield_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the most recent yield curve as arrays.

        Returns
        -------
        rates : np.ndarray
            Par rates in decimal (e.g., 0.045 = 4.5%)
        maturities : np.ndarray
            Maturity tenors in years [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        """
        df = self.get_treasury_yield_curve(
            start_date=(datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        )

        # Get last available row (drop NaN columns)
        latest = df.dropna(how="all").iloc[-1].dropna()

        maturities = np.array(list(FRED_SERIES.values()))
        # Align rates with our maturity grid
        rates = np.array([
            latest.get(f"{t:.4f}y", np.nan)
            for t in maturities
        ])

        # Fill any remaining NaN via interpolation
        valid_mask = ~np.isnan(rates)
        if valid_mask.sum() >= 2:
            rates = np.interp(
                maturities,
                maturities[valid_mask],
                rates[valid_mask]
            )

        # Convert from percentage to decimal
        rates = rates / 100.0

        logger.info(
            f"📈 Latest yield curve loaded: "
            f"{rates[0]*100:.2f}% (1M) → {rates[-1]*100:.2f}% (30Y)"
        )
        return rates, maturities

    def get_historical_short_rate(
        self,
        start_date: str = "2000-01-01"
    ) -> pd.Series:
        """
        Retrieve the historical US short-rate (Fed Funds Rate proxy: DGS3MO).

        Returns
        -------
        pd.Series
            Daily 3-month T-Bill rate in decimal form.
        """
        df = self.get_treasury_yield_curve(start_date=start_date)

        if "0.2500y" in df.columns:
            series = df["0.2500y"].dropna() / 100.0
        else:
            # Fallback: use synthetic short rate
            dates = pd.date_range(start_date, periods=500, freq="B")
            np.random.seed(42)
            r0 = 0.03
            series_data = [r0]
            for _ in range(499):
                r0 = max(0.001, r0 + np.random.normal(0, 0.001))
                series_data.append(r0)
            series = pd.Series(series_data, index=dates)

        logger.info(f"📊 Short rate series loaded: {len(series)} observations")
        return series

    # ─────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _download_from_fred(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download all treasury series from FRED and merge into a single DataFrame."""
        series_dfs: List[pd.Series] = []

        for series_id, maturity in FRED_SERIES.items():
            try:
                s = self._fred_client.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                s.name = f"{maturity:.4f}y"
                series_dfs.append(s)
                logger.debug(f"  ✓ {series_id} ({maturity}Y) — {len(s)} obs")
            except Exception as e:
                logger.warning(f"  ✗ Failed to download {series_id}: {e}")

        if not series_dfs:
            logger.error("All FRED downloads failed. Using synthetic data.")
            return self._generate_synthetic_curve(start_date, end_date)

        df = pd.concat(series_dfs, axis=1)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Forward-fill up to 5 business days (weekends / holidays)
        df.ffill(limit=5, inplace=True)

        return df

    def _generate_synthetic_curve(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic US Treasury yield curve data.

        Uses a Nelson-Siegel parametrized yield curve with calibrated parameters
        approximating the 2020-2024 US rate environment.
        """
        logger.info("🧪 Generating synthetic Nelson-Siegel yield curve...")

        dates = pd.bdate_range(start=start_date, end=end_date)
        maturities = list(FRED_SERIES.values())

        np.random.seed(42)
        n_dates = len(dates)

        # Nelson-Siegel base parameters (calibrated to approximate real levels)
        # β0 = level, β1 = slope, β2 = curvature, λ = decay factor
        beta0_base = 4.5   # Level (long-run rate %)
        beta1_base = -1.5  # Slope (spread: short vs long)
        beta2_base = 2.0   # Curvature (hump at medium maturities)
        lam = 1.5          # Decay factor

        records = []
        # Simulate evolution of NS parameters via mean-reverting processes
        beta0 = beta0_base
        beta1 = beta1_base
        beta2 = beta2_base

        for i in range(n_dates):
            # AR(1) dynamics with slight drift to simulate rate cycles
            beta0 += 0.001 * (beta0_base - beta0) + np.random.normal(0, 0.04)
            beta1 += 0.002 * (beta1_base - beta1) + np.random.normal(0, 0.03)
            beta2 += 0.003 * (beta2_base - beta2) + np.random.normal(0, 0.05)

            # Nelson-Siegel forward curve
            row = {}
            for mat in maturities:
                tau = mat / lam
                if tau < 1e-8:
                    tau = 1e-8
                ns_factor1 = (1 - np.exp(-tau)) / tau
                ns_factor2 = ns_factor1 - np.exp(-tau)
                rate = beta0 + beta1 * ns_factor1 + beta2 * ns_factor2
                rate = max(0.01, rate)  # Floor at 1 bp to avoid negative rates
                row[f"{mat:.4f}y"] = round(rate, 4)

            records.append(row)

        df = pd.DataFrame(records, index=dates)
        logger.info(
            f"✅ Synthetic curve generated: {len(df)} days × {len(maturities)} tenors"
        )
        return df

    def get_swaption_vol_surface(self) -> pd.DataFrame:
        """
        Return a representative swaption implied volatility surface.

        In production, this would be sourced from Bloomberg / Refinitiv.
        Here we provide a realistic calibrated surface in basis points (lognormal vol).

        Returns
        -------
        pd.DataFrame
            Indexed by option expiry (rows) × swap tenor (columns).
            Values are implied vol in percentage (e.g., 80 = 80 bps normal vol).
        """
        # Option expiries (years)
        expiries = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
        # Swap tenors (years)
        tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]

        # Calibrated normal implied vols (bps) from 2024 market data approximation
        # Pattern: higher vol for shorter expiries, hump in mid-tenors
        base_vols = np.array([
            [90, 95, 100, 110, 115, 120, 118, 115, 112],   # 3M expiry
            [85, 90,  95, 105, 110, 115, 113, 110, 108],   # 6M expiry
            [80, 85,  90,  98, 102, 108, 106, 103, 100],   # 1Y expiry
            [75, 80,  85,  92,  96, 100,  98,  96,  93],   # 2Y expiry
            [70, 74,  78,  85,  89,  93,  91,  89,  86],   # 3Y expiry
            [65, 68,  72,  78,  82,  86,  84,  82,  79],   # 5Y expiry
            [60, 63,  66,  72,  76,  80,  78,  76,  73],   # 7Y expiry
            [55, 58,  61,  67,  71,  75,  73,  71,  68],   # 10Y expiry
        ], dtype=float)

        df = pd.DataFrame(
            base_vols,
            index=[f"{e}Y" for e in expiries],
            columns=[f"{t}Y" for t in tenors]
        )
        df.index.name = "Option Expiry"
        df.columns.name = "Swap Tenor"

        logger.info(
            f"📐 Swaption vol surface loaded: "
            f"{len(expiries)} expiries × {len(tenors)} tenors"
        )
        return df

    def summary(self) -> None:
        """Print a summary of data availability."""
        print("\n" + "=" * 65)
        print("  FRED Data Loader — Configuration Summary")
        print("=" * 65)
        print(f"  API Key      : {'✅ Configured' if self.api_key else '❌ Not set'}")
        print(f"  FRED Client  : {'✅ Connected' if self._fred_client else '⚠️  Using synthetic'}")
        print(f"  Cache Dir    : {self.cache_dir}")
        print(f"  fredapi pkg  : {'✅ Installed' if FRED_AVAILABLE else '❌ Not installed'}")
        print(f"  yfinance pkg : {'✅ Installed' if YFINANCE_AVAILABLE else '❌ Not installed'}")
        print("=" * 65)
        print("  Series Available:")
        for sid, mat in FRED_SERIES.items():
            print(f"    {sid:<10} → {mat:.4f}Y maturity")
        print("=" * 65 + "\n")
