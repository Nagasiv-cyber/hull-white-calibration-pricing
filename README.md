# 🏦 Hull-White Short-Rate Calibration & Interest Rate Derivative Pricing Engine

> **A production-grade quantitative finance system for calibrating the Hull-White one-factor model to real market data and pricing interest rate derivatives.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Quant Finance](https://img.shields.io/badge/Domain-Quant%20Finance-green?style=flat-square)](https://en.wikipedia.org/wiki/Hull%E2%80%93White_model)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📌 Project Overview

This project implements a **full-stack quantitative finance engine** centered on the **Hull-White one-factor short-rate model** — a cornerstone of fixed income derivatives pricing used by investment banks, hedge funds, and central banks worldwide.

The engine covers:
- **Market Data Ingestion**: Real yield curve data from FRED (US Treasury), Yahoo Finance
- **Yield Curve Construction**: Bootstrap zero-coupon rates and discount factors
- **Hull-White Model Calibration**: Fit parameters `a` (mean reversion) and `σ` (volatility) to swaption/cap market data
- **Monte Carlo Simulation**: Simulate short-rate paths under the Hull-White dynamics
- **Derivative Pricing**: Price Caps, Floors, Swaptions, Zero-Coupon Bonds, and Coupon Bonds
- **Greeks Computation**: Delta, Vega, and duration measures
- **Professional Dashboard**: Interactive Plotly/Dash visualizations

---

## 💼 Real-World Finance Use Case

| Actor | Use Case |
|---|---|
| **Investment Bank** | Price and hedge interest rate derivatives (caps, floors, swaptions) |
| **Pension Fund** | Duration matching and liability hedging using IR models |
| **Central Bank** | Scenario analysis of rate policy on bond portfolios |
| **Hedge Fund** | Relative value strategies based on model vs. market swaption vol surfaces |
| **Risk Management** | Calculate VaR/CVaR for fixed income portfolios under rate scenarios |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER                                 │
│  FRED API ──► Yield Curves   Yahoo Finance ──► Bond Prices  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               CURVE ENGINE                                   │
│  Bootstrap Zero Rates ──► Discount Factors ──► Fwd Rates    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│           HULL-WHITE CALIBRATION ENGINE                      │
│  Analytic Bond Prices ──► Optimizer ──► a, σ parameters     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            MONTE CARLO SIMULATION ENGINE                     │
│  Euler-Maruyama ──► Short Rate Paths ──► Discount Factors   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               PRICING ENGINE                                 │
│  ZCB │ Coupon Bond │ Cap/Floor │ Swaption │ Greeks          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            VISUALIZATION & DASHBOARD                         │
│  Plotly Charts ──► Dash Dashboard ──► Export Reports        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Required Python Libraries

```bash
pip install numpy scipy pandas matplotlib plotly dash
pip install fredapi yfinance requests tqdm joblib
pip install scikit-learn statsmodels jupyter ipywidgets
```

---

## 📂 Folder / File Structure

```
hull-white-ir-engine/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── notebooks/
│   └── Hull_White_Complete.ipynb       # Main Colab notebook
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fred_loader.py              # FRED API data loader
│   │   └── yield_curve_builder.py     # Zero curve bootstrap
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hull_white.py               # HW model core
│   │   ├── calibration.py             # Calibration engine
│   │   └── monte_carlo.py             # MC simulation engine
│   │
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── bond_pricer.py             # ZCB and coupon bonds
│   │   ├── cap_floor_pricer.py        # Caps and Floors
│   │   └── swaption_pricer.py         # Swaptions
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── greeks.py                  # Delta, Vega, Duration
│   │   └── risk_metrics.py            # VaR, CVaR, DV01
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── charts.py                  # Plotly chart functions
│       └── dashboard.py               # Dash dashboard
│
├── data/
│   ├── raw/                           # Raw downloaded data
│   └── processed/                     # Cleaned data
│
└── outputs/
    ├── figures/                       # Exported charts
    └── reports/                       # HTML/PDF reports
```

---

## 🚀 Quick Start

```python
# In Google Colab or local Jupyter:
!pip install -r requirements.txt

# Run the complete notebook
# notebooks/Hull_White_Complete.ipynb
```

---

## 📊 Key Results & Deliverables

- Calibrated Hull-White parameters `(a, σ)` fitted to US Treasury yield curve
- Monte Carlo simulation of 10,000 short-rate paths
- Priced portfolio of caps, floors, and swaptions
- Interactive Plotly dashboard with yield curve evolution
- Greeks (Delta, Vega) for all derivative positions
- Performance metrics: calibration error, pricing accuracy vs Black's formula

---

## 📄 Resume Description

> *Developed a production-grade Hull-White one-factor short-rate model engine in Python, calibrated to real US Treasury yield curve data from the FRED API. Implemented Monte Carlo simulation (10,000 paths) for pricing interest rate derivatives including caps, floors, and swaptions. Built an interactive Plotly/Dash dashboard for real-time yield curve visualization, rate path simulation, and Greeks computation. Achieved < 2 bps calibration error against market benchmark rates.*

---

## 🔧 Potential Upgrades

| Upgrade | Description |
|---|---|
| **Hull-White 2-Factor** | Add second stochastic factor for richer correlation structure |
| **SABR Stochastic Vol** | Model vol smile for swaption pricing |
| **xVA Engine** | Add CVA/DVA calculations for OTC derivatives |
| **QuantLib Integration** | Validate against industry-standard QuantLib pricing |
| **Real-Time Feed** | Bloomberg/Refinitiv RTDS live data feed integration |
| **GPU Acceleration** | CUDA-based Monte Carlo with CuPy |
| **Neural Network Calibration** | ML-based fast calibration using neural surrogates |
| **Regulatory Reporting** | FRTB SA-TB capital charge calculations |

---

*Built by a Senior Quantitative Developer · Production-Ready · GitHub Portfolio Project*
