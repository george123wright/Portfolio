"""
CapIQ Monte Carlo valuation engines (FCFF DCF, DR-adjusted FCFE DCF, DDM, and Residual Income).

This module implements a valuation pipeline that:

1) loads Capital IQ consensus forecast tables (median/high/low/standard deviation/estimate counts),

2) aligns those forecasts to a consistent fiscal period grid (annual and, where available, quarterly),

3) simulates correlated future financial paths using heavy-tailed, potentially skewed marginal distributions,

4) applies sector-aware practical constraints and coherence checks, and

5) produces per-share valuation distributions for multiple valuation models.

The implementation is designed for robustness under incomplete analyst coverage and heterogeneous period
granularity. It therefore includes explicit missingness detection, imputation paths, and conservative
sanity checks intended to avoid unstable valuation outputs when individual drivers are sparse.

----------------------------
Key Units and Conventions
----------------------------
Cashflow drivers are modelled in currency units scaled by ``UNIT_MULT`` (default: 1e6). In practice,
the forecast sheets are assumed to be in "millions" for absolute financial statement line items.

Working capital sign convention:

- The historical and forecast driver ``dnwc`` is treated as the change in net working capital (ΔNWC),
  where a positive value is an inflow (a reduction in working capital) and a negative value is an outflow.

- For cashflow equations that require working-capital *outflow*, the transformation is:
  delta_wc_outflow = -dnwc

Discounting convention:

- Continuous-time notation is avoided in outputs, but discount factors are computed using:
    
    DF(t) = exp(-t * ln(1 + r)) = 1 / (1 + r)^t

  where t is the year fraction from the valuation date to the period end.

Terminal value convention (perpetuity in discrete compounding over a final step dt):

- Let one_r_dt = (1 + r)^dt and one_g_dt = (1 + g)^dt.

- TerminalValue = CF_T * one_g_dt / (one_r_dt - one_g_dt)
  When dt = 1 year, this reduces to: TerminalValue = CF_T * (1 + g) / (r - g).

----------------------------
Period Grid and Alignment
----------------------------
Forecast tables are aligned to a mixed grid containing:

- annual fiscal year-ends (FY) at the inferred fiscal month/day, and

- quarterly period-ends where quarter-level forecasts exist (including optional stub quarters before the
  first available annual period).

Alignment differentiates between:

- "flow" metrics (e.g., revenue, CapEx, interest, cashflow), which are aggregated or apportioned across
  periods, and

- "stock" metrics (e.g., net debt, BVPS), which are treated as point-in-time levels.

For quarterly flows, an adjustment is applied such that the fiscal-year-end quarter (Q4) becomes the
residual required for quarterly components to sum to the annual value when both granularities are mixed.

Seasonality support:
- Some flow metrics (notably CapEx) may be apportioned using empirically inferred quarterly weights
  derived from TTM quarters in the historical statements.

----------------------------
Marginal Simulation: Skew-t Distributions
----------------------------
Each forecast driver is simulated per period across ``N_SIMS`` paths using a skewed Student-t family.
The aim is to preserve:

- a central tendency (typically the consensus median or mean proxy),

- dispersion consistent with consensus standard deviation (or an inferred proxy from high/low ranges), and

- tail behaviour that is heavier than Gaussian where indicated by history or forecast ranges.

Advantages of a skew-t marginal:

- heavy tails improve robustness to forecast uncertainty that is not well described by a normal model,

- skewness accommodates asymmetric analyst distributions (e.g., limited downside, large upside),

- degrees-of-freedom control permits interpolation between normal-like and fat-tailed regimes.

----------------------------
Missingness and Imputation
----------------------------
Forecast coverage varies materially across tickers and drivers. When forecast tables are missing, or when
simulated draws contain non-finite values, imputation is used to fill a minimal driver set required to
evaluate valuation methods.

Imputation techniques include:

- history-anchored ratio priors (e.g., CapEx as a fraction of revenue),

- year-on-year link models for level series,

- random-walk simulations for level-like balance-sheet items when only history is available, and

- explicit driver derivations (e.g., deriving ΔNWC from a fitted relationship between revenue and ΔNWC).

Advantages of explicit imputation:

- increases model coverage without silently dropping tickers,

- reduces instability from sparse single-driver forecasts,

- allows method-level gating so that only coherent methods are used.

----------------------------
Dependence Modelling: Copula Reordering
----------------------------
Simulated marginals are combined using a dependence overlay derived from historical annual data.
The dependence step is implemented as a reordering (copula) procedure:

- marginal distributions are kept intact,

- simulated innovations are reordered to match a historical correlation structure, and

- a multivariate t innovation generator is used where feasible for robustness under fat tails.

Advantages of copula reordering:

- preserves the calibrated marginal distributions,

- introduces realistic cross-driver co-movement (e.g., revenue, margins, working capital),

- improves plausibility of cashflow paths relative to independently simulated drivers.

----------------------------
Practical Bounds and Coherence Checks
----------------------------
Sector policies bound key ratios (tax rates, CapEx ratios, D&A ratios, ΔNWC ratios) and gate method usage
when coherence is not evidenced in history (e.g., interest vs net debt, ΔNWC vs revenue).

Additional accounting coherence constraints are applied, for example:

- tax rates clipped to [tax_lo, tax_hi],

- CapEx, maintenance CapEx, D&A, and interest floored at zero,

- maintenance CapEx constrained to be no greater than total CapEx,

- EBITDA constrained to be at least EBIT + max(D&A, 0),

- EBIT optionally constrained by EBITDA - D&A.

----------------------------
Valuation Engines and Equations (Text Form)
----------------------------
FCFF DCF (enterprise valuation discounted at WACC):

- EnterpriseValue = PV(FCFF_t) + PV(TerminalValue_FCFF)

- EquityValue = EnterpriseValue - NetDebt

- PerShareValue = EquityValue / SharesOutstanding

Supported FCFF constructions (illustrative forms):

1) Direct forecast:

   FCFF = FCF

2) From cash from operations:

   FCFF = CFO - CapEx + Interest * (1 - TaxRate)

   FCFF = CFO - MaintenanceCapEx + Interest * (1 - TaxRate)

3) From operating profit:

   FCFF = EBIT * (1 - TaxRate) + D&A - CapEx + ΔNWC

   FCFF = (EBITDA - D&A) * (1 - TaxRate) + D&A - CapEx + ΔNWC

4) From net income:

   FCFF = NetIncome + D&A + Interest * (1 - TaxRate) - CapEx + ΔNWC

FCFE DCF (equity valuation discounted at CoE, DR-adjusted reinvestment):
Debt ratio is defined as:

   DR = Debt / (MarketCapitalisation + Debt)

In this module, DR is sourced from the capital-structure weight wD computed when forming WACC, such that:

   DR = wD = D / (E + D)

Supported FCFE constructions (all using delta_wc_outflow = -dnwc):

1) NI_DR:

   FCFE = NetIncome - (1 - DR) * (CapEx - D&A) - (1 - DR) * delta_wc_outflow

2) EBITDA_INT_TAX_DR (tax amount proxy based on positive EBT):

   TaxAmount = max((EBITDA - D&A) - Interest, 0) * TaxRate

   FCFE = (EBITDA - Interest - TaxAmount) - (1 - DR) * (CapEx - D&A) - (1 - DR) * delta_wc_outflow

3) FCFF_BRIDGE_DR:

   InterestAfterTax = Interest * (1 - TaxRate)

   FCFE = FCFF - InterestAfterTax + DR * ((CapEx - D&A) + delta_wc_outflow)

4) EBIT_INT_DR:

   FCFE = (EBIT - Interest) * (1 - TaxRate) - (1 - DR) * (CapEx - D&A) - (1 - DR) * delta_wc_outflow

DDM (dividend discount model discounted at CoE paths):

- Price = PV(Dividends_t) + PV(TerminalValue_Dividends)

Residual Income (Edwards–Bell–Ohlson form discounted at CoE):

- ResidualIncome_t = EPS_t - RequiredReturn_t * BVPS_begin_t
  where RequiredReturn_t is derived from CoE and the period year fraction.

- Price = BVPS_0 + PV(ResidualIncome_t) + PV(TerminalValue_ResidualIncome)

----------------------------
Determinism and Performance
----------------------------
Randomness is controlled by ``RunContext``, which constructs deterministic RNG streams keyed by:

- a global seed,

- an optional ticker identifier, and

- a label string describing the stochastic component.

For performance, FCFF and FCFE share a single per-ticker preparation context that contains:

- aligned periods,

- simulated driver panels,

- net-debt paths,

- caches for alignment and quarterly overrides, and

- shared terminal-growth uniforms that are transformed under model-specific caps.

"""

import re

import calendar

import numpy as np

import pandas as pd

import config

import math

from scipy.stats import norm

from scipy.stats import t, truncnorm, rankdata, kurtosis

import io

import contextlib

from pathlib import Path

import warnings

from dataclasses import dataclass, field

import gc

from typing import Any, Callable, Iterable, Sequence

from functools import lru_cache

import logging

import zlib

from collections import Counter

import time

from functions.export_forecast import export_results

from data_processing.financial_forecast_data import FinancialForecastData

from maps.ccy_exchange_map import _CCY_BY_SUFFIX, _USD_EXCEPTIONS

logging.getLogger('xlrd.compdoc').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

METRICS = ['Final Est.', 'Median', 'High', 'Low', 'Std. Dev.', 'No. of Estimates']

MIN_HEADER_NONNA = 3

FY_FREQ = 'Y-MAR'

UNIT_MULT = 1000000.0

SEED = 42


def _stable_u32(
    s: str
) -> int:
    """
    Return a stable unsigned 32-bit hash of a string.

    The hash is computed using CRC32 over UTF-8 bytes and then masked into the range
    ``0 <= h <= 2**32 - 1``.

    This helper exists primarily to support deterministic random-number seeding via ``RunContext``.
    CRC32 is used because it is:
   
    - stable across Python processes and platforms,
   
    - fast, and
   
    - sufficient for Monte Carlo stream partitioning.

    Parameters
    ----------
    s:
        Input text to hash.

    Returns
    -------
    int
        Unsigned 32-bit integer hash.

    Notes
    -----
    CRC32 is not a cryptographic hash. It must not be used for security-sensitive purposes.
    """
    
    return zlib.crc32(s.encode('utf-8')) & 4294967295


@dataclass(frozen = True)
class RunContext:
    """
    Deterministic random-number context for Monte Carlo valuation.

    The purpose of this class is to provide reproducible, label-addressable random streams that remain
    stable across refactors. A seed sequence is constructed from:
   
    - the global run seed (``seed``),
   
    - the ticker identifier (if present), and
   
    - a caller-provided label (a short string describing the stochastic component).

    This design supports:
   
    - per-ticker determinism (different tickers receive different streams), and
   
    - per-component determinism (different stochastic components do not share state).

    Advantages:
   
    - deterministic debugging of Monte Carlo outputs,
   
    - reduced risk of accidental correlation introduced by shared RNG state,
   
    - stable results under code motion provided the labels remain unchanged.
   
    """
    
    seed: int = SEED

    ticker: str | None = None


    def rng(
        self,
        label: str
    ) -> np.random.Generator:
        """
        Create a NumPy random generator uniquely keyed by (seed, ticker, label).

        Parameters
        ----------
        label:
            A stable identifier for the stochastic component (for example, ``"skewt:fcf"`` or
            ``"terminal_g:fcff_u"``). Labels should be treated as part of the model definition; changing
            a label changes the random stream and therefore the simulated distribution.

        Returns
        -------
        numpy.random.Generator
            An independent generator derived from a SeedSequence built from stable 32-bit hashes.

        Notes
        -----
        The hashing uses CRC32 and therefore produces a 32-bit unsigned integer. The resulting generator
        is suitable for Monte Carlo simulation but is not intended for cryptographic use.
        """
        
        parts = [int(self.seed)]

        if self.ticker is not None:

            parts.append(_stable_u32(
                s = str(self.ticker)
            ))

        parts.append(_stable_u32(
            s = label
        ))

        ss = np.random.SeedSequence(parts)

        return np.random.default_rng(ss)


def _ensure_ctx(
    ctx: 'RunContext | None'
) -> RunContext:
    """
    Ensure that a valid ``RunContext`` is available.

    Parameters
    ----------
    ctx:
        A ``RunContext`` instance or ``None``.

    Returns
    -------
    RunContext
        ``ctx`` if provided; otherwise a default context with ``seed=SEED`` and ``ticker=None``.

    Rationale
    ---------
    Many internal helpers accept an optional context for deterministic stochastic components. This
    wrapper centralises the default behaviour and avoids repeated conditional logic.
    """
    
    return ctx if ctx is not None else RunContext(seed = SEED, ticker = None)


N_SIMS = 50000

MIN_POINTS = 6

e6 = 1e-06

FLOOR = 0.0

G_CAP = 0.06

GF_CAP = FLOOR + 0.0001

e12 = 1e-12

Q_LO = 0.01

Q_HI = 0.99

PAD_IQR = 0.5

SAFETY_SPREAD = 0.01

SHRINK = 0.25

LAST_N = 50

ANCHOR = config.RF

NU_MIN = 1.5

NU_MAX = 200.0

USE_PRIMITIVE_FCFF = False

MONTH_ABBR = {m.lower(): i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start = 1)}

TODAY_TS = pd.Timestamp.today().normalize()

_Q_PAT = re.compile('(?:^|[^A-Z0-9])(?:F?Q[1-4]|[1-4]Q|Q[1-4])(?:[^A-Z0-9]|$)')

_HEADER_YEAR_MIN = 1900

_HEADER_YEAR_MAX = 2100

_EXCEL_SERIAL_MIN = 20000

_EXCEL_SERIAL_MAX = 80000

CANONICAL_SECTORS: tuple[str, ...] = ('Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Healthcare', 'Industrials', 'Materials', 'Real Estate', 'Technology', 'Utilities')

_FINANCIAL_SECTOR_KEYWORDS = ('bank', 'banks', 'banking', 'insurance', 'insurer', 'insurers', 'financial', 'financials', 'capital markets', 'asset management', 'brokerage')

_REAL_ESTATE_SECTOR_KEYWORDS = ('real estate', 'reit')

_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    'Communication Services': ('communication services', 'telecom', 'media', 'entertainment', 'interactive media'), 
    'Consumer Discretionary': ('consumer discretionary', 'retail', 'automobile', 'autos', 'leisure', 'hotels', 'restaurants', 'apparel'),
    'Consumer Staples': ('consumer staples', 'food', 'beverage', 'household', 'personal products', 'tobacco'), 
    'Energy': ('energy', 'oil', 'gas', 'exploration', 'midstream', 'downstream'), 
    'Healthcare': ('healthcare', 'pharma', 'biotech', 'medical', 'life sciences'), 
    'Industrials': ('industrials', 'industrial', 'aerospace', 'defense', 'transport', 'machinery', 'construction'), 
    'Materials': ('materials', 'chemical', 'chemicals', 'metals', 'mining', 'paper', 'packaging'), 
    'Technology': ('technology', 'information technology', 'software', 'semiconductor', 'hardware', 'it services'), 
    'Utilities': ('utilities', 'utility', 'electric', 'water', 'gas utility', 'independent power')}

@dataclass(frozen = True)
class SectorPolicy:
    """
    Sector-specific modelling policy and guardrails.

    This structure encodes:
  
    - method gating rules (for example, whether cash-from-operations methods require coherent interest
      and net-debt history),
  
    - ratio bounds used for practical constraints (tax, CapEx intensity, D&A intensity, working-capital
      intensity), and
  
    - terminal-growth shrinkage and dispersion multipliers used by terminal-growth estimation.

    The policy is used in:
  
    - practical clipping and sanity checks (see ``_apply_practical_checks_and_bounds``), and
  
    - selection of FCFF/FCFE method definitions when historical coherence evidence is required.

    Advantages:
  
    - reduces the incidence of implausible cashflow paths,
  
    - provides a transparent mechanism for sector-aware conservatism,
  
    - encourages graceful degradation when data completeness is limited.
  
    """
    
    name: str

    fcff_profile: str

    require_interest_debt_for_cfo: bool

    require_wc_for_dnwc: bool

    min_points_interest_debt: int

    min_points_wc: int

    min_abs_corr_interest_debt: float

    min_abs_corr_wc: float

    tax_lo: float

    tax_hi: float

    capex_ratio_lo: float

    capex_ratio_hi: float

    da_ratio_lo: float

    da_ratio_hi: float

    dnwc_ratio_lo: float

    dnwc_ratio_hi: float

    growth_shrink: float

    growth_sigma_mult: float

    ddm_payout_hi: float

    ri_payout_hi: float

SECTOR_POLICIES: dict[str, SectorPolicy] = {
    'Communication Services': 
        SectorPolicy(
            name = 'Communication Services', 
            fcff_profile = 'growth', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6,
            min_points_wc = 6,
            min_abs_corr_interest_debt = 0.1,
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.0,
            tax_hi = 0.4, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.3, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.22, 
            dnwc_ratio_lo = -0.25, 
            dnwc_ratio_hi = 0.25, 
            growth_shrink = 0.3, 
            growth_sigma_mult = 1.15,
            ddm_payout_hi = 1.4,
            ri_payout_hi = 1.4
        ), 
    'Consumer Discretionary':
        SectorPolicy(
            name = 'Consumer Discretionary', 
            fcff_profile = 'cyclical',
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False,
            min_points_interest_debt = 6,
            min_points_wc = 6,
            min_abs_corr_interest_debt = 0.1,
            min_abs_corr_wc = 0.1,
            tax_lo = 0.0, 
            tax_hi = 0.4,
            capex_ratio_lo = 0.0,
            capex_ratio_hi = 0.35,
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.25, 
            dnwc_ratio_lo = -0.3,
            dnwc_ratio_hi = 0.3, 
            growth_shrink = 0.25,
            growth_sigma_mult = 1.2,
            ddm_payout_hi = 1.35, 
            ri_payout_hi = 1.35
        ), 
    'Consumer Staples': 
        SectorPolicy(
            name = 'Consumer Staples', 
            fcff_profile = 'stable', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.05, 
            tax_hi = 0.35, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.2, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.18, 
            dnwc_ratio_lo = -0.15, 
            dnwc_ratio_hi = 0.2, 
            growth_shrink = 0.35, 
            growth_sigma_mult = 0.85, 
            ddm_payout_hi = 1.6, 
            ri_payout_hi = 1.45
        ), 
    'Energy':
        SectorPolicy(
            name = 'Energy', 
            fcff_profile = 'cyclical', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.0, 
            tax_hi = 0.45, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.5, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.35, 
            dnwc_ratio_lo = -0.35, 
            dnwc_ratio_hi = 0.35, 
            growth_shrink = 0.2, 
            growth_sigma_mult = 1.35, 
            ddm_payout_hi = 1.3, 
            ri_payout_hi = 1.3
        ), 
    'Financials': 
        SectorPolicy(
            name = 'Financials', 
            fcff_profile = 'financial', 
            require_interest_debt_for_cfo = True, 
            require_wc_for_dnwc = True, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.2, 
            min_abs_corr_wc = 0.15, 
            tax_lo = 0.0, 
            tax_hi = 0.4, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.12, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.12, 
            dnwc_ratio_lo = -0.1, 
            dnwc_ratio_hi = 0.1, 
            growth_shrink = 0.35, 
            growth_sigma_mult = 0.9, 
            ddm_payout_hi = 1.7, 
            ri_payout_hi = 1.7
        ), 
    'Healthcare':
        SectorPolicy(
            name = 'Healthcare', 
            fcff_profile = 'growth', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.0, 
            tax_hi = 0.4, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.25, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.2, 
            dnwc_ratio_lo = -0.2, 
            dnwc_ratio_hi = 0.25, 
            growth_shrink = 0.25, 
            growth_sigma_mult = 1.15, 
            ddm_payout_hi = 1.35, 
            ri_payout_hi = 1.45
        ), 
    'Industrials': 
        SectorPolicy(
            name = 'Industrials', 
            fcff_profile = 'wc_heavy', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = True, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.15, 
            tax_lo = 0.0, 
            tax_hi = 0.4, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.35, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.22, 
            dnwc_ratio_lo = -0.25, 
            dnwc_ratio_hi = 0.3, 
            growth_shrink = 0.25, 
            growth_sigma_mult = 1.1, 
            ddm_payout_hi = 1.35, 
            ri_payout_hi = 1.35
        ), 
    'Materials': 
        SectorPolicy(
            name = 'Materials', 
            fcff_profile = 'cyclical', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.0, 
            tax_hi = 0.4, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.45, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.3, 
            dnwc_ratio_lo = -0.3, 
            dnwc_ratio_hi = 0.35, 
            growth_shrink = 0.2, 
            growth_sigma_mult = 1.25, 
            ddm_payout_hi = 1.3, 
            ri_payout_hi = 1.3
        ), 
    'Real Estate': 
        SectorPolicy(
            name = 'Real Estate', 
            fcff_profile = 'asset_income', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = True, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.15, 
            tax_lo = 0.0, 
            tax_hi = 0.35, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.25, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.35, 
            dnwc_ratio_lo = -0.15, 
            dnwc_ratio_hi = 0.2, 
            growth_shrink = 0.35, 
            growth_sigma_mult = 0.9, 
            ddm_payout_hi = 1.8, 
            ri_payout_hi = 1.6
        ), 
    'Technology': 
        SectorPolicy(
            name = 'Technology', 
            fcff_profile = 'growth', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.0, 
            tax_hi = 0.35, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.22, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.15, 
            dnwc_ratio_lo = -0.25, 
            dnwc_ratio_hi = 0.2, 
            growth_shrink = 0.2, 
            growth_sigma_mult = 1.2, 
            ddm_payout_hi = 1.25, 
            ri_payout_hi = 1.35
        ), 
    'Utilities':
        SectorPolicy(
            name = 'Utilities', 
            fcff_profile = 'stable_income', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = 6, 
            min_points_wc = 6, 
            min_abs_corr_interest_debt = 0.1, 
            min_abs_corr_wc = 0.1, 
            tax_lo = 0.05, 
            tax_hi = 0.35, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.3, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.25, 
            dnwc_ratio_lo = -0.12, 
            dnwc_ratio_hi = 0.18, 
            growth_shrink = 0.4, 
            growth_sigma_mult = 0.8, 
            ddm_payout_hi = 1.9, 
            ri_payout_hi = 1.6
        ), 
    'Unknown': 
        SectorPolicy(
            name = 'Unknown', 
            fcff_profile = 'base', 
            require_interest_debt_for_cfo = False, 
            require_wc_for_dnwc = False, 
            min_points_interest_debt = MIN_POINTS, 
            min_points_wc = MIN_POINTS, 
            min_abs_corr_interest_debt = 0.0, 
            min_abs_corr_wc = 0.0, 
            tax_lo = 0.0, 
            tax_hi = 0.4, 
            capex_ratio_lo = 0.0, 
            capex_ratio_hi = 0.35, 
            da_ratio_lo = 0.0, 
            da_ratio_hi = 0.25, 
            dnwc_ratio_lo = -0.2, 
            dnwc_ratio_hi = 0.2, 
            growth_shrink = SHRINK, 
            growth_sigma_mult = 1.0, 
            ddm_payout_hi = 1.5, 
            ri_payout_hi = 1.5
        )
}

_DEBT_ROWS = ('Total Debt', 'Total Debt (incl. Leases)', 'Total Debt & Capital Lease Obligations', 'Total Debt & Lease Obligations')

_CASH_ROWS = ('Cash and Cash Equivalents', 'Cash & Cash Equivalents', 'Cash & Short Term Investments', 'Cash and Short Term Investments', 'Cash and Short-Term Investments')

_NET_DEBT_ROWS = ('Net Debt', 'Net Debt (incl. Leases)', 'Net Debt (Including Leases)', 'Net Debt / (Cash)')

_HIST_INC_ROWS_KEEP = ['Interest Expense', 'Interest Expense, Total', 'Effective Tax Rate %', 'EBIT', 'EBITDA', 'Net Income', 'Net Income to Company', 'Net Income (GAAP)', 'Net Income (Excl. Excep)', 'Net Income (Excl. Excep, GW)', 'Revenue', 'Total Revenue', 'EBT', 'Earnings Before Tax', 'Pre-Tax Income', 'Pretax Income', 'Depreciation & Amort.']

_HIST_CF_ROWS_KEEP = ['Unlevered Free Cash Flow', 'Levered Free Cash Flow', 'Cash from Ops.', 'Cash from Ops', 'Cash from Ops.', 'Cash From Operations', 'Operating Cash Flow', 'Capital Expenditure', 'Capital Expenditures', 'Depreciation & Amort.', 'Depreciation & Amort., Total', 'Change in Net Working Capital', 'Change in Working Capital']

_HIST_RAT_ROWS_KEEP = ['Gross Margin %', 'Gross Margin', '  Gross Margin %', 'ROE %', 'Return on Equity %', '  Return on Equity %', 'ROA %', 'Return on Assets %', '  Return on Assets %']

_SIMILAR_GROUPS = {'capex': {'maint_capex'}, 'maint_capex': {'capex'}, 'ebit': {'ebitda'}, 'ebitda': {'ebit'}, 'cfo': {'fcf'}, 'fcf': {'cfo'}}

FLOW_KEYS = {'revenue', 'cfo', 'fcf', 'capex', 'maint_capex', 'da', 'interest', 'ebit', 'ebitda', 'ebt', 'net_income'}

RATE_KEYS = {'tax', 'gross_margin', 'roe'}

STOCK_KEYS = {'net_debt', 'eps', 'dps', 'bvps'}

_EPS_HIST_ROWS: tuple[str, ...] = ('EPS Normalized', 'EPS', 'Diluted EPS', 'Basic EPS', 'Earnings Per Share')

_DPS_HIST_ROWS: tuple[str, ...] = ('DPS', 'Dividend Per Share', 'Dividends Per Share', 'Cash Dividends Paid Per Share')


def _extract_hist_ratios_series(
    df: pd.DataFrame | None,
    row_candidates: Sequence[str]
) -> pd.Series | None:
    """
    Extract a historical ratio series from a ratios table using multiple row-name candidates.

    Parameters
    ----------
    df:
        Historical ratios table, typically from a CapIQ financial statements workbook. The table is
        expected to be indexed by ratio name with columns that are date-like.
    row_candidates:
        Ordered candidate row labels to try (for example, multiple spellings of "EPS" or "DPS").

    Returns
    -------
    pandas.Series | None
        The first successfully matched row coerced to numeric values, with a best-effort conversion of
        column labels to datetimes. Returns ``None`` when no candidate row is present or when the input
        is empty.

    Notes
    -----
    The returned series is not rescaled; callers are responsible for applying any unit conventions.
    """
    
    if df is None or df.empty:

        return None

    row = _first_existing_row(
        df = df,
        candidates = row_candidates
    )

    if row is None:

        return None

    s = pd.to_numeric(df.loc[row], errors = 'coerce')

    try:

        s.index = pd.to_datetime(s.index, errors = 'ignore')

    except (TypeError, ValueError):

        pass

    return s


_HIST_PANEL_KEY_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    'fcf': [('cf', 'Unlevered Free Cash Flow'), ('cf', 'Levered Free Cash Flow')], 
    'cfo': [('cf', 'Cash from Ops.'), ('cf', 'Cash from Ops'), ('cf', 'Cash From Operations'), ('cf', 'Operating Cash Flow')], 
    'revenue': [('inc', 'Revenue'), ('inc', 'Total Revenue')], 
    'capex': [('cf', 'Capital Expenditure'), ('cf', 'Capital Expenditures')], 
    'maint_capex': [('cf', 'Capital Expenditure'), ('cf', 'Capital Expenditures')], 
    'interest': [('inc', 'Interest Expense'), ('inc', 'Interest Expense, Total')], 
    'tax': [('inc', 'Effective Tax Rate %')], 
    'da': [('inc', 'Depreciation & Amort.'), ('cf', 'Depreciation & Amort.'), ('cf', 'Depreciation & Amort., Total')], 
    'ebit': [('inc', 'EBIT'), ('inc', 'Operating Income')], 
    'ebitda': [('inc', 'EBITDA')], 
    'net_income': [('inc', 'Net Income'), ('inc', 'Net Income (GAAP)'), ('inc', 'Net Income to Company')], 
    'net_debt': [('nd', 'net_debt')], 
    'ebt': [('inc', 'EBT'), ('inc', 'Earnings Before Tax'), ('inc', 'Pre-Tax Income'), ('inc', 'Pretax Income')], 
    'gross_margin': [('rat', 'Gross Margin %'), ('rat', 'Gross Margin'), ('rat', '  Gross Margin %')], 
    'roe': [('rat', 'ROE %'), ('rat', 'Return on Equity %'), ('rat', '  Return on Equity %')], 
    'roa': [('rat', 'ROA %'), ('rat', 'Return on Assets %'), ('rat', '  Return on Assets %')], 
    'dnwc': [('cf', 'Change in Net Working Capital'), ('cf', 'Change in Working Capital')]
}

_FDATA_RUNTIME: FinancialForecastData | None = None

_MACRO_RUNTIME = None

macro = None


def _ensure_runtime_data() -> tuple[FinancialForecastData, object]:
    """
    Lazily initialise and cache runtime macro and reference data.

    Returns
    -------
    (FinancialForecastData, object)
        The shared ``FinancialForecastData`` instance and its associated macro container.

    Rationale
    ---------
    Several components (FX conversion, rate series selection, and cost-of-debt estimation) depend on
    macro data loaded by ``FinancialForecastData``. A module-level cache avoids repeated I/O and keeps
    behaviour consistent across valuations performed within a single Python process.
    """
    
    global _FDATA_RUNTIME, _MACRO_RUNTIME, macro

    if _FDATA_RUNTIME is None or _MACRO_RUNTIME is None:

        _FDATA_RUNTIME = FinancialForecastData()

        _MACRO_RUNTIME = _FDATA_RUNTIME.macro

        macro = _MACRO_RUNTIME

    return (_FDATA_RUNTIME, _MACRO_RUNTIME)


_KNOWN_CCY_ALIASES: dict[str, str] = {'GBX': 'GBP', 'GBPX': 'GBP', 'CNH': 'CNY', 'RMB': 'CNY'}

_EXTRA_SUFFIX_CCY: dict[str, str] = {'.T': 'JPY', '.TW': 'TWD', '.SW': 'CHF', '.SA': 'BRL', '.V': 'CAD', '.MI': 'EUR', '.HE': 'EUR', '.ST': 'SEK', '.OL': 'NOK', '.NZ': 'NZD'}

_FUTURE_RATIO_KEYS = {'tax', 'gross_margin', 'roe', 'roa', 'roe_pct', 'roa_pct'}

_FUTURE_META_ROWS = {'No_of_Estimates', 'period_type', 'period_label'}

_CCY_PAREN_PAT = re.compile('\\((?:[^)]*?\\|)?([A-Za-z]{3})\\)')


def _normalize_ccy_code(
    ccy: str | None
) -> str | None:
    """
    Normalise an input currency code to a canonical ISO-like 3-letter code.

    Parameters
    ----------
    ccy:
        Input currency code (for example, "usd", "GBX", "RMB") or ``None``.

    Returns
    -------
    str | None
        Upper-case 3-letter code when recognised; otherwise ``None``.

    Notes
    -----
    Common aliases are mapped to canonical forms, for example:
  
    - "GBX" and "GBPX" are mapped to "GBP".
  
    - "CNH" and "RMB" are mapped to "CNY".
  
    """
    
    if ccy is None:

        return None

    s = str(ccy).strip().upper()

    if not s:

        return None

    s = _KNOWN_CCY_ALIASES.get(s, s)

    return s if re.fullmatch('[A-Z]{3}', s) is not None else None


def _normalize_sector_label(
    raw: str | None
) -> str:
    """
    Map a raw sector/industry label to a canonical sector name.

    Parameters
    ----------
    raw:
        Raw sector or industry description (for example, from a metadata table).

    Returns
    -------
    str
        Canonical sector label, one of ``CANONICAL_SECTORS`` or "Unknown".

    Method
    ------
    The mapping is keyword-based:
  
    - real-estate and financial keywords are handled first (as these terms are often ambiguous),
  
    - remaining sectors are matched using a sector-specific keyword list,
  
    - unmatched inputs fall back to "Unknown".

    Advantages
    ----------
    Keyword mapping is tolerant of inconsistent upstream labelling and supports graceful behaviour when
    only coarse industry descriptions are available.
    """
    
    s = str(raw or '').strip().lower()

    if not s:

        return 'Unknown'

    if any((tok in s for tok in _REAL_ESTATE_SECTOR_KEYWORDS)):

        return 'Real Estate'

    if any((tok in s for tok in _FINANCIAL_SECTOR_KEYWORDS)):

        return 'Financials'

    for sector in CANONICAL_SECTORS:
        
        if sector in {'Real Estate', 'Financials'}:

            continue

        if any((tok in s for tok in _SECTOR_KEYWORDS.get(sector, ()))):

            return sector

    return 'Unknown'


def _policy_for_sector(
    sector_label: str | None
) -> SectorPolicy:
    """
    Retrieve the applicable ``SectorPolicy`` for a sector label.

    Parameters
    ----------
    sector_label:
        Sector label in arbitrary form.

    Returns
    -------
    SectorPolicy
        Sector policy mapped from ``SECTOR_POLICIES`` using ``_normalize_sector_label``. "Unknown" is
        returned when no match is found.
    """
    
    sector_name = _normalize_sector_label(
        raw = sector_label
    )

    return SECTOR_POLICIES.get(sector_name, SECTOR_POLICIES['Unknown'])


def _lookup_ticker_meta_value(
    meta_obj,
    ticker: str
):
    """
    Retrieve a metadata field for a ticker from a supported container type.

    Parameters
    ----------
    meta_obj:
        Supported types include ``pandas.Series`` and ``dict``. Any other type results in ``None``.
    ticker:
        Ticker identifier used as a lookup key.

    Returns
    -------
    object | None
        The retrieved value or ``None`` when unavailable.
    """
    
    if meta_obj is None:

        return None

    if isinstance(meta_obj, pd.Series):

        return meta_obj.get(ticker, None)

    if isinstance(meta_obj, dict):

        return meta_obj.get(ticker, None)

    return None


def _infer_sector_label_for_ticker(
    r_data,
    ticker: str
) -> str | None:
    """
    Infer a sector/industry label for a ticker from runtime metadata.

    Parameters
    ----------
    r_data:
        Runtime container with potential attributes such as ``sector``, ``industry``, ``sub_industry``,
        ``gics_sector``, and ``gics_industry``. Each attribute may be a series-like mapping keyed by
        ticker.
    ticker:
        Ticker identifier.

    Returns
    -------
    str | None
        The first non-empty label discovered. Returns ``None`` when no usable metadata is available.

    Notes
    -----
    A minimal heuristic fallback is applied when the ticker string itself contains obvious sector tokens
    (for example, "BANK", "INSUR", "REIT", "FIN"). This is intended only as a last resort.
    """
    
    for attr in ('sector', 'industry', 'sub_industry', 'gics_sector', 'gics_industry'):
    
        v = _lookup_ticker_meta_value(
            meta_obj = getattr(r_data, attr, None),
            ticker = ticker
        )

        if v is not None and str(v).strip():

            return str(v)

    t = str(ticker or '').strip().upper()

    if any((tok in t for tok in ('BANK', 'INSUR', 'REIT', 'FIN'))):

        return t

    return None


def _infer_quote_currency_from_ticker(
    ticker: str
) -> str | None:
    """
    Infer the quote currency for a ticker using suffix rules and explicit exceptions.

    Parameters
    ----------
    ticker:
        Ticker symbol, potentially including an exchange suffix (for example, "7203.T" or "NESN.SW").

    Returns
    -------
    str | None
        A 3-letter currency code (for example, "USD", "JPY") when inferable; otherwise ``None``.

    Method
    ------
   
    1) An explicit exception list is consulted for tickers quoted in USD despite non-US suffix patterns.
   
    2) Known suffix-to-currency maps are applied (CapIQ/market conventions).
   
    3) If no suffix exists, USD is assumed as a default.
   
    4) If an unrecognised suffix exists, ``None`` is returned to avoid guessing.
   
    """
    
    t = str(ticker).upper().strip()

    if not t:

        return None

    if t in _USD_EXCEPTIONS:

        return 'USD'

    for suf, ccy in sorted(_CCY_BY_SUFFIX.items(), key = lambda kv: len(kv[0]), reverse = True):
     
        if t.endswith(str(suf).upper()):

            return _normalize_ccy_code(
                ccy = ccy
            )

    for suf, ccy in sorted(_EXTRA_SUFFIX_CCY.items(), key = lambda kv: len(kv[0]), reverse = True):
       
        if t.endswith(str(suf).upper()):

            return _normalize_ccy_code(
                ccy = ccy
            )

    if '.' not in t:

        return 'USD'

    return None


@lru_cache(maxsize = 1)
def _latest_usd_per_ccy_map() -> dict[str, float]:
    """
    Return a mapping from currency code to the most recent USD-per-currency spot rate.

    Returns
    -------
    dict[str, float]
        Mapping ``CCY -> USD_per_CCY``. The USD base is always present with value 1.0.

    Data Sources
    ------------
    The function attempts, in order:
    1) a panel ``macro_obj.r.fx_usd_per_ccy`` with columns named like ``"USD_per_EUR"``, and
    2) a series ``macro_obj.currency`` with currency pairs encoded as concatenated ISO codes (for
       example, "EURUSD" or "USDEUR").

    Notes
    -----
    The mapping is cached (maxsize=1) because FX rates are only required at coarse granularity for
    forecast currency normalisation within a single process run.
    """
    
    out: dict[str, float] = {'USD': 1.0}

    try:

        _, macro_obj = _ensure_runtime_data()

    except (AttributeError, TypeError, ValueError):

        macro_obj = None

    fx_panel = getattr(getattr(macro_obj, 'r', None), 'fx_usd_per_ccy', None)

    if isinstance(fx_panel, pd.DataFrame) and (not fx_panel.empty):

        try:

            row = fx_panel.sort_index().ffill().iloc[-1]

            for col, v in row.items():
                m = re.fullmatch('USD_per_([A-Z]{3})', str(col).upper().strip())

                if not m:

                    continue

                if pd.notna(v) and np.isfinite(float(v)) and (float(v) > 0):

                    out[m.group(1)] = float(v)

        except (TypeError, ValueError, IndexError):

            pass

    fx_pairs = getattr(macro_obj, 'currency', None)

    if isinstance(fx_pairs, pd.Series) and (not fx_pairs.empty):

        for pair, v in fx_pairs.items():
            if pd.isna(v):

                continue

            try:

                vv = float(v)

            except (TypeError, ValueError, OverflowError):

                continue

            if not np.isfinite(vv) or vv <= 0:

                continue

            m = re.fullmatch('([A-Z]{3})([A-Z]{3})', str(pair).upper().strip())

            if not m:

                continue

            base, quote = (m.group(1), m.group(2))

            if quote == 'USD':

                out.setdefault(base, vv)
            elif base == 'USD':

                out.setdefault(quote, 1.0 / vv)

    return out


def _fx_target_per_source(
    source_ccy: str | None,
    target_ccy: str | None
) -> float | None:
    """
    Compute the FX conversion factor from a source currency into a target currency.

    Parameters
    ----------
    source_ccy:
        Source currency code.
    target_ccy:
        Target currency code.

    Returns
    -------
    float | None
        A conversion factor ``k`` such that:
        value_in_target = value_in_source * k
        where k = (USD_per_source) / (USD_per_target).
        Returns ``None`` when either currency is unknown or when rates are unavailable.

    Mathematics (Text Form)
    -----------------------
    Let:
 
    - USD_per_X be the number of USD corresponding to 1 unit of currency X.
 
    Then:
 
    - value_in_USD = value_in_source * USD_per_source
 
    - value_in_target = value_in_USD / USD_per_target
 
    Therefore:
 
    - value_in_target = value_in_source * (USD_per_source / USD_per_target)
 
    """
    
    src = _normalize_ccy_code(
        ccy = source_ccy
    )

    tgt = _normalize_ccy_code(
        ccy = target_ccy
    )

    if src is None or tgt is None:

        return None

    if src == tgt:

        return 1.0

    usd_per = _latest_usd_per_ccy_map()

    u_src = usd_per.get(src, None)

    u_tgt = usd_per.get(tgt, None)

    if u_src is None or u_tgt is None:

        return None

    if not np.isfinite(u_src) or not np.isfinite(u_tgt) or u_src <= 0 or (u_tgt <= 0):

        return None

    return float(u_src / u_tgt)


def _infer_consensus_currency_from_pred_file(
    pred_file: str,
    max_rows: int = 500
) -> str | None:
    """
    Infer the consensus forecast currency from a prediction workbook.

    Parameters
    ----------
    pred_file:
        Path to the CapIQ consensus workbook.
    max_rows:
        Maximum number of rows to scan in the first column for embedded currency codes.

    Returns
    -------
    str | None
        A 3-letter currency code inferred from textual annotations in the sheet (typically parentheses,
        for example "(USD)" or "(EUR)"). Returns ``None`` when no such annotations are found.

    Method
    ------
    The first column is scanned for substrings matching the pattern "(CCY)" where CCY is a 3-letter
    token. The most common valid currency token is returned.
   
    """
    
    try:

        df_raw = _load_consensus_sheet(
            file_path = pred_file
        )

    except (FileNotFoundError, OSError, ValueError, ImportError):

        return None

    if df_raw is None or df_raw.empty:

        return None

    c0 = df_raw.iloc[:max_rows, 0].astype(str).fillna('')

    valid_ccy = set(_latest_usd_per_ccy_map().keys())

    counts: Counter[str] = Counter()

    for txt in c0.tolist():
      
        for m in _CCY_PAREN_PAT.finditer(txt):
      
            ccy = _normalize_ccy_code(
                ccy = m.group(1)
            )

            if ccy is None:

                continue

            if valid_ccy and ccy not in valid_ccy:

                continue

            counts[ccy] += 1

    if not counts:

        return None

    return counts.most_common(1)[0][0]


def _row_is_ratio_like(
    label: object
) -> bool:
    """
    Heuristically classify a row label as ratio-like rather than currency-valued.

    This helper is used to avoid applying FX or unit scaling to rows that represent:
  
    - percentages (contain '%'),
  
    - ratios (contain 'ratio' or '/'), or

    - rates (contain the word 'rate').

    Parameters
    ----------
    label:
        Row label, typically from a DataFrame index.

    Returns
    -------
    bool
        True when the label appears ratio-like.
    """
    
    s = str(label).lower()

    return '%' in s or ' ratio' in s or ' rate ' in f' {s} ' or ('/' in s)


def _scale_numeric_table_rows(
    df: pd.DataFrame | None,
    factor: float,
    *,
    skip_rows: set[str] | None = None,
    skip_ratio_like_rows: bool = False
) -> pd.DataFrame | None:
    """
    Multiply numeric rows of a table by a scaling factor, with optional row exclusions.

    Parameters
    ----------
    df:
        Input table whose rows are to be scaled.
    factor:
        Multiplicative scaling factor. A factor of 1.0 leaves the table unchanged.
    skip_rows:
        Set of row labels (case-insensitive) to exclude from scaling. This is commonly used to exclude
        metadata rows such as "period_type" or "No_of_Estimates".
    skip_ratio_like_rows:
        When True, rows whose labels appear ratio-like are excluded using ``_row_is_ratio_like``.

    Returns
    -------
    pandas.DataFrame | None
        A scaled copy of the input DataFrame. Returns the original input when scaling is unnecessary.

    Notes
    -----
    Scaling is applied only when the row contains at least one finite numeric value after coercion.
    """
    
    if df is None or df.empty:

        return df

    if not np.isfinite(factor) or abs(float(factor) - 1.0) <= 1e-12:

        return df

    out = df.copy()

    skip_rows_norm = {str(r).strip().lower() for r in skip_rows or set()}

    for i, r in enumerate(out.index):
    
        r_s = str(r).strip()

        if r_s.lower() in skip_rows_norm:

            continue

        if skip_ratio_like_rows and _row_is_ratio_like(
            label = r
        ):

            continue

        num = pd.to_numeric(out.iloc[i], errors = 'coerce')

        arr = num.to_numpy(dtype = float, copy = False)

        if np.isfinite(arr).any():

            out.iloc[i, :] = (num * factor).to_numpy()

    return out


def _scale_future_tables_currency(
    future: dict[str, pd.DataFrame],
    factor: float
) -> dict[str, pd.DataFrame]:
    """
    Apply a currency conversion factor to all future forecast tables that represent absolute values.

    Parameters
    ----------
    future:
        Mapping of driver key to forecast table.
    factor:
        Multiplicative conversion factor from the source currency to the target currency.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Scaled mapping, where:
  
        - ratio-like drivers (tax, gross margin, ROE, etc.) are left unchanged, and
  
        - metadata rows are excluded from scaling.

    Rationale
    ---------
    FX conversion is required to express all cashflow drivers in a consistent reporting currency prior to
    Monte Carlo simulation. Ratio drivers must not be scaled because they are dimensionless.
    """
    
    if not np.isfinite(factor) or abs(float(factor) - 1.0) <= 1e-12:

        return future

    out: dict[str, pd.DataFrame] = {}

    for k, df in future.items():
       
        if df is None or df.empty:

            out[k] = df

            continue

        if str(k).lower() in _FUTURE_RATIO_KEYS:

            out[k] = df

            continue

        out[k] = _scale_numeric_table_rows(
            df = df,
            factor = factor,
            skip_rows = _FUTURE_META_ROWS,
            skip_ratio_like_rows = False
        )

    return out


def _load_consensus_sheet(
    file_path: str
) -> pd.DataFrame:
    """
    Load the "Consensus" worksheet from a CapIQ consensus workbook.

    Parameters
    ----------
    file_path:
        Path to a CapIQ workbook. Both XLSX and legacy XLS formats are supported.

    Returns
    -------
    pandas.DataFrame
        Raw worksheet content with no header inference (``header=None``).

    Implementation Notes
    --------------------
    The file signature is inspected to determine the format:
 
    - XLSX files start with the ZIP header "PK\\x03\\x04" and are loaded using ``openpyxl``.
 
    - Legacy XLS files are loaded using ``xlrd`` and any library warnings are suppressed by redirecting
      stdout and stderr.
 
    """
 
    with open(file_path, 'rb') as f:
    
        sig = f.read(8)

    is_xlsx = sig.startswith(b'PK\x03\x04')

    if is_xlsx:

        return pd.read_excel(file_path, sheet_name = 'Consensus', header = None, engine = 'openpyxl')

    buf = io.StringIO()

    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
  
        return pd.read_excel(file_path, sheet_name = 'Consensus', header = None, engine = 'xlrd')


def _coerce_year(
    y
) -> float:
    """
    Coerce a year-like value into a four-digit year.

    Parameters
    ----------
    y:
        Year value as string/number, potentially 2-digit.

    Returns
    -------
    float
        Four-digit year (as float for compatibility with vectorised numeric operations), or NaN when
        coercion fails.

    Rule (Text Form)
    ----------------
    Two-digit years are expanded using a cut-over:
   
    - 00..69 map to 2000..2069
   
    - 70..99 map to 1970..1999
   
    """
  
    yy = pd.to_numeric(y, errors = 'coerce')

    if pd.isna(yy):

        return np.nan

    if yy < 100:

        return 2000.0 + yy if yy < 70 else 1900.0 + yy

    return yy


def _coerce_year_vec(
    y: pd.Series
) -> pd.Series:
    """
    Vectorised version of ``_coerce_year`` for a pandas Series.

    Parameters
    ----------
    y:
        Series of year-like values.

    Returns
    -------
    pandas.Series
        Numeric year series with two-digit years expanded to four digits.
    """
   
    yy = pd.to_numeric(y, errors = 'coerce')

    m = yy.notna() & (yy < 100)

    if m.any():

        yy.loc[m] = np.where(yy.loc[m] < 70, 2000.0 + yy.loc[m], 1900.0 + yy.loc[m])

    return yy


@lru_cache(maxsize = 32)
def _fy_end_month(
    fy_freq: str
) -> int:
    """
    Extract the fiscal year-end month from a pandas-style fiscal frequency code.

    Parameters
    ----------
    fy_freq:
        Frequency string such as "Y-MAR" or "A-DEC".

    Returns
    -------
    int
        Fiscal year-end month (1..12). Defaults to 12 (December) when parsing fails.
    """
    
    s = str(fy_freq or '').upper().strip()

    if s.startswith('A-') and len(s) == 5:

        m = s[-3:].title()

        return MONTH_ABBR.get(m.lower(), 12)

    if s.startswith('Y-') and len(s) == 5:

        m = s[-3:].title()

        return MONTH_ABBR.get(m.lower(), 12)

    return 12


def _fy_freq_from_month(
    fy_month: int
) -> str:
    """
    Construct a "Y-MMM" fiscal frequency code from a month number.

    Parameters
    ----------
    fy_month:
        Month number (1..12).

    Returns
    -------
    str
        Fiscal frequency string of the form "Y-JAN", "Y-FEB", ..., "Y-DEC".
    """
    
    m = int(fy_month) if np.isfinite(pd.to_numeric(fy_month, errors = 'coerce')) else 12

    m = int(np.clip(m, 1, 12))

    return f'Y-{calendar.month_abbr[m].upper()}'


def _is_valid_header_year(
    y
) -> bool:
    """
    Validate that a year-like value lies within a conservative header parsing range.

    Parameters
    ----------
    y:
        Year-like value.

    Returns
    -------
    bool
        True when y coerces to an integer in the closed interval [1900, 2100].

    Rationale
    ---------
    Header parsing encounters a variety of non-date artefacts (for example, row counts, IDs, or Excel
    serials interpreted as integers). A strict range reduces false positives.
    """
    
    yy = pd.to_numeric(y, errors = 'coerce')

    if pd.isna(yy):

        return False

    yi = int(yy)

    return _HEADER_YEAR_MIN <= yi <= _HEADER_YEAR_MAX


def _excel_serial_to_timestamp(
    v
):
    """
    Convert an Excel serial day number into a normalised timestamp.

    Parameters
    ----------
    v:
        Excel serial day value.

    Returns
    -------
    pandas.Timestamp | pandas.NaT
        Normalised period-end date derived from the Excel origin "1899-12-30". Returns NaT when the
        value is out of range or invalid.
    """
    
    if not np.isfinite(v):

        return pd.NaT

    if v < _EXCEL_SERIAL_MIN or v > _EXCEL_SERIAL_MAX:

        return pd.NaT

    dt = pd.to_datetime(v, unit = 'D', origin = '1899-12-30', errors = 'coerce')

    if pd.isna(dt):

        return pd.NaT

    ts = pd.Timestamp(dt).normalize()

    if not _is_valid_header_year(
        y = ts.year
    ):

        return pd.NaT

    return ts


def _parse_header_cells_to_dates_vec(
    cells: pd.Series,
    fy_freq: str = FY_FREQ
) -> pd.Series:
    """
    Vectorised parsing of worksheet header cells into period-end dates.

    Parameters
    ----------
    cells:
        Series of raw header cell values, typically drawn from a row that contains period labels.
    fy_freq:
        Fiscal frequency code used to infer the fiscal year-end month when only a year is present
        (for example, "Y-MAR").

    Returns
    -------
    pandas.Series
        A series of dtype ``datetime64[ns]`` containing normalised timestamps where parsing succeeds and
        NaT where parsing fails.

    Parsing Strategy
    ---------------
    The header formats encountered in CapIQ-style workbooks are heterogeneous. This routine applies a
    layered parsing approach:
   
    1) Excel serial numbers in a conservative range are converted using origin "1899-12-30".
   
    2) Free-form date parsing is attempted on non-numeric strings (format="mixed").
   
    3) Pure years (for example, "2027" or numeric 2027) are mapped to fiscal year-end dates using
       the inferred fiscal year-end month.
   
    4) Fiscal year tokens ("FY27") and year tags ("2027E") are mapped similarly.
   
    5) Month-year tokens ("Mar 27") are parsed as month-ends.
   
    6) Quarter tokens ("FQ1 27", "CQ3 2027") are mapped to quarter-end dates using either fiscal or
       calendar quarter conventions.

    Special Handling
    ---------------
    Headers containing "12 months" are sometimes represented as the first day of the month in upstream
    exports. When this pattern is detected (and the label is not a trailing-twelve-month marker), the
    date is shifted back by one day to represent a month-end.

    Advantages
    ----------
    Vectorised parsing substantially reduces overhead compared with per-cell parsing and improves
    robustness by explicitly handling common CapIQ header artefacts.
    """
    
    raw = cells.copy()

    s = raw.astype('string').fillna('')

    s = s.str.replace('\xa0', ' ', regex = False).str.replace('\r\n', '\n', regex = False).str.replace('\r', '\n', regex = False).str.strip()

    s = s.str.replace('\\n+', '\n', regex = True)

    s = s.str.rsplit('\n', n = 1).str[-1].str.strip()

    s = s.str.replace('(?i)\\b(?:reclassified|restated)\\b', '', regex = True)

    s = s.str.replace('(?i)\\b(?:\\d{1,2}\\s*months?)\\b', '', regex = True)

    s = s.str.replace('(?i)^\\s*as\\s+of\\s+', '', regex = True)

    s = s.str.replace('\\s+', ' ', regex = True).str.strip()

    dt = pd.Series(pd.NaT, index = s.index, dtype = 'datetime64[ns]')

    numeric = pd.to_numeric(raw, errors = 'coerce')

    fy_end_m = _fy_end_month(
        fy_freq = fy_freq
    )

    serial_mask = numeric.notna() & numeric.between(_EXCEL_SERIAL_MIN, _EXCEL_SERIAL_MAX)

    if serial_mask.any():

        dt_serial = pd.to_datetime(numeric.where(serial_mask), unit = 'D', origin = '1899-12-30', errors = 'coerce')

        dt_serial = dt_serial.where(dt_serial.dt.year.between(_HEADER_YEAR_MIN, _HEADER_YEAR_MAX))

        m = dt_serial.notna()

        if m.any():

            dt.loc[m] = dt_serial.loc[m].dt.normalize()

    plain_num = s.str.fullmatch('[+-]?\\d+(?:\\.\\d+)?')

    dt_text = pd.to_datetime(s.mask(plain_num, ''), errors = 'coerce', format = 'mixed')

    dt_text = dt_text.where(dt_text.dt.year.between(_HEADER_YEAR_MIN, _HEADER_YEAR_MAX))

    m = dt.isna() & dt_text.notna()

    if m.any():

        dt.loc[m] = dt_text.loc[m].dt.normalize()

    miss = dt.isna()

    if miss.any():

        rounded = np.round(numeric)

        m = miss & numeric.notna() & np.isclose(numeric, rounded, atol = e6) & rounded.between(_HEADER_YEAR_MIN, _HEADER_YEAR_MAX)

        if m.any():

            base = pd.to_datetime(dict(year = rounded.loc[m].astype(int), month = fy_end_m, day = 1), errors = 'coerce')

            dt.loc[m] = base + pd.offsets.MonthEnd(0)

    miss = dt.isna()

    if miss.any():

        y_only = s.str.extract('(?i)^\\s*([12]\\d{3})(?:\\s*[A-Z]{1,3})?\\s*$')[0]

        y_only = pd.to_numeric(y_only, errors = 'coerce')

        m = miss & y_only.notna() & (y_only >= _HEADER_YEAR_MIN) & (y_only <= _HEADER_YEAR_MAX)

        if m.any():

            base = pd.to_datetime(dict(year = y_only.loc[m].astype(int), month = fy_end_m, day = 1), errors = 'coerce')

            dt.loc[m] = base + pd.offsets.MonthEnd(0)

    miss = dt.isna()

    if miss.any():

        y_tag = s.str.extract('(?i)\\b([12]\\d{3})\\s*(?:E|A|P|F)\\b')[0]

        y_tag = pd.to_numeric(y_tag, errors = 'coerce')

        m = miss & y_tag.notna() & (y_tag >= _HEADER_YEAR_MIN) & (y_tag <= _HEADER_YEAR_MAX)

        if m.any():

            base = pd.to_datetime(dict(year = y_tag.loc[m].astype(int), month = fy_end_m, day = 1), errors = 'coerce')

            dt.loc[m] = base + pd.offsets.MonthEnd(0)

    miss = dt.isna()

    if miss.any():

        my = s.str.extract('(?i)^\\s*([A-Za-z]{3})[-\\s]+(\\d{2,4})\\s*$')

        mon = my[0].str.lower().map(MONTH_ABBR)

        yy = _coerce_year_vec(
            y = my[1]
        )

        m = miss & mon.notna() & yy.notna() & yy.between(_HEADER_YEAR_MIN, _HEADER_YEAR_MAX)

        if m.any():

            base = pd.to_datetime(dict(year = yy.loc[m].astype(int), month = mon.loc[m].astype(int), day = 1), errors = 'coerce')

            dt.loc[m] = base + pd.offsets.MonthEnd(0)

    miss = dt.isna()

    if miss.any():

        fy = s.str.extract("(?i)\\bFY\\s*[’']?\\s*(\\d{2,4})(?:\\s*[A-Z]{1,3})?\\b")[0]

        yy = _coerce_year_vec(
            y = fy
        )

        m = miss & yy.notna() & yy.between(_HEADER_YEAR_MIN, _HEADER_YEAR_MAX)

        if m.any():

            base = pd.to_datetime(dict(year = yy.loc[m].astype(int), month = fy_end_m, day = 1), errors = 'coerce')

            dt.loc[m] = base + pd.offsets.MonthEnd(0)

    miss = dt.isna()

    if miss.any():

        qx = s.str.extract("(?i)\\b(?P<tag>FQ|CQ|Q)\\s*(?P<q>[1-4])(?:[^0-9]{0,4}(?:FY)?\\s*)?[’']?\\s*(?P<y>\\d{2,4})\\b")

        tag = qx['tag'].astype('string').str.upper()

        qq = pd.to_numeric(qx['q'], errors = 'coerce')

        yy = _coerce_year_vec(
            y = qx['y']
        )

        m = miss & tag.notna() & qq.notna() & yy.notna() & yy.between(_HEADER_YEAR_MIN, _HEADER_YEAR_MAX)

        if m.any():

            qq_i = qq.loc[m].astype(int)

            yy_i = yy.loc[m].astype(int)

            cal = tag.loc[m] == 'CQ'

            if cal.any():

                mm = qq_i.loc[cal] * 3

                base = pd.to_datetime(dict(year = yy_i.loc[cal], month = mm, day = 1), errors = 'coerce')

                dt.loc[m.index[m][cal.to_numpy()]] = (base + pd.offsets.MonthEnd(0)).to_numpy()

            fisc = ~cal

            if fisc.any():

                qqf = qq_i.loc[fisc]

                yyf = yy_i.loc[fisc]

                mm = (fy_end_m - 3 * (4 - qqf) - 1) % 12 + 1

                yf = np.where(mm <= fy_end_m, yyf, yyf - 1).astype(int)

                base = pd.to_datetime(dict(year = yf, month = mm, day = 1), errors = 'coerce')

                dt.loc[m.index[m][fisc.to_numpy()]] = (base + pd.offsets.MonthEnd(0)).to_numpy()

    raw_l = raw.astype('string').fillna('').str.replace('\xa0', ' ', regex = False).str.lower()

    has_12m = raw_l.str.contains('\\b12\\s*months?\\b', regex = True, na = False)

    has_ttm = raw_l.str.contains('\\b(?:ttm|ltm|trailing\\s*12|trailing\\s*twelve)\\b', regex = True, na = False)

    shift_prev_day = dt.notna() & has_12m & ~has_ttm & (dt.dt.day == 1)

    if shift_prev_day.any():

        dt.loc[shift_prev_day] = (dt.loc[shift_prev_day] - pd.Timedelta(days = 1)).dt.normalize()

    return dt


@lru_cache(maxsize = 256)
def _to_datetime_index_cached(
    cols_key: tuple[str, ...]
) -> pd.DatetimeIndex:
    """
    Cached conversion of an iterable of column labels into a normalised ``DatetimeIndex``.

    Parameters
    ----------
    cols_key:
        Tuple of stringified column labels.

    Returns
    -------
    pandas.DatetimeIndex
        Normalised datetime index with invalid entries coerced to NaT.

    Notes
    -----
    This function exists to memoise conversions for repeated alignments using the same header labels.
    """
    
    dt = pd.to_datetime(list(cols_key), errors = 'coerce')

    return pd.DatetimeIndex(dt).normalize()


def _to_datetime_index(
    cols
) -> pd.DatetimeIndex:
    """
    Convert an arbitrary column index to a normalised ``DatetimeIndex`` with caching.

    Parameters
    ----------
    cols:
        Column labels in any pandas-supported index form (Index, DatetimeIndex, list-like).

    Returns
    -------
    pandas.DatetimeIndex
        Normalised datetime index derived from the input.
    """
    
    if isinstance(cols, pd.DatetimeIndex):

        return cols.normalize()

    idx = pd.Index(cols)

    if isinstance(idx, pd.DatetimeIndex):

        return idx.normalize()

    key = tuple(('' if pd.isna(c) else str(c) for c in idx.tolist()))

    return _to_datetime_index_cached(
        cols_key = key
    )


def _quarter_end_for_fy(
    fq: int,
    fy_year: int,
    M: int
) -> pd.Timestamp:
    """
    Compute the quarter-end timestamp for a fiscal quarter within a fiscal year.

    Parameters
    ----------
    fq:
        Fiscal quarter number in {1, 2, 3, 4}.
    fy_year:
        Fiscal year label (calendar year of the fiscal year-end).
    M:
        Fiscal year-end month number (1..12).

    Returns
    -------
    pandas.Timestamp
        Normalised quarter-end timestamp corresponding to the fiscal quarter.

    Mathematics (Text Form)
    -----------------------
    The quarter-end month is determined by stepping back 0, 3, 6, or 9 months from the fiscal year-end
    month for FQ4, FQ3, FQ2, and FQ1 respectively (with wrap-around). The year is adjusted such that
    months greater than the fiscal year-end month belong to the previous calendar year.
    """
    
    q_months = {4: M, 3: (M - 3 - 1) % 12 + 1, 2: (M - 6 - 1) % 12 + 1, 1: (M - 9 - 1) % 12 + 1}

    m = q_months[fq]

    y = fy_year if m <= M else fy_year - 1

    return pd.Timestamp(year = y, month = m, day = 1) + pd.offsets.MonthEnd(0)


@lru_cache(None)
def _parse_header_cell_to_date(
    cell: str,
    fy_freq: str = FY_FREQ
) -> pd.Timestamp | None:
    """
    Parse a single worksheet header cell into a normalised period-end date.

    Parameters
    ----------
    cell:
        Header cell value, which may be numeric (Excel serial), a plain year, a date string, or a
        fiscal/quarter token.
    fy_freq:
        Fiscal frequency code used to infer fiscal year-end month when only a year is present.

    Returns
    -------
    pandas.Timestamp | pandas.NaT
        Parsed normalised timestamp when successful; otherwise NaT.

    Supported Header Forms (Non-exhaustive)
    --------------------------------------
    
    - Excel serials within a conservative range.
   
    - ISO-like dates ("2027-03-31") or locale-like dates ("31/03/2027").
   
    - Month-year tokens ("Mar 2027") mapped to month-end.
   
    - Fiscal year tokens ("FY27", "FY 2027E") mapped to fiscal year-end month-end.
   
    - Quarter tokens:
   
      - calendar quarter ("CQ1 2027") mapped to calendar quarter-end, and
   
      - fiscal quarter ("FQ1 2027") mapped using the fiscal year-end month.

    Notes
    -----
    Header cells may contain multi-line text. In such cases, the last non-empty line is preferred.
    Trailing-twelve-month markers (TTM/LTM) are removed prior to parsing.
    """
   
    if pd.isna(cell):

        return pd.NaT

    fy_end_m = _fy_end_month(
        fy_freq = fy_freq
    )

    if isinstance(cell, (int, np.integer, float, np.floating)) and (not isinstance(cell, bool)):

        v = float(cell)

        serial_dt = _excel_serial_to_timestamp(
            v = v
        )

        if pd.notna(serial_dt):

            return serial_dt

        iv = int(round(v))

        if abs(v - iv) <= e6 and _is_valid_header_year(
            y = iv
        ):

            return pd.Timestamp(year = iv, month = fy_end_m, day = 1) + pd.offsets.MonthEnd(0)

        return pd.NaT

    s = str(cell).replace('\xa0', ' ').strip()

    s = s.replace('\r\n', '\n').replace('\r', '\n')

    if not s:

        return pd.NaT

    if '\n' in s:

        parts = [p.strip() for p in s.split('\n') if p.strip()]

        for t in reversed(parts):
            
            dt = _parse_header_cell_to_date(
                cell = t,
                fy_freq = fy_freq
            )

            if pd.notna(dt):

                return dt

        return pd.NaT

    s_l = s.lower()

    if any((tag in s_l for tag in ['ttm', 'ltm', 'trailing 12', 'trailing twelve'])):

        s = re.sub('\\b(?:TTM|LTM|Trailing\\s*12(?:\\s*Months)?|Trailing\\s*Twelve(?:\\s*Months)?)\\b', '', s, flags = re.IGNORECASE)

        s = re.sub('\\s+', ' ', s).strip()

        if not s:

            return pd.NaT

        s_l = s.lower()

    if s_l.startswith('as of'):

        tail = s.split('of', 1)[-1].strip()

        return _parse_header_cell_to_date(
            cell = tail,
            fy_freq = fy_freq
        )

    if re.fullmatch('[+-]?\\d+(?:\\.\\d+)?', s):

        try:

            v = float(s)

        except (TypeError, ValueError):

            return pd.NaT

        serial_dt = _excel_serial_to_timestamp(
            v = v
        )

        if pd.notna(serial_dt):

            return serial_dt

        iv = int(round(v))

        if abs(v - iv) <= e6 and _is_valid_header_year(
            y = iv
        ):

            return pd.Timestamp(year = iv, month = fy_end_m, day = 1) + pd.offsets.MonthEnd(0)

        return pd.NaT

    pats_full = ['([A-Za-z]{3})[-\\s](\\d{1,2})[-\\s](\\d{2,4})', '(\\d{4})[-/](\\d{1,2})[-/](\\d{1,2})', '(\\d{1,2})/(\\d{1,2})/(\\d{4})']

    for p in pats_full:
   
        m = re.search(p, s, flags = re.IGNORECASE)

        if m:

            try:

                out = pd.to_datetime(m.group(0), errors = 'raise')

                out = pd.Timestamp(out).normalize()

                if _is_valid_header_year(
                    y = out.year
                ):

                    return out

                return pd.NaT

            except (TypeError, ValueError, KeyError):

                pass

    m2 = re.search('([A-Za-z]{3})[-\\s](\\d{2,4})', s, flags = re.IGNORECASE)

    if m2:

        mon = MONTH_ABBR.get(m2.group(1).lower(), None)

        if mon:

            yy = _coerce_year(
                y = m2.group(2)
            )

            if np.isfinite(yy) and _is_valid_header_year(
                y = int(yy)
            ):

                return pd.Timestamp(year = int(yy), month = mon, day = 1) + pd.offsets.MonthEnd(0)

            return pd.NaT

    m2y = re.search('(?i)^\\s*([12]\\d{3})(?:\\s*[A-Z]{1,3})?\\s*$', s)

    if m2y:

        y = _coerce_year(
            y = m2y.group(1)
        )
        
        y = int(y)

        if np.isfinite(y) and _is_valid_header_year(
            y = y
        ):

            return pd.Timestamp(year = y, month = fy_end_m, day = 1) + pd.offsets.MonthEnd(0)

        return pd.NaT

    m2y_tag = re.search('(?i)\\b([12]\\d{3})\\s*(?:E|A|P|F)\\b', s)

    if m2y_tag:

        y = _coerce_year(
            y = m2y_tag.group(1)
        )

        y = int(y)
        
        if np.isfinite(y) and _is_valid_header_year(
            y = y
        ):

            return pd.Timestamp(year = y, month = fy_end_m, day = 1) + pd.offsets.MonthEnd(0)

        return pd.NaT

    m3 = re.search("(?:^|\\b)(FQ|CQ|Q)\\s*([1-4])(?:[^0-9]{0,4}(?:FY)?\\s*)?([’\\']?\\s*\\d{2,4})", s, flags = re.IGNORECASE)

    if m3:

        qtag, qnum, yraw = m3.groups()

        q = int(qnum)

        y = _coerce_year(
            y = re.sub('[^0-9]', '', yraw)
        )

        y = int(y)
        
        if not np.isfinite(y) or not _is_valid_header_year(
            y = y
        ):

            return pd.NaT

        if qtag.upper() == 'CQ' or (qtag.upper() == 'Q' and 'CQ' in s.upper()):

            cal_q_months = {1: 3, 2: 6, 3: 9, 4: 12}

            m = cal_q_months[q]

            return pd.Timestamp(year = y, month = m, day = 1) + pd.offsets.MonthEnd(0)

        return _quarter_end_for_fy(
            fq = q,
            fy_year = y,
            M = fy_end_m
        )

    m4 = re.search("\\bFY\\s*([’\\']?\\s*\\d{2,4})(?:\\s*[A-Z]{1,3})?\\b", s, flags = re.IGNORECASE)

    if m4:

        y = _coerce_year(
            y = re.sub('[^0-9]', '', m4.group(1))
        )
        
        y = int(y)

        if not np.isfinite(y) or not _is_valid_header_year(
            y = y
        ):

            return pd.NaT

        m = fy_end_m

        return pd.Timestamp(year = y, month = m, day = 1) + pd.offsets.MonthEnd(0)

    return pd.NaT


def _clean_col(
    c
):
    """
    Clean and coerce a worksheet column into numeric values.

    Parameters
    ----------
    c:
        A pandas Series representing a column extracted from a raw sheet (strings, numbers, blanks).

    Returns
    -------
    pandas.Series
        Numeric series coerced with ``errors="coerce"``.

    Cleaning Rules
    --------------
    - Non-breaking spaces are removed.
   
    - Comma thousands separators are removed.
   
    - Parentheses are treated as negative numbers, for example "(123)" becomes "-123".
   
    - Percent signs are removed; the returned values remain in "percentage points" unless converted
      later by a caller.
    """
   
    s = c.astype(str).str.replace('\xa0', '', regex = False)

    s = s.str.replace(',', '')

    s = s.str.replace('^\\((.*)\\)$', '-\\1', regex = True)

    s = s.str.replace('%', '')

    return pd.to_numeric(s, errors = 'coerce')


def _parse_sheet_robust(
    xls: pd.ExcelFile,
    sheet_name: str,
    fy_freq: str = FY_FREQ,
    header_scan_rows: int = 60
) -> pd.DataFrame:
    """
    Parse a financial statement sheet with robust date-header detection.

    Parameters
    ----------
    xls:
        Open ``pandas.ExcelFile`` handle.
    sheet_name:
        Name of the sheet to parse.
    fy_freq:
        Fiscal frequency used for parsing year-only headers into fiscal year-end month-ends.
    header_scan_rows:
        Maximum number of top rows to scan for a plausible header row.

    Returns
    -------
    pandas.DataFrame
        Parsed table indexed by metric name, with datetime columns sorted ascending. Values are coerced
        to numeric where possible.

    Method
    ------
    1) The sheet is loaded with ``header=None`` to avoid premature header inference.
  
    2) The first ``header_scan_rows`` rows are scanned; a row is treated as a header when at least three
       date-like values can be parsed in columns 1..end.
  
    3) Only columns with parseable dates are retained.
  
    4) The first retained column is treated as the metric name, and remaining columns are the parsed
       dates.
  
    5) Numeric coercion is applied column-wise using ``_clean_col``.

    Advantages
    ----------
    This routine tolerates messy exports where:
  
    - header rows are not in the first row,
  
    - period labels contain mixed text and numeric formats, and
  
    - columns include non-date artefacts that must be dropped.
  
    """
  
    raw = pd.read_excel(xls, sheet_name = sheet_name, header = None)

    if raw.empty:

        return pd.DataFrame()

    header_row = None

    header_dates = None

    scan = min(header_scan_rows, len(raw))

    for r in range(scan):
        
        dates = _parse_header_cells_to_dates_vec(
            cells = raw.iloc[r, 1:],
            fy_freq = fy_freq
        )

        ok = dates.notna().to_numpy()

        if ok.sum() >= 3:

            header_row = r

            header_dates = dates

            break

    if header_row is None:

        header_row = 0

        header_dates = _parse_header_cells_to_dates_vec(
            cells = raw.iloc[0, 1:],
            fy_freq = fy_freq
        )

    keep = np.concatenate([[True], header_dates.notna().to_numpy()])

    use_cols = np.flatnonzero(keep)

    df = raw.iloc[header_row + 1:, use_cols].copy()

    cols = ['Metric'] + header_dates[header_dates.notna()].tolist()

    df.columns = cols

    df = df.dropna(how = 'all')

    df['Metric'] = df['Metric'].astype(str).str.replace('\xa0', ' ', regex = False).str.replace('\\s+', ' ', regex = True).str.strip()

    df = df.set_index('Metric')

    if df.shape[1] > 0:

        cleaned = df.apply(_clean_col, axis = 0)

        df = cleaned

    if df.columns.has_duplicates:

        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep = 'last')]

    df = df.sort_index(axis = 1)

    return df


def _pick_sheet(
    xls: pd.ExcelFile,
    want: str
) -> str:
    """
    Select a sheet name from an Excel file using exact or substring matching.

    Parameters
    ----------
    xls:
        Open ``pandas.ExcelFile`` handle.
    want:
        Desired sheet name (for example, "Income Statement").

    Returns
    -------
    str
        The actual sheet name present in the workbook.

    Matching Rules
    --------------
    - Case-insensitive exact match is preferred.
 
    - If no exact match exists, the shortest case-insensitive substring match is returned.
 
    - A ``KeyError`` is raised when no candidate is found.
 
    """
 
    want_norm = want.strip().lower()

    names_norm = {sh.strip().lower(): sh for sh in xls.sheet_names}

    if want_norm in names_norm:

        return names_norm[want_norm]

    cand = [real for norm, real in names_norm.items() if want_norm in norm]

    if cand:

        return sorted(cand, key = len)[0]

    raise KeyError(f"Sheet named '{want}' not found. Available: {xls.sheet_names}")


def _date_gap_stats(
    cols
):
    """
    Compute simple gap statistics for timestamp columns.

    Parameters
    ----------
    cols:
        Iterable of column labels, some of which may be ``pandas.Timestamp``.

    Returns
    -------
    tuple[float, float, int] | None
        (median_gap_days, mean_gap_days, n_timestamps) or ``None`` if fewer than two timestamps exist.

    Notes
    -----
    This helper is used to distinguish quarterly-like from annual-like time series based on period
    spacing.
    """
    
    ts = [c for c in cols if isinstance(c, pd.Timestamp)]

    lts = len(ts)

    if lts < 2:

        return None

    ts = sorted(ts)

    gaps = np.diff([t.value for t in ts]) / 86400000000

    return (np.median(gaps), np.mean(gaps), lts)


def _is_quarterly_like(
    cols
) -> bool:
    """
    Heuristically determine whether a set of date columns resembles a quarterly series.

    Parameters
    ----------
    cols:
        Iterable of column labels.

    Returns
    -------
    bool
        True when the median gap between timestamps is approximately quarterly.

    Decision Rule (Text Form)
    -------------------------
    Quarterly-like is defined as:
   
    - at least 8 timestamp columns, and
   
    - median gap between consecutive timestamps in [60, 120] days.
   
    """
   
    s = _date_gap_stats(
        cols = cols
    )

    if not s:

        return False

    med, mean, n = s

    return n >= 8 and 60.0 <= med <= 120.0


def load_statements_named(
    xls_path: Path,
    inc_name: str = 'Income Statement',
    bal_name: str = 'Balance Sheet',
    cf_name: str = 'Cash Flow',
    ratio_name: str = 'Ratios',
    ensure_quarterly: bool = False
):
    """
    Load standard financial statement tables from a workbook by sheet name.

    Parameters
    ----------
    xls_path:
        Path to the financial statements workbook.
    inc_name, bal_name, cf_name, ratio_name:
        Desired sheet names (or substrings) for the income statement, balance sheet, cashflow statement,
        and ratios sheet.
    ensure_quarterly:
        When True, raises if the parsed statements do not resemble quarterly data based on date-gap
        heuristics.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        (income_statement, balance_sheet, cash_flow, ratios).

    Notes
    -----
    Sheet parsing is performed by ``_parse_sheet_robust``, which attempts to detect and parse
    date-like header rows.
    """
    
    xls = pd.ExcelFile(xls_path)

    inc_sheet = _pick_sheet(
        xls = xls,
        want = inc_name
    )

    bal_sheet = _pick_sheet(
        xls = xls,
        want = bal_name
    )

    cf_sheet = _pick_sheet(
        xls = xls,
        want = cf_name
    )

    rat_sheet = _pick_sheet(
        xls = xls,
        want = ratio_name
    )

    inc = _parse_sheet_robust(
        xls = xls,
        sheet_name = inc_sheet
    )

    bal = _parse_sheet_robust(
        xls = xls,
        sheet_name = bal_sheet
    )

    cf = _parse_sheet_robust(
        xls = xls,
        sheet_name = cf_sheet
    )

    ratios = _parse_sheet_robust(
        xls = xls,
        sheet_name = rat_sheet
    )

    if ensure_quarterly:

        for kind, df in (('Income', inc), ('Cash Flow', cf), ('Balance', bal)):
       
            if not _is_quarterly_like(
                cols = df.columns
            ):

                raise ValueError(f"""{kind} sheet '{kind}' does not look quarterly (median gap stats = {_date_gap_stats(
                    cols = df.columns
                )}). Check you picked the right sheet.""")

    return (inc, bal, cf, ratios)


def _norm_label(
    x
) -> str:
    """
    Normalise a label for robust matching.

    Parameters
    ----------
    x:
        Input object (string-like or NaN).

    Returns
    -------
    str
        Lower-cased label with non-breaking spaces removed and internal whitespace collapsed.

    Rationale
    ---------
    CapIQ exports frequently contain non-breaking spaces and inconsistent spacing. Normalisation enables
    stable matching of metric labels across workbooks.
    """
    
    s = '' if pd.isna(x) else str(x)

    s = s.replace('\xa0', ' ').strip()

    s = re.sub('\\s+', ' ', s)

    return s.lower()


def _build_colmap(
    df_raw,
    header_row,
    start_col = 1
):
    """
    Construct a mapping from raw column indices to header labels for a given header row.

    Parameters
    ----------
    df_raw:
        Raw worksheet DataFrame.
    header_row:
        Row index believed to contain period labels.
    start_col:
        First column index to consider (default: 1), as column 0 typically contains metric names.

    Returns
    -------
    dict[int, object]
        Mapping of column indices to non-empty header cell values.
    """
    
    colmap = {}

    for j in range(start_col, df_raw.shape[1]):
       
        lab = df_raw.iloc[header_row, j]

        if pd.isna(lab) or lab == '':

            continue

        colmap[j] = lab

    return colmap


def _year_of(
    v
):
    """
    Extract a four-digit year from a header-like value.

    Parameters
    ----------
    v:
        Header cell value (numeric, string, etc.).

    Returns
    -------
    int | None
        Four-digit year in (1900, 2100) when identifiable; otherwise ``None``.

    Notes
    -----
    Values that appear to be quarter tokens (for example "FQ1", "Q3", "1Q") are explicitly rejected to
    avoid misclassifying quarterly headers as annual years.
    """
    
    if isinstance(v, (int, np.integer)):

        iv = int(v)

        return iv if 1900 < iv < 2100 else None

    if isinstance(v, (float, np.floating)) and np.isfinite(v):

        iv = int(round(v))

        if abs(v - iv) < e6 and 1900 < iv < 2100:

            return iv

    if isinstance(v, str):

        s = v.strip().replace('\xa0', ' ')

        if not s:

            return None

        if re.search('(?i)\\b(?:FQ|CQ|Q[1-4]|[1-4]Q|FH|H[12])\\b', s):

            return None

        if s.isdigit():

            iv = int(s)

            return iv if 1900 < iv < 2100 else None

        m_fy = re.search("(?i)\\bFY\\s*[’']?\\s*(\\d{2,4})(?:\\s*[A-Z]{1,3})?\\b", s)

        if m_fy:

            yy = _coerce_year(
                y = re.sub('[^0-9]', '', m_fy.group(1))
            )

            if np.isfinite(yy):

                iv = int(yy)

                return iv if 1900 < iv < 2100 else None

        m_year = re.search('(?i)^\\s*([12]\\d{3})(?:\\s*[A-Z]{1,3})?\\s*$', s)

        if m_year:

            iv = int(m_year.group(1))

            return iv if 1900 < iv < 2100 else None

        m_tag = re.search('(?i)\\b([12]\\d{3})\\s*(?:E|A|P|F)\\b', s)

        if m_tag:

            iv = int(m_tag.group(1))

            return iv if 1900 < iv < 2100 else None

    return None


def _is_quarter_label(
    v
) -> bool:
    """
    Identify whether a string label appears to refer to a quarter or half-year period.

    Parameters
    ----------
    v:
        Candidate label.

    Returns
    -------
    bool
        True when the label contains common quarter/half-year tokens.

    Notes
    -----
    Both explicit fiscal tags (for example "FQ") and generic quarter patterns (for example "Q1", "1Q")
    are supported.
    """
    
    if not isinstance(v, str):

        return False

    s = v.upper()

    if 'FQ' in s or 'FH' in s:

        return True

    if 'QUARTER' in s:

        return True

    return _Q_PAT.search(s) is not None


def _classify_header_row(
    df_raw,
    header_row,
    *,
    fy_freq: str = FY_FREQ
):
    """
    Classify a potential header row as "annual", "quarterly", or None.

    Parameters
    ----------
    df_raw:
        Raw consensus worksheet.
    header_row:
        Row index to evaluate.
    fy_freq:
        Fiscal frequency used when parsing dates.

    Returns
    -------
    str | None
        "annual", "quarterly", or ``None`` when the row does not resemble a period header.

    Method
    ------
    The classification uses a combination of signals:
   
    - the proportion of cells that resemble years,
   
    - the proportion of cells that resemble quarter tokens,
   
    - successful parsing of cells into timestamps, and
   
    - the median gap between parsed timestamps (approximately 90 days suggests quarterly; approximately
      one year suggests annual).

    Advantages
    ----------
    A hybrid rule-set is more robust than relying on a single parsing method because CapIQ exports can
    present either years, explicit dates, or quarter codes depending on the metric block.
    """
    
    c0 = _norm_label(
        x = df_raw.iloc[header_row, 0]
    )

    metrics_norm = {_norm_label(
        x = m
    ) for m in METRICS}

    if c0 in metrics_norm:

        return None

    vals = []

    years = []

    parsed_dates = []

    qcount = 0

    for j in range(1, df_raw.shape[1]):
      
        v = df_raw.iloc[header_row, j]

        if pd.isna(v) or v == '':

            continue

        vals.append(v)

        y = _year_of(
            v = v
        )

        if y is not None:

            years.append(y)

        dt = _parse_header_cell_to_date(
            cell = v,
            fy_freq = fy_freq
        )

        if pd.notna(dt):

            ts = pd.Timestamp(dt).normalize()

            if _is_valid_header_year(
                y = ts.year
            ):

                parsed_dates.append(ts)

        if _is_quarter_label(v = v):

            qcount += 1

    if len(vals) < 2:

        return None

    year_ratio = len(years) / len(vals)

    q_ratio = qcount / len(vals)

    if qcount >= 2 and q_ratio >= 0.25:

        return 'quarterly'

    if len(set(years)) >= 3 and year_ratio >= 0.45 and (qcount == 0):

        seq = []

        for j in range(1, df_raw.shape[1]):
        
            y = _year_of(
                v = df_raw.iloc[header_row, j]
            )

            if y is not None:

                seq.append(y)

        if len(seq) >= 3:

            diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]

            if all((d >= 0 for d in diffs)):

                pos = [d for d in diffs if d > 0]

                if not pos:

                    return 'annual'

                tight_share = sum((d <= 2 for d in pos)) / len(pos)

                if tight_share >= 0.6 and max(pos) <= 6:

                    return 'annual'

    if len(parsed_dates) >= 3:

        safe_dates: list[pd.Timestamp] = []

        for d in parsed_dates:
            try:

                ts = pd.Timestamp(d).normalize()

            except (ValueError, OverflowError, TypeError):

                continue

            if _is_valid_header_year(
                y = ts.year
            ):

                safe_dates.append(ts)

        if not safe_dates:

            return None

        d_unique = pd.DatetimeIndex(pd.Series(safe_dates, dtype = 'object').drop_duplicates().sort_values())

        if len(d_unique) >= 3:

            gaps = np.diff(d_unique.asi8) / 86400000000.0

            if len(gaps):

                med_gap = float(np.nanmedian(gaps))

                if 60.0 <= med_gap <= 120.0:

                    return 'quarterly'

                if qcount == 0 and 240.0 <= med_gap <= 450.0:

                    return 'annual'

    return None


def _find_period_header_above(
    df_raw,
    base_row,
    max_scan_up = 300,
    *,
    fy_freq: str = FY_FREQ
):
    """
    Search upwards from a row for the nearest plausible period header row.

    Parameters
    ----------
    df_raw:
        Raw worksheet.
    base_row:
        Starting row index (exclusive).
    max_scan_up:
        Maximum number of rows to scan upwards.
    fy_freq:
        Fiscal frequency used for date parsing.

    Returns
    -------
    tuple[int | None, str | None, dict[int, object] | None]
        (header_row_index, kind, colmap) where kind is "annual" or "quarterly". Returns
        (None, None, None) when no suitable header is found.
    """
    
    start = max(0, base_row - max_scan_up)

    for h in range(base_row - 1, start - 1, -1):
       
        if df_raw.iloc[h, 1:].notna().sum() < MIN_HEADER_NONNA:

            continue

        kind = _classify_header_row(
            df_raw = df_raw,
            header_row = h,
            fy_freq = fy_freq
        )

        if kind in {'annual', 'quarterly'}:

            return (h, kind, _build_colmap(
                df_raw = df_raw,
                header_row = h
            ))

    return (None, None, None)


def clean_series_vectorized(
    series
):
    """
    Vectorised cleaning of a row/series of worksheet values into numeric form.

    Parameters
    ----------
    series:
        Series of raw cell values.

    Returns
    -------
    pandas.Series
        Numeric series with common textual artefacts removed.

    Cleaning Rules
    --------------
    - Parentheses are treated as negative numbers.
   
    - Comma separators and percent signs are removed.
   
    - Placeholder tokens ("-", "—", empty strings) are treated as missing.
   
    """
   
    s = series.astype(str).str.replace('\xa0', ' ', regex = False).str.strip()

    s = s.str.replace('^\\((.*)\\)$', '-\\1', regex = True)

    s = s.str.replace(',', '', regex = False).str.replace('%', '', regex = False)

    s = s.mask(s.isin(['-', '—', '']))

    return pd.to_numeric(s, errors = 'coerce')


def _extract_metrics_df(
    df_raw,
    base_row,
    colmap,
    value_name: str,
    include_base_row = True
):
    """
    Extract a metric block (value row plus standard CapIQ metric rows) into a tidy DataFrame.

    Parameters
    ----------
    df_raw:
        Raw consensus worksheet DataFrame.
    base_row:
        Row index of the metric label (the row immediately preceding the standard metric rows).
    colmap:
        Mapping of column indices to period labels for the associated header row.
    value_name:
        Name to assign to the primary value row (for example, "Free_Cash_Flow").
    include_base_row:
        When True, includes the base metric row itself as the first numeric row.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by period labels (as provided in the header), with columns:
        value_name, Median, High, Low, Std_Dev, No_of_Estimates

    Notes
    -----
    The "No. of Estimates" row is retained as string-like content where appropriate because upstream
    exports may contain non-numeric placeholders.
    
    """
    
    metric_rows = {m: base_row + 1 + i for i, m in enumerate(METRICS)}

    col_indices = list(colmap.keys())

    col_labels = list(colmap.values())

    if not col_indices:

        return pd.DataFrame()

    data_dict = {}

    n_rows = len(df_raw)

    if include_base_row:

        if base_row < n_rows:

            raw_vals = df_raw.iloc[base_row, col_indices]

            data_dict[value_name] = clean_series_vectorized(
                series = raw_vals
            ).values

        else:

            data_dict[value_name] = np.full(len(col_indices), np.nan)

    for m, r in metric_rows.items():
        
        if r >= n_rows:

            data_dict[m] = np.full(len(col_indices), np.nan)

            continue

        raw_vals = df_raw.iloc[r, col_indices]

        if m == 'No. of Estimates':

            s = raw_vals.astype(str).str.replace('\xa0', ' ', regex = False).str.strip()

            s = s.replace(['-', '—', '', 'nan'], np.nan)

            data_dict[m] = s.values

        else:

            data_dict[m] = clean_series_vectorized(
                series = raw_vals
            ).values

    df = pd.DataFrame(data_dict, index = col_labels)

    df = df.rename(columns = {'Std. Dev.': 'Std_Dev', 'No. of Estimates': 'No_of_Estimates'})

    numeric_keys = [c for c in [value_name, 'Median', 'High', 'Low', 'Std_Dev'] if c in df.columns]

    if numeric_keys:

        df = df.dropna(subset = numeric_keys, how = 'all')

    col_order = [value_name, 'Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates']

    return df[[c for c in col_order if c in df.columns]]


def _parse_quarter_end(
    label,
    fy_freq: str = FY_FREQ
) -> pd.Timestamp:
    """
    Parse a quarter-like label into a normalised quarter-end date.

    Parameters
    ----------
    label:
        Quarter label (date-like, fiscal quarter token, or textual month-year token).
    fy_freq:
        Fiscal frequency used for fiscal quarter interpretation.

    Returns
    -------
    pandas.Timestamp | pandas.NaT
        Parsed quarter-end timestamp, normalised.
    """
    
    if pd.isna(label):

        return pd.NaT

    dt = _parse_header_cell_to_date(
        cell = label,
        fy_freq = fy_freq
    )

    if pd.notna(dt):

        return pd.Timestamp(dt).normalize()

    if isinstance(label, str):

        m = re.search('-\\s*([A-Za-z]{3})\\s*(\\d{4})', label)

        if m:

            mon, year = (m.group(1), m.group(2))

            base = pd.to_datetime(f'{mon} {year}', format = '%b %Y', errors = 'coerce')

            if pd.notna(base):

                return (base + pd.offsets.MonthEnd(0)).normalize()

    return pd.NaT


def _parse_fy_month_day(
    df_raw
):
    """
    Parse the "Current Fiscal Year End" month and day from a consensus worksheet.

    Parameters
    ----------
    df_raw:
        Raw consensus worksheet DataFrame.

    Returns
    -------
    tuple[str, int]
        (month_abbreviation, day_of_month). Defaults to (FY end month from ``FY_FREQ``, 31) when the
        expected annotation is absent.

    Expected Input Form
    -------------------
    A row in the first column containing text such as:
    "Current Fiscal Year End: Mar-31-2027"
    """
    
    default_mon = calendar.month_abbr[_fy_end_month(
        fy_freq = FY_FREQ
    )]

    col0 = df_raw.iloc[:, 0].astype(str)

    candidates = df_raw.index[col0.str.contains('Current Fiscal Year End:', na = False)].tolist()

    if not candidates:

        return (default_mon, 31)

    txt = str(df_raw.iloc[candidates[0], 0])

    m = re.search('Current Fiscal Year End:\\s*([A-Za-z]{3})-(\\d{1,2})-(\\d{4})', txt)

    if not m:

        return (default_mon, 31)

    return (m.group(1), m.group(2))


def _fy_end_date(
    fy_year,
    mon_abbr,
    day
):
    """
    Construct the fiscal year-end date for a given year, month abbreviation, and day.

    Parameters
    ----------
    fy_year:
        Fiscal year label (calendar year).
    mon_abbr:
        Month abbreviation (for example, "Mar").
    day:
        Day-of-month. Values larger than the month length are clamped to the month-end.

    Returns
    -------
    pandas.Timestamp
        Fiscal year-end timestamp.
    """
    
    month_num = pd.to_datetime(mon_abbr, format = '%b').month

    last_day = calendar.monthrange(fy_year, month_num)[1]

    d = min(day, last_day)

    return pd.Timestamp(fy_year, month_num, d)


def _safe_spearman(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Compute a defensive Spearman rank correlation with finite filtering and small-sample guards.

    Spearman's rho is defined as the Pearson correlation of the rank-transformed samples:

        rho_s = cov(rank(a), rank(b)) / (sd(rank(a)) * sd(rank(b)))

    where ``rank(.)`` assigns mid-ranks under ties (``rankdata(..., method="average")``). This
    implementation:

    - discards non-finite pairs,
    
    - returns 0.0 when fewer than 5 paired observations remain, and
    
    - returns 0.0 when the denominator is zero or the resulting statistic is not finite.

    The statistic is used throughout the model as a robust measure of monotonic association.
    Compared with Pearson correlation on levels, the rank correlation is less sensitive to
    heavy tails and outliers, which is advantageous when working with accounting series
    that exhibit regime shifts, unit discontinuities, or sparse coverage.

    Parameters
    ----------
    a, b:
        Arrays of observations. Only indices where both ``a`` and ``b`` are finite are used.

    Returns
    -------
    float
        Spearman rank correlation in [-1, 1], or 0.0 when the estimate is not well-defined.
    """
    
    a = np.asarray(a, dtype = float)

    b = np.asarray(b, dtype = float)

    m = np.isfinite(a) & np.isfinite(b)

    if m.sum() < 5:

        return 0.0

    aa = a[m]

    bb = b[m]

    ra = rankdata(aa, method = 'average')

    rb = rankdata(bb, method = 'average')

    ra = ra - ra.mean()

    rb = rb - rb.mean()

    den = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))

    if den <= 0.0:

        return 0.0

    r = np.sum(ra * rb) / den

    return r if np.isfinite(r) else 0.0


def _spearman_lag(
    x: np.ndarray,
    y: np.ndarray,
    lag: int
) -> float:
    """
    Compute a lagged Spearman rank correlation between two time-ordered arrays.

    The lag convention is defined on the raw array indices:

    - If ``lag > 0``: pairs are (x[t + lag], y[t]) for t = 0..n-lag-1.
   
    - If ``lag < 0``: pairs are (x[t], y[t - lag]) for t = 0..n+lag-1.
   
    - If ``lag == 0``: pairs are (x[t], y[t]).

    After aligning by the lag, the correlation is computed by ``_safe_spearman`` which applies
    finite filtering and small-sample guards. A minimum series length check is performed using
    ``MIN_POINTS`` before any alignment.

    Lagged rank correlation is used when selecting predictors for imputation. Allowing a lag
    accounts for economically plausible lead/lag relationships (for example, revenue changes
    leading margin changes), while retaining the robustness benefits of Spearman correlation.

    Parameters
    ----------
    x, y:
        Arrays interpreted as equally spaced, time-ordered observations.
    lag:
        Integer lag, in the same unit as the observation spacing (typically years for annual
        history). Positive values shift ``x`` forward relative to ``y``.

    Returns
    -------
    float
        Lagged Spearman correlation, or 0.0 when insufficient data are available.
    
    """
    
    x = np.asarray(x, dtype = float)

    y = np.asarray(y, dtype = float)

    n = min(len(x), len(y))

    if n < MIN_POINTS:

        return 0.0

    x = x[:n]

    y = y[:n]

    if lag > 0:

        if lag >= n:

            return 0.0

        return _safe_spearman(
            a = x[lag:],
            b = y[:-lag]
        )
        
    elif lag < 0:

        L = -lag

        if L >= n:

            return 0.0

        return _safe_spearman(
            a = x[:-L],
            b = y[L:]
        )

    else:

        return _safe_spearman(
            a = x,
            b = y
        )


def _nearest_psd_corr(
    R: np.ndarray,
    eps: float = e12
) -> np.ndarray:
    """
    Project a symmetric matrix onto a numerically safe correlation matrix.

    This helper performs a minimal "nearest positive semi-definite" repair suitable for
    correlation matrices used in Cholesky factorisation:

    1. Symmetrise: R <- (R + R.T) / 2.
   
    2. Eigen-decompose: R = V diag(w) V.T.
   
    3. Clip eigenvalues: w_i <- max(w_i, eps).
   
    4. Reconstruct: Rp = V diag(w) V.T.
   
    5. Re-normalise to unit diagonal: Rp_ij <- Rp_ij / sqrt(Rp_ii * Rp_jj).
   
    6. Symmetrise again and set diag to 1.

    The resulting matrix is guaranteed to be symmetric, to have ones on the diagonal, and to be
    positive semi-definite up to numerical tolerance. This repair is preferred to ad-hoc jittering
    at the point of factorisation because it preserves as much of the intended dependence structure
    as possible while preventing runtime failures.

    Parameters
    ----------
    R:
        Candidate correlation-like matrix.
    eps:
        Minimum eigenvalue and minimum diagonal scale used during repair.

    Returns
    -------
    numpy.ndarray
        A repaired correlation matrix with the same shape as ``R``.
    """
    
    R = np.asarray(R, dtype = float)

    R = 0.5 * (R + R.T)

    w, V = np.linalg.eigh(R)

    w = np.clip(w, max(float(eps), e12), None)

    Rp = V @ (w[:, None] * V.T)

    d = np.sqrt(np.clip(np.diag(Rp), max(float(eps), e12), None))

    Rp = Rp / np.outer(d, d)

    Rp = 0.5 * (Rp + Rp.T)

    np.fill_diagonal(Rp, 1.0)

    return Rp


def _first_existing_row(
    df: pd.DataFrame,
    candidates: tuple[str, ...]
) -> str | None:
    """
    Return the first row label present in a DataFrame from a sequence of candidate labels.

    This utility supports robust extraction from CapIQ-style statements where the same economic
    concept may appear under multiple possible row captions (for example, "Net Debt" versus
    "Net Debt (incl. Leases)").

    Parameters
    ----------
    df:
        Source table with row labels.
    candidates:
        Candidate row labels ordered by preference.

    Returns
    -------
    str | None
        The first matching label, or ``None`` when no candidates are present.
    """
    
    for r in candidates:
    
        if r in df.index:

            return r

    return None


def _as_numeric_series(
    s
) -> pd.Series:
    """
    Coerce a Series-like object to a numeric ``pandas.Series`` with an optional datetime index.

    The function is designed for statement row extraction where the input may be either:

    - a single row (Series)
    
    - a small DataFrame containing multiple plausible rows (for example, duplicated labels).

    When a DataFrame is provided, the "best" row is selected using a heuristic:

    - maximise the number of finite observations, then
   
    - break ties by preferring the row with the larger median absolute scale.

    This heuristic tends to select the most complete and economically meaningful row when
    statement parsing yields multiple candidates.

    Parameters
    ----------
    s:
        Series or DataFrame to be coerced to numeric values.

    Returns
    -------
    pandas.Series
        Numeric series (non-numeric cells coerced to NaN). If the index can be parsed to
        datetimes it is converted; otherwise it is preserved.
   
    """
   
    if isinstance(s, pd.DataFrame):

        best = None

        best_n = -1

        best_scale = -1.0

        for _, row in s.iterrows():
            
            num = pd.to_numeric(row, errors = 'coerce')

            arr = num.to_numpy(dtype = float, copy = False)

            n_finite = np.isfinite(arr).sum()

            scale = np.nanmedian(np.abs(arr)) if n_finite > 0 else -1.0

            if n_finite > best_n or (n_finite == best_n and scale > best_scale):

                best = num

                best_n = n_finite

                best_scale = scale

        out = best if best is not None else pd.Series(dtype = float)

    else:

        out = pd.to_numeric(s, errors = 'coerce')

    try:

        out.index = pd.to_datetime(out.index, errors = 'coerce')

    except (TypeError, ValueError):

        pass

    return out


def _hist_net_debt_debt_minus_cash(
    hist_bal: pd.DataFrame | None
) -> pd.Series | None:
    """
    Derive a historical net-debt series from a balance-sheet table.

    The function prefers an explicit "Net Debt" row when present; otherwise it constructs
    net debt as:

        NetDebt = GrossDebt - Cash

    Candidate row labels are taken from the module-level ``_NET_DEBT_ROWS``, ``_DEBT_ROWS``,
    and ``_CASH_ROWS`` lists. A simple sign sanitation is applied: if the median of the
    extracted debt or cash series is negative, the absolute value is taken (reflecting common
    statement conventions where liabilities are recorded as negatives).

    This series is used to detect and correct sign mismatches between historical statements
    and consensus forecast tables, which frequently encode net debt with inconsistent sign
    conventions across vendors.

    Parameters
    ----------
    hist_bal:
        Historical balance-sheet statement table, indexed by row captions with date-like columns.

    Returns
    -------
    pandas.Series | None
        Net debt series indexed by dates, or ``None`` when insufficient inputs are available.
   
    """
   
    if hist_bal is None or hist_bal.empty:

        return None

    nd_row = _first_existing_row(
        df = hist_bal,
        candidates = _NET_DEBT_ROWS
    )

    if nd_row is not None:

        nd = _as_numeric_series(
            s = hist_bal.loc[nd_row]
        ).dropna()

        if len(nd):

            return nd

    debt_row = _first_existing_row(
        df = hist_bal,
        candidates = _DEBT_ROWS
    )

    cash_row = _first_existing_row(
        df = hist_bal,
        candidates = _CASH_ROWS
    )

    if debt_row and cash_row:

        debt = _as_numeric_series(
            s = hist_bal.loc[debt_row]
        )

        cash = _as_numeric_series(
            s = hist_bal.loc[cash_row]
        )

        if np.nanmedian(debt.values) < 0:

            debt = debt.abs()

        if np.nanmedian(cash.values) < 0:

            cash = cash.abs()

        nd = (debt - cash).dropna()

        return nd if len(nd) else None

    return None


def _flip_future_metric_sign(
    dfT: pd.DataFrame,
    value_row: str
) -> pd.DataFrame:
    """
    Flip the sign of a forecast metric block and preserve the high/low ordering.

    A number of CapIQ consensus blocks report stock variables with either:

    - "debt is positive" convention, or
  
    - "debt is negative" convention.

    When a sign mismatch is detected against the historical statement convention, this helper
    multiplies the key estimate rows by -1:

        value_row, Median, High, Low

    If both "High" and "Low" are present, they are swapped after the sign flip to maintain the
    semantic ordering (High >= Low in the original units).

    Parameters
    ----------
    dfT:
        Forecast table with rows in the CapIQ consensus schema.
    value_row:
        Name of the primary value row to flip (for example, "Net_Debt").

    Returns
    -------
    pandas.DataFrame
        A modified copy of ``dfT`` with the sign flipped for the relevant rows.
   
    """
   
    dfT = dfT.copy()

    rows = [value_row, 'Median', 'High', 'Low']

    for r in rows:
    
        if r in dfT.index:

            dfT.loc[r] = -pd.to_numeric(dfT.loc[r], errors = 'coerce')

    if 'High' in dfT.index and 'Low' in dfT.index:

        hi = dfT.loc['High'].copy()

        dfT.loc['High'] = dfT.loc['Low']

        dfT.loc['Low'] = hi

    return dfT


def align_future_net_debt_sign_to_history(
    net_debt_future: pd.DataFrame | None,
    hist_bal: pd.DataFrame | None,
    value_row: str = 'Net_Debt',
    ratio_band: tuple[float, float] = (0.2, 4.0)
) -> pd.DataFrame:
    """
    Align the sign of a consensus net-debt forecast table to historical balance-sheet convention.

    CapIQ consensus tables may report net debt with a sign convention opposite to the historical
    statement tables. This function compares the most recent historical estimate of net debt,
    derived as either an explicit net-debt row or as (debt - cash), to the consensus forecast
    near the same date.

    Decision rule (high level)
    --------------------------
  
    1. Compute historical net debt series from ``hist_bal`` using ``_hist_net_debt_debt_minus_cash``.
  
    2. Find the forecast column whose timestamp is nearest to the last historical timestamp.
  
    3. Let ``hist_last`` be the last historical net debt and ``fut_near`` the nearest forecast value.
  
    4. If sign(hist_last) != sign(fut_near) and |hist_last| is non-trivial, compute:

           ratio = abs(fut_near / hist_last)

       If ``ratio`` lies within ``ratio_band`` (default 0.2..4.0), the forecast table is
       sign-flipped via ``_flip_future_metric_sign`` and a warning is emitted.

    The ratio band prevents flipping when the forecast magnitude is implausibly different from
    the last historical magnitude, which could indicate a genuine capital structure change rather
    than a sign convention error.

    Parameters
    ----------
    net_debt_future:
        Consensus table for net debt, indexed by CapIQ estimate rows and with date-like columns.
    hist_bal:
        Historical balance-sheet table used to infer the historical sign convention.
    value_row:
        Primary value row in the consensus table (default "Net_Debt").
    ratio_band:
        Acceptable magnitude ratio range required before applying a sign flip.

    Returns
    -------
    pandas.DataFrame
        Either the original table (when no action is required) or a sign-flipped copy.
    """
  
    if net_debt_future is None or net_debt_future.empty or hist_bal is None:

        return net_debt_future

    hist_nd = _hist_net_debt_debt_minus_cash(
        hist_bal = hist_bal
    )

    if hist_nd is None or hist_nd.empty:

        return net_debt_future

    hist_nd = hist_nd.sort_index()

    hist_last_date = pd.to_datetime(hist_nd.index.max())

    hist_last_val = hist_nd.dropna().iloc[-1]

    fut_cols = pd.to_datetime(net_debt_future.columns, errors = 'coerce')

    ok = pd.notna(fut_cols)

    if not ok.any() or value_row not in net_debt_future.index:

        return net_debt_future

    fut_cols_ok = fut_cols[ok]

    col_map = dict(zip(fut_cols_ok, net_debt_future.columns[ok]))

    nearest_dt = fut_cols_ok[np.argmin(np.abs((fut_cols_ok - hist_last_date).days))]

    nearest_col = col_map[nearest_dt]

    fut_val = pd.to_numeric(net_debt_future.loc[value_row, nearest_col], errors = 'coerce')

    if not np.isfinite(fut_val) or not np.isfinite(hist_last_val):

        return net_debt_future

    if np.sign(fut_val) != np.sign(hist_last_val) and abs(hist_last_val) > e12:

        ratio = abs(fut_val / hist_last_val)

        if ratio_band[0] <= ratio <= ratio_band[1]:

            warnings.warn(f'Net Debt sign mismatch vs (Debt-Cash) history. Flipping consensus Net Debt. hist={hist_last_val:.3g} (as of {hist_last_date.date()}), future={fut_val:.3g} (near {nearest_dt.date()}).')

            return _flip_future_metric_sign(
                dfT = net_debt_future,
                value_row = value_row
            )

    return net_debt_future


def _infer_fy_end_month_day_from_future(
    metric_future: pd.DataFrame | None
) -> tuple[int, int]:
    """
    Infer the fiscal year-end month/day from a forecast table's period columns.

    The function uses the mode of month and day across forecast column timestamps. When the
    table contains a ``period_type`` row, annual periods are preferred as they encode the
    fiscal year-end explicitly for mixed annual/quarterly tables.

    This inference is used for:

    - interpreting mixed annual/quarterly consensus tables,
  
    - constructing a reasonable fiscal year-end when creating fallback-zero forecast tables,
      and
  
    - determining the annual selection rule when only quarterly period ends are present.

    Parameters
    ----------
    metric_future:
        Consensus metric table with date-like columns and optional ``period_type`` row.

    Returns
    -------
    (int, int)
        Tuple ``(month, day)``. Defaults to (12, 31) on failure.
    """
  
    if metric_future is None or metric_future.empty:

        return (12, 31)

    try:

        cols = pd.to_datetime(metric_future.columns, errors = 'coerce')

    except (AttributeError, TypeError, ValueError):

        return (12, 31)

    cols = cols[pd.notna(cols)]

    if len(cols) == 0:

        return (12, 31)

    if 'period_type' in metric_future.index:

        types = metric_future.loc['period_type'].astype(str).str.lower().values

        ann = cols[types == 'annual']

        if len(ann) > 0:

            m = pd.Series(ann.month).mode().iat[0]

            d = pd.Series(ann.day).mode().iat[0]

            return (m, d)

    m = pd.Series(cols.month).mode().iat[0]

    d = pd.Series(cols.day).mode().iat[0]

    return (m, d)


def _zero_future_metric_table_from_df_raw(
    df_raw: pd.DataFrame,
    value_name: str,
    *,
    horizon_years: int = 3
) -> pd.DataFrame:
    """
    Construct a zero-filled consensus metric table when a workbook lacks usable forecasts.

    The output conforms to the schema expected by the simulation pipeline:

    - rows: value row, Median, High, Low, Std_Dev, No_of_Estimates, period_type, period_label
  
    - columns: future fiscal year-end dates (normalised to midnight)

    Period construction logic:

    - The fiscal year-end month/day is parsed from the raw worksheet when possible.
  
    - Candidate fiscal year-end dates are created for a window of calendar years around the
      current date, then filtered to those strictly after ``TODAY_TS``.
  
    - If no candidates are found, a simple Dec-31 ladder is used as a conservative default.

    The table is used as a "hard" fallback to ensure the overall valuation run remains robust
    even when the consensus workbook is missing a metric block entirely. Using explicit zeros is
    preferable to returning an empty table because downstream logic can still construct the
    period grid deterministically and can apply method gating consistently.

    Parameters
    ----------
    df_raw:
        Raw worksheet DataFrame, used only to infer fiscal year-end metadata.
    value_name:
        The name of the primary value row (for example, "Free_Cash_Flow").
    horizon_years:
        Number of annual periods to create.

    Returns
    -------
    pandas.DataFrame
        Zero-filled consensus metric table.
  
    """
  
    try:

        fy_mon_abbr, fy_d = _parse_fy_month_day(
            df_raw = df_raw
        )

    except (TypeError, ValueError):

        fy_mon_abbr, fy_d = ('Dec', 31)

    yrs = range(TODAY_TS.year - 1, TODAY_TS.year + horizon_years + 3)

    periods = []

    for y in yrs:
  
        try:

            d = _fy_end_date(
                fy_year = y,
                mon_abbr = fy_mon_abbr,
                day = fy_d
            )

        except (TypeError, ValueError):

            d = pd.Timestamp(y, 12, 31)

        if pd.notna(d) and d.normalize() > TODAY_TS:

            periods.append(d.normalize())

    if not periods:

        for k in range(1, horizon_years + 1):
            
            periods.append(pd.Timestamp(TODAY_TS.year + k, 12, 31).normalize())

    periods = pd.DatetimeIndex(periods[:horizon_years]).sort_values()

    idx = [value_name, 'Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates', 'period_type', 'period_label']

    out = pd.DataFrame(index = idx, columns = periods, dtype = object)

    for r in [value_name, 'Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates']:
        out.loc[r, :] = 0.0

    out.loc['period_type', :] = 'Annual'

    out.loc['period_label', :] = [f'FY{d.year}' for d in periods]

    return out


def _build_colmap_vec(
    df_raw: pd.DataFrame,
    header_row: int,
    start_col: int = 1
) -> dict[int, object]:
    """
    Build a column-index to header-label mapping for a detected period header row.

    CapIQ consensus sheets typically contain a "period header" row above each metric block.
    The header row contains period labels (for example, fiscal years or fiscal quarters) in the
    worksheet columns. This helper returns a sparse mapping from worksheet column positions to the
    original cell values for the non-empty header cells.

    Parameters
    ----------
    df_raw:
        Raw worksheet DataFrame as loaded by ``_load_consensus_sheet``.
    header_row:
        Row index in ``df_raw`` containing the period header.
    start_col:
        Column index at which period labels begin (CapIQ sheets typically use column 0 for labels).

    Returns
    -------
    dict[int, object]
        Mapping from integer column index to the original header cell value.
 
    """
    
    labs = df_raw.iloc[header_row, start_col:]

    s = labs.astype('string').fillna('').str.strip()

    m = s.ne('').to_numpy()

    if not m.any():

        return {}

    col_idx = np.arange(start_col, df_raw.shape[1])[m].tolist()

    vals = labs.to_numpy(dtype = object, copy = False)[m].tolist()

    return dict(zip(col_idx, vals))


@dataclass(frozen = True)
class _ConsensusParsed:
    """
    Parsed representation of a raw CapIQ-style consensus worksheet.

    Instances of this dataclass are produced by ``_parse_consensus_file_core`` and contain:
  
    - the raw worksheet as loaded into a DataFrame,
  
    - a normalised view of the first column (row labels),
  
    - detected metric blocks keyed by label,
  
    - inferred fiscal year-end month/day information, and
  
    - cached header metadata required for robust period parsing.

    This intermediate representation separates I/O concerns from the extraction of specific metrics and
    facilitates scoring and selection of annual and quarterly blocks.
  
    """
  
    df_raw: pd.DataFrame

    col0_norm: pd.Series

    blocks_by_label: dict[str, list[tuple[str, int, int, dict[int, object]]]]

    fy_mon_abbr: str

    fy_d: int

    fy_freq: str

    prev_header: np.ndarray | None

    header_kind: dict[int, str] | None

    header_colmap: dict[int, dict[int, object]] | None


def _parse_consensus_file_core(
    file_path: str
) -> _ConsensusParsed:
    """
    Parse a CapIQ-style consensus workbook into a block-indexed intermediate representation.

    The CapIQ consensus worksheet is treated as a semi-structured table containing multiple
    metric blocks. Each block is assumed to have:

    - a metric label in the first column (for example, "Revenue"), followed by
  
    - a fixed sequence of estimate statistic rows (``METRICS``), and
  
    - a period header row above the block that classifies the block as annual or quarterly.

    Extraction strategy
    -------------------
  
    1. Load the worksheet into a raw DataFrame.
  
    2. Infer fiscal year-end month/day metadata from the worksheet; derive a local fiscal-year
       frequency label (for example, "FY" or "Q-DEC") used by period parsing.
  
    3. Normalise first-column labels using ``_norm_label`` to facilitate matching under spacing
       and punctuation variation.
  
    4. Identify candidate header rows by scanning for rows with sufficient non-null coverage and
       classifying them via ``_classify_header_row`` as annual or quarterly.
  
    5. For each row i, record the nearest preceding header row index using ``prev_header``.
  
    6. Scan down the sheet looking for rows where the next ``len(METRICS)`` rows match exactly the
       normalised metrics sequence. Each match forms a block keyed by the metric label at row i.

    The returned ``_ConsensusParsed`` object decouples the expensive I/O and header detection from
    the subsequent per-metric extraction functions. This improves performance when multiple metrics
    are extracted from the same workbook and reduces duplicated header parsing logic.

    Parameters
    ----------
    file_path:
        Path to the consensus Excel file.

    Returns
    -------
    _ConsensusParsed
        Parsed workbook representation containing block metadata and cached header structures.
    """
  
    df_raw = _load_consensus_sheet(
        file_path = file_path
    )

    default_mon = calendar.month_abbr[_fy_end_month(
        fy_freq = FY_FREQ
    )]

    if df_raw is None or df_raw.empty:

        return _ConsensusParsed(df_raw = pd.DataFrame(), col0_norm = pd.Series(dtype = object), blocks_by_label = {}, fy_mon_abbr = default_mon, fy_d = 31, fy_freq = FY_FREQ, prev_header = None, header_kind = None, header_colmap = None)

    try:

        fy_mon_abbr_raw, fy_d_raw = _parse_fy_month_day(
            df_raw = df_raw
        )

        fy_mon_abbr = str(fy_mon_abbr_raw)

        fy_d = int(pd.to_numeric(fy_d_raw, errors = 'coerce'))

    except (ValueError, TypeError):

        fy_mon_abbr, fy_d = (default_mon, 31)

    fy_mon = MONTH_ABBR.get(str(fy_mon_abbr).strip().lower(), 12)

    fy_freq_local = _fy_freq_from_month(
        fy_month = fy_mon
    )

    col0_norm = df_raw.iloc[:, 0].map(_norm_label)

    blocks_by_label: dict[str, list[tuple[str, int, int, dict[int, object]]]] = {}

    n = len(df_raw)

    m = len(METRICS)

    header_kind: dict[int, str] = {}

    header_colmap: dict[int, dict[int, object]] = {}

    nonna_counts = df_raw.iloc[:, 1:].notna().sum(axis = 1).to_numpy()

    for h in np.where(nonna_counts >= MIN_HEADER_NONNA)[0]:
        
        kind = _classify_header_row(
            df_raw = df_raw,
            header_row = h,
            fy_freq = fy_freq_local
        )

        if kind in {'annual', 'quarterly'}:

            header_kind[h] = kind

            header_colmap[h] = _build_colmap_vec(
                df_raw = df_raw,
                header_row = h
            )

    prev_header = np.full(n, -1, dtype = int)

    last = -1

    for i in range(n):
        if i in header_kind:

            last = i

        prev_header[i] = last

    _METRICS_NORM_LOCAL = tuple((_norm_label(
        x = m
    ) for m in METRICS))

    for i in range(0, n - (m + 1)):
        
        lab = col0_norm.iat[i]

        if not lab or lab in _METRICS_NORM_LOCAL:

            continue

        seq = tuple(col0_norm.iloc[i + 1:i + 1 + m].tolist())

        if seq != _METRICS_NORM_LOCAL:

            continue

        h = prev_header[i] if 0 <= i < n else -1

        if h < 0 or i - h > 300:

            continue

        kind = header_kind.get(h)

        if kind is None:

            continue

        colmap = header_colmap.get(h, {})

        blocks_by_label.setdefault(lab, []).append((kind, i, h, colmap))

    return _ConsensusParsed(df_raw = df_raw, col0_norm = col0_norm, blocks_by_label = blocks_by_label, fy_mon_abbr = str(fy_mon_abbr), fy_d = fy_d, fy_freq = fy_freq_local, prev_header = prev_header, header_kind = header_kind, header_colmap = header_colmap)


def _filter_blocks_by_ticker(
    blocks,
    df_raw,
    ticker: str
):
    """
    Apply a heuristic filter to consensus metric blocks based on ticker suffix conventions.

    Some consensus workbooks contain multiple variants of the same metric label, typically
    corresponding to different currency listings or ADR representations. This helper assigns a
    score to each candidate block by scanning a small text window above the block's header row for
    currency and exchange markers inferred from the ticker suffix (for example, ".L" implies GBP
    and LSE markers).

    The returned list contains the blocks tied for the maximum score. The heuristic is intentionally
    conservative: when no expected or avoid tokens are configured for the ticker suffix, the blocks
    are returned unchanged.

    Parameters
    ----------
    blocks:
        List of block tuples as produced by ``_parse_consensus_file_core``.
    df_raw:
        Raw worksheet DataFrame.
    ticker:
        Ticker symbol used to infer expected currency/exchange markers.

    Returns
    -------
    list
        Filtered list of blocks, potentially identical to the input.
   
    """
   
    if not ticker or not blocks:

        return blocks

    tick = ticker.upper()

    expected = []

    avoid = []

    if tick.endswith('.L'):

        expected = ['(GBP)', ' GBP', 'Great British Pound', 'Pence', 'LSE:']

        avoid = ['(USD)', ' USD', ' ADR ', 'American Depository']
   
    elif tick.endswith('.TO') or tick.endswith('.V'):

        expected = ['(CAD)', ' CAD', 'Toronto', 'TSX']

        avoid = ['(USD)', ' USD']
        
    elif tick.endswith('.AX'):

        expected = ['(AUD)', ' AUD', 'Australian']

        avoid = ['(USD)']
        
    elif tick.endswith('.HK'):

        expected = ['(HKD)', ' HKD', 'Hong Kong']

        avoid = ['(USD)']
        
    elif tick.endswith('.PA') or tick.endswith('.MC') or tick.endswith('.DE'):

        expected = ['(EUR)', ' EUR', 'Euro']

        avoid = ['(USD)']

    if not expected and (not avoid):

        return blocks

    scored = []

    for b in blocks:
        
        h = b[2]

        start = max(0, h - 15)

        text_segment = df_raw.iloc[start:h + 1, 0].astype(str).str.cat(sep = ' ').upper()

        score = 0

        for t in expected:
            
            if t.upper() in text_segment:

                score += 10

        for t in avoid:
            
            if t.upper() in text_segment:

                score -= 100

        scored.append((score, b))

    if not scored:

        return blocks

    best_score = max((s for s, _ in scored))

    return [b for s, b in scored if s == best_score]


def _score_annual_block(
    b,
    df_raw: pd.DataFrame | None = None,
    *,
    fy_freq: str = FY_FREQ
):
    """
    Score an annual consensus block for selection when multiple blocks match the same metric label.

    The score tuple is designed for lexicographic maximisation. It prioritises blocks that:

    - contain more usable (finite) values in the primary value row, and
    - extend further into the future.

    When ``df_raw`` is provided, the value row referenced by the block is examined to count
    finite entries aligned to the block's header columns.

    Parameters
    ----------
    b:
        Block tuple ``(kind, row, header_row, colmap)``.
    df_raw:
        Raw worksheet DataFrame used to check for actual finite values.
    fy_freq:
        Fiscal-year frequency label used when parsing header cells into dates.

    Returns
    -------
    tuple
        Score tuple ``(usable_count, usable_max_year, total_count, header_max_year)`` where
        higher is better. A negative tuple is returned when no period labels can be parsed.
    
    """
    
    _, row, _, colmap = b

    if not colmap:

        return (-1, -1, -1, -1)

    cols = list(colmap.keys())

    labels = list(colmap.values())

    years: list[int] = []

    for v in labels:
        
        y = _year_of(
            v = v
        )

        if y is not None:

            years.append(int(y))

            continue

        dt = _parse_header_cell_to_date(
            cell = v,
            fy_freq = fy_freq
        )

        if pd.notna(dt):

            years.append(int(pd.Timestamp(dt).year))

    if not years:

        return (-1, -1, -1, -1)

    if df_raw is not None and 0 <= row < len(df_raw):

        try:

            vals = clean_series_vectorized(
                series = df_raw.iloc[row, cols]
            ).to_numpy(dtype = float, copy = False)

        except (TypeError, ValueError, IndexError):

            vals = np.full(len(cols), np.nan, dtype = float)

        usable_years: list[int] = []

        for y, v in zip(years, vals):
            
            if np.isfinite(v):

                usable_years.append(int(y))

        if usable_years:

            return (len(usable_years), max(usable_years), len(years), max(years))

    return (0, -1, len(years), max(years))


def _score_quarter_block(
    b,
    df_raw: pd.DataFrame | None = None,
    *,
    fy_freq: str = FY_FREQ
):
    """
    Score a quarterly consensus block for selection when multiple blocks match the same metric label.

    The score tuple is designed for lexicographic maximisation. It prioritises blocks that:

    - contain more usable (finite) values in the primary value row, and
 
    - extend to later quarter-end dates.

    The quarter-end dates are inferred from header labels using ``_parse_quarter_end``. When
    ``df_raw`` is provided, the block's value row is checked for finite values at the block's
    header columns.

    Parameters
    ----------
    b:
        Block tuple ``(kind, row, header_row, colmap)``.
    df_raw:
        Raw worksheet DataFrame used to check for actual finite values.
    fy_freq:
        Fiscal-year frequency label used when parsing quarter headers.

    Returns
    -------
    tuple
        Score tuple ``(usable_count, usable_max_dt, total_count, header_max_dt)`` where higher
        is better. A negative tuple is returned when no quarter labels can be parsed.
  
    """
  
    _, row, _, colmap = b

    if not colmap:

        return (-1, pd.Timestamp.min, -1, pd.Timestamp.min)

    cols = list(colmap.keys())

    labels = list(colmap.values())

    dts = [_parse_quarter_end(
        label = lbl,
        fy_freq = fy_freq
    ) for lbl in labels]

    dts = [pd.Timestamp(d).normalize() for d in dts if pd.notna(d)]

    if not dts:

        return (-1, pd.Timestamp.min, -1, pd.Timestamp.min)

    if df_raw is not None and 0 <= row < len(df_raw):

        try:

            vals = clean_series_vectorized(
                series = df_raw.iloc[row, cols]
            ).to_numpy(dtype = float, copy = False)

        except (TypeError, ValueError, IndexError):

            vals = np.full(len(cols), np.nan, dtype = float)

        usable_dts = [d for d, v in zip(dts, vals) if np.isfinite(v)]

        if usable_dts:

            return (len(usable_dts), max(usable_dts), len(dts), max(dts))

    return (0, pd.Timestamp.min, len(dts), max(dts))


def _use_additive_annual_residual(
    metric_key: str | None,
    value_name: str
) -> bool:
    """
    Decide whether annual values should be converted into residual (annual minus summed quarterly) values.

    CapIQ consensus data may include both quarterly forecasts and annual forecasts for the same
    fiscal year. For additive flow quantities (for example, revenue, CFO, CapEx), combining both
    naively would double count. A conservative approach is to treat the annual figure as the full
    fiscal-year total and replace it by the residual:

        AnnualResidual = AnnualTotal - sum(QuarterlyTotalsInFY)

    This adjustment is appropriate for additive flows and inappropriate for rates or stock variables.
    The decision is based on ``metric_key`` and ``value_name`` heuristics.

    Parameters
    ----------
    metric_key:
        Internal metric key (for example, "revenue", "eps", "net_debt").
    value_name:
        Name of the primary value row in the extracted consensus table.

    Returns
    -------
    bool
        ``True`` when additive residual adjustment is appropriate; ``False`` otherwise.
   
    """
   
    k = str(metric_key or '').strip().lower()

    if k:

        if k in RATE_KEYS or k in {'net_debt', 'bvps'}:

            return False

        if k in FLOW_KEYS or k in {'eps', 'dps', 'ebt', 'ebt_future'}:

            return True

    vn = str(value_name or '').strip().lower()

    if not vn:

        return False

    if vn in {'net_debt', 'bvps'}:

        return False

    if any((tok in vn for tok in ('pct', 'percent', 'margin', 'rate', 'roe'))):

        return False

    return True


def _adjust_partial_fy_annual_to_residual(
    df: pd.DataFrame,
    *,
    metric_key: str | None,
    value_name: str,
    fy_m: int,
    fy_d: int
) -> pd.DataFrame:
    """
    Replace annual additive-flow forecasts by fiscal-year residuals when overlapping quarter forecasts exist.

    The consensus extraction stage may produce a mixed table containing both quarterly and annual
    periods. When both appear for the same fiscal year and the metric is an additive flow (as
    determined by ``_use_additive_annual_residual``), the annual entry is transformed into a residual
    amount so that the combination of quarterly and annual rows for the fiscal year sums back to the
    originally quoted annual total.

    For each annual period end A and its set of quarters Q(A) within that fiscal year, the adjustment
    applied to each estimate row r in {value_name, Median, High, Low} is:

        out[A, r] = out[A, r] - sum_{q in Q(A)} out[q, r]

    For the dispersion proxy ``Std_Dev``, a variance subtraction is applied under an independence
    approximation:

        var_residual = max( var_annual - sum(var_quarters), 0 )
   
        sd_residual  = sqrt(var_residual)

    This transformation reduces double counting risk and makes the subsequent annualisation logic
    coherent when quarterly coverage is partial (for example, three quarters available and a fourth
    implied by the annual total).

    Parameters
    ----------
    df:
        Mixed-period DataFrame indexed by period end timestamps and containing a ``period_type`` column.
    metric_key:
        Internal metric key used to determine whether residualisation is appropriate.
    value_name:
        Primary estimate column name.
    fy_m, fy_d:
        Fiscal year-end month and day, used to map quarter ends to fiscal years.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with annual rows adjusted to residual values when applicable.
   
    """
   
    if df is None or df.empty:

        return df

    if 'period_type' not in df.columns:

        return df

    if not _use_additive_annual_residual(metric_key = metric_key, value_name = value_name):

        return df

    out = df.copy()

    idx_dt = pd.to_datetime(out.index, errors = 'coerce')

    if pd.Series(idx_dt).notna().sum() == 0:

        return out

    out = out.loc[pd.notna(idx_dt)].copy()

    out.index = pd.DatetimeIndex(idx_dt[pd.notna(idx_dt)]).normalize()

    out = out.sort_index()

    ptype = out['period_type'].astype(str).str.lower()

    q_idx = pd.DatetimeIndex(out.index[ptype.eq('quarterly')]).normalize().sort_values().unique()

    a_idx = pd.DatetimeIndex(out.index[ptype.eq('annual')]).normalize().sort_values().unique()

    if len(q_idx) == 0 or len(a_idx) == 0:

        return out

    q_by_fy: dict[pd.Timestamp, list[pd.Timestamp]] = {}

    for q in q_idx:
     
        fy_end = pd.Timestamp(_fiscal_year_end_for_date(
            d = q,
            fy_m = fy_m,
            fy_d = fy_d
        )).normalize()

        q_by_fy.setdefault(fy_end, []).append(pd.Timestamp(q).normalize())

    value_rows = [r for r in [value_name, 'Median', 'High', 'Low'] if r in out.columns]

    has_sd = 'Std_Dev' in out.columns

    for a_end in a_idx:
        qs = sorted(set(q_by_fy.get(pd.Timestamp(a_end).normalize(), [])))

        if len(qs) == 0:

            continue

        for r in value_rows:
      
            a_val = pd.to_numeric(pd.Series([out.at[a_end, r]]), errors = 'coerce').iloc[0]

            if not np.isfinite(a_val):

                continue

            q_vals = pd.to_numeric(out.loc[qs, r], errors = 'coerce').to_numpy(dtype = float, copy = False)

            q_sum = float(np.nansum(q_vals)) if np.isfinite(q_vals).any() else 0.0

            out.at[a_end, r] = float(a_val - q_sum)

        if has_sd:

            a_sd = pd.to_numeric(pd.Series([out.at[a_end, 'Std_Dev']]), errors = 'coerce').iloc[0]

            if np.isfinite(a_sd):

                q_sd = pd.to_numeric(out.loc[qs, 'Std_Dev'], errors = 'coerce').to_numpy(dtype = float, copy = False)

                q_var = float(np.nansum(np.square(q_sd[np.isfinite(q_sd)]))) if np.isfinite(q_sd).any() else 0.0

                resid_var = max(float(a_sd) * float(a_sd) - q_var, 0.0)

                out.at[a_end, 'Std_Dev'] = float(np.sqrt(resid_var))

    return out


def _extract_future_metric_estimates_from_core(
    core: _ConsensusParsed,
    metric_label: str,
    value_name: str,
    *,
    fy_freq: str = FY_FREQ,
    on_empty: str = 'raise',
    ticker: str,
    metric_key: str | None = None
) -> pd.DataFrame:
    """
    Extract a single metric forecast table (annual and/or quarterly) from a parsed consensus workbook.

    This function is the core of CapIQ consensus metric extraction. Given an intermediate
    ``_ConsensusParsed`` representation and a metric label, it performs:

    - candidate block discovery (using cached block indices when available, or a fallback search),
 
    - optional ticker-specific block filtering (currency/listing heuristics),
 
    - annual and quarterly block selection via scoring functions,
 
    - row extraction into a canonical schema using ``_extract_metrics_df``,
 
    - conversion of header labels into normalised period-end timestamps,
 
    - selection of periods that are plausibly "forward" relative to ``TODAY_TS``, and
 
    - de-duplication and annual-residual adjustment for additive flows.

    Output schema
    -------------
    The returned DataFrame has the CapIQ-consensus orientation expected by the simulation stage:

    - index rows: value_name, Median, High, Low, Std_Dev, No_of_Estimates, period_type, period_label
 
    - columns: period-end timestamps (annual fiscal year-ends and/or quarterly ends)

    The table is transposed to match the legacy convention used elsewhere in the codebase.

    Parameters
    ----------
    core:
        Parsed consensus workbook representation.
    metric_label:
        Human-readable metric label to locate in the workbook (for example, "Revenue").
    value_name:
        Canonical value row name to use in the output table (for example, "Revenue").
    fy_freq:
        Fiscal-year frequency label used for parsing ambiguous header cells.
    on_empty:
        Behaviour when extraction yields no usable periods. "raise" raises an exception; "zeros"
        returns a zero-filled table with a compatible schema.
    ticker:
        Optional ticker symbol used by block filtering heuristics.
    metric_key:
        Internal metric key used for annual residual logic and logging.

    Returns
    -------
    pandas.DataFrame
        Extracted forecast table, oriented with estimate rows on the index.
 
    """
 
    df_raw = core.df_raw

    fy_freq_eff = str(getattr(core, 'fy_freq', fy_freq) or fy_freq)

    if df_raw is None or df_raw.empty:

        msg = f'Consensus sheet empty for file.'

        if str(on_empty).lower() == 'zeros':

            return _zero_future_metric_table_from_df_raw(
                df_raw = df_raw if df_raw is not None else pd.DataFrame(),
                value_name = value_name
            )

        raise ValueError(msg)

    target = _norm_label(
        x = metric_label
    )

    blocks = list(core.blocks_by_label.get(target, []))

    if not blocks:

        metric_rows = df_raw.index[core.col0_norm == target].tolist()

        if not metric_rows:

            msg = f"Metric '{metric_label}' not found anywhere in first column."

            if str(on_empty).lower() == 'zeros':

                return _zero_future_metric_table_from_df_raw(
                    df_raw = df_raw,
                    value_name = value_name
                )

            raise ValueError(msg)

        for r in metric_rows:
           
            if core.prev_header is None or core.header_kind is None or core.header_colmap is None:

                h, kind, colmap = _find_period_header_above(
                    df_raw = df_raw,
                    base_row = r,
                    fy_freq = fy_freq_eff
                )

            else:

                h = core.prev_header[r] if 0 <= r < len(core.prev_header) else -1

                if h >= 0 and r - h <= 300:

                    kind = core.header_kind.get(h)

                    colmap = core.header_colmap.get(h, {})

                else:

                    kind, colmap = (None, None)

            if kind is None:

                h2, kind2, colmap2 = _find_period_header_above(
                    df_raw = df_raw,
                    base_row = r,
                    fy_freq = fy_freq_eff
                )

                if kind2 is not None:

                    h, kind, colmap = (h2, kind2, colmap2)

            if kind is not None:

                blocks.append((kind, r, h, colmap))

    if ticker:

        blocks = _filter_blocks_by_ticker(
            blocks = blocks,
            df_raw = df_raw,
            ticker = ticker
        )

    annual = [b for b in blocks if b[0] == 'annual']

    quarterly = [b for b in blocks if b[0] == 'quarterly']

    if not annual and (not quarterly):

        msg = f"Found '{metric_label}' but couldn't identify annual OR quarterly blocks. Identified blocks = {[(k, r, h) for k, r, h, _ in blocks]}"

        if str(on_empty).lower() == 'zeros':

            return _zero_future_metric_table_from_df_raw(
                df_raw = df_raw,
                value_name = value_name
            )

        raise ValueError(msg)

    annual_df = None

    quarterly_df = None

    if annual:

        _, annual_row, _, annual_colmap = max(annual, key = lambda b: _score_annual_block(
            b = b,
            df_raw = df_raw,
            fy_freq = fy_freq_eff
        ))

        annual_df = _extract_metrics_df(
            df_raw = df_raw,
            base_row = annual_row,
            colmap = annual_colmap,
            value_name = value_name
        )

    if quarterly:

        _, quarterly_row, _, quarter_colmap = max(quarterly, key = lambda b: _score_quarter_block(
            b = b,
            df_raw = df_raw,
            fy_freq = fy_freq_eff
        ))

        quarterly_df = _extract_metrics_df(
            df_raw = df_raw,
            base_row = quarterly_row,
            colmap = quarter_colmap,
            value_name = value_name
        )

    parts = []

    if quarterly_df is not None:

        quarterly_df['period_end'] = [_parse_quarter_end(
            label = lbl,
            fy_freq = fy_freq_eff
        ) for lbl in quarterly_df.index]

        quarterly_df = quarterly_df.dropna(subset = ['period_end']).sort_values('period_end')

        if not quarterly_df.empty:

            max_q = quarterly_df['period_end'].max()

            if pd.notna(max_q) and TODAY_TS - max_q <= pd.Timedelta(days = 540):

                asof_eff = max_q if pd.notna(max_q) and TODAY_TS > max_q else TODAY_TS

                q_future = quarterly_df[quarterly_df['period_end'] >= asof_eff].copy()

                if q_future.empty:

                    q_future = quarterly_df.tail(1).copy()

                q_out = q_future.copy()

                q_out['period_type'] = 'Quarterly'

                q_out['period_label'] = q_out.index

                cols = [value_name, 'Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates', 'period_type', 'period_label']

                parts.append(q_out.set_index('period_end')[cols])

    if annual_df is not None:

        fy_mon, fy_day2 = (core.fy_mon_abbr, int(core.fy_d))

        annual_ends: list[pd.Timestamp | None] = []

        for lbl in pd.Index(annual_df.index):
            
            y = _year_of(
                v = lbl
            )

            if y is not None:

                try:

                    annual_ends.append(_fy_end_date(
                        fy_year = int(y),
                        mon_abbr = fy_mon,
                        day = fy_day2
                    ))

                except (TypeError, ValueError):

                    annual_ends.append(pd.NaT)

                continue

            dt = _parse_header_cell_to_date(
                cell = lbl,
                fy_freq = fy_freq_eff
            )

            if pd.notna(dt):

                annual_ends.append(pd.Timestamp(dt).normalize())

            else:

                annual_ends.append(pd.NaT)

        annual_df['period_end'] = annual_ends

        annual_df = annual_df.dropna(subset = ['period_end']).sort_values('period_end')

        anchor = TODAY_TS

        a_after = annual_df[annual_df['period_end'] > anchor].copy()

        if a_after.empty and (not annual_df.empty):

            max_a = annual_df['period_end'].max()

            if pd.notna(max_a) and anchor - max_a <= pd.Timedelta(days = 370):

                a_after = annual_df.tail(1).copy()

        if not a_after.empty:

            a_out = a_after.copy()

            a_out['period_type'] = 'Annual'

            a_out['period_label'] = [f'FY{d.year}' for d in a_out['period_end']]

            cols = [value_name, 'Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates', 'period_type', 'period_label']

            parts.append(a_out.set_index('period_end')[cols])

    if not parts:

        msg = f'{metric_label}: no usable annual or quarterly periods after parsing.'

        if str(on_empty).lower() == 'zeros':

            return _zero_future_metric_table_from_df_raw(
                df_raw = df_raw,
                value_name = value_name
            )

        raise ValueError(msg)

    final = pd.concat([p for p in parts if p is not None and (not p.empty)], axis = 0).sort_index()

    if final.empty:

        msg = f'{metric_label}: no usable annual or quarterly periods after parsing.'

        if str(on_empty).lower() == 'zeros':

            return _zero_future_metric_table_from_df_raw(
                df_raw = df_raw,
                value_name = value_name
            )

        raise ValueError(msg)

    if final.index.has_duplicates:

        pref = final['period_type'].astype(str).str.lower().map({'quarterly': 1, 'annual': 0}).fillna(0).astype(int)

        final = final.assign(_pref = pref).sort_values('_pref').groupby(level = 0, sort = True).tail(1).drop(columns = '_pref').sort_index()

    fy_m = _fy_end_month(
        fy_freq = fy_freq_eff
    )

    fy_d = int(core.fy_d) if np.isfinite(pd.to_numeric(core.fy_d, errors = 'coerce')) else 31

    final = _adjust_partial_fy_annual_to_residual(
        df = final,
        metric_key = metric_key,
        value_name = value_name,
        fy_m = fy_m,
        fy_d = fy_d
    )

    return final.T


def _looks_like_missing_stub(
    _df: pd.DataFrame,
    _value_row: str
) -> bool:
    """
    Detect a "missing metric" stub table produced by a zero-fill fallback path.

    A number of workbooks embed empty-looking blocks for metrics that are not covered for a
    particular ticker. These blocks frequently contain the standard set of estimate-statistic
    rows but with zeros in the value row and a lack of analyst count and dispersion information.

    This function treats a table as a stub when:

    - the value row exists,
  
    - the standard estimate rows are present,
  
    - the value row is effectively all zeros (via ``_is_zero_like_future_table``),
  
    - No_of_Estimates is all zero/NaN, and
  
    - Std_Dev is all zero/NaN (within a small tolerance).

    Parameters
    ----------
    _df:
        Candidate forecast table.
    _value_row:
        Name of the primary value row to check.

    Returns
    -------
    bool
        True when the table appears to be a placeholder stub rather than a genuine forecast.
  
    """
  
    want = ['Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates']

    if _value_row is None or _value_row not in _df.index:

        return False

    if not all((r in _df.index for r in want)):

        return False

    if not _is_zero_like_future_table(dfT = _df, value_row = str(_value_row)):

        return False

    ne = pd.to_numeric(_df.loc['No_of_Estimates'], errors = 'coerce')

    if np.isfinite(ne.to_numpy(dtype = float)).any() and float(np.nanmax(np.abs(ne.to_numpy(dtype = float)))) > 0.0:

        return False

    sd = pd.to_numeric(_df.loc['Std_Dev'], errors = 'coerce')

    if np.isfinite(sd.to_numpy(dtype = float)).any() and float(np.nanmax(np.abs(sd.to_numpy(dtype = float)))) > 1e-12:

        return False

    return True


def extract_many_future_metric_estimates(
    pred_file: str,
    specs: dict[str, tuple[list[str], str]],
    *,
    fy_freq: str = FY_FREQ,
    ticker: str = None
) -> dict[str, pd.DataFrame]:
    """
    Extract a collection of forecast metric tables from a single CapIQ consensus workbook.

    The input ``specs`` mapping defines, for each internal metric key, a list of alternative
    workbook labels and a canonical output value-row name. Each metric is attempted in label
    priority order until a usable forecast is found. Usability is determined by:

    - successful parsing (no ValueError), and
  
    - presence of at least one finite value in the future columns for the value row or common
      estimate rows (``_has_usable_future_values``), and
  
    - not being a detected stub block (``_looks_like_missing_stub``), for non-dividend metrics.

    When no usable block exists for a metric, a zero-filled table with a compatible schema is
    produced as a fallback. This design ensures the downstream valuation pipeline has deterministic
    period grids and can apply method gating without special-casing missing tables.

    Parameters
    ----------
    pred_file:
        Path to the CapIQ consensus workbook.
    specs:
        Mapping ``metric_key -> (labels, value_name)``.
    fy_freq:
        Fiscal-year frequency label used when parsing header cells.
    ticker:
        Optional ticker used for block filtering and logging context.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from internal metric keys to forecast tables (possibly zero-filled fallbacks).
  
    """
  
    core = _parse_consensus_file_core(
        file_path = pred_file
    )

    out: dict[str, pd.DataFrame] = {}

    for key, (labels, value_name) in specs.items():
        dfT = None

        fallback_dfT = None

        last_value_error: ValueError | None = None

        for lab in labels:
            try:

                cand = _extract_future_metric_estimates_from_core(
                    core = core,
                    metric_label = lab,
                    value_name = value_name,
                    fy_freq = fy_freq,
                    ticker = ticker,
                    metric_key = key
                )

            except ValueError as err:

                last_value_error = err

                continue

            except (TypeError, KeyError) as err:

                raise RuntimeError(f"{key}: unexpected parsing failure for label '{lab}' in {pred_file}") from err

            if cand is None or cand.empty:

                continue

            if key not in {'dps'}:

                try:

                    _row0 = cand.index[0] if len(cand.index) else None

                    if _row0 is not None and _looks_like_missing_stub(_df = cand, _value_row = str(_row0)):

                        if ticker is not None:

                            logger.warning("%s forecast for '%s' not found in workbook.", ticker, key)

                        continue

                except (TypeError, ValueError):

                    pass

            if fallback_dfT is None:

                fallback_dfT = cand

            if _has_usable_future_values(
                dfT = cand,
                value_row = value_name, 
                today = TODAY_TS
            ):

                dfT = cand

                break

        if dfT is None:

            dfT = fallback_dfT

        if dfT is None or dfT.empty:

            lab0 = labels[0] if labels else value_name

            try:

                dfT = _extract_future_metric_estimates_from_core(
                    core = core,
                    metric_label = lab0,
                    value_name = value_name,
                    fy_freq = fy_freq,
                    on_empty = 'zeros',
                    ticker = ticker,
                    metric_key = key
                )

            except (TypeError, KeyError) as err:

                raise RuntimeError(f"{key}: failed to build fallback-zero table for '{lab0}' in {pred_file}") from err

            if (dfT is None or dfT.empty) and last_value_error is not None:

                raise last_value_error

        if dfT is not None and key not in {'dps'}:

            try:

                _row0 = dfT.index[0] if len(dfT.index) else None

                if _row0 is not None and _looks_like_missing_stub(_df = dfT, _value_row = str(_row0)):

                    if ticker is not None:

                        logger.warning("%s forecast for '%s' not found in workbook.", ticker, key)

                    dfT = None

            except (TypeError, ValueError):

                pass

        out[key] = dfT

    del core

    gc.collect()

    return out


def _has_usable_future_values(
    dfT: pd.DataFrame | None,
    *,
    value_row: str,
    today: pd.Timestamp = TODAY_TS
) -> bool:
    """
    Determine whether a forecast table contains any usable finite future values.

    The check is applied after removing any columns that are not strictly in the future relative
    to ``today`` (via ``_drop_nonfuture_columns``). A table is considered usable when at least one
    of the key estimate rows contains a finite numeric value in the remaining columns.

    Parameters
    ----------
    dfT:
        Forecast table in CapIQ-consensus schema.
    value_row:
        Name of the primary value row (for example, "Revenue").
    today:
        Anchor timestamp used to define "future" columns.

    Returns
    -------
    bool
        True when at least one finite future value exists; False otherwise.
  
    """
  
    if dfT is None or dfT.empty:

        return False

    dff = _drop_nonfuture_columns(
        df = dfT,
        today = today
    )

    if dff is None or dff.empty:

        return False

    for r in (value_row, 'Median', 'High', 'Low'):
    
        if r not in dff.index:

            continue

        vals = pd.to_numeric(dff.loc[r], errors = 'coerce').to_numpy(dtype = float, copy = False)

        if np.isfinite(vals).any():

            return True

    return False


def _is_zero_like_future_table(
    dfT: pd.DataFrame | None,
    value_row: str
) -> bool:
    """
    Check whether a forecast table's primary value row is effectively all zeros.

    This utility is used when distinguishing genuine forecast blocks from placeholder stubs.
    A row is treated as "zero-like" when:

    - the table or row is missing, or
   
    - the finite entries in the value row have maximum absolute magnitude <= e12.

    Parameters
    ----------
    dfT:
        Forecast table (or None).
    value_row:
        Primary value row name.

    Returns
    -------
    bool
        True when the table behaves like an all-zero stub for the primary value row.
    
    """
    
    if dfT is None or dfT.empty:

        return True

    if value_row not in dfT.index:

        return True

    v = pd.to_numeric(dfT.loc[value_row], errors = 'coerce').to_numpy(dtype = float)

    v = v[np.isfinite(v)]

    if len(v) == 0:

        return True

    return np.nanmax(np.abs(v)) <= e12


def _nu_from_excess_kurt(
    exk: float
) -> float:
    """
    Convert an excess kurtosis estimate into a Student-t degrees-of-freedom parameter.

    For a centred Student-t distribution with degrees of freedom nu, the excess kurtosis is:

        excess_kurtosis = 6 / (nu - 4)   for nu > 4

    Solving for nu yields:

        nu = 6 / excess_kurtosis + 4

    This function applies the above relationship as a pragmatic mapping from empirical
    excess kurtosis to a heavy-tail parameter used elsewhere in the model, with clipping to
    ``[NU_MIN, NU_MAX]``. Very small or non-finite excess kurtosis values are treated as
    approximately normal and mapped to ``NU_MAX``.

    Parameters
    ----------
    exk:
        Excess kurtosis estimate (kurtosis minus 3). Negative values are treated as 0.

    Returns
    -------
    float
        Degrees of freedom parameter nu used for t-distributed draws.
   
    """
   
    if not np.isfinite(exk) or exk <= e6:

        return NU_MAX

    nu = 6.0 / exk + 4.0

    return np.clip(nu, NU_MIN, NU_MAX)


def _sample_skew_exkurt(
    x: np.ndarray
) -> tuple[float, float]:
    """
    Estimate sample skewness and excess kurtosis using central moments with finite filtering.

    Given a finite sample x_i, define the central moments:

        m2 = mean( (x - mu)^2 )
 
        m3 = mean( (x - mu)^3 )
 
        m4 = mean( (x - mu)^4 )

    The moment-based skewness and kurtosis are:

        skewness = m3 / m2^(3/2)
 
        kurtosis = m4 / m2^2
 
        excess_kurtosis = kurtosis - 3

    The estimates are clipped to avoid destabilising downstream calibration in small samples.

    Parameters
    ----------
    x:
        Sample array; non-finite values are discarded.

    Returns
    -------
    (float, float)
        Tuple ``(skewness, excess_kurtosis)``.
  
    """
  
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    n = x.size

    if n < MIN_POINTS:

        return (0.0, 0.0)

    mu = np.mean(x)

    c = x - mu

    m2 = np.mean(c ** 2)

    if m2 <= 1e-18:

        return (0.0, 0.0)

    m3 = np.mean(c ** 3)

    m4 = np.mean(c ** 4)

    skew = m3 / m2 ** 1.5

    kurt = m4 / m2 ** 2

    exk = kurt - 3.0

    if not np.isfinite(skew):

        skew = 0.0

    if not np.isfinite(exk):

        exk = 0.0

    skew = np.clip(skew, -20, 20)

    exk = np.clip(exk, 0.0, 50.0)

    return (skew, exk)


def _robust_loc_scale(
    x: np.ndarray
) -> tuple[float, float]:
    """
    Compute a robust location and scale estimate using the median and MAD.

    Location is defined as the sample median. Scale is defined as:

        sd ~= 1.4826 * median( |x - median(x)| )

    where the constant 1.4826 makes the MAD consistent with the standard deviation under a
    normal model. If the MAD is degenerate (zero or non-finite), the unbiased sample standard
    deviation is used as a fallback, and finally a unit scale is used when insufficient data
    remain.

    Robust estimates are preferred throughout the simulation pipeline because accounting series
    commonly exhibit outliers, heavy tails, and occasional unit discontinuities. Median/MAD
    estimates reduce sensitivity to such features while remaining computationally cheap.

    Parameters
    ----------
    x:
        Array of observations; non-finite entries are discarded.

    Returns
    -------
    (float, float)
        Tuple ``(median, robust_sd)``.
 
    """
 
    x = np.asarray(x, dtype = float)

    x = x[np.isfinite(x)]

    if x.size == 0:

        return (0.0, 1.0)

    med = float(np.median(x))

    mad = float(np.median(np.abs(x - med)))

    sd = 1.4826 * mad

    if not np.isfinite(sd) or sd <= 0.0:

        sd = float(np.std(x, ddof = 1)) if x.size > 1 else 1.0

    if not np.isfinite(sd) or sd <= 0.0:

        sd = 1.0

    return (med, sd)


def skew_of_delta(
    delta
):
    """
    Compute the skewness implied by the skew-normal "delta" shape parameter.

    In the Azzalini skew-normal parameterisation, delta is a shape parameter in (-1, 1).
    The implied skewness is:

        a   = delta * sqrt(2 / pi)
     
        var = 1 - (2 * delta^2) / pi
     
        skewness = ((4 - pi) / 2) * a^3 / var^(3/2)

    The implementation includes a guard for degenerate variance at extreme delta values.

    Parameters
    ----------
    delta:
        Skew-normal delta in (-1, 1).

    Returns
    -------
    float
        The corresponding distribution skewness.
  
    """
  
    a = delta * math.sqrt(2.0 / math.pi)

    var = 1.0 - 2.0 * delta * delta / math.pi

    if var <= 0:

        return np.sign(delta) * 0.999

    num = (4.0 - math.pi) / 2.0 * a ** 3

    den = var ** 1.5

    return num / den


def _delta_from_target_skew_skewnormal(
    target_skew: float
) -> float:
    """
    Invert the skew-normal skewness mapping to obtain a delta parameter for a target skewness.

    The mapping ``delta -> skew_of_delta(delta)`` is monotone for delta in (-1, 1), allowing a
    bisection search to find a delta producing a desired skewness. The target skewness is clipped
    to a conservative interval to avoid numerical issues at extreme skewness.

    This inversion is used when calibrating skew-t innovations from consensus estimates where
    asymmetry is inferred via mean/median differences.

    Parameters
    ----------
    target_skew:
        Desired skewness (moment-based). Values outside [-0.95, 0.95] are clipped.

    Returns
    -------
    float
        Delta parameter in approximately [-0.995, 0.995] that matches the target skewness.
  
    """
  
    target = np.clip(target_skew, -0.95, 0.95)

    lo, hi = (-0.995, 0.995) if target >= 0 else (0.995, -0.995)

    for _ in range(60):
        mid = 0.5 * (lo + hi)

        s_mid = skew_of_delta(
            delta = mid
        )

        if s_mid < target and target >= 0 or (s_mid > target and target < 0):

            lo = mid

        else:

            hi = mid

    return np.clip(0.5 * (lo + hi), -0.995, 0.995)


def _skewt_standard_mean_var(
    delta: float,
    nu: float
):
    """
    Compute the mean and variance of the unstandardised skew-t innovation used by the model.

    Innovation construction
    -----------------------
    A skew-t random variable is constructed as a scale mixture of a skew-normal variable:

        Z0, Z1 ~ Normal(0, 1) independent
 
        U = delta * abs(Z0) + sqrt(1 - delta^2) * Z1

    and a chi-square scale mixture:

        W ~ ChiSquare(nu)
 
        S = sqrt(nu / W)
 
        T = S * U

    The skew-t draw used by the simulation is standardised afterwards, therefore the mean and
    variance of ``T`` are required.

    Moments
    -------
    For the skew-normal component:

        m_u = E[U]   = delta * sqrt(2 / pi)
      
        v_u = Var(U) = 1 - (2 * delta^2) / pi

    For the scale mixture (nu > 2):

        E[S]  = sqrt(nu / 2) * Gamma((nu - 1) / 2) / Gamma(nu / 2)
    
        E[S^2] = nu / (nu - 2)
    
        Var(S) = E[S^2] - (E[S])^2

    Combining the independent components yields:

        mean(T) = E[S] * m_u
    
        var(T)  = E[S^2] * v_u + Var(S) * m_u^2

    Parameters
    ----------
    delta:
        Skew-normal delta parameter in (-1, 1).
    nu:
        Student-t degrees of freedom. Values <= 2 are lifted to avoid infinite variance.

    Returns
    -------
    (float, float)
        Mean and variance of ``T``.
   
    """
   
    m_u = delta * math.sqrt(2.0 / math.pi)

    v_u = 1.0 - 2.0 * delta * delta / math.pi

    if nu <= 2.0:

        nu = 2.01

    lg = math.lgamma

    log_Es = 0.5 * math.log(nu / 2.0) + lg((nu - 1.0) / 2.0) - lg(nu / 2.0)

    Es = math.exp(log_Es)

    Es2 = nu / (nu - 2.0)

    VarS = Es2 - Es * Es

    m_t = Es * m_u

    v_t = Es2 * v_u + VarS * m_u ** 2

    return (m_t, v_t)


def _draw_skewt_standard(
    n_sims: int,
    delta: float,
    nu: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Draw standardised skew-t innovations with approximately zero mean and unit variance.

    The draw is constructed as described in ``_skewt_standard_mean_var``:

        Z0, Z1 ~ Normal(0, 1)
   
        W ~ ChiSquare(nu)
   
        S = sqrt(nu / W)
   
        U = delta * abs(Z0) + sqrt(1 - delta^2) * Z1
   
        T = S * U

    Standardisation is then applied using the analytic mean and variance of T:

        X = (T - mean(T)) / sqrt(var(T))

    The resulting ``X`` is used as an innovation term in multiple locations:

    - in the skew-t consensus draw generator (period-by-period),
  
    - in link-model residual generation for imputation, and
  
    - in AR(1) / copula-like reordering steps where a standardised latent variable is required.

    Parameters
    ----------
    n_sims:
        Number of independent draws.
    delta:
        Skew-normal delta parameter.
    nu:
        Degrees of freedom parameter for the chi-square mixture.
    rng:
        NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sims,)`` containing standardised skew-t draws.
    """
 
    z0 = rng.standard_normal(n_sims)

    z1 = rng.standard_normal(n_sims)

    w = rng.chisquare(df = nu, size = n_sims)

    s = np.sqrt(nu / w)

    u = delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1

    t = s * u

    m_t, v_t = _skewt_standard_mean_var(
        delta = delta,
        nu = nu
    )

    x = (t - m_t) / np.sqrt(v_t)

    return x


def _draw_skewt(
    loc: float,
    scale: float,
    delta: float,
    nu: float,
    size: tuple[int, ...],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Draw a location-scale skew-t random array.

    The standardised skew-t draw X from ``_draw_skewt_standard`` is transformed as:

        Y = loc + scale * X

    Parameters
    ----------
    loc:
        Location parameter (mean-like centre).
    scale:
        Scale parameter (standard deviation-like dispersion).
    delta:
        Skew-normal delta parameter controlling asymmetry.
    nu:
        Degrees of freedom controlling tail thickness.
    size:
        Output shape.
    rng:
        NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Skew-t draws with shape ``size``.
    """
  
    n = np.prod(size)

    z = _draw_skewt_standard(
        n_sims = n,
        delta = delta,
        nu = nu,
        rng = rng
    ).reshape(size)

    return loc + scale * z


def max_abs_lag_corr(
    b_arr: np.ndarray,
    max_lag_years: int,
    a
) -> float:
    """
    Compute the maximum absolute lagged Spearman correlation between two arrays over a lag window.

    For each integer lag k in [-max_lag_years, +max_lag_years], the lagged Spearman correlation is
    computed using ``_spearman_lag`` and the maximum absolute value is returned.

    This statistic is used when selecting predictors for imputation, favouring variables that have
    consistently strong monotonic association with the target even if the association is shifted
    by a small lag.

    Parameters
    ----------
    b_arr:
        Candidate predictor array.
    max_lag_years:
        Maximum absolute lag to consider.
    a:
        Target array.

    Returns
    -------
    float
        Maximum absolute lagged Spearman correlation.
    """
   
    best = 0.0

    for k in range(-max_lag_years, max_lag_years + 1):
      
        best = max(best, abs(_spearman_lag(
            x = a,
            y = b_arr,
            lag = k
        )))

    return best


def _choose_predictor_for_imputation(
    target: str,
    candidates: list[str],
    hist_yoy: pd.DataFrame | None,
    *,
    max_lag_years: int = 5,
    num_cache: dict[str, np.ndarray] | None = None
) -> str | None:
    """
    Select an imputation predictor variable using lagged rank correlation on historical innovations.

    The imputation system frequently needs to infer a missing future driver (for example, CapEx)
    from other simulated drivers. This helper chooses a predictor from ``candidates`` by:

    - extracting numeric innovation arrays from ``hist_yoy`` for the target and each candidate,
 
    - computing the maximum absolute Spearman rank correlation over a lag window, and
 
    - selecting the candidate with the highest score after applying a mild penalty to "similar"
      group members (to reduce circularity and near-duplicate predictors).

    The method is intentionally non-parametric (Spearman) and lag-aware, which is advantageous when
    historical relationships are monotone but not linear and when accounting recognition may lead or
    lag across line items.

    Parameters
    ----------
    target:
        Name of the target variable to be imputed.
    candidates:
        List of candidate predictor variable names.
    hist_yoy:
        DataFrame of historical innovations (year-on-year deltas), indexed by dates and with
        columns for each driver.
    max_lag_years:
        Maximum lag window used by ``max_abs_lag_corr``.
    num_cache:
        Optional cache mapping variable names to numeric arrays to avoid repeated coercion.

    Returns
    -------
    str | None
        Selected predictor name, or ``None`` when a suitable predictor cannot be identified.
 
    """
 
    if hist_yoy is None or hist_yoy.empty:

        return None

    if target not in hist_yoy.columns:

        return None

    if num_cache is None:

        num_cache = {}

    bad = _SIMILAR_GROUPS.get(target, set())

    if target in num_cache:

        a = num_cache[target]

    else:

        a = pd.to_numeric(hist_yoy[target], errors = 'coerce').to_numpy(dtype = float, copy = False)

        num_cache[target] = a

    best = None

    best_s = -1.0

    for pred in candidates:
        
        if pred == target:

            continue

        if pred not in hist_yoy.columns:

            continue

        if pred in num_cache:

            b = num_cache[pred]

        else:

            b = pd.to_numeric(hist_yoy[pred], errors = 'coerce').to_numpy(dtype = float, copy = False)

            num_cache[pred] = b

        if np.isfinite(b).sum() < MIN_POINTS or np.isfinite(a).sum() < MIN_POINTS:

            continue

        s = max_abs_lag_corr(
            b_arr = b,
            max_lag_years = max_lag_years,
            a = a
        )

        if pred in bad:

            s *= 0.7

        if s > best_s:

            best_s = s

            best = pred

    return best


def _hist_ratio_prior_from_history(
    y: pd.Series,
    x: pd.Series,
    *,
    nonneg: bool
) -> tuple[float, float, float, float] | None:
    """
    Construct a robust prior for the ratio y/x using historical observations.

    This helper is used to build priors for imputing missing drivers by scaling revenue.
    Given paired series y and x (for example, CapEx and Revenue), the ratio series is formed as:

        r_t = y_t / x_t

    with finite filtering and x_t = 0 treated as missing. A robust prior is then computed:

    - location: median(r)
   
    - dispersion: MAD-based robust sd(r)
   
    - bounds: [quantile(r, Q_LO) - pad, quantile(r, Q_HI) + pad]

    where ``pad`` is proportional to the interquartile range when available (``PAD_IQR * IQR``),
    otherwise ``3 * sd`` is used. When ``nonneg`` is True, absolute ratios are used and the lower
    bound is clipped at 0.

    Parameters
    ----------
    y, x:
        Series aligned on a datetime index.
    nonneg:
        If True, ratios are treated as non-negative magnitudes.

    Returns
    -------
    (float, float, float, float) | None
        Tuple ``(mu, sd, lo, hi)`` or ``None`` when insufficient data are available.
   
    """
   
    y = pd.to_numeric(y, errors = 'coerce')

    x = pd.to_numeric(x, errors = 'coerce').replace(0.0, np.nan)

    df = pd.concat([y, x], axis = 1).dropna()

    if len(df) < MIN_POINTS:

        return None

    df = df.iloc[-LAST_N:] if len(df) > LAST_N else df

    yv = df.iloc[:, 0]

    xv = df.iloc[:, 1].replace(0.0, np.nan)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
     
        ratio = (yv / xv).replace([np.inf, -np.inf], np.nan).dropna()

    if nonneg:

        ratio = ratio.abs()

    if len(ratio) < MIN_POINTS:

        return None

    mu = np.median(ratio)

    mad = 1.4826 * np.median(np.abs(ratio - mu)) if len(ratio) >= 2 else 0.0

    sd = max(mad, e6)

    q1 = ratio.quantile(0.25)

    q3 = ratio.quantile(0.75)

    iqr = q3 - q1 if np.isfinite(q3) and np.isfinite(q1) else np.nan

    lo = ratio.quantile(Q_LO)

    hi = ratio.quantile(Q_HI)

    pad = PAD_IQR * iqr if np.isfinite(iqr) and iqr > 0 else 3.0 * sd

    lo = lo - pad

    hi = hi + pad

    if nonneg:

        lo = max(0.0, lo)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:

        lo = mu - 3.0 * sd

        hi = mu + 3.0 * sd

        if nonneg:

            lo = max(0.0, lo)

    return (mu, sd, lo, hi)


def _derive_rev_priors_from_history(
    hist_ttm: pd.DataFrame | None
) -> dict[str, tuple[float, float, bool, float, float]]:
    """
    Derive per-driver revenue-scaling priors from historical TTM/FY panel data.

    For each driver column k other than revenue and selected rate-like or stock-like keys, a prior
    for the ratio k / revenue is estimated using ``_hist_ratio_prior_from_history``. The resulting
    priors are used by the imputation engine to synthesise missing future driver draws by:

        driver_draws ~= revenue_draws * ratio_draws

    where ratio_draws are drawn from a robust heavy-tailed distribution centred at the historical
    median ratio and clipped to plausible bounds.

    Parameters
    ----------
    hist_ttm:
        Historical panel of level series (TTM or annual) including a "revenue" column.

    Returns
    -------
    dict[str, tuple[float, float, bool, float, float]]
        Mapping ``driver_key -> (mu, sd, nonneg, lo, hi)``.
   
    """
   
    if hist_ttm is None or hist_ttm.empty:

        return {}

    if 'revenue' not in hist_ttm.columns:

        return {}

    rev = pd.to_numeric(hist_ttm['revenue'], errors = 'coerce').replace(0.0, np.nan).dropna()

    if len(rev) < MIN_POINTS:

        return {}

    NONNEG = {'capex', 'maint_capex', 'da', 'interest'}

    skip = {'revenue', 'net_debt'}

    skip |= {'tax', 'gross_margin', 'roe', 'roa', 'roe_pct', 'roa_pct'}

    out: dict[str, tuple[float, float, bool, float, float]] = {}

    for k in list(hist_ttm.columns):
     
        k = str(k)

        if k in skip:

            continue

        y = pd.to_numeric(hist_ttm[k], errors = 'coerce')

        nonneg = k in NONNEG

        pr = _hist_ratio_prior_from_history(
            y = y,
            x = rev,
            nonneg = nonneg
        )

        if pr is None:

            continue

        mu, sd, lo, hi = pr

        out[k] = (mu, sd, nonneg, lo, hi)

    return out


def _robust_ratio_prior(
    x: np.ndarray,
    *,
    nonneg: bool,
    lo: float,
    hi: float
) -> tuple[float, float, float, float]:
    """
    Compute a robust (median, sd) prior for a ratio series, retaining caller-supplied bounds.

    The function returns:

    - mu: median(x)
    - sd: max(1.4826 * MAD(x), e6, 0.10 * abs(mu))

    The additional ``0.10 * abs(mu)`` floor prevents unrealistically tight priors when the ratio
    is nearly constant historically but forward uncertainty should not collapse completely.

    Parameters
    ----------
    x:
        Numeric ratio sample.
    nonneg:
        If True, the absolute value is taken before estimating the prior.
    lo, hi:
        Bounds to be returned unchanged.

    Returns
    -------
    (float, float, float, float)
        Tuple ``(mu, sd, lo, hi)``.
  
    """
  
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    if x.size == 0:

        return (0.0, 0.0, lo, hi)

    if nonneg:

        x = np.abs(x)

    mu = np.median(x)

    mad = 1.4826 * np.median(np.abs(x - mu)) if x.size >= 2 else 0.0

    sd = max(mad, e6, 0.1 * abs(mu))

    return (mu, sd, lo, hi)


def _hist_scaled(
    name: str,
    unit_mult,
    hist_ttm,
    SCALE_KEYS
) -> pd.Series | None:
    """
    Extract and scale a historical series from a panel, applying a cash-unit multiplier when required.

    A number of drivers are represented in statement units that require scaling to a consistent cash
    unit system (for example, thousands or millions). The scaling set is supplied by ``SCALE_KEYS``.

    Parameters
    ----------
    name:
        Column name to extract.
    unit_mult:
        Multiplier applied when ``name`` is in ``SCALE_KEYS``.
    hist_ttm:
        Historical panel DataFrame.
    SCALE_KEYS:
        Set of keys that should be scaled by ``unit_mult``.

    Returns
    -------
    pandas.Series | None
        Scaled numeric series, or None when unavailable.
 
    """
 
    if hist_ttm is None or hist_ttm.empty or name not in hist_ttm.columns:

        return None

    s = pd.to_numeric(hist_ttm[name], errors = 'coerce').dropna()

    if name in SCALE_KEYS:

        s = s * unit_mult

    return s if len(s) else None


def _pct_to_dec_if_needed(
    x: np.ndarray
) -> np.ndarray:
    """
    Convert a percentage-point array to a decimal fraction array using a magnitude heuristic.

    CapIQ sources are inconsistent about whether rates are expressed as:

    - decimals (for example, 0.25), or
   
    - percentage points (for example, 25.0).

    The conversion rule applied is:

    - If the median absolute finite magnitude exceeds 1.5, treat the series as percentage points
      and divide by 100.
  
    - Otherwise, return the series unchanged.

    Parameters
    ----------
    x:
        Input array.

    Returns
    -------
    numpy.ndarray
        Converted array in decimal units when the heuristic triggers.
   
    """
   
    x = np.asarray(x, dtype = float)

    m = np.nanmedian(np.abs(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan

    if np.isfinite(m) and m > 1.5:

        return x / 100.0

    return x


def _safe_pos_growth(
    x: np.ndarray
) -> np.ndarray:
    """
    Compute simple growth rates for consecutive positive observations with finite filtering.

    For consecutive points (x0, x1) where both are finite and strictly positive, the growth rate is:

        g = x1 / x0 - 1

    The function returns an array of such growth rates, excluding any invalid pairs.

    Parameters
    ----------
    x:
        Level series.

    Returns
    -------
    numpy.ndarray
        Array of finite growth rates. May be empty when insufficient valid pairs exist.
   
    """
   
    x = np.asarray(x, dtype = float)

    if x.size < 2:

        return np.array([], dtype = float)

    x0 = x[:-1]

    x1 = x[1:]

    m = np.isfinite(x0) & np.isfinite(x1) & (x0 > 0) & (x1 > 0)

    if not m.any():

        return np.array([], dtype = float)

    g = x1[m] / x0[m] - 1.0

    g = g[np.isfinite(g)]

    return g


def estimate_terminal_growth_for_ddm(
    dps_future: pd.DataFrame | None,
    eps_future: pd.DataFrame | None = None,
    *,
    g_cap: float,
    sector_policy: SectorPolicy | None = None
):
    """
    Estimate a terminal dividend-per-share growth rate distribution for the dividend discount model.

    Estimation approach
    -------------------
    The input forecast table is expected to contain a "DPS" row. Annual periods are extracted using
    ``_annual_cols`` and a pool of recent positive DPS growth rates is formed:

        g_t = DPS_t / DPS_{t-1} - 1

    The raw terminal growth anchor is the median of the most recent ``LAST_N`` growth observations
    (when available). This raw estimate is then shrunk towards a long-run anchor ``ANCHOR``:

        g_faded = ANCHOR + (1 - shrink) * (g_raw - ANCHOR)

    Finally the terminal growth is clipped to:

        g_term = clip(g_faded, FLOOR, g_cap)

    where ``g_cap`` is typically derived from the cost of equity to enforce a no-arbitrage style
    constraint (growth cannot exceed the discount rate in perpetuity).

    Dispersion proxy
    ---------------
    A robust dispersion estimate is computed from the growth pool via MAD and scaled by an optional
    sector multiplier. The sigma is used later when simulating terminal growth draws.

    Advantages
    ----------
    - Using median growth and MAD reduces sensitivity to one-off dividend changes.
   
    - Shrinkage towards an anchor improves stability when only a short forecast history exists.
   
    - Capping by ``g_cap`` prevents numerical instability in the perpetuity formula.

    Parameters
    ----------
    dps_future:
        Consensus forecast table containing a "DPS" row.
    eps_future:
        Optional EPS table (not required by the current heuristic, retained for API symmetry).
    g_cap:
        Upper bound for the terminal growth rate.
    sector_policy:
        Optional sector policy controlling shrinkage and dispersion scaling.

    Returns
    -------
    (float, float)
        Tuple ``(g_term, sigma)``: terminal growth rate and a robust dispersion proxy.
  
    """
  
    if dps_future is None or dps_future.empty or 'DPS' not in dps_future.index:

        return (ANCHOR, 0.01)

    cols = _annual_cols(
        metric_df = dps_future
    )

    dps = pd.to_numeric(dps_future.loc['DPS', cols], errors = 'coerce').to_numpy(dtype = float)

    dps = dps[np.isfinite(dps)]

    g_pool = []

    if dps.size >= 2:

        g = _safe_pos_growth(
            x = dps
        )

        if g.size:

            g_last = g[-LAST_N:] if g.size >= LAST_N else g

            g_pool.extend(list(g_last))

    g_raw = float(np.median(g_pool)) if len(g_pool) else ANCHOR

    shrink = float(sector_policy.growth_shrink) if sector_policy is not None else SHRINK

    g_faded = float(ANCHOR + (1.0 - shrink) * (g_raw - ANCHOR))

    g_term = float(np.clip(g_faded, FLOOR, g_cap))

    gp = np.array([x for x in g_pool if np.isfinite(x)], dtype = float)

    if gp.size >= 2:

        sigma = float(1.4826 * np.median(np.abs(gp - np.median(gp))))

    else:

        sigma = 0.01

    if sector_policy is not None and np.isfinite(float(sector_policy.growth_sigma_mult)):

        sigma *= max(float(sector_policy.growth_sigma_mult), 0.0)

    return (g_term, sigma)


def estimate_terminal_growth_for_residual_income(
    eps_future: pd.DataFrame | None,
    dps_future: pd.DataFrame | None,
    roe_future: pd.DataFrame | None = None,
    *,
    g_cap: float,
    sector_policy: SectorPolicy | None = None
):
    """
    Estimate a terminal growth rate distribution for the residual income model.

    Candidate growth signals
    ------------------------
    Residual income valuation depends on the long-run growth of earnings and book value. The
    estimator forms a pooled signal from:

    1. EPS growth derived from annual EPS forecasts:

           g_eps_t = EPS_t / EPS_{t-1} - 1

    2. Sustainable growth implied by ROE and retention (when ROE, EPS, and DPS forecasts coexist):

           payout_t    = DPS_t / EPS_t
           retention_t = 1 - payout_t
           g_sust      = ROE_t * retention_t

       where ROE is converted to decimal units when provided in percentage points.

    The terminal growth is the median of the pooled candidates (when available), shrunk towards
    ``ANCHOR`` using the sector policy (or default shrinkage), and clipped to ``[FLOOR, g_cap]``.

    Dispersion is estimated robustly from the pooled candidates via MAD, with optional sector
    scaling.

    Advantages
    ----------
    - Incorporating ROE and retention provides an economically motivated anchor when EPS growth is
      noisy or sparse.

    - Robust statistics (median/MAD) reduce sensitivity to extreme forecast revisions.

    - The cap ``g_cap`` supports numerical stability in long-horizon discounting.

    Parameters
    ----------
    eps_future:
        Consensus EPS forecast table containing "EPS_Normalized".
    dps_future:
        Consensus DPS forecast table containing "DPS".
    roe_future:
        Optional ROE forecast table containing "ROE_pct".
    g_cap:
        Upper bound for terminal growth.
    sector_policy:
        Optional sector policy controlling shrinkage and dispersion scaling.

    Returns
    -------
    (float, float)
        Tuple ``(g_term, sigma)``: terminal growth rate and dispersion proxy.
 
    """
 
    g_pool = []

    if eps_future is not None and (not eps_future.empty) and ('EPS_Normalized' in eps_future.index):

        cols = _annual_cols(
            metric_df = eps_future
        )

        eps = pd.to_numeric(eps_future.loc['EPS_Normalized', cols], errors = 'coerce').to_numpy(dtype = float)

        eps = eps[np.isfinite(eps)]

        if eps.size >= 2:

            g = _safe_pos_growth(
                x = eps
            )

            if g.size:

                g_last = g[-LAST_N:] if g.size >= LAST_N else g

                g_pool.extend(list(g_last))

    if roe_future is not None and (not roe_future.empty) and ('ROE_pct' in roe_future.index) and (eps_future is not None) and (not eps_future.empty) and ('EPS_Normalized' in eps_future.index) and (dps_future is not None) and (not dps_future.empty) and ('DPS' in dps_future.index):

        cols = pd.Index(_annual_cols(
            metric_df = roe_future
        )).intersection(_annual_cols(
            metric_df = eps_future
        )).intersection(_annual_cols(
            metric_df = dps_future
        ))

        if len(cols) > 0:

            cols = pd.to_datetime(cols, errors = 'coerce')

            cols = pd.DatetimeIndex(cols[pd.notna(cols)]).sort_values()

            if len(cols) > 0:

                roe = pd.to_numeric(roe_future.reindex(columns = cols).loc['ROE_pct'], errors = 'coerce').to_numpy(dtype = float)

                roe = _pct_to_dec_if_needed(
                    x = roe
                )

                eps = pd.to_numeric(eps_future.reindex(columns = cols).loc['EPS_Normalized'], errors = 'coerce').to_numpy(dtype = float)

                dps = pd.to_numeric(dps_future.reindex(columns = cols).loc['DPS'], errors = 'coerce').to_numpy(dtype = float)

                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    payout = dps / eps

                retention = 1.0 - payout

                ok = np.where(np.isfinite(roe) & np.isfinite(retention) & (retention >= 0) & (retention <= 1) & (eps > 0))[0]

                if len(ok) > 0:

                    g_sust = float(roe[ok[-1]] * retention[ok[-1]])

                    if np.isfinite(g_sust):

                        g_pool.append(g_sust)

    g_raw = float(np.median(g_pool)) if len(g_pool) else ANCHOR

    shrink = float(sector_policy.growth_shrink) if sector_policy is not None else SHRINK

    g_faded = float(ANCHOR + (1.0 - shrink) * (g_raw - ANCHOR))

    g_term = float(np.clip(g_faded, FLOOR, g_cap))

    gp = np.array([x for x in g_pool if np.isfinite(x)], dtype = float)

    if gp.size >= 2:

        sigma = float(1.4826 * np.median(np.abs(gp - np.median(gp))))

    else:

        sigma = 0.01

    if sector_policy is not None and np.isfinite(float(sector_policy.growth_sigma_mult)):

        sigma *= max(float(sector_policy.growth_sigma_mult), 0.0)

    return (g_term, sigma)


def _simulate_link_model_yoy(
    model,
    x_yoy_draws: np.ndarray,
    *,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate target innovations from a fitted link model given predictor innovations.

    Two model forms are supported:

    1. Independent skew-t innovations:

           y = eps

       represented as ``("indep", loc, sc, delta, nu)``.

    2. Linear link with skew-t residuals:

           y = a + b * x + eps

       represented as ``(coef, loc, sc, delta, nu)`` where ``coef = [a, b]``.

    The residual ``eps`` is drawn from a location-scale skew-t distribution using ``_draw_skewt``.
    This allows asymmetric and heavy-tailed innovation behaviour, which is advantageous for
    accounting drivers where downside shocks and fat tails are empirically common.

    Parameters
    ----------
    model:
        Model tuple produced by ``_fit_link_model_yoy``.
    x_yoy_draws:
        Predictor innovations array of shape ``(T, n_sims)``.
    rng:
        NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Simulated target innovations of shape ``(T, n_sims)``.
   
    """
   
    x = np.asarray(x_yoy_draws, float)

    T, n = x.shape

    if isinstance(model, (tuple, list)) and len(model) and isinstance(model[0], str) and (model[0] == 'indep'):

        _, loc, sc, delta, nu = model

        eps = _draw_skewt(
            loc = loc,
            scale = sc,
            delta = delta,
            nu = nu,
            size = (T, n),
            rng = rng
        )

        return eps

    coef, loc, sc, delta, nu = model

    a, b = (coef[0], coef[1])

    eps = _draw_skewt(
        loc = loc,
        scale = sc,
        delta = delta,
        nu = nu,
        size = (T, n),
        rng = rng
    )

    return a + b * x + eps


def _fit_link_model_yoy(
    hist_y: pd.Series,
    hist_x: pd.Series
):
    """
    Fit a simple link model between two historical innovation series.

    The primary use is imputation: when a driver is missing in the forecast set, it can be
    synthesised from a related driver using an innovation-level relationship.

    Fitted model
    ------------
    When at least 8 paired observations exist:

        y_t = a + b * x_t + e_t

    is fitted by ordinary least squares. The residuals e_t are then calibrated to a skew-t
    distribution:

    - location and scale are estimated robustly (median and MAD),
   
    - skewness and excess kurtosis are estimated by moments, mapped to (delta, nu).

    When insufficient paired observations exist, an "independent" model is produced which models
    y innovations directly as skew-t without using x.

    Advantages
    ----------
    - Innovation-level modelling reduces sensitivity to level shifts and unit scaling.
   
    - A skew-t residual allows asymmetric shocks and fat tails, improving robustness relative to
      Gaussian residual assumptions.
   
    - The linear form is cheap and stable, appropriate for a large-scale Monte Carlo pipeline.

    Parameters
    ----------
    hist_y, hist_x:
        Historical innovation series aligned on a datetime index.

    Returns
    -------
    tuple
        Either ``("indep", loc, sc, delta, nu)`` or ``(coef, loc, sc, delta, nu)``.
  
    """
  
    y = pd.to_numeric(hist_y, errors = 'coerce').dropna()

    x = pd.to_numeric(hist_x, errors = 'coerce').dropna()

    idx = y.index.intersection(x.index)

    y = y.reindex(idx)

    x = x.reindex(idx)

    if len(idx) < 8:

        loc, sc = _robust_loc_scale(
            x = y.to_numpy(dtype = float)
        )

        skew, exk = _sample_skew_exkurt(
            x = y.to_numpy(dtype = float)
        )

        delta = _delta_from_target_skew_skewnormal(
            target_skew = skew
        )

        nu = _nu_from_excess_kurt(
            exk = exk
        )

        return ('indep', loc, max(sc, e12), delta, nu)

    X = np.column_stack([np.ones(len(idx)), x.to_numpy(dtype = float)])

    coef, *_ = np.linalg.lstsq(X, y.to_numpy(dtype = float), rcond = None)

    resid = y.to_numpy(dtype = float) - X @ coef

    loc, sc = _robust_loc_scale(
        x = resid
    )

    sc = max(sc, e12)

    skew, exk = _sample_skew_exkurt(
        x = resid
    )

    delta = _delta_from_target_skew_skewnormal(
        target_skew = skew
    )

    nu = _nu_from_excess_kurt(
        exk = exk
    )

    return (coef, loc, sc, delta, nu)


def _fiscal_quarter_num(
    d: pd.Timestamp,
    M: int
) -> int:
    """
    Convert a calendar month into a fiscal quarter number given a fiscal year-end month.

    The fiscal quarter number is computed by counting backwards from the fiscal year-end month M.
    For example, when M is December (12), the mapping is:

    - March -> Q1
    - June  -> Q2
    - September -> Q3
    - December  -> Q4

    Parameters
    ----------
    d:
        Date whose month determines the fiscal quarter.
    M:
        Fiscal year-end month (1..12).

    Returns
    -------
    int
        Fiscal quarter number in {1, 2, 3, 4}.
  
    """
  
    m = pd.Timestamp(d).month

    m_to_end = (M - m) % 12

    q = 4 - m_to_end // 3

    return np.clip(q, 1, 4)


def _yoy_kind_for_series(
    s: pd.Series,
    name: str
) -> str:
    """
    Choose an innovation transform ("log" or "diff") for a historical level series.

    For most strictly positive flow and stock series, log differences provide a scale-free measure
    of change:

        d_t = log(x_t) - log(x_{t-1})

    For rate-like series or series that are not reliably positive, arithmetic differences are used:

        d_t = x_t - x_{t-1}

    The decision is based on:

    - sufficient data availability,
  
    - whether the series is mostly positive, and
   
    - membership in key sets such as ``RATE_KEYS`` and ``FLOW_KEYS``.

    Parameters
    ----------
    s:
        Level series.
    name:
        Driver name used for heuristics.

    Returns
    -------
    str
        Either "log" or "diff".
   
    """
   
    x = pd.to_numeric(s, errors = 'coerce').to_numpy(dtype = float)

    x = x[np.isfinite(x)]

    if x.size < MIN_POINTS:

        return 'diff'

    mostly_pos = x.size >= MIN_POINTS and np.mean(x > 0) >= 0.85 and (np.mean(np.abs(x) < e12) < 0.1)

    if name in RATE_KEYS:

        return 'diff'

    if mostly_pos and (name in FLOW_KEYS or name in STOCK_KEYS):

        return 'log'

    return 'diff'


def _build_yoy_from_ttm_fy(
    hist_panel: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Build a year-on-year innovation panel from a historical level panel.

    For each column in ``hist_panel``, an innovation transform is selected using
    ``_yoy_kind_for_series`` and then applied:

    - "log":  d_t = log(x_t) - log(x_{t-1}) for x_t > 0
   
    - "diff": d_t = x_t - x_{t-1}

    Innovations are then winsorised by clipping to the [Q_LO, Q_HI] quantiles when sufficient data
    exist. This reduces the influence of extreme one-off changes on correlation estimation and
    imputation model fitting.

    Parameters
    ----------
    hist_panel:
        Historical panel of level series indexed by period end timestamps.

    Returns
    -------
    (pandas.DataFrame, dict[str, str])
        Innovation DataFrame and mapping ``column_name -> kind`` where kind is "log" or "diff".
  
    """
  
    if hist_panel is None or hist_panel.empty:

        return (pd.DataFrame(), {})

    yoy: dict[str, pd.Series] = {}

    kind: dict[str, str] = {}

    for c in hist_panel.columns:
     
        s = pd.to_numeric(hist_panel[c], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

        k = _yoy_kind_for_series(
            s = s,
            name = str(c)
        )

        kind[str(c)] = k

        if k == 'log':

            s_safe = s.where(s > e12)

            d = np.log(s_safe).diff()

        else:

            d = s.diff()

        d = d.replace([np.inf, -np.inf], np.nan)

        arr = d.to_numpy(dtype = float)

        z = arr[np.isfinite(arr)]

        if z.size >= 8:

            lo = np.nanquantile(z, Q_LO)

            hi = np.nanquantile(z, Q_HI)

            arr = np.clip(arr, lo, hi)

            d = pd.Series(arr, index = d.index, name = d.name)

        yoy[str(c)] = d

    yoy_df = pd.concat(yoy, axis = 1).sort_index()

    yoy_df.index = pd.DatetimeIndex(yoy_df.index).normalize()

    return (yoy_df, kind)


def _yoy_from_levels(
    levels: np.ndarray,
    base_prev: float,
    kind: str
) -> np.ndarray:
    """
    Compute innovations from a simulated level path given a previous-period anchor.

    The input ``levels`` is interpreted as a time-by-simulation matrix. The first innovation uses
    ``base_prev`` as the prior level:

    - kind == "log":

          d_0 = log(levels_0) - log(base_prev)
   
          d_t = log(levels_t) - log(levels_{t-1})

      with non-positive levels treated as missing (NaN) for the log transform.

    - kind != "log" ("diff"):

          d_0 = levels_0 - base_prev
    
          d_t = levels_t - levels_{t-1}

    Parameters
    ----------
    levels:
        Level array of shape ``(T, n_sims)``.
    base_prev:
        Prior level used for the first innovation.
    kind:
        Innovation transform ("log" or "diff").

    Returns
    -------
    numpy.ndarray
        Innovation array of shape ``(T, n_sims)``.
 
    """
 
    X = np.asarray(levels, float)

    T, n = X.shape

    prev = np.full((1, n), base_prev, dtype = float)

    prev = np.where(np.isfinite(prev) & (np.abs(prev) > e12), prev, np.nan)

    if kind == 'log':

        Xs = np.where(X > e12, X, np.nan)

        prevs = np.where(prev > e12, prev, np.nan)

        out0 = np.log(Xs[0:1, :]) - np.log(prevs)

        out = np.vstack([out0, np.diff(np.log(Xs), axis = 0)])

    else:

        out0 = X[0:1, :] - prev

        out = np.vstack([out0, np.diff(X, axis = 0)])

    return out


def _levels_from_yoy(
    yoy: np.ndarray,
    base_prev: float,
    kind: str,
    *,
    floor_at_zero: bool = False
) -> np.ndarray:
    """
    Reconstruct a level path from innovations given a previous-period anchor.

    This is the inverse operation of ``_yoy_from_levels`` under the chosen transform:

    - kind == "log":

          level_t = level_{t-1} * exp(d_t)

    - kind != "log" ("diff"):

          level_t = level_{t-1} + d_t

    Optionally the reconstructed levels are floored at zero, which is appropriate for quantities
    that are theoretically non-negative (for example, revenue, CapEx, depreciation).

    A log-innovation step d_t represents a multiplicative change exp(d_t). When the
    innovation distribution is heavy-tailed (for example, skew-t residuals during
    imputation), extremely large positive draws can cause exp(d_t) to overflow in float64.
    Such magnitudes correspond to implausible one-period growth factors and typically
    degrade the simulation by injecting infinities that propagate into NaNs. A conservative
    cap on the log step avoids overflow whilst preserving a wide range of growth outcomes.
    exp(50) is approximately 3e21, which is already far beyond realistic one-year changes
    # for statement items used in this model.
    
    Parameters
    ----------
    yoy:
        Innovation array of shape ``(T, n_sims)``.
    base_prev:
        Prior level used as the starting point.
    kind:
        Innovation transform ("log" or "diff").
    floor_at_zero:
        If True, apply ``max(level, 0)`` elementwise.

    Returns
    -------
    numpy.ndarray
        Level array of shape ``(T, n_sims)``.
  
    """
  
    d = np.asarray(yoy, float)

    T, n = d.shape

    out = np.empty((T, n), dtype = float)

    prev = np.full((n,), base_prev, dtype = float)

    for t in range(T):
      
        if kind == 'log':

            d_step = np.clip(d[t, :], -50.0, 50.0)

            prev = prev * np.exp(d_step)

        else:

            prev = prev + d[t, :]

        out[t, :] = prev

    if floor_at_zero:

        out = np.maximum(out, 0.0)

    return out


def _fy_end_timestamp(
    fy_end_year: int,
    fy_m: int,
    fy_d: int
) -> pd.Timestamp:
    """
    Construct a normalised fiscal year-end timestamp from year/month/day inputs.

    The day-of-month is clamped to the month-end when it exceeds the number of days in the month.
    The result is normalised to midnight.

    Parameters
    ----------
    fy_end_year:
        Fiscal year label (calendar year in which the fiscal year ends).
    fy_m:
        Fiscal year-end month (1..12).
    fy_d:
        Fiscal year-end day (1..31), clamped to month length.

    Returns
    -------
    pandas.Timestamp
        Normalised fiscal year-end timestamp.
        
    """
    
    fy_m = np.clip(fy_m, 1, 12)

    fy_d = max(1, fy_d)

    month_end = (pd.Timestamp(fy_end_year, fy_m, 1) + pd.offsets.MonthEnd(0)).normalize()

    day = min(fy_d, month_end.day)

    return pd.Timestamp(fy_end_year, fy_m, day).normalize()


def _fiscal_year_end_for_date(
    d: pd.Timestamp,
    fy_m: int,
    fy_d: int
) -> pd.Timestamp:
    """
    Determine the fiscal year-end date that contains a given calendar date.

    Given a fiscal year that ends on month/day (fy_m, fy_d), the fiscal year-end for a date d is:

    - fy_end(d.year) if d is on or before that year-end, else
   
    - fy_end(d.year + 1)

    where fy_end(y) is constructed by ``_fy_end_timestamp``.

    Parameters
    ----------
    d:
        Calendar date.
    fy_m, fy_d:
        Fiscal year-end month and day.

    Returns
    -------
    pandas.Timestamp
        Normalised fiscal year-end timestamp for the fiscal year containing d.
   
    """
   
    d = pd.Timestamp(d).normalize()

    fy_this = _fy_end_timestamp(
        fy_end_year = d.year,
        fy_m = fy_m,
        fy_d = fy_d
    )

    fy_year = d.year if d <= fy_this else d.year + 1

    return _fy_end_timestamp(
        fy_end_year = fy_year,
        fy_m = fy_m,
        fy_d = fy_d
    )


def _impute_missing_driver_draws(
    sim: dict[str, np.ndarray],
    hist_panel: pd.DataFrame | None,
    missing: set[str],
    ctx: RunContext,
    *,
    periods: pd.DatetimeIndex | None = None,
    period_types: list[str] | None = None,
    fy_m: int,
    fy_d: int,
    seasonal_flow_weights_q1_q4: np.ndarray | None = None,
    net_debt_draws: np.ndarray | None = None,
    rev_priors: dict[str, tuple[float, float, bool, float, float]] | None = None
):
    """
    Impute missing simulated driver paths required by downstream valuation formulas.

    Purpose
    -------
    The valuation engines (FCFF, FCFE, DDM, residual income) are expressed as algebraic combinations
    of a set of fundamental drivers (for example, revenue, EBIT, tax rate, CapEx, working capital).
    In practice, consensus workbooks are incomplete: some drivers may be missing entirely or may be
    present only as partial-period coverage. This function fills gaps by synthesising driver draw
    matrices so that method evaluation can proceed without fragile special-casing.

    Inputs and conventions
    ----------------------
    ``sim`` is a mapping ``driver_key -> draws`` where each draws matrix has shape (T, n_sims) and
    corresponds to the period grid produced earlier in the pipeline. All imputed outputs adhere to
    the same shape.

    A hierarchical imputation strategy is used, favouring identities and history-informed priors:

    1. Accounting identities (deterministic transforms)
    
       - EBITDA = EBIT + DA
    
       - EBIT   = EBITDA - DA
    
       - DA     = max(EBITDA - EBIT, 0)
    
       - EBIT   = EBT + Interest
    
       - EBT    = EBIT - Interest
    
       - FCF    = CFO - CapEx
    
       - Maintenance CapEx ~= max(DA, 0) or 0.75 * CapEx

       These transforms are used when the required operands are already available. Identities are
       preferred because they preserve internal consistency across the driver set.

    2. Driver-specific statistical priors
       - Tax rate:

             tax_draws = clip( t_draws(df=10, loc=median(tax_hist), scale=MAD(tax_hist)), 0, 0.40 )

         where percentage-point history is converted to decimals when required. A fallback prior of
         approximately 25% with 2% dispersion is used when history is unavailable.

       - Interest expense:

             cod_hist = abs(interest_hist) / abs(net_debt_hist)
             cod_draws = clip( t_draws(df=10, loc=median(cod_hist), scale=MAD(cod_hist)), 0, 0.25 )
             interest_draws = abs(net_debt_draws) * cod_draws

         When net debt draws are not available, a scale-anchored non-negative fallback is used.

       - Depreciation and amortisation:
       
         Prefer a revenue ratio prior when revenue is available:

             da_ratio_hist = abs(da_hist) / abs(revenue_hist)
             da_draws = max(revenue_draws * t_prior(da_ratio_hist), 0)

    3. Link-model imputation using historical innovations
    
       When historical panel data exist, year-on-year innovations are constructed for each driver
       using ``_build_yoy_from_ttm_fy``. A predictor is selected by maximising lagged Spearman
       correlation (``_choose_predictor_for_imputation``). A simple innovation link model is then
       fitted and simulated:

           y_t = a + b * x_t + eps_t

       where eps_t follows a skew-t distribution calibrated to residual skewness and kurtosis.
       The simulated innovations are integrated back into level paths using ``_levels_from_yoy``.

       For mixed period grids (annual + quarterly), annual levels are mapped back to quarterly rows:
    
       - additive flows are allocated across quarters, optionally using seasonal weights for fiscal
         quarters 1..4,
    
       - stock variables are carried through to each quarter in the fiscal year.

       Link-model imputation provides a mechanism for injecting economically plausible co-movement
       while remaining computationally inexpensive. Using innovations rather than levels reduces
       sensitivity to unit scaling and level shifts.

    4. Revenue-scaling priors
    
       When a driver has a historical ratio prior relative to revenue (``REV_PRIORS``), an imputed
       level path is produced as:

           ratio_draws = clip( t_draws(df=10, loc=mu, scale=sd), lo, hi )
        
           driver_draws = revenue_draws * ratio_draws

       with optional non-negativity enforcement.

    5. Fallback noise
     
       As a final guardrail, a heavy-tailed noise draw is produced with scale anchored to the median
       absolute magnitude of an available reference series (revenue, CFO, EBIT, EBITDA, or the first
       available simulated array). For inherently non-negative drivers (CapEx, DA, interest), the
       fallback is floored at zero.

    Advantages of the approach
    --------------------------
    - Robustness: valuation can proceed even under sparse forecast coverage.
    
    - Internal consistency: identities are exploited to reduce contradictory driver combinations.
    
    - Tail realism: Student-t and skew-t innovations avoid excessive reliance on Gaussian noise.
    
    - Plausible dependence: predictor selection and link models preserve monotone co-movement without
      requiring full multivariate calibration.

    Parameters
    ----------
    sim:
        Mapping of available driver draw matrices.
    hist_panel:
        Historical panel of level series used to infer priors and link relationships.
    missing:
        Set of driver keys that should be imputed when absent from ``sim``.
    ctx:
        Run context providing deterministic RNG streams.
    periods:
        Period end timestamps aligned to the rows of the draw matrices, used for annual/quarter
        mapping in mixed grids.
    period_types:
        List of period type strings ("Annual" or "Quarterly") aligned to ``periods``.
    fy_m, fy_d:
        Fiscal year-end month and day, used to map quarter ends to fiscal years.
    seasonal_flow_weights_q1_q4:
        Optional array of 4 seasonal weights used to allocate annual flow totals to quarters.
    net_debt_draws:
        Optional net debt draws used for interest expense imputation.
    rev_priors:
        Optional precomputed revenue ratio priors, used to avoid recomputation across calls.

    Returns
    -------
    (dict[str, numpy.ndarray], set[str])
        Updated simulation mapping and the set of keys that were imputed by this call.
  
    """
  
    imputed: set[str] = set()

    if not missing:

        return (sim, imputed)

    first = None

    for v in sim.values():
      
        if isinstance(v, np.ndarray):

            first = v

            break

    if first is None:

        return (sim, imputed)

    T, n = first.shape

    hist_scaled_cache: dict[tuple[str, float], pd.Series | None] = {}

    hist_num_cache: dict[str, np.ndarray] = {}


    def _hist_scaled_cached(
        name: str
    ) -> pd.Series | None:
        """
        Fetch a scaled historical series from ``hist_panel`` with memoisation.

        The helper caches the scaled series keyed by (name, UNIT_MULT). It is used to avoid
        repeated ``pandas.to_numeric`` coercion and unit scaling inside the driver loop.

        Parameters
        ----------
        name:
            Historical panel column name to extract.

        Returns
        -------
        pandas.Series | None
            Scaled historical series, or None when unavailable.
     
        """
     
        key = (name, UNIT_MULT)

        if key in hist_scaled_cache:

            return hist_scaled_cache[key]

        s = _hist_scaled(
            name = name,
            unit_mult = UNIT_MULT,
            hist_ttm = hist_panel,
            SCALE_KEYS = SCALE_KEYS
        )

        hist_scaled_cache[key] = s

        return s


    keys_avail = [k for k in sim.keys() if isinstance(sim.get(k), np.ndarray)]

    SCALE_KEYS = {'fcf', 'cfo', 'capex', 'maint_capex', 'interest', 'da', 'ebit', 'ebitda', 'ebt', 'net_income', 'revenue', 'net_debt'}

    if 'ebitda' in missing and 'ebit' in sim and ('da' in sim):

        sim['ebitda'] = sim['ebit'] + sim['da']

        imputed.add('ebitda')

    if 'ebit' in missing and 'ebitda' in sim and ('da' in sim):

        sim['ebit'] = sim['ebitda'] - sim['da']

        imputed.add('ebit')

    if 'da' in missing and 'ebitda' in sim and ('ebit' in sim):

        sim['da'] = np.maximum(sim['ebitda'] - sim['ebit'], 0.0)

        imputed.add('da')

    if 'ebit' in missing and 'ebt' in sim and ('interest' in sim):

        sim['ebit'] = sim['ebt'] + sim['interest']

        imputed.add('ebit')

    if 'fcf' in missing and 'cfo' in sim and ('capex' in sim):

        sim['fcf'] = sim['cfo'] - sim['capex']

        imputed.add('fcf')

    if 'ebt' in missing and 'ebit' in sim and ('interest' in sim):

        sim['ebt'] = sim['ebit'] - sim['interest']

        imputed.add('ebt')

    if 'maint_capex' in missing and 'da' in sim:

        sim['maint_capex'] = np.maximum(sim['da'], 0.0)

        imputed.add('maint_capex')

    if 'maint_capex' in missing and 'capex' in sim and ('maint_capex' not in sim):

        sim['maint_capex'] = np.maximum(0.75 * sim['capex'], 0.0)

        imputed.add('maint_capex')

    REV_PRIORS = rev_priors if rev_priors is not None else _derive_rev_priors_from_history(
        hist_ttm = hist_panel
    )

    for v in list(missing):
        if v in sim:

            continue

        rng_v = ctx.rng(f'impute:{v}:base')

        if v == 'tax':

            tx_hist = _hist_scaled_cached(
                name = 'tax'
            )

            if tx_hist is not None and len(tx_hist) >= 4:

                tx = tx_hist.to_numpy(dtype = float)

                med0 = np.median(tx)

                if med0 > 1.5:

                    tx = tx / 100.0

                tx = tx[np.isfinite(tx)]

                if tx.size:

                    mu = np.median(tx)

                    mad = 1.4826 * np.median(np.abs(tx - mu)) if tx.size >= 2 else 0.02

                    eps = rng_v.standard_t(df = 10, size = (T, n)) * max(mad, 0.01) + mu

                    sim[v] = np.clip(eps, 0.0, 0.4)

                    imputed.add(v)

                    continue

            eps = rng_v.standard_t(df = 10, size = (T, n)) * 0.02 + 0.25

            sim[v] = np.clip(eps, 0.0, 0.4)

            imputed.add(v)

            continue

        if v == 'interest':

            nd = net_debt_draws

            if nd is None:

                nd = sim.get('net_debt', None)

            cod_mu = np.nan

            cod_sd = np.nan

            int_hist = _hist_scaled(
                name = 'interest',
                unit_mult = UNIT_MULT,
                hist_ttm = hist_panel,
                SCALE_KEYS = SCALE_KEYS
            )

            nd_hist = _hist_scaled(
                name = 'net_debt',
                unit_mult = UNIT_MULT,
                hist_ttm = hist_panel,
                SCALE_KEYS = SCALE_KEYS
            )

            if int_hist is not None and nd_hist is not None:

                dfh = pd.concat([int_hist.abs(), nd_hist.abs()], axis = 1).dropna()

                if len(dfh) >= 5:

                    with np.errstate(divide = 'ignore', invalid = 'ignore'):
                        cod = (dfh.iloc[:, 0] / dfh.iloc[:, 1]).replace([np.inf, -np.inf], np.nan).dropna()

                    if len(cod) >= 5:

                        cod_mu = np.median(cod)

                        cod_mu = np.clip(cod_mu, 0.0, 0.25)

                        mad = 1.4826 * np.median(np.abs(cod - cod_mu)) if len(cod) >= 2 else 0.005

                        cod_sd = np.clip(max(mad, 0.002), 0.002, 0.05)

            if not np.isfinite(cod_mu):

                cod_mu = 0.04

                cod_sd = 0.01

            if nd is not None:

                cod_draw = rng_v.standard_t(df = 10, size = (T, n)) * cod_sd + cod_mu

                cod_draw = np.clip(cod_draw, 0.0, 0.25)

                sim[v] = np.abs(nd) * cod_draw

                imputed.add(v)

                continue

            anchor = None

            for k in ('revenue', 'cfo', 'ebit', 'ebitda'):
                if k in sim:

                    anchor = sim[k]

                    break

            if anchor is None:

                anchor = first

            scale = np.nanmedian(np.abs(anchor)) if np.isfinite(anchor).any() else 1.0

            sim[v] = np.maximum(rng_v.standard_t(df = 10, size = (T, n)) * (0.01 * max(scale, 1.0)), 0.0)

            imputed.add(v)

            continue

        if v == 'da':

            if 'revenue' in sim:

                da_hist = _hist_scaled(
                    name = 'da',
                    unit_mult = UNIT_MULT,
                    hist_ttm = hist_panel,
                    SCALE_KEYS = SCALE_KEYS
                )

                rev_hist = _hist_scaled(
                    name = 'revenue',
                    unit_mult = UNIT_MULT,
                    hist_ttm = hist_panel,
                    SCALE_KEYS = SCALE_KEYS
                )

                if da_hist is not None and rev_hist is not None:

                    dfh = pd.concat([da_hist.abs(), rev_hist.replace(0.0, np.nan).abs()], axis = 1).dropna()

                    if len(dfh) >= 5:

                        with np.errstate(divide = 'ignore', invalid = 'ignore'):
                            ratio = (dfh.iloc[:, 0] / dfh.iloc[:, 1]).replace([np.inf, -np.inf], np.nan).dropna()

                        if len(ratio) >= 5:

                            mu, sd, lo, hi = _robust_ratio_prior(
                                x = ratio.to_numpy(),
                                nonneg = True,
                                lo = 0.0,
                                hi = 0.25
                            )

                            rdraw = rng_v.standard_t(df = 10, size = (T, n)) * sd + mu

                            rdraw = np.clip(rdraw, lo, hi)

                            sim[v] = np.maximum(sim['revenue'] * rdraw, 0.0)

                            imputed.add(v)

                            continue

            if 'revenue' in sim and v in REV_PRIORS:

                mu, sd, nonneg, lo, hi = REV_PRIORS[v]

                ratio = rng_v.standard_t(df = 10, size = (T, n)) * sd + mu

                ratio = np.clip(ratio, lo, hi)

                sim[v] = np.maximum(sim['revenue'] * ratio, 0.0)

                imputed.add(v)

                continue

        if hist_panel is not None and (not hist_panel.empty):

            hist_yoy, yoy_kind = _build_yoy_from_ttm_fy(
                hist_panel = hist_panel
            )

            pred = _choose_predictor_for_imputation(
                target = v,
                candidates = keys_avail,
                hist_yoy = hist_yoy,
                num_cache = hist_num_cache
            )

            if pred is not None and v in hist_yoy.columns and (pred in hist_yoy.columns):

                y_hist = pd.to_numeric(hist_yoy[v], errors = 'coerce')

                x_hist = pd.to_numeric(hist_yoy[pred], errors = 'coerce')

                idx = y_hist.dropna().index.intersection(x_hist.dropna().index)

                if len(idx) >= MIN_POINTS:

                    rng_local = ctx.rng(f'impute:{v}:link')

                    model = _fit_link_model_yoy(
                        hist_y = y_hist,
                        hist_x = x_hist
                    )

                    X_full = np.asarray(sim[pred], float)

                    if periods is not None and period_types is not None and (len(period_types) == X_full.shape[0]):

                        ann_idx = np.array([i for i, t in enumerate(period_types) if str(t).lower() == 'annual'], dtype = int)

                        q_idx = np.array([i for i, t in enumerate(period_types) if str(t).lower() == 'quarterly'], dtype = int)

                    else:

                        ann_idx = np.arange(X_full.shape[0], dtype = int)

                        q_idx = np.array([], dtype = int)

                    X_ann = X_full[ann_idx, :]

                    base_x = pd.to_numeric(hist_panel.get(pred, pd.Series(dtype = float)), errors = 'coerce').dropna().iloc[-1] if pred in hist_panel.columns and pd.to_numeric(hist_panel[pred], errors = 'coerce').dropna().size else np.nanmedian(X_ann)

                    kx = yoy_kind.get(pred, 'diff')

                    x_yoy_draws = _yoy_from_levels(
                        levels = X_ann,
                        base_prev = base_x,
                        kind = kx
                    )

                    y_yoy_draws = _simulate_link_model_yoy(
                        model = model,
                        x_yoy_draws = x_yoy_draws,
                        rng = rng_local
                    )

                    base_y = pd.to_numeric(hist_panel.get(v, pd.Series(dtype = float)), errors = 'coerce').dropna().iloc[-1] if v in hist_panel.columns and pd.to_numeric(hist_panel[v], errors = 'coerce').dropna().size else np.nanmedian(X_ann)

                    ky = yoy_kind.get(v, 'diff')

                    floor0 = v in {'capex', 'maint_capex', 'da', 'interest'}

                    Y_ann = _levels_from_yoy(
                        yoy = y_yoy_draws,
                        base_prev = base_y,
                        kind = ky,
                        floor_at_zero = floor0
                    )

                    Y_full = np.full_like(X_full, np.nan, dtype = float)

                    Y_full[ann_idx, :] = Y_ann

                    if q_idx.size and periods is not None and (period_types is not None):

                        periods_dt = pd.DatetimeIndex(pd.to_datetime(periods, errors = 'coerce')).normalize()

                        ann_dates = periods_dt[ann_idx]

                        ann_map = {pd.Timestamp(d).normalize(): j for j, d in enumerate(ann_dates)}

                        q_by_fy: dict[pd.Timestamp, list[int]] = {}

                        for i_q in q_idx.tolist():
                        
                            qd = pd.Timestamp(periods_dt[i_q]).normalize()

                            fy = pd.Timestamp(_fiscal_year_end_for_date(
                                d = qd,
                                fy_m = fy_m,
                                fy_d = fy_d
                            )).normalize()

                            q_by_fy.setdefault(fy, []).append(i_q)

                        for fy_end, qpos_list in q_by_fy.items():
                          
                            if fy_end not in ann_map:

                                continue

                            jA = ann_map[fy_end]

                            A_level = Y_ann[jA, :]

                            qpos_list = sorted(qpos_list, key = lambda i_: periods_dt[i_])

                            qs = [pd.Timestamp(periods_dt[i_]).normalize() for i_ in qpos_list]

                            if v in FLOW_KEYS:

                                if seasonal_flow_weights_q1_q4 is not None and len(seasonal_flow_weights_q1_q4) == 4:

                                    fq_nums = np.array([_fiscal_quarter_num(
                                        d = q,
                                        M = fy_m
                                    ) for q in qs], dtype = int)

                                    w_raw = np.array([seasonal_flow_weights_q1_q4[qnum - 1] for qnum in fq_nums], dtype = float)

                                    w_raw = np.where(np.isfinite(w_raw) & (w_raw > 0), w_raw, 0.0)

                                    w = w_raw / w_raw.sum() if w_raw.sum() > e12 else np.full(len(qs), 1.0 / len(qs))

                                else:

                                    w = np.full(len(qs), 1.0 / len(qs))

                                for jj, i_q in enumerate(qpos_list):
                                 
                                    Y_full[i_q, :] = A_level * w[jj]

                            else:

                                for i_q in qpos_list:
                                 
                                    Y_full[i_q, :] = A_level

                    sim[v] = Y_full

                    imputed.add(v)

                    continue

        if 'revenue' in sim and v in REV_PRIORS:

            mu, sd, nonneg, lo, hi = REV_PRIORS[v]

            rng_local = ctx.rng(f'impute:{v}:revprior')

            ratio = rng_local.standard_t(df = 10, size = (T, n)) * sd + mu

            ratio = np.clip(ratio, lo, hi)

            out = sim['revenue'] * ratio

            if nonneg:

                out = np.maximum(out, 0.0)

            sim[v] = out

            imputed.add(v)

            continue

        anchor = None

        for k in ('revenue', 'cfo', 'ebit', 'ebitda'):
            if k in sim:

                anchor = sim[k]

                break

        if anchor is None:

            anchor = first

        rng_local = ctx.rng(f'impute:{v}:fallback')

        scale = np.nanmedian(np.abs(anchor)) if np.isfinite(anchor).any() else 1.0

        noise = rng_local.standard_t(df = 10, size = (T, n)) * (0.05 * max(scale, 1.0))

        if v in {'capex', 'maint_capex', 'da', 'interest'}:

            noise = np.maximum(noise, 0.0)

        sim[v] = noise

        imputed.add(v)

    return (sim, imputed)


def _spearman_to_latent_pearson(
    rho_s: float
) -> float:
    """
    Approximate a latent Pearson correlation from an observed Spearman rank correlation.

    Background
    ----------
    In a Gaussian copula setting, the Spearman rank correlation rho_s and the Pearson linear
    correlation rho are related by:

        rho_s = (6 / pi) * arcsin(rho / 2)

    Inverting this mapping yields:

        rho = 2 * sin(pi * rho_s / 6)

    This function applies the inverse mapping as a pragmatic conversion when a rank correlation is
    estimated robustly from historical innovations (Spearman) but a linear correlation matrix is
    required for Cholesky factorisation of a multivariate latent innovation model.

    Practical considerations
    ------------------------
    - The input is clipped to [-0.99, 0.99] to avoid unstable behaviour near the boundaries.
 
    - Non-finite inputs are treated as 0.0.

    Parameters
    ----------
    rho_s:
        Spearman rank correlation estimate.

    Returns
    -------
    float
        Approximate latent Pearson correlation.
  
    """
   
    rho_s = np.clip(rho_s, -0.99, 0.99) if np.isfinite(rho_s) else 0.0

    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def _hist_change_for_corr(
    x: np.ndarray,
    name: str
) -> np.ndarray:
    """
    Compute a robust innovation series from historical levels for correlation estimation.

    Correlation in this model is estimated on innovations rather than levels to reduce:

    - sensitivity to unit scaling and level shifts, and
   
    - spurious correlation induced by shared trends.

    Innovation transform
    --------------------
    A transform is selected using a simple positivity heuristic:

    - If the series is "mostly positive", use log differences:

          d_t = log(x_t) - log(x_{t-1})

      where x_t <= 0 is treated as missing for the log.

    - Otherwise, use arithmetic differences:

          d_t = x_t - x_{t-1}

    Certain rate-like series (tax rate, ROE) are forced to use differences even if positive, as
    log transforms are not meaningful for bounded rates.

    Robustification
    ---------------
    When sufficient finite innovations exist, the innovations are clipped to the quantile band
    [Q_LO, Q_HI]. This winsorisation mitigates the impact of extreme outliers on dependence
    estimation.

    Parameters
    ----------
    x:
        Level series array.
    name:
        Variable name used for transform heuristics.

    Returns
    -------
    numpy.ndarray
        Innovation array with the same shape as ``x``. The first element is NaN by construction.
 
    """
 
    x = np.asarray(x, float)

    out = np.full_like(x, np.nan)

    finite = np.isfinite(x)

    x0 = x[finite]

    if x0.size < 2:

        return out

    mostly_pos = x0.size >= MIN_POINTS and np.mean(x0 > 0) >= 0.85 and (np.mean(np.abs(x0) < e12) < 0.1)

    if name in {'tax', 'roe', 'roe_pct'}:

        mostly_pos = False

    if mostly_pos:

        x_safe = np.where(x > e12, x, np.nan)

        out[1:] = np.diff(np.log(x_safe))

    else:

        out[1:] = np.diff(x)

    z = out[np.isfinite(out)]

    if z.size >= 8:

        lo = np.nanquantile(z, Q_LO)

        hi = np.nanquantile(z, Q_HI)

        out = np.clip(out, lo, hi)

    return out


def _build_joint_corr_multi_from_history(
    *,
    var_periods: dict[str, pd.DatetimeIndex],
    hist_annual: pd.DataFrame | None,
    min_points: int = 12
) -> tuple[np.ndarray, list[str], float] | None:
    """
    Estimate a joint dependence structure for multiple drivers using historical annual data.

    Purpose
    -------
    A large portion of the Monte Carlo engine simulates each driver marginally, period by period,
    from consensus estimates (or imputed priors). To avoid unrealistic combinations (for example,
    simultaneously extreme revenue growth and collapsing margins), a dependence adjustment is
    applied later by reordering draws. This helper provides the correlation matrix and an optional
    degrees-of-freedom parameter for a multivariate t latent innovation model used in that step.

    Estimation workflow
    -------------------
    1. Candidate variable selection:
  
       Only variables present in ``hist_annual`` are considered, based on the keys in ``var_periods``.

    2. Innovation construction:
   
       For each candidate variable v, a historical innovation series is computed using
       ``_hist_change_for_corr`` (log differences for mostly-positive series, differences otherwise),
       then robustly standardised using the median and MAD:

           z_t = (d_t - median(d)) / (1.4826 * median(|d - median(d)|))

       with fallbacks to sample standard deviation when MAD is degenerate.

    3. Pairwise dependence estimation:
   
       For each pair (i, j), Spearman rank correlation is computed on time-aligned innovations,
       then converted to an approximate latent Pearson correlation using ``_spearman_to_latent_pearson``.
       Pairwise correlations are clipped to a conservative bound to improve numerical stability.

    4. PSD repair:
   
       The resulting matrix is repaired to a valid correlation matrix using ``_nearest_psd_corr``.

    5. Tail parameter calibration:
   
       A single degrees-of-freedom parameter nu is derived from the median of the per-variable
       excess kurtosis estimates. The mapping used is:

           nu ~= 4 + 6 / median_excess_kurtosis

       clipped to a reasonable range. When kurtosis information is unavailable, nu defaults to 8.

    Advantages
    ----------
   
    - Rank correlation is robust to heavy tails and outliers.
   
    - Innovation-level dependence avoids spurious correlations from common trends.
   
    - PSD repair prevents Cholesky failures downstream without ad-hoc jittering.
   
    - A t-copula style latent model captures tail dependence more realistically than a Gaussian model
      at comparable computational cost.

    Parameters
    ----------
    var_periods:
        Mapping from variable name to the valuation period grid where the variable is used. Only the
        keys are required here; the periods are retained for API symmetry with later steps.
    hist_annual:
        Historical annual panel used to compute innovations.
    min_points:
        Minimum number of time-aligned innovation points required for each pairwise estimate.

    Returns
    -------
    (numpy.ndarray, list[str], float) | None
        Tuple ``(R, keep_vars, nu)`` where:
 
        - R is a correlation matrix for the variables in keep_vars,
 
        - keep_vars is the variable ordering corresponding to R, and
 
        - nu is the degrees-of-freedom parameter for a multivariate t latent model.
 
        Returns ``None`` when insufficient historical data are available.
 
    """
 
    if hist_annual is None or hist_annual.empty:

        logger.info('No historical annual data available for correlation estimation.')

        return None

    cand = [v for v in var_periods.keys() if v in hist_annual.columns]

    if len(cand) < 2:

        logger.info('Not enough variables with historical annual data for correlation estimation.')

        return None

    z_series: dict[str, pd.Series] = {}

    exk_list: list[float] = []

    for v in cand:
        
        s = pd.to_numeric(hist_annual[v], errors = 'coerce').replace([np.inf, -np.inf], np.nan).dropna()

        if s.shape[0] < min_points:

            continue

        innov = pd.Series(_hist_change_for_corr(
            x = s.to_numpy(dtype = float),
            name = v
        ), index = s.index)

        innov = innov.replace([np.inf, -np.inf], np.nan).dropna()

        if innov.shape[0] < min_points:

            continue

        x = innov.to_numpy(dtype = float)

        med = float(np.nanmedian(x))

        mad = float(np.nanmedian(np.abs(x - med)))

        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale <= 0.0:

            scale = float(np.nanstd(x, ddof = 1))

        if not np.isfinite(scale) or scale <= 0.0:

            scale = 1.0

        z = (innov - med) / scale

        z_series[v] = z

        try:

            exk = float(kurtosis(z.to_numpy(dtype = float), fisher = True, bias = False))

            if np.isfinite(exk):

                exk_list.append(exk)

        except (TypeError, ValueError, FloatingPointError):

            pass

    keep_vars = list(z_series.keys())

    if len(keep_vars) < 2:

        logger.info('Not enough variables with sufficient historical data for correlation estimation after processing.')

        return None

    d = len(keep_vars)

    R = np.eye(d, dtype = float)

    for i in range(d):
        
        for j in range(i + 1, d):
        
            zi = z_series[keep_vars[i]]

            zj = z_series[keep_vars[j]]

            idx_i = pd.to_datetime(zi.index, errors = 'coerce')

            idx_j = pd.to_datetime(zj.index, errors = 'coerce')

            m_i = pd.notna(idx_i)

            m_j = pd.notna(idx_j)

            si = pd.Series(pd.to_numeric(zi.to_numpy(dtype = float, copy = False)[m_i], errors = 'coerce'), index = pd.Index(pd.DatetimeIndex(idx_i[m_i]).view('i8'), dtype = 'int64'), name = 'x')

            sj = pd.Series(pd.to_numeric(zj.to_numpy(dtype = float, copy = False)[m_j], errors = 'coerce'), index = pd.Index(pd.DatetimeIndex(idx_j[m_j]).view('i8'), dtype = 'int64'), name = 'y')

            si = si[~si.index.duplicated(keep = 'last')]

            sj = sj[~sj.index.duplicated(keep = 'last')]

            aligned = pd.concat([si, sj], axis = 1).dropna()

            if aligned.shape[0] < min_points:

                rho_s = 0.0

            else:

                rho_s = float(aligned.corr(method = 'spearman').iloc[0, 1])

            rho = float(_spearman_to_latent_pearson(
                rho_s = rho_s
            ))

            R[i, j] = R[j, i] = float(np.clip(rho, -0.95, 0.95))

    R = _nearest_psd_corr(
        R = R
    )

    nu = 8.0

    if exk_list:

        exk_med = float(np.nanmedian([e for e in exk_list if np.isfinite(e)]))

        if np.isfinite(exk_med) and exk_med > 1e-06:

            nu = float(np.clip(4.0 + 6.0 / exk_med, 4.5, 60.0))

    return (R, keep_vars, nu)


def _estimate_ar1_phi_multi_from_history(
    *,
    hist_annual: pd.DataFrame,
    vars_: Sequence[str],
    min_points: int = 10,
    phi_cap: float = 0.97
) -> dict[str, float]:
    """
    Estimate a per-variable AR(1) persistence parameter from historical innovations.

    Role in the model
    -----------------
    The dependence adjustment step uses a latent innovation representation to reorder simulated
    marginal draws to better match historical co-movement. Empirically, innovations exhibit mild
    persistence. This helper estimates a simple lag-1 persistence parameter for each variable and
    is used to introduce temporal correlation into the latent innovations:

        z_t = phi * z_{t-1} + sqrt(1 - phi^2) * e_t

    where e_t is an i.i.d. latent innovation draw with the cross-sectional dependence structure
    supplied by the correlation matrix.

    Estimation method
    -----------------
    For each variable:

    1. Construct innovations using ``_hist_change_for_corr``.
   
    2. Robustly standardise innovations by subtracting the median and dividing by MAD-based scale.
   
    3. Estimate phi as the Pearson correlation between z_t and z_{t-1}.
   
    4. Clip phi to [0, phi_cap] and default to 0 when insufficient data exist.

    The use of robust standardisation reduces the influence of outliers on persistence estimates.

    Parameters
    ----------
    hist_annual:
        Historical annual panel.
    vars_:
        Sequence of variable names for which to estimate phi.
    min_points:
        Minimum number of observations required for each variable.
    phi_cap:
        Upper bound applied to phi to prevent near-unit-root behaviour.

    Returns
    -------
    dict[str, float]
        Mapping ``variable -> phi``.
   
    """
   
    out: dict[str, float] = {}

    if hist_annual is None or hist_annual.empty:

        return {str(v): 0.0 for v in vars_}

    for v in vars_:
        v = str(v)

        if v not in hist_annual.columns:

            out[v] = 0.0

            continue

        s = pd.to_numeric(hist_annual[v], errors = 'coerce').to_numpy(dtype = float)

        s = s[np.isfinite(s)]

        if s.size < min_points + 2:

            out[v] = 0.0

            continue

        innov = _hist_change_for_corr(
            x = s,
            name = v
        )

        innov = innov[np.isfinite(innov)]

        if innov.size < min_points + 2:

            out[v] = 0.0

            continue

        mu = float(np.nanmedian(innov))

        mad = float(1.4826 * np.nanmedian(np.abs(innov - mu))) if innov.size >= 2 else 0.0

        sc = max(mad, 1e-08)

        z = (innov - mu) / sc

        z = z[np.isfinite(z)]

        if z.size < min_points + 2:

            out[v] = 0.0

            continue

        z0 = z[:-1]

        z1 = z[1:]

        if z0.size < min_points:

            out[v] = 0.0

            continue

        phi = float(np.corrcoef(z0, z1)[0, 1])

        if not np.isfinite(phi):

            phi = 0.0

        phi = max(0.0, min(phi, phi_cap))

        out[v] = phi

    return out


def _mv_t_innovations(
    *,
    n_sims: int,
    R: np.ndarray,
    nu: float | None,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Draw latent multivariate innovations with correlation R and optional t tails.

    Construction
    ------------
    A multivariate t distribution can be constructed as a scale mixture of a multivariate normal:

        z ~ Normal(0, R)
  
        w ~ ChiSquare(nu)
  
        x = z * sqrt(nu / w)

    where the scalar factor sqrt(nu / w) is shared across dimensions for each draw, introducing
    heavier tails and tail dependence relative to the Gaussian case.

    Implementation details
    ----------------------
  
    - The input matrix is repaired to a PSD correlation matrix using ``_nearest_psd_corr``.
  
    - Cholesky factorisation is used to induce the correlation structure efficiently.
  
    - When ``nu`` is None or not valid (> 2), the Gaussian case is used.

    Parameters
    ----------
    n_sims:
        Number of draws.
    R:
        Correlation matrix of shape (d, d).
    nu:
        Degrees of freedom. When provided and > 2, a t-mixture scaling is applied.
    rng:
        NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_sims, d) containing latent innovations.
   
    """
   
    d = int(R.shape[0])

    if d == 1:

        z = rng.standard_normal((n_sims, 1))

        if nu is not None and np.isfinite(nu) and (nu > 2):

            w = rng.chisquare(df = float(nu), size = (n_sims,))

            z = z * np.sqrt(float(nu) / np.maximum(w, 1e-12))[:, None]

        return z

    R_psd = _nearest_psd_corr(
        R = R
    )

    try:

        L = np.linalg.cholesky(R_psd)

    except np.linalg.LinAlgError:

        L = np.linalg.cholesky(R_psd + 1e-10 * np.eye(d))

    z = rng.standard_normal((n_sims, d)) @ L.T

    if nu is not None and np.isfinite(nu) and (nu > 2):

        w = rng.chisquare(df = float(nu), size = (n_sims,))

        z = z * np.sqrt(float(nu) / np.maximum(w, 1e-12))[:, None]

    return z


def _robust_loc_scale_1d(
    x: np.ndarray
) -> tuple[float, float]:
    """
    Compute robust location and scale for a one-dimensional array using median and MAD.

    The estimates are:

        mu = median(x)
        sd = 1.4826 * median(|x - mu|)

    with finite filtering. The scale is floored at ``e12`` to avoid division-by-zero in subsequent
    standardisation.

    Parameters
    ----------
    x:
        Input array.

    Returns
    -------
    (float, float)
        Tuple ``(mu, sd)``.
 
    """
 
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    if x.size == 0:

        return (0.0, 0.0)

    mu = float(np.nanmedian(x))

    sd = float(1.4826 * np.nanmedian(np.abs(x - mu)))

    return (mu, max(sd, e12))


def _excess_kurtosis_1d(
    x: np.ndarray
) -> float:
    """
    Compute the excess kurtosis of a one-dimensional array with finite filtering.

    Excess kurtosis is defined as:

        excess_kurtosis = E[(x - mu)^4] / (Var(x)^2) - 3

    The function returns NaN when fewer than 8 finite observations are available or when the
    variance is not positive.

    Excess kurtosis is used as a diagnostic for tail thickness and to calibrate degrees of freedom
    parameters in t-based innovations.

    Parameters
    ----------
    x:
        Input array.

    Returns
    -------
    float
        Excess kurtosis estimate, or NaN when not well-defined.
   
    """
   
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    if x.size < 8:

        return np.nan

    mu = float(np.nanmean(x))

    v = float(np.nanvar(x))

    if not np.isfinite(v) or v <= 0.0:

        return np.nan

    c4 = float(np.nanmean((x - mu) ** 4))

    return c4 / (v * v) - 3.0


def _estimate_ar1_params_from_levels(
    *,
    x: np.ndarray,
    target: float | None = None,
    min_points: int = 10,
    phi_cap: float = 0.97
) -> tuple[float, float, float]:
    """
    Estimate AR(1) persistence + innovation scale + (optional) t df from a historical level series.

    Model: y_t = phi * y_{t-1} + e_t, where y_t = x_t - target.
    Returns:
      phi_hat in [0, phi_cap],
      noise_scale_hat such that std(e_t) ~= noise_scale_hat * sd_level,
      nu_hat (t degrees-of-freedom) from residual excess kurtosis (fallback 8.0).
  
    """
  
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    if x.size < min_points + 2:

        return (0.0, 1.0, 8.0)

    if target is None or not np.isfinite(target):

        target = float(np.nanmedian(x))

    y = x - float(target)

    if y.size >= 8:

        lo = float(np.nanquantile(y, Q_LO))

        hi = float(np.nanquantile(y, Q_HI))

        y = np.clip(y, lo, hi)

    y0 = y[:-1]

    y1 = y[1:]

    m = np.isfinite(y0) & np.isfinite(y1)

    y0 = y0[m]

    y1 = y1[m]

    if y0.size < min_points:

        return (0.0, 1.0, 8.0)

    denom = float(np.dot(y0, y0))

    phi = float(np.dot(y0, y1) / denom) if denom > 0.0 else 0.0

    phi = float(np.clip(phi, 0.0, phi_cap))

    resid = y1 - phi * y0

    _, sd_level = _robust_loc_scale_1d(
        x = y
    )

    _, sd_resid = _robust_loc_scale_1d(
        x = resid
    )

    noise_scale = float(sd_resid / max(sd_level, e12))

    noise_scale = float(np.clip(noise_scale, 0.05, 3.0))

    exk = _excess_kurtosis_1d(
        x = resid / max(sd_resid, e12)
    )

    nu = 8.0

    if np.isfinite(exk) and exk > 1e-06:

        nu = float(np.clip(4.0 + 6.0 / exk, 4.5, 60.0))

    return (phi, noise_scale, nu)


def _pick_macro_rate_series(
    macro_source
) -> tuple[pd.Series | None, str]:
    """
    Pick a reasonable interest-rate / risk-free proxy series from multiple macro source types.
    
    """
    
    macro_df: pd.DataFrame | None = None

    rate_series: pd.Series | None = None

    if macro_source is None:

        return (None, 'no_macro_source')

    if isinstance(macro_source, pd.Series):

        s = pd.to_numeric(macro_source, errors = 'coerce')

        s.index = pd.to_datetime(macro_source.index, errors = 'coerce', format = 'mixed')

        s = s.dropna().sort_index()

        return (s if len(s) else None, 'series_empty' if len(s) == 0 else 'series')

    if isinstance(macro_source, pd.DataFrame):

        macro_df = macro_source

    else:

        try:

            ir = getattr(macro_source, 'interest', None)

            if isinstance(ir, pd.DataFrame) and (not ir.empty):

                rate_series = ir.iloc[:, -1] if ir.shape[1] else None
       
            elif isinstance(ir, pd.Series):

                rate_series = ir

        except (AttributeError, TypeError, ValueError):

            rate_series = None

        if rate_series is None:

            try:

                ah = getattr(macro_source, 'assign_macro_history', None)

                macro_df = ah() if callable(ah) else None

            except (AttributeError, TypeError, ValueError):

                macro_df = None

        else:

            s = pd.to_numeric(rate_series, errors = 'coerce')

            s.index = pd.to_datetime(rate_series.index, errors = 'coerce', format = 'mixed')

            s = s.dropna().sort_index()

            return (s if len(s) else None, 'provider_interest_empty' if len(s) == 0 else 'provider_interest')

    if macro_df is None or getattr(macro_df, 'empty', True):

        return (None, 'macro_dataframe_empty')

    cols = {str(c).lower(): c for c in macro_df.columns}

    candidates = ['interest', 'riskfree', 'risk_free', 'rf', 'r_f', 'risk-free', 'us10y', 'ust10y', '10y', 'treasury10y', 'treasury_10y', 'gov10y', 'fedfunds', 'fed_funds', 'sofr']

    for key in candidates:
       
        if key in cols:

            s = pd.to_numeric(macro_df[cols[key]], errors = 'coerce')

            s.index = pd.to_datetime(macro_df.index, errors = 'coerce', format = 'mixed')

            s = s.dropna().sort_index()

            if len(s):

                return (s, f'macro_column:{key}')

    for c in macro_df.columns:
      
        s = pd.to_numeric(macro_df[c], errors = 'coerce')

        if s.notna().sum() >= 20:

            s.index = pd.to_datetime(macro_df.index, errors = 'coerce', format = 'mixed')

            s = s.dropna().sort_index()

            if len(s):

                return (s, f'macro_fallback:{c}')

    return (None, 'no_usable_macro_rate_column')


def _estimate_corr_from_innovations(
    a: pd.Series,
    b: pd.Series
) -> float:
    """
    Estimate Pearson correlation on robust innovations (no level reordering).
    """
    
    da = a.to_numpy(dtype = float)

    db = b.to_numpy(dtype = float)

    ia = _hist_change_for_corr(
        x = da,
        name = str(a.name or 'a')
    )

    ib = _hist_change_for_corr(
        x = db,
        name = str(b.name or 'b')
    )

    m = np.isfinite(ia) & np.isfinite(ib)

    ia = ia[m]

    ib = ib[m]

    if ia.size < 10:

        return 0.0

    ma, sa = _robust_loc_scale_1d(
        x = ia
    )

    mb, sb = _robust_loc_scale_1d(
        x = ib
    )

    za = (ia - ma) / max(sa, e12)

    zb = (ib - mb) / max(sb, e12)

    rho = float(np.corrcoef(za, zb)[0, 1])

    if not np.isfinite(rho):

        return 0.0

    return float(np.clip(rho, -0.95, 0.95))


def _calibrate_ddm_rhos(
    *,
    hist_ratios: pd.DataFrame | None,
    macro_df
) -> tuple[float, float]:
    """
    rho_g_div: corr(innov(EPS), innov(DPS)) using company history (or 0 if insufficient).
    rho_rg:    corr(innov(EPS), innov(rate)) using macro rate proxy aligned to EPS dates (or 0).
  
    """
  
    if hist_ratios is None or hist_ratios.empty:

        return (0.0, 0.0)

    hist_eps = _extract_hist_ratios_series(
        df = hist_ratios,
        row_candidates = _EPS_HIST_ROWS
    )

    hist_dps = _extract_hist_ratios_series(
        df = hist_ratios,
        row_candidates = _DPS_HIST_ROWS
    )

    rho_g_div = 0.0

    if hist_eps is not None and hist_dps is not None:

        df = pd.concat([hist_eps.rename('eps'), hist_dps.rename('dps')], axis = 1).dropna()

        if len(df) >= 10:

            rho_g_div = _estimate_corr_from_innovations(
                a = df['eps'],
                b = df['dps']
            )

    rho_rg = 0.0

    rate, rate_reason = _pick_macro_rate_series(
        macro_source = macro_df
    )

    if rate is None:

        logger.info('[DDM-RHO] rate proxy unavailable (%s); rho_rg set to 0.', rate_reason)

    if rate is not None and hist_eps is not None:

        eps_s = pd.to_numeric(hist_eps, errors = 'coerce').dropna()

        eps_s.index = pd.to_datetime(eps_s.index, errors = 'coerce')

        eps_s = eps_s.dropna().sort_index()

        if len(eps_s) >= 10 and len(rate) >= 50:

            rate_df = rate.rename('rate').to_frame()

            eps_df = eps_s.rename('eps').to_frame()

            aligned = pd.merge_asof(eps_df.sort_index(), rate_df.sort_index(), left_index = True, right_index = True, direction = 'backward').dropna()

            if len(aligned) >= 10:

                rho_rg = _estimate_corr_from_innovations(
                    a = aligned['eps'],
                    b = aligned['rate']
                )

            else:

                logger.info('[DDM-RHO] insufficient aligned eps/rate points (%s); rho_rg set to 0.', len(aligned))

        else:

            logger.info('[DDM-RHO] insufficient history for rho_rg (eps=%s, rate=%s; source=%s); rho_rg set to 0.', len(eps_s), len(rate), rate_reason)

    return (float(rho_g_div), float(rho_rg))


def _reorder_sim_and_netdebt_by_history(
    *,
    sim: dict[str, np.ndarray],
    sim_components: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]],
    nd_draws: np.ndarray,
    nd_components: tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None,
    fcf_periods: pd.DatetimeIndex,
    nd_use_cols: pd.DatetimeIndex,
    hist_annual: pd.DataFrame | None,
    rng: np.random.Generator,
    tax_is_percent: bool
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """
    Impose cross-driver dependence and mild time persistence on simulated driver draws using history.

    Conceptual role
    ---------------
    The primary driver simulation step constructs marginal draw matrices for each driver independently
    by combining:

        draws = mu_sims + sigma_sims * x_std

    where:
  
    - mu_sims is a period-by-simulation matrix of forecast means (itself simulated to reflect
      analyst dispersion in the mean estimate),
  
    - sigma_sims is a period-by-simulation matrix of forecast uncertainty (itself simulated to
      reflect uncertainty in the dispersion estimate), and
  
    - x_std is a period-by-simulation matrix of standardised innovations with approximately zero
      mean and unit variance.

    Independence across drivers in x_std is computationally convenient but economically unrealistic.
    This function replaces the independent latent innovations with correlated latent innovations
    calibrated from historical annual data, while leaving mu_sims and sigma_sims unchanged. The
    resulting draw matrices preserve the consensus-implied location and scale period by period but
    exhibit more realistic co-movement.

    Dependence and persistence model
    -------------------------------
    Cross-sectional dependence is modelled using a correlation matrix R estimated from historical
    innovations (rank-based Spearman correlations mapped to a latent Pearson correlation matrix).
    Tail dependence is approximated via a multivariate t scale mixture with degrees of freedom nu.

    For each period t, and for the active subset of variables in that period, latent innovations are
    drawn as:

        e_t ~ MV-t(0, R_sub, nu)

    A per-variable AR(1) persistence parameter phi is estimated from historical innovations and used
    to introduce temporal correlation in the latent variables:

        z_t = phi * z_{t-1} + sqrt(1 - phi^2) * e_t

    This update preserves unit variance for each marginal z_t when e_t is standardised.

    Net debt handling
    -----------------
    Net debt may have a period grid different from the cash-flow driver grid. When net debt
    components are provided, the latent innovations for "net_debt" are written into the net debt
    latent matrix at the positions corresponding to ``nd_use_cols``.

    Post-processing
    ---------------
    Certain variables require unit sanitation after reconstruction:

    - Percent-like series may be expressed as percentage points rather than decimals. When the
      median magnitude suggests percentage points, division by 100 is applied.
   
    - Tax rate is clipped to [0, 0.40] and gross margin / payout ratios are clipped to [0, 1].

    Advantages
    ----------
    - Preserves per-period consensus mean and uncertainty (mu_sims and sigma_sims) while improving
      joint realism.
   
    - Uses robust rank correlation and innovation transforms, reducing sensitivity to level trends.
   
    - PSD repair enables stable Cholesky factorisation without fragile numerical tuning.
   
    - Optional t tails provide a simple mechanism for tail dependence at low computational cost.

    Limitations
    -----------
    - The dependence adjustment operates on latent innovations and primarily targets second moments;
      higher-moment features (for example, skewness) are not preserved exactly for all drivers.
   
    - Estimation uses annual history; when the valuation grid is quarterly, dependence is applied at
      the period level but calibrated from annual innovations.

    Parameters
    ----------
    sim:
        Mapping ``driver_key -> draws`` to be updated.
    sim_components:
        Mapping ``driver_key -> (mu_sims, sigma_sims, x_std, floor_at_zero)`` where x_std is the
        latent innovation matrix to be overwritten.
    nd_draws:
        Existing net debt draw matrix.
    nd_components:
        Optional tuple ``(mu_nd, sigma_nd, x_nd, floor_at_zero)`` describing the net debt latent
        innovation representation.
    fcf_periods:
        Period grid for the cash-flow drivers.
    nd_use_cols:
        Period grid for net debt.
    hist_annual:
        Historical annual panel used to estimate dependence and persistence.
    rng:
        Random generator used to draw latent innovations.
    tax_is_percent:
        Flag indicating whether tax inputs are expected in percentage-point units.

    Returns
    -------
    (dict[str, numpy.ndarray], numpy.ndarray, dict[str, object])
        Updated ``sim`` mapping, updated net debt draws, and a status dictionary describing whether
        the adjustment was attempted and applied.
  
    """
  
    status: dict[str, object] = {'attempted': False, 'used': False, 'reason': 'not_attempted'}

    if hist_annual is None or hist_annual.empty:

        status['reason'] = 'hist_annual_missing'

        return (sim, nd_draws, status)


    def _postprocess(
        key: str,
        draws: np.ndarray
    ) -> np.ndarray:
        """
        Apply unit sanitation and clipping for rate-like variables after reconstruction.

        The dependence adjustment produces draws on the same scale as the underlying mu/sigma
        simulation, but certain drivers are ambiguous between decimal and percentage-point units.
        A magnitude heuristic is used to convert percentage points to decimals when required.

        Parameters
        ----------
        key:
            Driver key.
        draws:
            Draw matrix.

        Returns
        -------
        numpy.ndarray
            Sanitised draw matrix.
    
        """
    
        if key in ('gross_margin', 'eff_tax_rate', 'payout', 'payout_ratio'):

            if np.nanmedian(draws) > 1.5:

                draws = draws / 100.0

            if key == 'gross_margin':

                draws = np.clip(draws, 0.0, 1.0)

            if key in ('payout', 'payout_ratio'):

                draws = np.clip(draws, 0.0, 1.0)

        if key == 'tax':

            if tax_is_percent or np.nanmedian(draws) > 1.5:

                draws = draws / 100.0

            draws = np.clip(draws, 0.0, 0.4)

        return draws


    sim_vars = [k for k in sim_components.keys()]

    if not sim_vars:

        status['reason'] = 'sim_components_empty'

        return (sim, nd_draws, status)

    any_x = next(iter(sim_components.values()))[2]

    N = int(any_x.shape[1])

    has_nd = nd_components is not None and nd_use_cols is not None and (len(nd_use_cols) > 0)

    nd_pos: dict[pd.Timestamp, int] = {}

    if has_nd:

        nd_pos = {pd.Timestamp(d).normalize(): i for i, d in enumerate(pd.to_datetime(nd_use_cols))}

    var_periods: dict[str, pd.DatetimeIndex] = {v: fcf_periods for v in sim_vars}

    if has_nd and 'net_debt' in hist_annual.columns:

        var_periods['net_debt'] = nd_use_cols

    status['attempted'] = True

    corr_res = _build_joint_corr_multi_from_history(
        var_periods = var_periods,
        hist_annual = hist_annual
    )

    if corr_res is None:

        status['reason'] = 'insufficient_corr_history'

        return (sim, nd_draws, status)

    R_full, keep_vars, nu = corr_res

    var_to_idx = {v: i for i, v in enumerate(keep_vars)}

    phi_map = _estimate_ar1_phi_multi_from_history(
        hist_annual = hist_annual,
        vars_ = keep_vars
    )

    prev: dict[str, np.ndarray] = {v: np.zeros(N, dtype = float) for v in keep_vars}

    fcf_periods_norm = pd.to_datetime(fcf_periods).normalize()

    for t, dt_t in enumerate(fcf_periods_norm):
        active = [v for v in sim_vars if v in var_to_idx]

        if has_nd and dt_t in nd_pos and ('net_debt' in var_to_idx):

            active.append('net_debt')

        if len(active) == 0:

            continue

        idxs = [var_to_idx[v] for v in active]

        R_sub = R_full[np.ix_(idxs, idxs)]

        z = _mv_t_innovations(
            n_sims = N,
            R = R_sub,
            nu = nu,
            rng = rng
        )

        for j, v in enumerate(active):
       
            phi = float(phi_map.get(v, 0.0))

            if phi > 0.0:

                z[:, j] = phi * prev[v] + np.sqrt(max(1.0 - phi * phi, 0.0)) * z[:, j]

            prev[v] = z[:, j]

            if v == 'net_debt':

                _mu, _sig, x_nd, _floor = nd_components

                x_nd[nd_pos[dt_t], :] = z[:, j]

            else:

                _mu, _sig, x_std, _floor = sim_components[v]

                x_std[t, :] = z[:, j]

    for v in sim_vars:
    
        mu_sims, sigma_sims, x_std, floor0 = sim_components[v]

        draws = mu_sims + sigma_sims * x_std

        if floor0:

            draws = np.maximum(draws, 0.0)

        sim[v] = _postprocess(
            key = v,
            draws = draws
        )

    if has_nd:

        mu_nd, sigma_nd, x_nd, floor0 = nd_components

        nd_new = mu_nd + sigma_nd * x_nd

        if floor0:

            nd_new = np.maximum(nd_new, 0.0)

        nd_draws = nd_new

    status['used'] = True

    status['reason'] = 'applied'

    return (sim, nd_draws, status)


def match_ticker_interest_rate(
    tickers,
    country,
    macro_source = None
) -> pd.Series:
    """
    Map tickers to an interest-rate / risk-free proxy based on country membership.

    The macro provider is expected to expose an ``interest`` attribute, either as:

    - a Series indexed by country name, or
  
    - a DataFrame with a country-indexed last column representing the latest value.

    For each ticker, the country label is taken from the ``country`` input (aligned to the
    ``tickers`` index) and used to reindex the macro interest series. Missing values are filled
    using the United States proxy when available; otherwise the cross-country median is used as a
    fallback.

    This mapping supports downstream cost-of-debt estimation when security-specific bond yields are
    unavailable.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    country:
        Country labels aligned to tickers (for example, a Series or dict).
    macro_source:
        Optional macro data provider. When None, runtime macro data are loaded via
        ``_ensure_runtime_data``.

    Returns
    -------
    pandas.Series
        Series indexed by ticker with name "Interest Rate". Values may be NaN when no proxy is
        available.
   
    """
   
    tickers = pd.Index(list(tickers), dtype = object)

    ctry = pd.Series(country, dtype = object).reindex(tickers)

    if macro_source is None:

        _, macro_source = _ensure_runtime_data()

    interest = getattr(macro_source, 'interest', None)

    if interest is None:

        ir = pd.Series(np.nan, index = tickers, name = 'Interest Rate')

        ir.index.name = 'Ticker'

        return ir

    if isinstance(interest, pd.DataFrame):

        src = interest.iloc[:, -1]

    else:

        src = interest

    base_us = _as_scalar(
        x = src.get('United States', np.nan)
    )

    if not np.isfinite(pd.to_numeric(base_us, errors = 'coerce')):

        src_num = pd.to_numeric(src, errors = 'coerce')

        src_num = src_num[np.isfinite(src_num)]

        base_us = float(src_num.median()) if len(src_num) else np.nan

    ir = src.reindex(ctry).copy()

    ir = ir.where(ir.notna(), base_us)

    ir.index = tickers

    ir.name = 'Interest Rate'

    return ir


def _as_scalar(
    x
):
    """
    Coerce a value container into a single scalar, favouring the most recent finite element.

    This helper is used to normalise inputs that may be provided as:

    - a pandas.Series (use the last non-null value),
   
    - a list/tuple/ndarray (use the last finite value), or
   
    - an already-scalar value (returned unchanged).

    Parameters
    ----------
    x:
        Value container.

    Returns
    -------
    object
        Scalar value or NaN when no finite element is available.
  
    """
  
    if isinstance(x, pd.Series):

        x = x.dropna()

        return x.iloc[-1] if len(x) else np.nan

    if isinstance(x, (list, tuple, np.ndarray)):

        arr = np.asarray(x, dtype = float)

        arr = arr[np.isfinite(arr)]

        return arr[-1] if len(arr) else np.nan

    return x


def cod(
    tickers,
    country,
    tax_rate_source,
    macro_source
):
    """
    Estimate an after-tax cost of debt proxy per ticker using a country interest-rate series.

    The cost of debt proxy is constructed as:

        cost_of_debt_after_tax = interest_rate * (1 - tax_rate)

    where:
 
    - interest_rate is obtained from ``match_ticker_interest_rate`` and coerced to decimal units
      when provided in percentage points, and
 
    - tax_rate is taken from ``tax_rate_source`` and coerced to decimal units when provided in
      percentage points, with a fallback of 0.25 when unavailable.

    The resulting series is intended for WACC construction, where the after-tax cost of debt is
    used:

        WACC = wE * cost_of_equity + wD * cost_of_debt_after_tax

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    country:
        Country labels aligned to tickers.
    tax_rate_source:
        Tax rate inputs aligned to tickers.
    macro_source:
        Macro data provider passed through to ``match_ticker_interest_rate``.

    Returns
    -------
    pandas.Series
        After-tax cost of debt proxy indexed by ticker with name "Cost of Debt".
  
    """
  
    tickers = pd.Index(list(tickers), dtype = object)

    ir = match_ticker_interest_rate(
        tickers = tickers,
        country = country,
        macro_source = macro_source
    ).apply(_as_scalar)

    ir = pd.to_numeric(ir, errors = 'coerce').to_numpy(dtype = float)

    tau_s = pd.Series(tax_rate_source).reindex(tickers).apply(_as_scalar)

    tau = pd.to_numeric(tau_s, errors = 'coerce').to_numpy(dtype = float)

    tau = np.where(np.isfinite(tau) & (tau > 1.0), tau / 100.0, tau)

    tau = np.where(~np.isfinite(tau), 0.25, tau)

    ir = np.where(np.isfinite(ir) & (ir > 1.0), ir / 100.0, ir)

    out = np.where(np.isfinite(ir), ir * (1.0 - tau), np.nan)

    s = pd.Series(out, index = tickers, name = 'Cost of Debt')

    s.index.name = 'Ticker'

    return s


def _build_discount_factor_vector(
    coe: float,
    E: float,
    cost_of_debt: float,
    D: float,
    *,
    return_components: bool = False
):
    """
    Construct WACC and capital-structure weights from market capitalisation and debt inputs.

    Definitions
    -----------
    Let:
  
    - E be market capitalisation,
  
    - D be a debt proxy for the market value of debt used for discounting,
  
    - V = E + D be total firm value,
  
    - coe be the cost of equity, and
  
    - cod be the (after-tax) cost of debt.

    Then the capital weights are:

        wE = E / (E + D)
  
        wD = D / (E + D)

    and the weighted average cost of capital is:

        WACC = wE * coe + wD * cod

    Practical considerations
    ------------------------
    - Negative D values are treated as sign artefacts and converted to abs(D).
  
    - Non-finite D is treated as 0.
  
    - V must be positive; otherwise an exception is raised.

    The optional component return is used to:
  
    - supply WACC for FCFF discounting, and
  
    - expose wD as a debt ratio proxy (DR) for DR-adjusted FCFE formulations.

    Parameters
    ----------
    coe:
        Cost of equity (decimal).
    E:
        Market capitalisation.
    cost_of_debt:
        After-tax cost of debt (decimal).
    D:
        Debt proxy used in discounting.
    return_components:
        When True, return a dictionary with keys "wacc", "wE", "wD". When False, return WACC only.

    Returns
    -------
    float | dict[str, float]
        WACC when ``return_components`` is False, else a component dictionary.
  
    """
  
    if np.isfinite(D) and D < 0:

        D = abs(D)

    if not np.isfinite(D):

        D = 0.0

    V = E + D

    if V <= 0:

        raise ValueError(f'[WACC] Invalid capital structure: E={E:.3g}, D={D:.3g} => V={V:.3g}')

    wE = E / V

    wD = D / V

    WACC = wE * coe + wD * cost_of_debt

    if return_components:

        return {'wacc': float(WACC), 'wE': float(wE), 'wD': float(wD)}

    return WACC


def _parse_n_analysts(
    x
) -> float:
    """
    Parse an analyst-count cell into a numeric value.

    CapIQ consensus sheets sometimes represent the number of analyst estimates as:

    - a plain integer (for example, 12),
 
    - a string "used / total" (for example, "12 / 18"), or
 
    - a string with surrounding whitespace.

    This helper extracts the "used" count when the split form is provided and returns NaN when no
    numeric value can be parsed.

    Parameters
    ----------
    x:
        Input cell value.

    Returns
    -------
    float
        Parsed analyst count or NaN.
  
    """
  
    if pd.isna(x):

        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):

        return x

    s = str(x).strip()

    m = re.match('^\\s*(\\d+)\\s*/\\s*(\\d+)\\s*$', s)

    if m:

        return m.group(1)

    m2 = re.match('^\\s*(\\d+)\\s*$', s)

    return m2.group(1) if m2 else np.nan


def _estimate_skewness_from_mean_median(
    mu,
    med,
    sigma
) -> float:
    """
    Estimate distribution skewness from mean, median, and standard deviation using Pearson's rule.

    Pearson's second coefficient of skewness provides the approximation:

        skewness ~= 3 * (mean - median) / standard_deviation

    This heuristic is used when calibrating skewed innovations from consensus tables that provide
    both mean-like (value row) and median estimates. It is computationally cheap and robust enough
    for a large-scale Monte Carlo pipeline.

    Parameters
    ----------
    mu:
        Mean-like estimate.
    med:
        Median estimate.
    sigma:
        Standard deviation estimate.

    Returns
    -------
    float
        Approximate skewness. Returns 0.0 when inputs are not finite or sigma is non-positive.
  
    """
  
    if not np.isfinite(mu) or not np.isfinite(med) or (not np.isfinite(sigma)) or (sigma <= 0):

        return 0.0

    return 3.0 * (mu - med) / sigma


def _normal_ppf(
    p: float
) -> float:
    """
    Compute the inverse CDF (quantile function) of the standard normal distribution.

    This wrapper exists primarily to keep the quantile function usage centralised for calibration
    routines that infer tail parameters from high/low summary statistics.

    Parameters
    ----------
    p:
        Probability in (0, 1).

    Returns
    -------
    float
        Standard normal quantile at probability p.
 
    """
 
    return norm.ppf(p)


def _t_ppf(
    p: float,
    nu: float
) -> float:
    """
    Compute the inverse CDF (quantile function) of the Student-t distribution.

    Parameters
    ----------
    p:
        Probability in (0, 1).
    nu:
        Degrees of freedom.

    Returns
    -------
    float
        Student-t quantile at probability p.
  
    """
  
    return t.ppf(p, df = nu)


def _infer_df_from_high_low(
    mu,
    sigma,
    high,
    low,
    n_analysts
) -> float:
    """
    Infer a Student-t degrees-of-freedom parameter from high/low forecast summary statistics.

    Rationale
    ---------
    CapIQ consensus tables often provide (Mean, High, Low, Std_Dev, No_of_Estimates) for each period.
    Under a thin-tailed normal model, observed (High, Low) values are unlikely to deviate far from
    the mean in standard-deviation units. When high/low ranges are wider than would be expected
    under normality, heavier tails are implied. A Student-t distribution provides a tractable
    heavy-tailed family indexed by degrees of freedom nu.

    Heuristic
    ---------
   
    1. Define the larger side z-score in standard deviation units:

           z_target = max( (high - mu) / sigma, (mu - low) / sigma )

    2. Approximate the expected maximum order statistic quantile for n samples using Blom's formula:

           p_max = (n - 0.375) / (n + 0.25)

       clipped to a conservative interval.

    3. If z_target is no larger than the corresponding standard normal quantile at p_max, treat the
       distribution as effectively normal and return NU_MAX.

    4. Otherwise, solve for nu such that:

           t_ppf(p_max, nu) = z_target

       using a bisection search on [NU_MIN, NU_MAX].

    Parameters
    ----------
    mu, sigma:
        Mean and standard deviation estimates for the period.
    high, low:
        High and low summary statistics for the period.
    n_analysts:
        Number of analyst estimates (used to choose p_max). Values < 3 default to NU_MAX.

    Returns
    -------
    float
        Inferred degrees of freedom nu clipped to [NU_MIN, NU_MAX].
  
    """
  
    if not (np.isfinite(mu) and np.isfinite(sigma) and (sigma > 0) and np.isfinite(high) and np.isfinite(low)):

        return NU_MAX

    if not np.isfinite(n_analysts) or n_analysts < 3:

        return NU_MAX

    z_hi = (high - mu) / sigma

    z_lo = (mu - low) / sigma

    z_target = max(z_hi, z_lo)

    if not np.isfinite(z_target) or z_target <= 0:

        return NU_MAX

    n = n_analysts

    p_max = (n - 0.375) / (n + 0.25)

    p_max = min(max(p_max, 0.7), 0.999)

    z_norm = _normal_ppf(
        p = p_max
    )

    if np.isfinite(z_norm) and z_target <= z_norm:

        return NU_MAX

    lo, hi = (NU_MIN, NU_MAX)

    for _ in range(60):
        mid = 0.5 * (lo + hi)

        val = _t_ppf(
            p = p_max,
            nu = mid
        )

        if not np.isfinite(val):

            lo = mid

            continue

        if val > z_target:

            lo = mid

        else:

            hi = mid

    nu = 0.5 * (lo + hi)

    return np.clip(nu, NU_MIN, NU_MAX)


def _annual_cols(
    metric_df: pd.DataFrame | None
) -> pd.Index:
    """
    Identify the annual-period columns in a mixed annual/quarterly consensus table.

    CapIQ consensus tables may contain both annual and quarterly period columns. When a
    ``period_type`` row is present, it is treated as the authoritative classifier for each column.
    This helper returns the subset of columns marked as annual.

    When no ``period_type`` row exists, all columns are returned, which preserves behaviour for
    legacy tables and for pre-filtered annual-only tables.

    Parameters
    ----------
    metric_df:
        Forecast table with optional ``period_type`` row.

    Returns
    -------
    pandas.Index
        Column labels corresponding to annual periods.
 
    """
 
    if metric_df is None or not isinstance(metric_df, pd.DataFrame) or getattr(metric_df, 'empty', True):

        return pd.Index([])

    if 'period_type' not in metric_df.index:

        return pd.Index(metric_df.columns)

    try:

        pt = metric_df.loc['period_type']

        mask = pt.astype(str).str.lower().values == 'annual'

        return pd.Index(metric_df.columns[mask])

    except (KeyError, TypeError, ValueError, IndexError):

        return pd.Index(metric_df.columns)


def _safe_growth_rates(
    x: np.ndarray
):
    """
    Compute simple one-step growth rates with finite filtering.

    Growth is computed as:

        g_t = x_t / x_{t-1} - 1

    Division-by-zero and invalid arithmetic are suppressed using NumPy error states, and only
    finite growth values are returned.

    Parameters
    ----------
    x:
        Level series array.

    Returns
    -------
    numpy.ndarray
        Array of finite growth rates. May be empty.
  
    """
  
    x = np.asarray(x, dtype = float)

    if len(x) < 2:

        return np.array([])

    denom = x[:-1]

    numer = x[1:]

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        g = numer / denom - 1.0

    g = g[np.isfinite(g)]

    return g


def _dedupe_cols(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove duplicated column labels from a DataFrame by keeping the last occurrence.

    Forecast tables can contain duplicated period columns after concatenation or due to workbook
    artefacts. Duplicates can break alignment logic and growth estimation. This helper keeps the
    last instance of each duplicated column label.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with duplicate columns removed (or the original if no duplicates exist).
   
    """
   
    if df is None or not df.columns.has_duplicates:

        return df

    return df.loc[:, ~df.columns.duplicated(keep = 'last')]


def estimate_terminal_growth_from_forecasts(
    fcf_future: pd.DataFrame | None,
    value_row: str = 'Free_Cash_Flow',
    revenue_future: pd.DataFrame | None = None,
    roe_future: pd.DataFrame | None = None,
    eps_future: pd.DataFrame | None = None,
    dps_future: pd.DataFrame | None = None,
    g_cap: float = G_CAP,
    sector_policy: SectorPolicy | None = None
):
    """
    Estimate a terminal growth rate and dispersion proxy from forecast tables.

    Role in valuation
    -----------------
    The DCF engines require a terminal growth rate g for the perpetuity terminal value. This
    function produces:

    - a central terminal growth estimate ``g_term``; and
   
    - a dispersion proxy ``sigma`` used to simulate terminal growth draws.

    Estimation sources
    ------------------
    A pooled growth signal is formed from:

    1. Free cash flow growth from annual FCF forecasts (value_row):

           g_fcf_t = FCF_t / FCF_{t-1} - 1

    2. Revenue growth from annual revenue forecasts:

           g_rev_t = Rev_t / Rev_{t-1} - 1

    3. Sustainable growth from ROE and retention (when ROE, EPS, and DPS forecasts coexist):

           payout_t    = DPS_t / EPS_t
           retention_t = 1 - payout_t
           g_sust      = ROE_t * retention_t

       with ROE converted from percentage points to decimals when required.

    The central estimate is taken as the median of the available candidates (FCF, revenue, and
    sustainable growth), then shrunk towards a long-run anchor ``ANCHOR``:

        g_faded = ANCHOR + (1 - shrink) * (g_raw - ANCHOR)

    where ``shrink`` is taken from the sector policy when provided (otherwise ``SHRINK``).

    Caps and floors
    ---------------
    The terminal growth rate is clipped to:

        g_term = clip(g_faded, FLOOR, g_cap)

    where g_cap is typically derived from the discount rate (for example, cost of equity or WACC)
    to preserve numerical stability in perpetuity valuation.

    Dispersion proxy
    ---------------
    ``sigma`` is estimated robustly from the pooled growth observations using MAD, with an optional
    sector multiplier. This sigma is later used to transform base uniform draws into correlated
    terminal growth draws under model-specific caps.

    Advantages
    ----------
    - Combines multiple economic signals rather than relying on a single noisy series.
  
    - Robust statistics (median/MAD) reduce sensitivity to outliers and forecast discontinuities.
  
    - Shrinkage and capping improve stability under sparse forecast horizons.

    Parameters
    ----------
    fcf_future:
        Forecast table for free cash flow.
    value_row:
        Row name for the FCF value series.
    revenue_future:
        Forecast table for revenue.
    roe_future, eps_future, dps_future:
        Forecast tables used for sustainable growth estimation.
    g_cap:
        Maximum permissible terminal growth rate.
    sector_policy:
        Optional policy affecting shrinkage and sigma scaling.

    Returns
    -------
    (float, float)
        Tuple ``(g_term, sigma)``.
    """
    g_fcf = np.nan

    g_rev = np.nan

    g_sust = np.nan

    g_pool: list[float] = []

    if fcf_future is not None and isinstance(fcf_future, pd.DataFrame) and (not fcf_future.empty):

        fcf_future = _dedupe_cols(
            df = fcf_future
        )

        cols = _annual_cols(
            metric_df = fcf_future
        )

        if len(cols) > 0 and value_row in fcf_future.index:

            fcf_ann = pd.to_numeric(fcf_future.loc[value_row, cols], errors = 'coerce').to_numpy(dtype = float)

            fcf_ann = fcf_ann[np.isfinite(fcf_ann)]

            if len(fcf_ann) >= 2:

                g = _safe_growth_rates(
                    x = fcf_ann
                )

                if len(g) > 0:

                    g_last = g[-LAST_N:] if len(g) >= LAST_N else g

                    g_fcf = float(np.median(g_last))

                    g_pool.extend([float(x) for x in g_last if np.isfinite(x)])

    if revenue_future is not None and isinstance(revenue_future, pd.DataFrame) and (not revenue_future.empty):

        revenue_future = _dedupe_cols(
            df = revenue_future
        )

        cols_r = _annual_cols(
            metric_df = revenue_future
        )

        rev_row = 'Revenue' if 'Revenue' in revenue_future.index else 'revenue' if 'revenue' in revenue_future.index else None

        if rev_row is not None and len(cols_r) > 0:

            rev_ann = pd.to_numeric(revenue_future.loc[rev_row, cols_r], errors = 'coerce').to_numpy(dtype = float)

            rev_ann = rev_ann[np.isfinite(rev_ann)]

            if len(rev_ann) >= 2:

                g = _safe_growth_rates(
                    x = rev_ann
                )

                if len(g) > 0:

                    g_last = g[-LAST_N:] if len(g) >= LAST_N else g

                    g_rev = float(np.median(g_last))

                    g_pool.extend([float(x) for x in g_last if np.isfinite(x)])

    if roe_future is not None and eps_future is not None and (dps_future is not None):

        if isinstance(roe_future, pd.DataFrame) and isinstance(eps_future, pd.DataFrame) and isinstance(dps_future, pd.DataFrame) and (not roe_future.empty) and (not eps_future.empty) and (not dps_future.empty):

            if 'ROE_pct' in roe_future.index and 'EPS_Normalized' in eps_future.index and ('DPS' in dps_future.index):

                cols_roe = pd.Index(_annual_cols(
                    metric_df = roe_future
                ))

                cols_eps = pd.Index(_annual_cols(
                    metric_df = eps_future
                ))

                cols_dps = pd.Index(_annual_cols(
                    metric_df = dps_future
                ))

                cols_s = cols_roe.intersection(cols_eps).intersection(cols_dps)

                try:

                    cols_s = pd.to_datetime(cols_s).sort_values()

                except (TypeError, ValueError):

                    pass

                if len(cols_s) > 0:

                    roe = pd.to_numeric(roe_future.reindex(columns = cols_s).loc['ROE_pct'], errors = 'coerce').to_numpy(dtype = float)

                    roe = _pct_to_dec_if_needed(
                        x = roe
                    )

                    eps = pd.to_numeric(eps_future.reindex(columns = cols_s).loc['EPS_Normalized'], errors = 'coerce').to_numpy(dtype = float)

                    dps = pd.to_numeric(dps_future.reindex(columns = cols_s).loc['DPS'], errors = 'coerce').to_numpy(dtype = float)

                    with np.errstate(divide = 'ignore', invalid = 'ignore'):
                 
                        payout = dps / eps

                    retention = 1.0 - payout

                    idx = np.where(np.isfinite(roe) & np.isfinite(retention) & np.isfinite(eps) & (eps > 0) & (retention >= 0) & (retention <= 1))[0]

                    if len(idx) > 0:

                        g_sust = float(roe[idx[-1]] * retention[idx[-1]])

                        if np.isfinite(g_sust):

                            g_pool.append(g_sust)

    candidates = [g for g in [g_fcf, g_rev, g_sust] if np.isfinite(g)]

    g_raw = float(np.median(candidates)) if candidates else float(ANCHOR)

    shrink = float(sector_policy.growth_shrink) if sector_policy is not None else SHRINK

    g_faded = float(ANCHOR + (1.0 - shrink) * (g_raw - ANCHOR))

    g_term = float(np.clip(g_faded, FLOOR, g_cap))

    g_pool_arr = np.array([x for x in g_pool if np.isfinite(x)], dtype = float)

    if len(g_pool_arr) >= 2:

        sigma = float(1.4826 * np.median(np.abs(g_pool_arr - np.median(g_pool_arr))))

    else:

        sigma = 0.01

    if sector_policy is not None and np.isfinite(float(sector_policy.growth_sigma_mult)):

        sigma *= max(float(sector_policy.growth_sigma_mult), 0.0)

    return (g_term, sigma)


def _hist_band(
    x: np.ndarray,
    *,
    nonneg: bool = False
) -> tuple[float, float] | None:
    """
    Compute a robust historical plausibility band for a numeric sample.

    The band is derived from quantiles with optional IQR padding:

        lo = quantile(x, Q_LO) - pad
   
        hi = quantile(x, Q_HI) + pad

    where:

        pad = PAD_IQR * IQR(x)   if IQR is positive and finite
     
        pad = 0                 otherwise

    When ``nonneg`` is True, the absolute value is taken before band construction and the lower
    bound is floored at 0.

    The band is used to clip simulated ratios and margins to company-informed ranges while
    retaining a margin beyond the raw quantile band. This approach is a compromise between:

    - using strict historical quantiles (which can be too tight), and
    
    - using broad global bounds (which can permit implausible draws).

    Parameters
    ----------
    x:
        Sample array.
    nonneg:
        Whether to treat the sample as a non-negative magnitude.

    Returns
    -------
    (float, float) | None
        Lower and upper bounds, or None when insufficient data exist.
  
    """
  
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    if x.size < MIN_POINTS:

        return None

    if nonneg:

        x = np.abs(x)

    lo = np.quantile(x, Q_LO)

    hi = np.quantile(x, Q_HI)

    q1 = np.quantile(x, 0.25)

    q3 = np.quantile(x, 0.75)

    iqr = q3 - q1

    pad = PAD_IQR * iqr if np.isfinite(iqr) and iqr > 0 else 0.0

    lo -= pad

    hi += pad

    if nonneg:

        lo = max(0.0, lo)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:

        return None

    return (lo, hi)


def _intersect_bounds(
    lo_a: float,
    hi_a: float,
    lo_b: float,
    hi_b: float
) -> tuple[float, float]:
    """
    Intersect two numeric intervals, returning a safe fallback when the intersection is invalid.

    Given two intervals [lo_a, hi_a] and [lo_b, hi_b], the intersection is:

        lo = max(lo_a, lo_b)
  
        hi = min(hi_a, hi_b)

    If the intersection is not finite or has hi <= lo, the function returns the second interval
    [lo_b, hi_b] as a conservative fallback. This behaviour is intended for situations where a
    history-derived band is pathological, in which case the policy-derived guardrail should apply.

    Parameters
    ----------
    lo_a, hi_a:
        First interval.
    lo_b, hi_b:
        Second interval (fallback interval).

    Returns
    -------
    (float, float)
        Intersected interval, or the second interval when the intersection is invalid.
 
    """
 
    lo = max(float(lo_a), float(lo_b))

    hi = min(float(hi_a), float(hi_b))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:

        return (float(lo_b), float(hi_b))

    return (lo, hi)


def _coherence_flag_from_history(
    x: pd.Series,
    y: pd.Series,
    *,
    min_points: int,
    min_abs_corr: float,
    use_abs: bool = False
) -> tuple[bool, int, float]:
    """
    Compute a simple historical coherence flag based on Spearman rank correlation.

    Certain valuation methods rely on relationships that should be coherent historically. Examples:

    - interest expense should scale positively with debt magnitude, and
   
    - working capital investment should relate to revenue activity.

    This helper computes Spearman correlation between two historical series after aligning on the
    index and dropping missing values. It then evaluates a threshold condition:

        score = abs(corr)  if use_abs else corr
   
        coherent = (score >= min_abs_corr) and (n >= min_points)

    Parameters
    ----------
    x, y:
        Series to compare.
    min_points:
        Minimum number of paired observations required.
    min_abs_corr:
        Threshold applied to the correlation score.
    use_abs:
        If True, absolute correlation is used; otherwise the signed correlation is used.

    Returns
    -------
    (bool, int, float)
        Tuple ``(coherent, n_pairs, corr)``.
  
    """
  
    paired = pd.concat([x.rename('x'), y.rename('y')], axis = 1).dropna()

    n = int(len(paired))

    if n < int(min_points):

        return (False, n, float('nan'))

    corr = float(paired.corr(method = 'spearman').iloc[0, 1])

    if not np.isfinite(corr):

        return (False, n, corr)

    score = abs(corr) if use_abs else corr

    return (bool(score >= float(min_abs_corr)), n, corr)


def _base_fcff_methods(
    *,
    fcf_is_stub: bool
) -> dict[str, tuple[str, set[str]]]:
    """
    Define the base set of FCFF construction methods and their required drivers.

    Each method is represented as:

        method_key -> (display_label, required_driver_set)

    The required driver sets correspond to the inputs needed to compute FCFF under each method.
    Downstream logic uses these sets to gate methods based on available simulated drivers and on
    coherence flags derived from history.

    Parameters
    ----------
    fcf_is_stub:
        If True, the direct CapIQ FCF forecast is treated as imputed/stubbed and the method label
        is adjusted for transparency.

    Returns
    -------
    dict[str, tuple[str, set[str]]]
        Base FCFF method definitions.
  
    """
  
    return {
        'fcf': ('CapIQ_FCF' if not fcf_is_stub else 'Imputed_FCF', {'fcf'}), 
        'cfo_capex': ('CFO-CapEx+Int(1-T)', {'cfo', 'capex', 'interest', 'tax'}), 
        'cfo_maint': ('CFO-MaintCapEx+Int(1-T)', {'cfo', 'maint_capex', 'interest', 'tax'}),
        'ebit': ('EBIT_FCFF', {'ebit', 'tax', 'da', 'capex', 'dnwc'}), 
        'ebitda': ('EBITDA_FCFF', {'ebitda', 'da', 'tax', 'capex', 'dnwc'}), 
        'ni': ('NI_FCFF', {'net_income', 'da', 'interest', 'tax', 'capex', 'dnwc'})
    }


def _build_fcff_method_defs(
    policy: SectorPolicy,
    *,
    fcf_is_stub: bool,
    has_interest_debt_coherence: bool,
    has_wc_coherence: bool
) -> list[tuple[str, set[str]]]:
    """
    Construct an ordered list of FCFF method definitions based on sector policy and data coherence.

    The FCFF engine supports multiple algebraically related constructions (for example, direct FCF,
    CFO minus CapEx plus after-tax interest, EBIT-based build-up). Different sectors and different
    data availability patterns make different constructions more reliable.

    Selection logic
    ---------------
 
    - The direct FCF method is always included as the primary anchor.
 
    - CFO-based methods require interest/debt coherence when the sector policy demands it.
 
    - DNWC-based methods (EBIT/EBITDA/NI build-ups) require working-capital coherence when demanded.
 
    - The sector policy ``fcff_profile`` determines the preferred mix and order of methods:
      "financial", "asset_income", "wc_heavy", "cyclical", or default.

    The returned list is ordered. The evaluation pipeline uses this ordering to determine which
    methods are attempted first and to report method usage.

    Advantages
    ----------
 
    - Provides graceful degradation under sparse inputs while retaining sector-specific preferences.
 
    - Avoids methods known to be unreliable when coherence diagnostics fail.
 
    - Makes method composition explicit and auditable.

    Parameters
    ----------
    policy:
        Sector policy controlling gating thresholds and method preferences.
    fcf_is_stub:
        Whether the direct FCF forecast is stubbed/imputed.
    has_interest_debt_coherence:
        Historical coherence flag for interest vs debt magnitude.
    has_wc_coherence:
        Historical coherence flag for DNWC vs revenue activity.

    Returns
    -------
    list[tuple[str, set[str]]]
        List of method definitions as (label, required_driver_set).
 
    """
 
    base = _base_fcff_methods(
        fcf_is_stub = fcf_is_stub
    )

    cfo_methods = [base['cfo_capex'], base['cfo_maint']]

    dnwc_methods = [base['ebit'], base['ebitda'], base['ni']]

    allow_cfo = not policy.require_interest_debt_for_cfo or has_interest_debt_coherence

    allow_dnwc = not policy.require_wc_for_dnwc or has_wc_coherence

    methods: list[tuple[str, set[str]]] = [base['fcf']]

    if policy.fcff_profile == 'financial':

        if allow_cfo:

            methods.extend(cfo_methods)

        if allow_cfo and allow_dnwc:

            methods.extend(dnwc_methods)

        return methods

    if policy.fcff_profile == 'asset_income':

        methods.append(base['cfo_maint'])

        if has_interest_debt_coherence:

            methods.append(base['cfo_capex'])

        if allow_dnwc:

            methods.extend(dnwc_methods)

        return methods

    if policy.fcff_profile == 'wc_heavy':

        if allow_cfo:

            methods.extend(cfo_methods)

        if allow_dnwc:

            methods.extend(dnwc_methods)

        return methods

    if policy.fcff_profile == 'cyclical':

        if allow_cfo:

            methods.extend(cfo_methods)

        if allow_dnwc:

            methods.extend([base['ebit'], base['ebitda']])

        return methods

    if allow_cfo:

        methods.extend(cfo_methods)

    if allow_dnwc:

        methods.extend(dnwc_methods)

    return methods


def _base_fcfe_methods_dr(
    *,
    fcf_is_stub: bool
) -> dict[str, tuple[str, set[str], str]]:
    """
    Define the base set of DR-adjusted FCFE construction methods and their required drivers.

    FCFE here is defined as an equity cash flow obtained by adjusting investment cash flows using
    a debt ratio DR. The engine supports multiple equivalent (or near-equivalent) algebraic forms
    to improve robustness under missing drivers.

    Each method definition is represented as:

        method_key -> (display_label, required_driver_set, formula_key)

    where formula_key selects the formula implementation in the FCFE engine.

    Parameters
    ----------
    fcf_is_stub:
        Whether FCFF used in the bridge method is considered imputed/stubbed (affects labelling).

    Returns
    -------
    dict[str, tuple[str, set[str], str]]
        Base FCFE method definitions.
 
    """
 
    fcff_bridge_label = 'FCFF_BRIDGE_DR' if not fcf_is_stub else 'Imputed_FCFF_BRIDGE_DR'

    return {
        'ni_dr': ('NI_DR', {'net_income', 'capex', 'da', 'dnwc'}, 'ni_dr'),
        'ebitda_int_tax_dr': ('EBITDA_INT_TAX_DR', {'ebitda', 'da', 'interest', 'tax', 'capex', 'dnwc'}, 'ebitda_int_tax_dr'),  
        'fcff_bridge_dr': (fcff_bridge_label, {'fcf', 'interest', 'tax', 'capex', 'da', 'dnwc'}, 'fcff_bridge_dr'), 
        'ebit_int_dr': ('EBIT_INT_DR', {'ebit', 'interest', 'tax', 'capex', 'da', 'dnwc'}, 'ebit_int_dr')
    }


def _build_fcfe_method_defs_dr(
    policy: SectorPolicy,
    *,
    fcf_is_stub: bool,
    has_interest_debt_coherence: bool,
    has_wc_coherence: bool
) -> tuple[list[tuple[str, set[str], str]], dict[str, str]]:
    """
    Construct an ordered list of DR-adjusted FCFE method definitions with availability gating.

    Gating rationale
    ---------------
    The DR-adjusted FCFE formulas require DNWC and, for some variants, interest expense. Historical
    coherence checks are used to avoid using methods that would be poorly identified given the
    observed historical relationships:

    - When DNWC coherence is required by the policy and fails, DNWC-based methods are skipped.
 
    - When interest/debt coherence is required by the policy and fails, interest-based methods are
      skipped.

    The function returns both the accepted method list and a dictionary of skipped methods with
    reasons, which supports transparent diagnostics in verbose runs.

    Parameters
    ----------
    policy:
        Sector policy controlling coherence requirements.
    fcf_is_stub:
        Whether FCFF is stubbed/imputed for labelling of the bridge method.
    has_interest_debt_coherence:
        Coherence flag for interest vs debt.
    has_wc_coherence:
        Coherence flag for DNWC vs revenue.

    Returns
    -------
    (list[tuple[str, set[str], str]], dict[str, str])
        Tuple of (method definitions, skipped mapping). Each method definition is
        (label, required_driver_set, formula_key).
  
    """
  
    base = _base_fcfe_methods_dr(
        fcf_is_stub = fcf_is_stub
    )

    methods: list[tuple[str, set[str], str]] = []

    skipped: dict[str, str] = {}

    allow_interest = not policy.require_interest_debt_for_cfo or has_interest_debt_coherence

    allow_dnwc = not policy.require_wc_for_dnwc or has_wc_coherence


    def _add(
        k: str
    ):
        """
        Append a base method definition to the method list by key.

        Parameters
        ----------
        k:
            Key in the base method mapping returned by ``_base_fcfe_methods_dr``.
      
        """
      
        label, req, key = base[k]

        methods.append((label, req, key))


    if allow_dnwc:

        _add(
            k = 'ni_dr'
        )

    else:

        skipped['ni_dr'] = 'missing_dnwc_coherence'

    if allow_dnwc and allow_interest:

        _add(
            k = 'ebitda_int_tax_dr'
        )

        _add(
            k = 'ebit_int_dr'
        )

        _add(
            k = 'fcff_bridge_dr'
        )

        return (methods, skipped)

    if not allow_dnwc:

        skipped['ebitda_int_tax_dr'] = 'missing_dnwc_coherence'

        skipped['ebit_int_dr'] = 'missing_dnwc_coherence'

        skipped['fcff_bridge_dr'] = 'missing_dnwc_coherence'

        return (methods, skipped)

    skipped['ebitda_int_tax_dr'] = 'missing_interest_coherence'

    skipped['ebit_int_dr'] = 'missing_interest_coherence'

    skipped['fcff_bridge_dr'] = 'missing_interest_coherence'

    return (methods, skipped)


def _apply_practical_checks_and_bounds(
    sim: dict[str, np.ndarray],
    *,
    hist_annual: pd.DataFrame | None = None,
    revenue_key: str = 'revenue',
    tax_key: str = 'tax',
    sector_policy: SectorPolicy | None = None
):
    """
    Apply practical sanity checks, unit conversions, and plausibility bounds to simulated drivers.

    Purpose
    -------
    The marginal simulation and imputation steps are designed to be robust under sparse and noisy
    inputs, but they can still produce economically implausible combinations, especially in the
    tails. This function constrains simulated drivers to ranges that are consistent with:

    - generic sector policy guardrails, and
 
    - company-specific historical ratios and margins where sufficient history is available.

    The intent is to improve numerical stability and economic plausibility in downstream cash-flow
    construction and valuation, at the cost of some tail truncation.

    Applied checks (high level)
    ---------------------------
   
    1. Unit conversion and clipping for rates:
   
       - Tax rates are converted from percentage points to decimals when necessary and clipped to
         [policy.tax_lo, policy.tax_hi], optionally intersected with a history-derived band.

    2. Non-negativity and basic accounting coherence:
   
       - CapEx, maintenance CapEx, depreciation/amortisation, and interest are floored at zero.
   
       - Maintenance CapEx is capped at CapEx.
   
       - EBITDA is enforced to be at least EBIT + max(DA, 0).

    3. Ratio and margin plausibility using revenue as scale:
   
       For each driver k that should behave like a ratio to revenue (for example, CapEx/Revenue,
       DA/Revenue, DNWC/Revenue) or a margin (for example, EBIT/Revenue), the function:
   
       - computes a history-derived band using quantiles and optional IQR padding, and
   
       - clips simulated ratios/margins to the intersection of the history band and policy bounds.

       When insufficient history exists, policy-only bounds are applied.

    4. Gross margin coherence:
   
       - Gross margin is converted from percentage points when necessary, clipped to [0, 1], and
         optionally clipped to a history band.

       - EBIT and EBITDA margins are constrained not to exceed gross margin.

    Advantages
    ----------
  
    - Prevents extreme but data-driven draws from dominating valuation outputs through non-linear
      terminal value effects.
  
    - Uses company history where available, improving plausibility relative to generic clipping.
  
    - Maintains a clear separation between probabilistic modelling and deterministic guardrails,
      supporting auditability.

    Parameters
    ----------
    sim:
        Mapping ``driver_key -> draws`` to be modified in place.
    hist_annual:
        Optional historical annual panel used to build ratio and margin bands.
    revenue_key:
        Key for revenue in ``sim``.
    tax_key:
        Key for effective tax rate in ``sim``.
    sector_policy:
        Optional sector policy providing default bounds and behavioural assumptions.

    Returns
    -------
    dict[str, numpy.ndarray]
        The modified ``sim`` mapping.
 
    """
 
    policy = sector_policy if sector_policy is not None else SECTOR_POLICIES['Unknown']

    if tax_key in sim:

        sim_tax = _pct_to_dec_if_needed(
            x = sim[tax_key]
        )

        sim[tax_key] = np.clip(sim_tax, policy.tax_lo, policy.tax_hi)

    for k in ('capex', 'maint_capex', 'da', 'interest'):
        if k in sim:

            sim[k] = np.maximum(sim[k], 0.0)

    if 'capex' in sim and 'maint_capex' in sim:

        sim['maint_capex'] = np.minimum(sim['maint_capex'], sim['capex'])

    if 'ebit' in sim and 'ebitda' in sim and ('da' in sim):

        sim['ebitda'] = np.maximum(sim['ebitda'], sim['ebit'] + np.maximum(sim['da'], 0.0))

    hist_rev = None

    if hist_annual is not None and (not hist_annual.empty) and ('revenue' in hist_annual.columns):

        hist_rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').replace(0.0, np.nan).to_numpy(dtype = float)

    if tax_key in sim and hist_annual is not None and ('tax' in getattr(hist_annual, 'columns', [])):

        tx = pd.to_numeric(hist_annual['tax'], errors = 'coerce').to_numpy(dtype = float)

        tx = tx[np.isfinite(tx)]

        if tx.size >= MIN_POINTS:

            if np.nanmedian(np.abs(tx)) > 1.5:

                tx = tx / 100.0

            b = _hist_band(
                x = tx,
                nonneg = True
            )

            if b is not None:

                lo, hi = b

                lo = np.clip(lo, 0.0, 1.0)

                hi = np.clip(hi, 0.0, 1.0)

                lo, hi = _intersect_bounds(
                    lo_a = lo,
                    hi_a = hi,
                    lo_b = policy.tax_lo,
                    hi_b = policy.tax_hi
                )

                if hi > lo:

                    sim[tax_key] = np.clip(sim[tax_key], lo, hi)

    if revenue_key not in sim:

        return sim

    rev = sim[revenue_key]

    rev_safe = np.maximum(np.abs(rev), e12)

    ratio_keys = {'capex': True, 'maint_capex': True, 'da': True, 'interest': True, 'dnwc': False}

    margin_keys = {'cfo': False, 'ebit': False, 'ebitda': False, 'ebt': False, 'net_income': False, 'fcf': False}

    hist_ratio_bands: dict[str, tuple[float, float]] = {}

    hist_margin_bands: dict[str, tuple[float, float]] = {}

    if hist_rev is not None and np.isfinite(hist_rev).sum() >= MIN_POINTS:

        hist_rev_safe = np.where(np.abs(hist_rev) > e12, np.abs(hist_rev), np.nan)

        for k, nonneg in ratio_keys.items():
        
            if hist_annual is not None and k in hist_annual.columns:

                y = pd.to_numeric(hist_annual[k], errors = 'coerce').to_numpy(dtype = float)

                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    r = y / hist_rev_safe

                r = r[np.isfinite(r)]

                b = _hist_band(
                    x = r,
                    nonneg = nonneg
                )

                if b is not None:

                    hist_ratio_bands[k] = b

        for k, nonneg in margin_keys.items():
    
            if hist_annual is not None and k in hist_annual.columns:

                y = pd.to_numeric(hist_annual[k], errors = 'coerce').to_numpy(dtype = float)

                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    m = y / hist_rev_safe

                m = m[np.isfinite(m)]

                b = _hist_band(
                    x = m,
                    nonneg = nonneg
                )

                if b is not None:

                    hist_margin_bands[k] = b

    for k, (lo, hi) in hist_ratio_bands.items():
      
        if k in sim:

            if k in {'capex', 'maint_capex'}:

                lo, hi = _intersect_bounds(
                    lo_a = lo,
                    hi_a = hi,
                    lo_b = policy.capex_ratio_lo,
                    hi_b = policy.capex_ratio_hi
                )
         
            elif k == 'da':

                lo, hi = _intersect_bounds(
                    lo_a = lo,
                    hi_a = hi,
                    lo_b = policy.da_ratio_lo,
                    hi_b = policy.da_ratio_hi
                )
         
            elif k == 'dnwc':

                lo, hi = _intersect_bounds(
                    lo_a = lo,
                    hi_a = hi,
                    lo_b = policy.dnwc_ratio_lo,
                    hi_b = policy.dnwc_ratio_hi
                )

            r = sim[k] / rev_safe

            sim[k] = np.clip(r, lo, hi) * rev_safe

    for k in ('capex', 'maint_capex', 'da', 'dnwc'):
     
        if k not in sim:

            continue

        if k in {'capex', 'maint_capex'}:

            lo, hi = (policy.capex_ratio_lo, policy.capex_ratio_hi)
     
        elif k == 'da':

            lo, hi = (policy.da_ratio_lo, policy.da_ratio_hi)

        else:

            lo, hi = (policy.dnwc_ratio_lo, policy.dnwc_ratio_hi)

        if hi <= lo:

            continue

        r = sim[k] / rev_safe

        sim[k] = np.clip(r, lo, hi) * rev_safe

    for k, (lo, hi) in hist_margin_bands.items():
     
        if k in sim:

            m = sim[k] / rev_safe

            sim[k] = np.clip(m, lo, hi) * rev_safe

    if 'gross_margin' in sim:

        gm = sim['gross_margin']

        m0 = np.nanmedian(np.abs(gm)) if np.isfinite(gm).any() else np.nan

        if np.isfinite(m0) and m0 > 1.5:

            gm = gm / 100.0

        if hist_annual is not None and 'gross_margin' in hist_annual.columns:

            h = pd.to_numeric(hist_annual['gross_margin'], errors = 'coerce').to_numpy(dtype = float)

            h = h[np.isfinite(h)]

            if h.size >= MIN_POINTS and np.nanmedian(np.abs(h)) > 1.5:

                h = h / 100.0

            b = _hist_band(
                x = h,
                nonneg = True
            )

            if b is not None:

                lo, hi = b

                gm = np.clip(gm, max(0.0, lo), min(1.0, hi))

            else:

                gm = np.clip(gm, 0.0, 1.0)

        else:

            gm = np.clip(gm, 0.0, 1.0)

        sim['gross_margin'] = gm

        for k in ('ebit', 'ebitda'):
        
            if k in sim:

                m = sim[k] / rev_safe

                m = np.minimum(m, gm)

                sim[k] = m * rev_safe

    return sim


def _simulate_ratio_rw_from_history(
    hist_ratio: pd.Series,
    *,
    T: int,
    rng: np.random.Generator,
    lo: float,
    hi: float,
    kappa: float = 0.15,
    nu: float = 8.0
) -> np.ndarray:
    """
    Simulate a bounded ratio path using a mean-reverting random walk in levels.

    Model form
    ----------
    The ratio r_t is simulated as a discrete-time Ornstein-Uhlenbeck style process:

        r_t = r_{t-1} + kappa * (target - r_{t-1}) + eps_t

    where:
   
    - target is the historical median of the ratio series,
   
    - kappa controls the speed of mean reversion, and
   
    - eps_t are heavy-tailed innovations drawn from a Student-t distribution with degrees of freedom
      nu and location/scale calibrated from historical first differences:

          d_hist = diff(r_hist)
   
          med = median(d_hist)
   
          sd  = MAD(d_hist) scaled to sd units
   
          eps_t ~ t_nu(loc=med, scale=sd)

    The simulated path is clipped to the interval [lo, hi] at each step.

    Small-sample fallback
    ---------------------
    When fewer than 6 historical observations are available, the simulation degenerates to a
    constant base value (historical median or 0.0) plus small t noise.

    Typical use
    -----------
    This process is used when deriving primitive driver ratios such as EBIT margin or DNWC/revenue
    in the fallback FCFF derivation pathway.

    Parameters
    ----------
    hist_ratio:
        Historical ratio series.
    T:
        Number of simulated periods.
    rng:
        Random generator.
    lo, hi:
        Hard bounds applied to the simulated ratio.
    kappa:
        Mean reversion speed.
    nu:
        Student-t degrees of freedom controlling tail thickness.

    Returns
    -------
    numpy.ndarray
        Array of shape (T, N_SIMS) containing simulated ratio paths.

    """

    s = pd.to_numeric(hist_ratio, errors = 'coerce').replace([np.inf, -np.inf], np.nan).dropna()

    if s.size < 6:

        base = float(np.clip(np.nanmedian(s.to_numpy(dtype = float)) if s.size else 0.0, lo, hi))

        eps = rng.standard_t(df = nu, size = (T, N_SIMS)) * 0.01

        return np.clip(base + eps, lo, hi)

    x = s.to_numpy(dtype = float)

    med, sd = _robust_loc_scale(
        x = np.diff(x)
    )

    level0 = float(np.clip(x[-1], lo, hi))

    target = float(np.clip(np.nanmedian(x), lo, hi))

    eps = rng.standard_t(df = nu, size = (T, N_SIMS)) * sd + med

    out = np.empty((T, N_SIMS), dtype = float)

    prev = np.full(N_SIMS, level0, dtype = float)

    for t in range(T):
       
        prev = prev + kappa * (target - prev) + eps[t, :]

        prev = np.clip(prev, lo, hi)

        out[t, :] = prev

    return out


def _simulate_pos_ratio_rw_from_history(
    hist_ratio: pd.Series,
    *,
    T: int,
    rng: np.random.Generator,
    lo: float,
    hi: float,
    kappa: float = 0.1,
    nu: float = 8.0
) -> np.ndarray:
    """
    Simulate a bounded strictly-positive ratio path using mean reversion in log space.

    Model form
    ----------
    For strictly positive ratios, modelling in log space is numerically stable and enforces
    positivity. Let y_t = log(r_t). The process is:

        y_t = y_{t-1} + kappa * (log(target) - y_{t-1}) + eps_t

    which implies:

        r_t = r_{t-1} * exp( kappa * (log(target) - log(r_{t-1})) + eps_t )

    Innovations eps_t are drawn from a Student-t distribution calibrated from historical log
    differences:

        d_hist = diff(log(r_hist))
     
        med = median(d_hist)
     
        sd  = MAD(d_hist) scaled to sd units
     
        eps_t ~ t_nu(loc=med, scale=sd)

    The resulting ratio is clipped to [lo, hi] at each step.

    Small-sample fallback
    ---------------------
    When fewer than 6 positive historical observations are available, the simulation uses a
    constant base ratio plus multiplicative t noise:

        r_t = base * exp(eps_t)

    Parameters
    ----------
    hist_ratio:
        Historical ratio series (expected to be positive).
    T:
        Number of simulated periods.
    rng:
        Random generator.
    lo, hi:
        Bounds for the ratio.
    kappa:
        Mean reversion speed in log space.
    nu:
        Student-t degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Array of shape (T, N_SIMS) containing simulated positive ratio paths.
  
    """
  
    s = pd.to_numeric(hist_ratio, errors = 'coerce').replace([np.inf, -np.inf], np.nan).dropna()

    s = s[s > 0.0]

    if s.size < 6:

        base = float(np.clip(np.nanmedian(s.to_numpy(dtype = float)) if s.size else max(lo, 1e-06), lo, hi))

        eps = rng.standard_t(df = nu, size = (T, N_SIMS)) * 0.05

        return np.clip(base * np.exp(eps), lo, hi)

    x = np.log(np.clip(s.to_numpy(dtype = float), 1e-12, None))

    d = np.diff(x)

    med, sd = _robust_loc_scale(
        x = d
    )

    level0 = float(np.clip(np.exp(x[-1]), lo, hi))

    target = float(np.clip(np.exp(np.median(x)), lo, hi))

    eps = rng.standard_t(df = nu, size = (T, N_SIMS)) * sd + med

    out = np.empty((T, N_SIMS), dtype = float)

    prev = np.full(N_SIMS, level0, dtype = float)

    for t in range(T):
     
        prev = prev * np.exp(kappa * (np.log(target) - np.log(np.clip(prev, 1e-12, None))) + eps[t, :])

        prev = np.clip(prev, lo, hi)

        out[t, :] = prev

    return out


def _derive_fcff_from_primitives(
    *,
    sim: dict[str, np.ndarray],
    hist_annual: pd.DataFrame | None,
    fcf_periods: pd.DatetimeIndex,
    nd_draws: np.ndarray | None,
    ctx: RunContext,
    sector_policy: SectorPolicy | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray | None, bool]:
    """
    Derive an internally consistent FCFF driver set from primitive history when forecasts are incomplete.

    Purpose
    -------
    The FCFF engine prefers direct consensus forecasts. However, some tickers exhibit missing or
    stubbed forecast blocks. This helper provides a deterministic fallback that synthesises a
    minimum viable set of drivers required to compute FCFF:

        FCFF = EBIT * (1 - tax_rate) + DA - CapEx + DNWC

    where DNWC is the change in net working capital (consistent with the sign convention used
    elsewhere: DNWC is negative for a working-capital outflow).

    Derivation strategy
    -------------------
    1. Revenue:
   
       When absent from ``sim``, revenue is simulated from historical annual revenue using a
       history-based random walk (``_simulate_rw_from_history_levels``) with non-negativity enforced.

    2. Tax rate:
   
       When absent, tax is taken from historical effective tax rate (converted to decimals if needed)
       and simulated as a tight heavy-tailed perturbation around the historical median, clipped to
       policy bounds. A default base of 0.21 is used when history is unavailable.

    3. Operating margin and investment ratios:
   
       The following ratios are simulated primarily as mean-reverting processes using
       ``_simulate_ratio_rw_from_history`` / ``_simulate_pos_ratio_rw_from_history`` when sufficient
       history exists:

       - EBIT margin: EBIT / Revenue (bounded, may be negative)
   
       - DA ratio:    DA / Revenue (positive)
   
       - CapEx ratio: CapEx / Revenue (positive)
   
       - DNWC ratio:  DNWC / Revenue (bounded, may be negative)

       When history is insufficient, an implied ratio is obtained from any existing simulated
       numerator in ``sim`` (median over simulations of numerator / revenue) or a default. A small
       t noise term is added to avoid degenerate paths.

    4. Interest expense (optional):
   
       When net debt draws are provided, interest expense is imputed using a simulated cost-of-debt
       ratio derived from historical interest and net debt:

           rd_hist ~= abs(interest_t) / abs(net_debt_{t-1})
   
           interest_t = rd_t * max(net_debt_{t-1}, 0)

       When history is unavailable, a default rd of 5% plus noise is used.

    5. FCFF construction:
   
       With EBIT, DA, CapEx, and DNWC constructed, FCFF is computed by the standard build-up
       identity shown above.

    Advantages
    ----------
   
    - Provides a coherent fallback that respects key accounting relationships and revenue scaling.
   
    - Uses robust, bounded, heavy-tailed ratio dynamics to avoid explosive paths.
   
    - Produces inputs sufficient for FCFF valuation even when consensus coverage is sparse.

    Parameters
    ----------
    sim:
        Mapping of existing driver draw matrices (updated in place).
    hist_annual:
        Historical annual panel used to infer revenue levels and ratio priors.
    fcf_periods:
        Period grid for the simulation output.
    nd_draws:
        Optional net debt draw matrix aligned to ``fcf_periods``.
    ctx:
        Run context providing deterministic RNG streams.
    sector_policy:
        Optional sector policy controlling ratio bounds and tax bounds.

    Returns
    -------
    (dict[str, numpy.ndarray], numpy.ndarray | None, bool)
        Updated ``sim``, updated ``nd_draws`` (unchanged or used for interest imputation), and a
        boolean flag indicating whether derivation succeeded.
   
    """
   
    policy = sector_policy if sector_policy is not None else SECTOR_POLICIES['Unknown']

    T = len(fcf_periods)

    if T == 0:

        return (sim, nd_draws, False)

    if nd_draws is not None:

        nd_draws = np.asarray(nd_draws, dtype = float)

        if nd_draws.ndim != 2:

            raise ValueError(f'nd_draws must be 2D [T,n_sims], got shape {nd_draws.shape}')

        if nd_draws.shape[0] != T:

            raise ValueError(f'nd_draws period mismatch in primitive FCFF: nd_draws has {nd_draws.shape[0]} rows but fcf_periods has {T}. Pass nd_draws aligned to fcf_periods.')

    revenue = sim.get('revenue', None)

    if revenue is None:

        if hist_annual is None or 'revenue' not in hist_annual.columns:

            return (sim, nd_draws, False)

        srev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').dropna()

        if srev.size < 6:

            return (sim, nd_draws, False)

        revenue = _simulate_rw_from_history_levels(
            hist = srev,
            T = T,
            floor_at_zero = True,
            rng = ctx.rng('prim:revenue:rw')
        )

        sim['revenue'] = revenue

    tax = sim.get('tax', None)

    if tax is None:

        if hist_annual is not None and 'tax' in hist_annual.columns:

            trat = pd.to_numeric(hist_annual['tax'], errors = 'coerce').dropna()

            base = float(np.clip(np.nanmedian(_pct_to_dec_if_needed(
                x = trat.to_numpy(dtype = float)
            )), policy.tax_lo, policy.tax_hi))

        else:

            base = 0.21

        eps = ctx.rng('prim:tax').standard_t(df = 8.0, size = (T, N_SIMS)) * 0.01

        tax = np.clip(base + eps, policy.tax_lo, policy.tax_hi)

        sim['tax'] = tax

    else:

        tax = _pct_to_dec_if_needed(
            x = tax
        )

        tax = np.clip(tax, policy.tax_lo, policy.tax_hi)

        sim['tax'] = tax


    def _hist_ratio(
        num_key: str
    ) -> pd.Series | None:
        """
        Compute a historical ratio series num/revenue for a given numerator key.

        Parameters
        ----------
        num_key:
            Numerator column name in ``hist_annual``.

        Returns
        -------
        pandas.Series | None
            Ratio series with non-finite values removed, or None when unavailable.
       
        """
     
        if hist_annual is None or 'revenue' not in hist_annual.columns or num_key not in hist_annual.columns:

            return None

        rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

        num = pd.to_numeric(hist_annual[num_key], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

        rev_safe = rev.where(np.abs(rev) > e12)

        r = (num / rev_safe).replace([np.inf, -np.inf], np.nan).dropna()

        return r if r.size else None


    def _implied_ratio(
        num_key: str,
        default: float,
        lo: float,
        hi: float
    ) -> np.ndarray:
        """
        Construct an implied ratio path from existing simulated numerators or a default.

        When the numerator draw matrix is already present in ``sim``, a base ratio is estimated as
        the median over simulations of (numerator / revenue). Otherwise, the supplied default is
        used. A small heavy-tailed perturbation is then added:

            ratio = clip(base + t_noise, lo, hi)

        The result has shape (T, N_SIMS) and is suitable for scaling revenue to obtain numerator
        levels.

        Parameters
        ----------
        num_key:
            Key of the numerator variable in ``sim``.
        default:
            Default base ratio when numerator draws are unavailable.
        lo, hi:
            Bounds applied to the ratio.

        Returns
        -------
        numpy.ndarray
            Ratio draw matrix of shape (T, N_SIMS).
     
        """
     
        a = sim.get(num_key, None)

        if a is None:

            base = default

        else:

            with np.errstate(divide = 'ignore', invalid = 'ignore'):
        
                r = a / np.where(np.abs(revenue) > e12, revenue, np.nan)

            base = float(np.clip(np.nanmedian(r), lo, hi))

        eps = ctx.rng(f'prim:{num_key}:implied').standard_t(df = 8.0, size = (T, N_SIMS)) * 0.01

        return np.clip(base + eps, lo, hi)


    ebit_m_hist = _hist_ratio(
        num_key = 'ebit'
    )

    if ebit_m_hist is not None and ebit_m_hist.size >= 6:

        ebit_margin = _simulate_ratio_rw_from_history(
            hist_ratio = ebit_m_hist,
            T = T,
            rng = ctx.rng('prim:ebit_margin'),
            lo = -0.3,
            hi = 0.6
        )

    else:

        ebit_margin = _implied_ratio(
            num_key = 'ebit',
            default = 0.15,
            lo = -0.3,
            hi = 0.6
        )

    da_r_hist = _hist_ratio(
        num_key = 'da'
    )

    if da_r_hist is not None and da_r_hist.size >= 6:

        da_ratio = _simulate_pos_ratio_rw_from_history(
            hist_ratio = da_r_hist.abs(),
            T = T,
            rng = ctx.rng('prim:da_ratio'),
            lo = policy.da_ratio_lo,
            hi = policy.da_ratio_hi
        )

    else:

        da_default = min(max(0.05, policy.da_ratio_lo), policy.da_ratio_hi)

        da_ratio = np.abs(_implied_ratio(
            num_key = 'da',
            default = da_default,
            lo = policy.da_ratio_lo,
            hi = policy.da_ratio_hi
        ))

    capex_r_hist = _hist_ratio(
        num_key = 'capex'
    )

    if capex_r_hist is not None and capex_r_hist.size >= 6:

        capex_ratio = _simulate_pos_ratio_rw_from_history(
            hist_ratio = capex_r_hist.abs(),
            T = T,
            rng = ctx.rng('prim:capex_ratio'),
            lo = policy.capex_ratio_lo,
            hi = policy.capex_ratio_hi
        )

    else:

        capex_default = min(max(0.07, policy.capex_ratio_lo), policy.capex_ratio_hi)

        capex_ratio = np.abs(_implied_ratio(
            num_key = 'capex',
            default = capex_default,
            lo = policy.capex_ratio_lo,
            hi = policy.capex_ratio_hi
        ))

    dnwc = sim.get('dnwc', None)

    if dnwc is None or not np.isfinite(dnwc).any():

        dnwc_r_hist = _hist_ratio(
            num_key = 'dnwc'
        )

        if dnwc_r_hist is not None and dnwc_r_hist.size >= 6:

            dnwc_ratio = _simulate_ratio_rw_from_history(
                hist_ratio = dnwc_r_hist,
                T = T,
                rng = ctx.rng('prim:dnwc_ratio'),
                lo = policy.dnwc_ratio_lo,
                hi = policy.dnwc_ratio_hi
            )

        else:

            dnwc_ratio = _implied_ratio(
                num_key = 'dnwc',
                default = 0.0,
                lo = policy.dnwc_ratio_lo,
                hi = policy.dnwc_ratio_hi
            )

        sim['dnwc'] = revenue * dnwc_ratio

    sim['ebit'] = revenue * ebit_margin

    sim['da'] = np.maximum(revenue * da_ratio, 0.0)

    sim['capex'] = np.maximum(revenue * capex_ratio, 0.0)

    if nd_draws is not None:

        rd = None

        if hist_annual is not None and 'interest' in hist_annual.columns and ('net_debt' in hist_annual.columns):

            nd = pd.to_numeric(hist_annual['net_debt'], errors = 'coerce').to_numpy(dtype = float)

            it = pd.to_numeric(hist_annual['interest'], errors = 'coerce').to_numpy(dtype = float)

            if nd.size >= 7 and it.size >= 7:

                nd_lag = np.roll(nd, 1)

                nd_lag[0] = nd_lag[1] if nd.size > 1 else nd_lag[0]

                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    r = np.abs(it) / np.where(np.abs(nd_lag) > e12, np.abs(nd_lag), np.nan)

                r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()

                if r.size >= 6:

                    rd = _simulate_pos_ratio_rw_from_history(
                        hist_ratio = r,
                        T = T,
                        rng = ctx.rng('prim:cost_of_debt'),
                        lo = 0.0,
                        hi = 0.25
                    )

        if rd is None:

            base = 0.05

            eps = ctx.rng('prim:cost_of_debt:fallback').standard_t(df = 8.0, size = (T, N_SIMS)) * 0.01

            rd = np.clip(base + eps, 0.0, 0.25)

        nd0 = None

        if hist_annual is not None and 'net_debt' in hist_annual.columns:

            nd_last = float(pd.to_numeric(hist_annual['net_debt'], errors = 'coerce').dropna().iloc[-1]) if pd.to_numeric(hist_annual['net_debt'], errors = 'coerce').dropna().size else np.nan

            if np.isfinite(nd_last):

                nd0 = np.full(N_SIMS, nd_last, dtype = float)

        if nd0 is None:

            nd0 = nd_draws[0, :].copy()

        nd_lag = np.vstack([nd0[None, :], nd_draws[:-1, :]])

        sim['interest'] = np.maximum(rd * np.maximum(nd_lag, 0.0), 0.0)

    one_minus_tax = 1.0 - sim['tax']

    sim['fcf'] = sim['ebit'] * one_minus_tax + sim['da'] - sim['capex'] + sim['dnwc']

    return (sim, nd_draws, True)


def _unit_sanity_warning(
    *,
    sim: dict[str, np.ndarray],
    cash_unit_mult: float,
    market_cap: float | None = None
):
    """
    Emit a heuristic warning when simulated revenue scale appears inconsistent with market capitalisation.

    The valuation pipeline relies on a cash-unit multiplier to normalise CapIQ statement units
    (for example, thousands versus millions). When the unit multiplier is incorrect, simulated
    level series can be off by orders of magnitude, which can silently corrupt valuations.

    This function compares the median simulated revenue level to market capitalisation and warns
    when the implied revenue-to-market-cap ratio is implausibly small or large:

        ratio = abs(median(revenue_draws)) / market_cap

    A warning is emitted when ratio < 0.01 or ratio > 100.0.

    Parameters
    ----------
    sim:
        Simulation mapping containing a "revenue" draw matrix.
    cash_unit_mult:
        Cash-unit multiplier applied to consensus level series.
    market_cap:
        Market capitalisation used as a scale reference.

    Returns
    -------
    None
        The function is diagnostic only.
    """
    if 'revenue' not in sim:

        return

    rev_med = np.nanmedian(sim['revenue']) if np.isfinite(sim['revenue']).any() else np.nan

    if not np.isfinite(rev_med):

        return

    if market_cap is not None and np.isfinite(market_cap) and (market_cap > 0):

        ratio = abs(rev_med) / market_cap

        if ratio < 0.01 or ratio > 100.0:

            warnings.warn(f'[units] revenue median {rev_med:.3g} vs mcap {market_cap:.3g} looks scale-mismatched. Check cash_unit_mult={cash_unit_mult} and CapIQ units.')


def _simulate_skewt_from_rows(
    dfT: pd.DataFrame,
    value_row: str,
    unit_mult: float,
    floor_at_zero: bool,
    *,
    rng: np.random.Generator,
    return_components: bool = False
):
    """
    Simulate period-by-period forecast draws from CapIQ consensus summary rows using a skew-t model.

    Inputs and interpretation
    -------------------------
    CapIQ forecast blocks provide, for each period, a set of summary statistics:

    - Mean-like estimate (the value row, often the average across analysts),
   
    - Median estimate,
   
    - High and Low (typically analyst range extremes), and
   
    - Standard deviation and number of estimates.

    The objective is to convert these summaries into a Monte Carlo draw matrix of shape
    (T_periods, N_SIMS) that captures:

    - uncertainty in the "true" mean estimate (analyst sampling error),
   
    - uncertainty in the dispersion estimate, and
   
    - asymmetric and heavy-tailed outcome uncertainty consistent with the high/low range.

    Hierarchical simulation structure
    ---------------------------------
    For each period i:

    1. Analyst-count normalisation:
      
       The effective analyst count n_eff is taken from ``No_of_Estimates`` when available, with a
       conservative fallback to the median analyst count (and a minimum of 3).

    2. Mean uncertainty (mu_sims):
      
       The reported mean mu_i is treated as an estimator with standard error approximately:

           se(mu_i) = sd_i / sqrt(n_eff_i)

       and simulated as:

           mu_sims_i ~ Normal(mu_i, se(mu_i))

       This represents uncertainty in the estimated consensus mean.

    3. Dispersion uncertainty (sigma_sims):
      
       The reported standard deviation sd_i is treated as uncertain. A scaled-inverse-chi-square
       style draw is used:

           df_sig = max(3, n_eff_i - 1)
       
           chi ~ ChiSquare(df_sig)
       
           sigma_sims_i = sd_i * sqrt(df_sig / chi)

       This introduces dispersion uncertainty that increases when analyst coverage is low.

    4. Skew-t innovations (x_std):
      
       Outcome uncertainty is modelled using a standardised skew-t innovation constructed as:

           Z0, Z1 ~ Normal(0, 1)
      
           W ~ ChiSquare(nu_i)
      
           S = sqrt(nu_i / W)
      
           U = delta_i * abs(Z0) + sqrt(1 - delta_i^2) * Z1
      
           T = S * U

       The innovation is then standardised to zero mean and unit variance using analytic moments
       from ``_skewt_standard_mean_var``:

           x_std = (T - mean(T)) / sqrt(var(T))

       Skewness calibration:
      
       A target skewness is inferred via Pearson's mean-median rule:

           target_skew_i = 3 * (mu_i - median_i) / sd_i

       and mapped to the skew-normal delta parameter using ``_delta_from_target_skew_skewnormal``.

       Tail calibration:
      
       Degrees of freedom nu_i are inferred from the (High, Low) range and analyst count using
       ``_infer_df_from_high_low``.

    5. Final draws:
      
       The simulated period draw matrix is:

           draws_i = mu_sims_i + sigma_sims_i * x_std_i

       If ``floor_at_zero`` is True, the draws are floored at 0 to enforce non-negativity.

    Missing-statistic handling
    --------------------------
    When sd_i is missing or non-positive, a proxy sd is constructed from the high/low range:

        sd_i ~= (High_i - Low_i) / 4

    which corresponds to a rough normal 95% interval heuristic.

    Outputs
    -------
    The function returns:
    
    - draws: simulated outcomes,
    
    - calib: a per-period DataFrame reporting mu, sigma, delta, nu_df, and target_skew.

    When ``return_components`` is True, the latent components (mu_sims, sigma_sims, x_std) are also
    returned. These components are reused later for dependence adjustments.

    Advantages
    ----------
    
    - Captures parameter uncertainty (mu and sigma) as well as outcome uncertainty.
    
    - Allows asymmetric outcomes via skewness inferred from mean/median disagreement.
    
    - Allows heavy tails and range-consistency via nu inferred from high/low extremes.
    
    - Provides latent standardised innovations for downstream dependence modelling.

    Parameters
    ----------
    dfT:
        Forecast table with consensus statistic rows.
    value_row:
        Name of the primary value row.
    unit_mult:
        Unit multiplier applied to convert table units into model cash units.
    floor_at_zero:
        Whether to floor draws at zero.
    rng:
        Random generator.
    return_components:
        Whether to return latent simulation components.

    Returns
    -------
    (numpy.ndarray, pandas.DataFrame) or tuple
        By default returns (draws, calib). When ``return_components`` is True, returns
        (draws, calib, mu_sims, sigma_sims, x_std).
  
    """


    def _fast_get(
        row_name
    ):
        """
        Retrieve and scale a numeric consensus row quickly, returning NaNs when absent.

        Parameters
        ----------
        row_name:
            Row label in the consensus table.

        Returns
        -------
        numpy.ndarray
            Scaled numeric array with length equal to the number of periods (table columns).
      
        """
      
        if row_name not in dfT.index:

            return np.full(len(dfT.columns), np.nan, dtype = float)

        vals = dfT.loc[row_name].values

        try:

            return vals.astype(float) * unit_mult

        except (ValueError, TypeError):

            return pd.to_numeric(vals, errors = 'coerce').to_numpy(dtype = float) * unit_mult


    mu = _fast_get(
        row_name = value_row
    )

    med = _fast_get(
        row_name = 'Median'
    )

    mu = np.where(np.isfinite(mu), mu, med)

    hi = _fast_get(
        row_name = 'High'
    )

    lo = _fast_get(
        row_name = 'Low'
    )

    sd = _fast_get(
        row_name = 'Std_Dev'
    )

    if 'No_of_Estimates' in dfT.index:

        n_est_raw = dfT.loc['No_of_Estimates'].astype(str).values

    else:

        n_est_raw = np.full(len(mu), '3', dtype = object)

    n_est = np.empty(len(n_est_raw), dtype = float)

    for i, x in enumerate(n_est_raw):
  
        n_est[i] = _parse_n_analysts(
            x = x
        )

    T_periods = len(mu)

    target_skew = np.array([_estimate_skewness_from_mean_median(
        mu = mu[i],
        med = med[i],
        sigma = sd[i]
    ) for i in range(T_periods)])

    delta = np.array([_delta_from_target_skew_skewnormal(
        target_skew = s
    ) for s in target_skew])

    nu = np.array([_infer_df_from_high_low(
        mu = mu[i],
        sigma = sd[i],
        high = hi[i],
        low = lo[i],
        n_analysts = n_est[i]
    ) for i in range(T_periods)])

    missing_sd = ~np.isfinite(sd) | (sd <= 0)

    range_proxy = (hi - lo) / 4.0

    sd[missing_sd] = np.where((np.isfinite(hi) & np.isfinite(lo) & (hi > lo))[missing_sd], range_proxy[missing_sd], 0.0)

    n_est_finite = n_est[np.isfinite(n_est)]

    n_est_med = np.median(n_est_finite) if n_est_finite.size else 3.0

    n_est_med = max(3.0, n_est_med)

    n_eff = np.where(np.isfinite(n_est) & (n_est >= 3), n_est, n_est_med)

    sigma_mean = sd / np.sqrt(n_eff)

    mu_sims = rng.normal(loc = mu[:, None], scale = sigma_mean[:, None], size = (T_periods, N_SIMS))

    df_sig = np.maximum(3.0, n_eff - 1.0)

    chi = rng.chisquare(df = df_sig[:, None], size = (T_periods, N_SIMS))

    sigma_sims = sd[:, None] * np.sqrt(df_sig[:, None] / np.clip(chi, e12, None))

    z0 = rng.standard_normal((T_periods, N_SIMS))

    z1 = rng.standard_normal((T_periods, N_SIMS))

    w = rng.chisquare(df = nu[:, None], size = (T_periods, N_SIMS))

    s = np.sqrt(nu[:, None] / w)

    u = delta[:, None] * np.abs(z0) + np.sqrt(1.0 - delta[:, None] ** 2) * z1

    t_innov = s * u

    m_t_vec = np.empty(T_periods)

    v_t_vec = np.empty(T_periods)

    for i in range(T_periods):
      
        m, v = _skewt_standard_mean_var(
            delta = delta[i],
            nu = nu[i]
        )

        m_t_vec[i] = m

        v_t_vec[i] = v

    x_std = (t_innov - m_t_vec[:, None]) / np.sqrt(v_t_vec[:, None])

    draws = mu_sims + sigma_sims * x_std

    if floor_at_zero:

        draws = np.maximum(draws, 0.0)

    calib = pd.DataFrame({'mu': mu, 'sigma': sd, 'delta': delta, 'nu_df': nu, 'target_skew': target_skew}, index = dfT.columns)

    if return_components:

        return (draws, calib, mu_sims, sigma_sims, x_std)

    return (draws, calib)


def _annual_periods(
    dfT: pd.DataFrame,
    *,
    fy_m: int | None = None
) -> pd.DatetimeIndex:
    """
    Extract annual period-end timestamps from a consensus forecast table.

    The function attempts to interpret the forecast columns as period end dates and return a set of
    annual fiscal year-end timestamps.

    Selection logic
    ---------------
  
    1. Convert columns to datetimes, sort, and normalise to midnight.
  
    2. Determine the fiscal year-end month fy_end_m using, in order of preference:
  
       - the explicit ``fy_m`` argument when valid,
  
       - inference from the table via ``_infer_fy_end_month_day_from_future``, and
  
       - the global default derived from ``FY_FREQ``.
  
    3. If a ``period_type`` row exists:
  
       - return columns explicitly labelled "Annual" when present.
  
       - otherwise, if quarterly columns exist, infer annual ends as either:
  
         - the subset of quarter ends whose month equals fy_end_m (preferred), or
  
         - every fourth quarter end (Q4 selection) as a fallback.
  
    4. If no ``period_type`` row exists:
  
       - if the columns look quarterly-like, apply the same Q4 selection heuristic,
  
       - otherwise return all parsed columns (treated as annual by default).

    This extraction is used when constructing the valuation period grid and when aligning mixed
    quarterly/annual consensus tables.

    Parameters
    ----------
    dfT:
        Forecast table with date-like columns and optional ``period_type`` row.
    fy_m:
        Optional fiscal year-end month override.

    Returns
    -------
    pandas.DatetimeIndex
        Annual period end timestamps (normalised).
 
    """
 
    if dfT is None or dfT.empty:

        return pd.DatetimeIndex([])

    cols = _to_datetime_index(
        cols = dfT.columns
    )

    cols = cols[pd.notna(cols)].sort_values().normalize()

    if len(cols) == 0:

        return pd.DatetimeIndex([])

    fy_end_m = None

    if fy_m is not None:

        try:

            fy_end_m = int(fy_m)

        except (TypeError, ValueError):

            fy_end_m = None

    if fy_end_m is None or not 1 <= fy_end_m <= 12:

        try:

            fy_end_m, _ = _infer_fy_end_month_day_from_future(
                metric_future = dfT
            )

        except (TypeError, ValueError):

            fy_end_m = None

    if fy_end_m is None or not 1 <= fy_end_m <= 12:

        fy_end_m = _fy_end_month(
            fy_freq = FY_FREQ
        )

    if 'period_type' in dfT.index:

        types = dfT.loc['period_type'].astype(str).str.lower().reindex(dfT.columns).values

        types = np.array([t if isinstance(t, str) else '' for t in types], dtype = object)

        ann_mask = types == 'annual'

        if ann_mask.any():

            ann_cols = _to_datetime_index(
                cols = pd.Index(dfT.columns)[ann_mask]
            )

            ann_cols = ann_cols[pd.notna(ann_cols)].sort_values().normalize()

            if len(ann_cols) > 0:

                return ann_cols

        q_mask = types == 'quarterly'

        q_cols = _to_datetime_index(
            cols = pd.Index(dfT.columns)[q_mask]
        )

        q_cols = q_cols[pd.notna(q_cols)].sort_values().normalize()

        if len(q_cols) >= 4:

            cand = q_cols[q_cols.month == fy_end_m]

            if len(cand) >= 1:

                return pd.DatetimeIndex(sorted(set(cand))).normalize()

            return q_cols[3::4].normalize()

    if _is_quarterly_like(
        cols = cols
    ) and len(cols) >= 4:

        cand = cols[cols.month == fy_end_m]

        if len(cand) >= 1:

            return pd.DatetimeIndex(sorted(set(cand))).normalize()

        return cols[3::4].normalize()

    return cols.normalize()


def _quarterly_periods(
    dfT: pd.DataFrame,
    *,
    fy_m: int
) -> list[pd.Timestamp]:
    """
    Extract quarterly period-end timestamps from a consensus forecast table.

    The function returns a list of quarter-end timestamps inferred from the forecast table columns.
    When a ``period_type`` row is present, it is used directly to identify quarterly columns.
    Otherwise, quarter ends are inferred by testing whether each column timestamp equals the end
    time of a fiscal quarter under the frequency "Q-{FY_END_MONTH}".

    Parameters
    ----------
    dfT:
        Forecast table with date-like columns and optional ``period_type`` row.
    fy_m:
        Fiscal year-end month (1..12), used to define the fiscal quarter frequency.

    Returns
    -------
    list[pandas.Timestamp]
        Sorted list of unique quarter-end timestamps (normalised).
  
    """
  
    cols = pd.to_datetime(dfT.columns, errors = 'coerce')

    cols = cols[pd.notna(cols)]

    if len(cols) == 0:

        return []

    if 'period_type' in dfT.index:

        pt = dfT.loc['period_type'].astype(str).str.strip().str.lower()

        q = cols[pt.eq('quarterly')]

        if len(q) > 0:

            return pd.DatetimeIndex(q).sort_values().unique().tolist()

    qfreq = f'Q-{calendar.month_abbr[fy_m].upper()}'

    out: list[pd.Timestamp] = []

    for c in cols:
        
        ts = pd.Timestamp(c).normalize()

        try:

            if pd.Period(ts, freq = qfreq).end_time.normalize() == ts:

                out.append(ts)

        except (TypeError, ValueError):

            continue

    return pd.DatetimeIndex(out).sort_values().unique().tolist()


def _quarterly_cols_from_future(
    dfT: pd.DataFrame | None,
    *,
    fy_m: int
) -> pd.DatetimeIndex:
    """
    Return the set of quarterly period columns present in a forecast table.

    This helper standardises quarterly column extraction across multiple drivers. It prefers the
    explicit ``period_type`` row when present; otherwise it falls back to ``_quarterly_periods``.

    Parameters
    ----------
    dfT:
        Forecast table (or None).
    fy_m:
        Fiscal year-end month used for quarter inference when ``period_type`` is absent.

    Returns
    -------
    pandas.DatetimeIndex
        Quarterly period end timestamps present in the table.
   
    """
   
    if dfT is None or dfT.empty:

        return pd.DatetimeIndex([])

    src_cols = pd.to_datetime(dfT.columns, errors = 'coerce')

    ok = pd.notna(src_cols)

    if not ok.any():

        return pd.DatetimeIndex([])

    cols = pd.DatetimeIndex(src_cols[ok]).normalize()

    cols = cols[~cols.duplicated()].sort_values()

    if 'period_type' in dfT.index:

        pt = dfT.loc['period_type'].astype(str).str.strip().str.lower()

        pt = pt.reindex(dfT.columns)

        m = pt.eq('quarterly').to_numpy()

        q = pd.to_datetime(pd.Index(dfT.columns)[m], errors = 'coerce')

        q = pd.DatetimeIndex(q).dropna().normalize()

        return q[~q.duplicated()].sort_values()

    q = _quarterly_periods(
        dfT = dfT,
        fy_m = fy_m
    )

    return pd.DatetimeIndex(q).dropna().normalize().sort_values().unique()


def _build_quarterly_override_period_types(
    *,
    periods: pd.DatetimeIndex,
    period_types_global: list[str],
    required_futures: dict[str, pd.DataFrame | None],
    fy_m: int,
    fy_d: int
) -> tuple[list[str], bool, dict[str, list[pd.Timestamp]]]:
    """
    Override a mixed period-type vector to quarterly only when all required drivers support the quarter.

    Motivation
    ----------
    Mixed-period valuation grids may include quarterly periods, but quarterly valuation requires
    consistent coverage across all drivers used by a method. If some required driver lacks a
    quarterly forecast for a given quarter, treating that period as quarterly would force an
    imputation or silently mix periodicities.

    This helper converts each period to "quarterly" only when every non-empty required future table
    contains that quarter in its quarterly column set; otherwise the period is forced to "annual".

    Outputs
    -------
    The function returns:
  
    - a revised list of period types (lower-case strings),
  
    - a flag indicating whether any quarterly periods remain after gating, and
  
    - a dictionary mapping each driver key to the list of quarters that were requested but missing.

    Parameters
    ----------
    periods:
        Period grid (annual and/or quarterly).
    period_types_global:
        Baseline period type vector ("annual"/"quarterly") aligned to ``periods``.
    required_futures:
        Mapping of driver keys to their forecast tables; only non-empty tables are checked.
    fy_m, fy_d:
        Fiscal year-end month/day (retained for API symmetry; fy_m is used for quarter inference).

    Returns
    -------
    (list[str], bool, dict[str, list[pandas.Timestamp]])
        Revised period types, quarterly-allowed flag, and per-driver missing-quarter diagnostics.
  
    """
  
    periods = pd.DatetimeIndex(pd.to_datetime(periods, errors = 'coerce')).dropna().normalize().sort_values().unique()

    ptg = [str(x).lower() for x in period_types_global]

    if len(ptg) != len(periods):

        raise ValueError('period_types_global must be same length as periods')

    qcols_by_key: dict[str, set[pd.Timestamp]] = {}

    for k, df in required_futures.items():
   
        qcols_by_key[k] = set(_quarterly_cols_from_future(
            dfT = df,
            fy_m = fy_m
        ).tolist()) if df is not None else set()

    ptq = []

    for p, t in zip(periods, ptg):
    
        p_norm = pd.Timestamp(p).normalize()

        has_all_q = True

        any_checked = False

        for k, df in required_futures.items():
        
            if df is None or df.empty:

                continue

            any_checked = True

            qset = qcols_by_key.get(k, set())

            if p_norm not in qset:

                has_all_q = False

                break

        has_all_q = has_all_q and any_checked

        if t == 'quarterly':

            ptq.append('quarterly' if has_all_q else 'annual')

        else:

            ptq.append('quarterly' if has_all_q else 'annual')

    q_needed = [pd.Timestamp(p).normalize() for p, t in zip(periods, ptq) if t == 'quarterly']

    missing: dict[str, list[pd.Timestamp]] = {}

    for k in required_futures.keys():
      
        qset = qcols_by_key.get(k, set())

        miss_k = [p for p in q_needed if p not in qset]

        if miss_k:

            missing[k] = miss_k

    allow = any((t == 'quarterly' for t in ptq))

    return (ptq, allow, missing)


def _build_mixed_valuation_periods(
    *,
    base_future: pd.DataFrame,
    driver_futures: dict[str, pd.DataFrame | None],
    include_stub_quarters: bool = True,
    fy_m: int,
    fy_d: int
) -> tuple[pd.DatetimeIndex, list[str], pd.DatetimeIndex]:
    """
    Construct a mixed annual/quarterly valuation period grid from consensus forecast tables.

    Overview
    --------
    The valuation engines operate on a single period grid. CapIQ consensus inputs, however, may
    provide:

    - annual fiscal year-end periods,
  
    - quarterly periods for some drivers, and
  
    - "stub" quarterly periods prior to the first annual forecast year.

    This helper builds a mixed grid that:
  
    - always includes the annual fiscal year-end periods from the base forecast table, and
  
    - includes quarterly periods when they are present for at least one driver in a given fiscal
      year (Q1-Q3), plus optional stub quarters before the first annual period.

    Construction details
    --------------------
  
    1. Annual anchors:
  
       Annual periods are extracted from ``base_future`` using ``_annual_periods`` and filtered to
       those matching the fiscal year-end month/day (fy_m, fy_d).

    2. Quarterly union:
  
       For each future table (base and driver futures), quarterly period ends are extracted and any
       period ends that also appear as annual ends in that table are removed. The union over all
       tables forms q_union.

    3. Stub quarters:
  
       When ``include_stub_quarters`` is True, quarters in q_union that occur before the first
       annual period are included as stub quarters. These can improve near-term valuation when only
       quarterly data are available immediately ahead.

    4. Per-fiscal-year quarter inclusion:
  
       For each annual year-end a, the Q1-Q4 quarter ends are constructed using a fiscal quarter
       frequency "Q-{fy_m}". If any of Q1-Q3 are present in q_union for that fiscal year, all of
       Q1-Q3 are included in the grid (Q4 equals the annual end and is already included).

    The function returns the union of annual and selected quarterly periods, along with a period
    type vector and the subset of quarterly stub periods.

    Parameters
    ----------
    base_future:
        Base forecast table providing the annual anchor periods (typically FCF or a primary driver).
    driver_futures:
        Additional forecast tables used to discover quarterly availability.
    include_stub_quarters:
        Whether to include quarters prior to the first annual period when available.
    fy_m, fy_d:
        Fiscal year-end month and day.

    Returns
    -------
    (pandas.DatetimeIndex, list[str], pandas.DatetimeIndex)
        Tuple ``(periods, period_types, q_stub)`` where period_types are lower-case strings and
        q_stub contains quarterly periods strictly before the first annual anchor.
 
    """
  
    ann = pd.DatetimeIndex(_annual_periods(
        dfT = base_future,
        fy_m = fy_m
    )).sort_values().unique()

    if len(ann) == 0:

        return (pd.DatetimeIndex([]), [], pd.DatetimeIndex([]))

    ann = pd.DatetimeIndex([pd.Timestamp(a).normalize() for a in ann])

    ann = ann[(ann.month == fy_m) & (ann.day == fy_d)].sort_values().unique()

    if len(ann) == 0:

        return (pd.DatetimeIndex([]), [], pd.DatetimeIndex([]))

    q_union = pd.DatetimeIndex([])

    for fut in [base_future] + [v for v in driver_futures.values() if v is not None]:
   
        q_raw = set((pd.Timestamp(x).normalize() for x in _quarterly_periods(
            dfT = fut,
            fy_m = fy_m
        )))

        a_raw = set((pd.Timestamp(x).normalize() for x in _annual_periods(
            dfT = fut,
            fy_m = fy_m
        )))

        q_union = q_union.union(pd.DatetimeIndex(sorted(q_raw - a_raw)))

    q_union = q_union.sort_values().unique()

    if not include_stub_quarters or len(q_union) == 0:

        return (ann, ['annual'] * len(ann), pd.DatetimeIndex([]))

    qfreq = f'Q-{calendar.month_abbr[int(fy_m)].upper()}'

    q_union_set = set((pd.Timestamp(q).normalize() for q in q_union))

    ann_min = pd.Timestamp(ann.min()).normalize()

    quarter_periods: set[pd.Timestamp] = set()

    if include_stub_quarters and len(q_union):

        quarter_periods.update((pd.Timestamp(q).normalize() for q in q_union if pd.Timestamp(q).normalize() < ann_min))

    for a in ann:
    
        a = pd.Timestamp(a).normalize()

        q_list = pd.period_range(end = a, periods = 4, freq = qfreq).to_timestamp(how = 'end').normalize()

        q_list = [pd.Timestamp(q).normalize() for q in q_list]

        q_list_no_q4 = [q for q in q_list if q != a]

        if any((q in q_union_set for q in q_list_no_q4)):

            quarter_periods.update(q_list_no_q4)

    q_keep = pd.DatetimeIndex(sorted(quarter_periods)).sort_values().unique()

    periods = ann.union(q_keep).sort_values().unique()

    q_set = set(q_keep.tolist())

    period_types = ['quarterly' if pd.Timestamp(p).normalize() in q_set else 'annual' for p in periods]

    q_stub = q_keep[q_keep < ann_min]

    return (periods, period_types, q_stub)


def _terminal_value_perpetuity(
    cf_T: np.ndarray,
    r: float | np.ndarray,
    g: np.ndarray,
    dt_years: float,
    eps: float = 1e-08
) -> np.ndarray:
    """
    Compute a discrete-compounding perpetuity terminal value for a cash flow at horizon T.

    Terminal value formula
    ----------------------
    For a terminal cash flow CF_T observed at time T and a perpetual growth rate g, the discrete
    perpetuity value at time T (with cash flows arriving with a time step dt) is:

        TV_T = CF_T * (1 + g)^dt / ((1 + r)^dt - (1 + g)^dt)

    where r is the discount rate. The dt exponent allows the terminal step to represent a fraction
    of a year (for example, quarterly dt = 0.25).

    Numerical stability
    -------------------
  
    - The computation uses log1p and exp to avoid precision loss for small rates:

          (1 + r)^dt = exp(log(1 + r) * dt)

    - g is clipped to a conservative range to avoid invalid log1p inputs.
   
    - The denominator is floored at eps to avoid division by zero when r approaches g.

    Parameters
    ----------
    cf_T:
        Horizon cash flow array (typically shape (n_sims,)).
    r:
        Discount rate (scalar or array broadcastable to cf_T).
    g:
        Terminal growth rate array (typically shape (n_sims,)).
    dt_years:
        Time step in years between cash flow periods at the horizon.
    eps:
        Small positive floor used for dt and denominator.

    Returns
    -------
    numpy.ndarray
        Terminal value array aligned to cf_T.
 
    """
 
    dt = max(dt_years, eps)

    cf_T = np.asarray(cf_T, dtype = float)

    g = np.asarray(g, dtype = float)

    r = np.asarray(r, dtype = float)

    one_r_dt = np.exp(np.log1p(r) * dt)

    one_g_dt = np.exp(np.log1p(np.clip(g, -0.99, 10.0)) * dt)

    denom = np.maximum(one_r_dt - one_g_dt, eps)

    return cf_T * one_g_dt / denom


def _filter_future_periods(
    periods: pd.DatetimeIndex,
    period_types: list[str],
    today: pd.Timestamp
) -> tuple[pd.DatetimeIndex, list[str]]:
    """
    Filter a period grid to retain only periods strictly after the valuation date.

    Parameters
    ----------
    periods:
        Candidate period end timestamps.
    period_types:
        Period type strings aligned to periods.
    today:
        Valuation date.

    Returns
    -------
    (pandas.DatetimeIndex, list[str])
        Filtered periods and corresponding period types. Returns empty outputs when no future
        periods remain.
  
    """
  
    if periods is None or len(periods) == 0:

        return (pd.DatetimeIndex([]), [])

    if len(period_types) != len(periods):

        raise ValueError('period_types must be same length as periods')

    today = pd.Timestamp(today).normalize()

    mask = pd.DatetimeIndex(periods).normalize() > today

    if not mask.any():

        return (pd.DatetimeIndex([]), [])

    keep_idx = np.where(mask)[0]

    new_periods = pd.DatetimeIndex(periods)[keep_idx]

    new_types = [period_types[i] for i in keep_idx.tolist()]

    return (new_periods, new_types)


def _net_debt_at_valuation_date(
    *,
    hist_bal: pd.DataFrame | None,
    net_debt_future: pd.DataFrame | None,
    today: pd.Timestamp,
    cost_of_debt: float | None = None
) -> float | None:
    """
    Estimate net debt at the valuation date using historical statements and/or consensus forecasts.

    Role in FCFF valuation
    ----------------------
    FCFF DCF produces an enterprise value. Converting enterprise value to equity value requires a
    net debt estimate at the valuation date:

        EquityValue ~= EnterpriseValue - NetDebt

    where net debt is defined in the prevailing statement convention (typically debt minus cash).

    Estimation order
    ---------------
   
    1. Historical balance sheet:
   
       When a historical net debt series can be derived from ``hist_bal`` (either explicit net debt
       or debt minus cash), the last observation at or before ``today`` is used. The value is scaled
       by ``UNIT_MULT`` to match model units.

    2. Consensus net debt table:
   
       When historical data are unavailable, the consensus forecast table is used. Columns are
       coerced to datetimes, duplicates are removed, and the value row is selected as:
     
       - "Net_Debt" when present, otherwise the first row.
     
       The forecast value at the column nearest to ``today`` is selected and scaled by UNIT_MULT.

       If the nearest forecast period is after ``today`` and a cost of debt is provided, the value
       is discounted back to the valuation date:

           dt_years = (nearest - today) / 365
     
           nd_today ~= nd_nearest / (1 + cost_of_debt)^dt_years

       This discounting is a pragmatic adjustment to avoid overstating the impact of a forward-dated
       net debt observation when it must be applied at time 0.

    Parameters
    ----------
    hist_bal:
        Historical balance-sheet statement table.
    net_debt_future:
        Consensus net debt forecast table.
    today:
        Valuation date.
    cost_of_debt:
        Optional discount rate used when the nearest consensus net debt is forward-dated.

    Returns
    -------
    float | None
        Net debt estimate in model cash units, or None when no estimate can be obtained.
   
    """
   
    today = pd.Timestamp(today).normalize()

    nd_hist = _hist_net_debt_debt_minus_cash(
        hist_bal = hist_bal
    ) if hist_bal is not None else None

    if nd_hist is not None and (not nd_hist.empty):

        s = pd.to_numeric(nd_hist, errors = 'coerce')

        try:

            s.index = pd.to_datetime(s.index, errors = 'coerce')

        except (TypeError, ValueError):

            pass

        s = s.dropna()

        if len(s):

            s = s.sort_index()

            s_valid = s[s.index <= today]

            if len(s_valid):

                return float(s_valid.iloc[-1]) * UNIT_MULT

    if net_debt_future is None or net_debt_future.empty:

        return None

    cols = pd.to_datetime(net_debt_future.columns, errors = 'coerce')

    ok = pd.notna(cols)

    if not ok.any():

        return None

    cols = pd.DatetimeIndex(cols[ok]).normalize()

    df = net_debt_future.loc[:, ok].copy()

    df.columns = cols

    df = df.loc[:, ~df.columns.duplicated(keep = 'last')]

    row = 'Net_Debt' if 'Net_Debt' in df.index else df.index[0] if len(df.index) else None

    if row is None:

        return None

    vals = pd.to_numeric(df.loc[row], errors = 'coerce')

    if vals.dropna().empty:

        return None

    nearest = cols[np.argmin(np.abs((cols - today).days))]

    nd = float(pd.to_numeric(df.loc[row, nearest], errors = 'coerce'))

    if not np.isfinite(nd):

        return None

    nd = nd * UNIT_MULT

    if nearest > today and np.isfinite(cost_of_debt) and (cost_of_debt > -1.0):

        dt_years = float((nearest - today).days) / 365.0

        disc = (1.0 + cost_of_debt) ** max(dt_years, 0.0)

        if np.isfinite(disc) and disc > 0:

            nd = nd / disc

    return nd


def _infer_bvps0_from_history(
    *,
    hist_bal: pd.DataFrame | None,
    hist_ratios: pd.DataFrame | None,
    shares_outstanding: float | None
) -> float | None:
    """
    Infer an initial book value per share (BVPS_0) from historical tables.

    Residual income valuation requires an initial book value per share. This helper attempts to
    obtain BVPS_0 using, in order:

    1. Historical ratios sheet:
  
       Extract a "Book Value Per Share" style series from ``hist_ratios`` and use the last finite
       value.

    2. Balance sheet equity divided by shares:
    
       When BVPS is unavailable in ratios, total equity is extracted from ``hist_bal`` using a set
       of candidate equity row labels. The most recent equity value is scaled by UNIT_MULT and
       divided by ``shares_outstanding``.

    Parameters
    ----------
    hist_bal:
        Historical balance-sheet statement table.
    hist_ratios:
        Historical ratios table.
    shares_outstanding:
        Shares outstanding used to convert total equity to per-share values.

    Returns
    -------
    float | None
        Initial book value per share, or None when insufficient inputs exist.

    """

    if hist_ratios is not None and (not hist_ratios.empty):

        bvps_rows = ('Book Value Per Share', 'Book Value / Share', 'Book Value/Share', 'Net Asset Value Per Share', 'BVPS')

        s_bvps = _extract_hist_ratios_series(
            df = hist_ratios,
            row_candidates = bvps_rows
        )

        if s_bvps is not None and len(s_bvps.dropna()):

            return float(pd.to_numeric(s_bvps, errors = 'coerce').dropna().iloc[-1])

    if hist_bal is None or hist_bal.empty:

        return None

    if shares_outstanding is None or not np.isfinite(shares_outstanding) or shares_outstanding <= 0:

        return None

    equity_rows = ('Total Equity', "Total Stockholders' Equity", "Total Shareholders' Equity", 'Total Common Equity', 'Total Equity (Including Minority Interest)', 'Total Equity (incl. Minority Interest)')

    row = _first_existing_row(
        df = hist_bal,
        candidates = equity_rows
    )

    if row is None:

        return None

    s_eq = _as_numeric_series(
        s = hist_bal.loc[row]
    ).dropna()

    if s_eq.empty:

        return None

    eq_val = float(s_eq.iloc[-1]) * UNIT_MULT

    return eq_val / float(shares_outstanding)


def _future_to_annual_aligned(
    dfT: pd.DataFrame,
    periods: pd.DatetimeIndex,
    mode: str = 'flow'
) -> pd.DataFrame:
    """
    Align a forecast table to a specified set of annual period ends, aggregating quarters when needed.

    Purpose
    -------
    CapIQ forecast tables may contain annual periods, quarterly periods, or a mixture. Many model
    components require an annual-aligned table on a specific annual period grid. This helper
    constructs such a table by copying annual values when available and otherwise aggregating the
    most recent four quarters ending at the annual period.

    Modes
    -----
    The aggregation behaviour depends on the economic type of the metric:

    - mode == "flow":
    
      Quarterly values represent additive flows. Aggregation is a sum:

          AnnualValue = sum(q1, q2, q3, q4)

      For "Std_Dev", variances are summed under an independence approximation:

          AnnualStdDev = sqrt( sum(StdDev_q^2) )

    - mode == "ratio":
      
      Quarterly values represent rates or ratios. Aggregation is an average:

          AnnualValue = mean(q1, q2, q3, q4)

    - mode == "stock":
     
      Quarterly values represent stocks/levels. Aggregation selects the last available quarter-end
      within the fiscal year, with backfilling to the latest finite value if the final quarter is
      missing.

    When a matching annual column exists for a target period, it is used directly. When quarterly
    aggregation is not feasible (for example, fewer than 4 quarters available), the nearest available
    column by date is used as a fallback.

    Input/Output schema
    -------------------
    The output retains the same index rows as the input (including estimate statistic rows) and uses
    ``periods`` as the columns. Numeric rows are converted to numeric arrays internally for speed.
    ``No_of_Estimates`` is propagated using the last quarter in the aggregated block when available.

    Parameters
    ----------
    dfT:
        Forecast table with date-like columns.
    periods:
        Target annual period end timestamps.
    mode:
        One of "flow", "ratio", or "stock".

    Returns
    -------
    pandas.DataFrame
        Annual-aligned forecast table with columns equal to ``periods``.
   
    """
   
    if dfT is None or dfT.empty:

        return dfT

    periods = pd.DatetimeIndex(pd.to_datetime(periods, errors = 'coerce')).dropna().normalize().sort_values()

    if len(periods) == 0:

        return dfT

    orig_cols = pd.Index(dfT.columns)

    dt_all = _to_datetime_index(
        cols = orig_cols
    )

    ok = pd.notna(dt_all)

    if not bool(np.any(ok)):

        return dfT.reindex(columns = periods)

    kept_orig = orig_cols[ok]

    dt = pd.DatetimeIndex(dt_all[ok]).normalize()

    df = dfT.loc[:, kept_orig]

    df = df.copy(deep = False)

    df.columns = dt

    df = df.sort_index(axis = 1)

    if df.columns.has_duplicates:

        df = df.loc[:, ~df.columns.duplicated(keep = 'last')]

    annual_cols = pd.DatetimeIndex([])

    quarterly_cols = pd.DatetimeIndex([])

    if 'period_type' in dfT.index:

        types = dfT.loc['period_type', kept_orig].astype(str).str.lower().to_numpy()

        annual_cols = df.columns[types == 'annual']

        quarterly_cols = df.columns[types == 'quarterly']

    else:

        quarterly_cols = df.columns

        annual_cols = df.columns

    annual_cols = pd.DatetimeIndex(annual_cols).sort_values().normalize()

    quarterly_cols = pd.DatetimeIndex(quarterly_cols).sort_values().normalize()

    out = pd.DataFrame(index = df.index, columns = periods, dtype = object)

    meta_rows = {'period_type', 'period_label'}

    no_est_row = 'No_of_Estimates'

    num_rows = [r for r in df.index if r not in meta_rows and r != no_est_row]

    has_no_est = no_est_row in df.index

    df_num = df.loc[num_rows].apply(pd.to_numeric, errors = 'coerce').to_numpy(dtype = float, copy = False)

    num_row_pos = {r: i for i, r in enumerate(num_rows)}

    sd_pos = num_row_pos.get('Std_Dev', None)

    cols_all = df.columns

    colsD = cols_all.values.astype('datetime64[D]')

    q_cols = quarterly_cols

    q_colsD = q_cols.values.astype('datetime64[D]') if len(q_cols) else None

    out_num = np.full((len(num_rows), len(periods)), np.nan, dtype = float)

    out_no_est = np.empty(len(periods), dtype = object) if has_no_est else None

    for j, d in enumerate(periods):
        
        dD = np.datetime64(d.normalize(), 'D')

        if d in annual_cols:

            k = np.searchsorted(colsD, dD, side = 'left')

            if 0 <= k < len(colsD) and colsD[k] == dD:

                out_num[:, j] = df_num[:, k]

                if has_no_est:

                    out_no_est[j] = df.loc[no_est_row, cols_all[k]]

                continue

        use_quarters = False

        qs_idx = None

        if q_colsD is not None and len(q_colsD) >= 4:

            i1 = np.searchsorted(q_colsD, dD, side = 'right')

            if i1 > 0 and q_colsD[i1 - 1] == dD:

                qs_start = i1 - 4

                if qs_start >= 0:

                    if dD - q_colsD[qs_start] <= np.timedelta64(370, 'D'):

                        qs_idx = np.arange(qs_start, i1, dtype = int)

                        use_quarters = qs_idx.size == 4

        if use_quarters:

            qsel = q_cols[qs_idx]

            col_pos = df.columns.get_indexer(qsel)

            col_pos = col_pos[col_pos >= 0]

            if col_pos.size >= 3:

                block = df_num[:, col_pos]

                if mode == 'flow':

                    s = np.nansum(block, axis = 1)

                    if sd_pos is not None:

                        s[sd_pos] = np.sqrt(np.nansum(block[sd_pos, :] ** 2))

                    out_num[:, j] = s
                    
                elif mode == 'ratio':

                    out_num[:, j] = np.nanmean(block, axis = 1)
                    
                elif mode == 'stock':

                    last = block[:, -1].copy()

                    for kk in range(block.shape[1] - 2, -1, -1):
                        
                        v = block[:, kk]

                        m = ~np.isfinite(last) & np.isfinite(v)

                        last[m] = v[m]

                    out_num[:, j] = last

                else:

                    out_num[:, j] = np.nanmean(block, axis = 1)

                if has_no_est:

                    out_no_est[j] = df.loc[no_est_row, qsel[-1]]

                continue

        k = int(np.argmin(np.abs((colsD - dD).astype('timedelta64[D]').astype(int))))

        out_num[:, j] = df_num[:, k]

        if has_no_est:

            out_no_est[j] = df.loc[no_est_row, cols_all[k]]

    out.loc[num_rows, :] = pd.DataFrame(out_num, index = num_rows, columns = periods).astype(float).to_numpy(dtype = object)

    if has_no_est:

        out.loc[no_est_row, :] = out_no_est

    if 'period_type' in out.index:

        out.loc['period_type', :] = 'Annual'

    if 'period_label' in out.index:

        out.loc['period_label', :] = [f'FY{d.year}' for d in periods]

    return out


def _future_to_period_aligned(
    dfT: pd.DataFrame,
    periods: pd.DatetimeIndex,
    period_types: list[str],
    fy_m: int,
    fy_d: int,
    mode: str = 'flow',
    *,
    seasonal_flow_weights_q1_q4: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Align a forecast table to an arbitrary mixed annual/quarterly period grid.

    Overview
    --------
    The valuation period grid may contain both annual and quarterly periods. Consensus forecast
    tables may:

    - provide values directly for some of these periods,
   
    - omit values for some periods, or
   
    - provide values in a different periodicity (for example, only annual when quarterly is needed).

    This helper produces an aligned table with:
   
    - columns equal to the supplied ``periods`` grid, and
   
    - a ``period_type`` row describing the intended periodicity for each column.

    Alignment steps
    ---------------
   
    1. Direct copy where available:
   
       Values are copied directly from the source table for columns that exist and match the target
       periodicity (annual or quarterly), based on either the source ``period_type`` row or inferred
       period sets.

    2. Annual anchor construction:
   
       An annual-aligned helper table annA is constructed on an anchor set containing:
   
       - the target annual periods, and
   
       - the fiscal year ends corresponding to all target quarterly periods.

       annA is built using ``_future_to_annual_aligned`` with the specified ``mode``.

    3. Fill target annual periods:
   
       Target annual columns are filled from annA.

    4. Fill quarterly periods:
   
       Quarterly periods are grouped by fiscal year end. When mode == "flow", missing quarterly
       values within a fiscal year are filled so that the quarterly sum matches the annual anchor:

           residual = AnnualTotal - sum(present_quarters)
   
           missing_quarters <- residual * weights

       where weights are uniform by default or derived from ``seasonal_flow_weights_q1_q4`` by fiscal
       quarter number. For "Std_Dev", residual variance is allocated similarly:

           residual_var = max(AnnualStdDev^2 - sum(present_q_std^2), 0)
   
           missing_q_std <- sqrt(residual_var * weights)

       For non-flow modes (ratio/stock), missing quarters are filled with the annual anchor value
       when available, otherwise with the mean of existing quarters or 0.0 as a final fallback.

    5. Analyst count propagation:
   
       When "No_of_Estimates" exists and annA provides an annual value, it is propagated to missing
       quarterly cells.

    Parameters
    ----------
    dfT:
        Source forecast table.
    periods:
        Target period grid (annual and/or quarterly ends).
    period_types:
        Period type vector aligned to periods; elements should be "annual" or "quarterly".
    fy_m, fy_d:
        Fiscal year-end month/day used to group quarters into fiscal years.
    mode:
        One of "flow", "ratio", or "stock", controlling aggregation and fill behaviour.
    seasonal_flow_weights_q1_q4:
        Optional array of 4 weights used to allocate annual flow residuals across missing quarters.

    Returns
    -------
    pandas.DataFrame
        Table aligned to ``periods`` with compatible schema for downstream simulation.
   
    """
   
    if dfT is None or dfT.empty:

        return dfT

    periods = pd.DatetimeIndex(pd.to_datetime(periods, errors = 'coerce')).dropna().normalize().sort_values().unique()

    if len(periods) == 0:

        return dfT.iloc[:, :0].copy()

    if len(period_types) != len(periods):

        raise ValueError('period_types must be same length as periods')

    src_cols = pd.to_datetime(dfT.columns, errors = 'coerce')

    ok = pd.notna(src_cols)

    dfS = dfT.loc[:, ok].copy()

    dfS.columns = pd.DatetimeIndex(src_cols[ok]).normalize()

    dfS = dfS.loc[:, ~dfS.columns.duplicated()]

    out_index = list(dfS.index)

    if 'period_type' not in out_index:

        out_index.append('period_type')

    out = pd.DataFrame(index = out_index, columns = periods, dtype = object)

    pt = [str(x).lower() for x in period_types]

    out.loc['period_type', :] = pt

    src_ann = pd.DatetimeIndex([])

    src_q = pd.DatetimeIndex([])

    if 'period_type' in dfS.index:

        spt = dfS.loc['period_type'].astype(str).str.lower()

        src_q = pd.DatetimeIndex(spt.index[spt.eq('quarterly')]).normalize().sort_values().unique()

        src_ann = pd.DatetimeIndex(spt.index[spt.eq('annual')]).normalize().sort_values().unique()

    if len(src_q) == 0 and len(src_ann) == 0:

        src_ann = pd.DatetimeIndex(_annual_periods(
            dfT = dfS,
            fy_m = fy_m
        )).sort_values().unique()

        src_q = pd.DatetimeIndex(_quarterly_periods(
            dfT = dfS,
            fy_m = fy_m
        )).sort_values().unique()

    data_rows = [r for r in dfS.index if r != 'period_type']

    for p, t in zip(periods, pt):
      
        if p not in dfS.columns:

            continue

        if t == 'annual':

            if len(src_ann) == 0 or p in src_ann or p not in src_q:

                out.loc[data_rows, p] = dfS.loc[data_rows, p].values
        elif t == 'quarterly':

            if len(src_q) == 0 or p in src_q:

                out.loc[data_rows, p] = dfS.loc[data_rows, p].values

    ann_target = pd.DatetimeIndex([p for p, t in zip(periods, pt) if t == 'annual']).sort_values().unique()

    q_periods = pd.DatetimeIndex([p for p, t in zip(periods, pt) if t == 'quarterly']).sort_values().unique()

    q_by_fy: dict[pd.Timestamp, list[pd.Timestamp]] = {}

    for q in q_periods:
        
        fy = pd.Timestamp(_fiscal_year_end_for_date(
            d = q,
            fy_m = fy_m,
            fy_d = fy_d
        )).normalize()

        q_by_fy.setdefault(fy, []).append(pd.Timestamp(q).normalize())

    ann_anchor = ann_target.union(pd.DatetimeIndex(list(q_by_fy.keys()))).sort_values().unique()

    annA = _future_to_annual_aligned(
        dfT = dfS,
        periods = ann_anchor,
        mode = mode
    ) if len(ann_anchor) else None

    if len(ann_target) and annA is not None:

        out.loc[data_rows, ann_target] = annA.loc[data_rows, ann_target].values

    if len(q_periods):

        meta_rows = {'period_type', 'period_label'}

        no_est = 'No_of_Estimates'

        num_rows = [r for r in out.index if r not in meta_rows and r != no_est]

        out_num = out.loc[num_rows].apply(pd.to_numeric, errors = 'coerce')


        def _quarter_weights(
            qs_local: list[pd.Timestamp]
        ) -> np.ndarray:
            """
            Compute quarter allocation weights for a set of quarter-end dates within a fiscal year.

            When seasonal weights are not provided, quarters receive equal weight. When seasonal
            weights are provided, the fiscal quarter number is computed for each quarter end and the
            corresponding weight is applied, with normalisation and fallback to equal weights when
            weights are invalid.

            Parameters
            ----------
            qs_local:
                List of quarter-end timestamps belonging to a single fiscal year.

            Returns
            -------
            numpy.ndarray
                Weight vector of length len(qs_local) summing to 1.
          
            """
          
            m = len(qs_local)

            if m == 0:

                return np.array([], dtype = float)

            if seasonal_flow_weights_q1_q4 is None or len(seasonal_flow_weights_q1_q4) != 4:

                return np.full(m, 1.0 / m, dtype = float)

            qnums = np.array([_fiscal_quarter_num(
                d = pd.Timestamp(q),
                M = fy_m
            ) for q in qs_local], dtype = int)

            w_raw = np.array([seasonal_flow_weights_q1_q4[q - 1] for q in qnums], dtype = float)

            w_raw = np.where(np.isfinite(w_raw) & (w_raw > 0), w_raw, 0.0)

            s = w_raw.sum()

            return np.full(m, 1.0 / m, dtype = float) if s <= e12 else w_raw / s


        if annA is not None and len(q_by_fy):

            if mode == 'flow':

                for fy_end, qs in q_by_fy.items():
                
                    fy_end = pd.Timestamp(fy_end).normalize()

                    qs = sorted(set((pd.Timestamp(q).normalize() for q in qs)))

                    if fy_end not in qs:

                        continue

                    if fy_end not in annA.columns:

                        continue

                    w = _quarter_weights(
                        qs_local = qs
                    )

                    for r in out_num.index:
                       
                        a = pd.to_numeric(annA.at[r, fy_end], errors = 'coerce') if r in annA.index else np.nan

                        q_vals = out_num.loc[r, qs].to_numpy(dtype = float, copy = True)

                        present = np.isfinite(q_vals)

                        missing = ~present

                        if not missing.any():

                            continue

                        w_m = w[missing]

                        sw = w_m.sum()

                        w_m = np.full(missing.sum(), 1.0 / missing.sum(), dtype = float) if sw <= e12 else w_m / sw

                        if np.isfinite(a):

                            if r == 'Std_Dev':

                                target_var = a * a

                                present_var = np.nansum(q_vals[present] ** 2) if present.any() else 0.0

                                resid_var = max(target_var - present_var, 0.0)

                                q_vals[missing] = np.sqrt(resid_var * w_m)

                            else:

                                resid = a - np.nansum(q_vals[present]) if present.any() else a

                                q_vals[missing] = resid * w_m

                        else:

                            if present.any():

                                fill = float(np.nanmean(q_vals[present]))

                            else:

                                fill = 0.0

                            q_vals[missing] = fill

                        out_num.loc[r, qs] = q_vals

                out.loc[num_rows, :] = out_num.values

            else:

                for fy_end, qs in q_by_fy.items():
                 
                    fy_end = pd.Timestamp(fy_end).normalize()

                    qs = sorted(set((pd.Timestamp(q).normalize() for q in qs)))

                    if fy_end not in qs:

                        continue

                    if fy_end not in annA.columns:

                        continue

                    vA = annA.loc[num_rows, fy_end].apply(pd.to_numeric, errors = 'coerce').to_numpy(dtype = float, copy = False)

                    fallback = out_num.loc[:, qs].apply(pd.to_numeric, errors = 'coerce')

                    fb = fallback.mean(axis = 1, skipna = True).to_numpy(dtype = float, copy = False)

                    for q in qs:
                 
                        vq = out_num[q].to_numpy(dtype = float, copy = False)

                        need = ~np.isfinite(vq)

                        fill_val = np.where(np.isfinite(vA), vA, np.where(np.isfinite(fb), fb, 0.0))

                        if np.any(need):

                            out_num.loc[:, q] = np.where(need, fill_val, vq)

                out.loc[num_rows, :] = out_num.values

            if no_est in out.index and annA is not None and (no_est in annA.index):

                for fy_end, qs in q_by_fy.items():
                 
                    fy_end = pd.Timestamp(fy_end).normalize()

                    qs = sorted(set((pd.Timestamp(q).normalize() for q in qs)))

                    if fy_end not in qs:

                        continue

                    if fy_end not in annA.columns:

                        continue

                    for q in qs:
                    
                        if pd.isna(out.at[no_est, q]):

                            out.at[no_est, q] = annA.at[no_est, fy_end]

    return out


def _fit_skewt_innovations(
    x: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Calibrate a location-scale skew-t innovation model to a one-dimensional sample.

    The skew-t innovation family used by this model is parameterised by:

    - loc:   location (median-like centre),
    
    - sc:    scale (MAD-like dispersion),
    
    - delta: skew-normal delta controlling asymmetry, and
    
    - nu:    degrees of freedom controlling tail thickness.

    Estimation method
    -----------------
    
    - Location and scale are estimated robustly using ``_robust_loc_scale`` (median and MAD).
    
    - Sample skewness and excess kurtosis are estimated by moments using ``_sample_skew_exkurt``.
    
    - delta is inferred from the target skewness using ``_delta_from_target_skew_skewnormal``.
    
    - nu is inferred from excess kurtosis using ``_nu_from_excess_kurt``.

    This calibration is used primarily for simulating random-walk increments from historical
    level series, providing a cheap but more realistic alternative to Gaussian increments.

    Parameters
    ----------
    x:
        Innovation sample (for example, first differences of a level series).

    Returns
    -------
    (float, float, float, float)
        Tuple ``(loc, sc, delta, nu)``. Returns a near-degenerate parameter set when insufficient
        sample points exist.
   
    """
   
    x = np.asarray(x, float)

    x = x[np.isfinite(x)]

    if x.size < MIN_POINTS:

        return (0.0, 0.0, 0.0, 30.0)

    loc, sc = _robust_loc_scale(
        x = x
    )

    sc = max(sc, e12)

    skew, exk = _sample_skew_exkurt(
        x = x
    )

    delta = _delta_from_target_skew_skewnormal(
        target_skew = skew
    )

    nu = _nu_from_excess_kurt(
        exk = exk
    )

    return (loc, sc, delta, nu)


def _simulate_rw_from_history_levels(
    hist: pd.Series,
    *,
    T: int,
    floor_at_zero: bool = False,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate a level path using a random walk calibrated from historical levels.

    Model
    -----
    Let the historical level series be x_t. Define increments:

        d_t = x_t - x_{t-1}

    The future increments are simulated from a calibrated skew-t distribution (via
    ``_fit_skewt_innovations`` and ``_draw_skewt``), then integrated forward:

        x_{t+1} = x_t + eps_{t+1}

    where eps_{t+1} are i.i.d. skew-t draws with location/scale/skew/tail parameters fitted to the
    historical increment sample.

    Scaling guardrail
    -----------------
    The increment scale is floored relative to the last observed level to avoid an unrealistically
    tight random walk when historical increments are near-zero but future uncertainty should not
    collapse completely.

    Parameters
    ----------
    hist:
        Historical level series.
    T:
        Number of simulated future periods.
    floor_at_zero:
        Whether to floor simulated levels at zero.
    rng:
        Random generator.

    Returns
    -------
    numpy.ndarray
        Level draw matrix of shape (T, N_SIMS).
  
    """
  
    x = pd.to_numeric(hist, errors = 'coerce').dropna().to_numpy(dtype = float)

    if len(x) < 2:

        base = x[-1] if len(x) else 0.0

        out = np.full((T, N_SIMS), base, dtype = float)

        return np.maximum(out, 0.0) if floor_at_zero else out

    last = x[-1]

    d = np.diff(x)

    loc, sc, delta, nu = _fit_skewt_innovations(
        x = d
    )

    sc = max(sc, 1e-09 * (abs(last) + 1.0), e12)

    eps = _draw_skewt(
        loc = loc,
        scale = sc,
        delta = delta,
        nu = nu,
        size = (T, N_SIMS),
        rng = rng
    )

    out = last + np.cumsum(eps, axis = 0)

    return np.maximum(out, 0.0) if floor_at_zero else out


def _pick_fy_points(
    idx: pd.DatetimeIndex,
    fy_d,
    fy_m,
    window_days: int = 21
) -> pd.DatetimeIndex:
    """
    Select representative fiscal year-end timestamps from an arbitrary datetime index.

    Historical statement tables are often reported at quarterly frequency or irregularly. Many
    modelling components require an annualised panel at fiscal year ends. This helper selects, for
    each calendar year present in the index, the observation closest to the fiscal year-end date.

    Selection rule for each year y
    ------------------------------
   
    1. Construct the fiscal year-end target date:

           target = Timestamp(y, fy_m, min(fy_d, month_end_day(y, fy_m)))

    2. Search for index points within +/- window_days of target and select the closest.
   
    3. If none exist, fall back to the last observation in month fy_m for that year, when present.

    Parameters
    ----------
    idx:
        Datetime index of available observation dates.
    fy_d, fy_m:
        Fiscal year-end day and month.
    window_days:
        Search half-window around the fiscal year-end target.

    Returns
    -------
    pandas.DatetimeIndex
        Unique sorted set of selected fiscal-year representative timestamps.
  
    """
  
    idx = pd.DatetimeIndex(pd.to_datetime(idx, errors = 'coerce')).dropna().sort_values()

    if len(idx) == 0:

        return idx

    picked = []

    for y in sorted(set(idx.year)):
     
        last_day = calendar.monthrange(y, fy_m)[1]

        d_target = min(fy_d, last_day)

        target = pd.Timestamp(y, fy_m, d_target)

        lo = target - pd.Timedelta(days = window_days)

        hi = target + pd.Timedelta(days = window_days)

        cand = idx[(idx >= lo) & (idx <= hi)]

        if len(cand) > 0:

            best = cand[np.argmin(np.abs((cand - target).days))]

            picked.append(best)

        else:

            cand2 = idx[(idx.year == y) & (idx.month == fy_m)]

            if len(cand2) > 0:

                picked.append(cand2.max())

    return pd.DatetimeIndex(sorted(set(picked)))


def _extract_hist_annual_series(
    hist_inc: pd.DataFrame | None,
    hist_cf: pd.DataFrame | None,
    hist_bal: pd.DataFrame | None,
    hist_ratios: pd.DataFrame | None,
    row_candidates: list[tuple[str, str]],
    fy_m: int,
    fy_d: int,
    cashflow_is_ttm: bool = True
) -> pd.Series | None:
    """
    Extract a historical annual series for a given driver using multiple candidate sheet/row sources.

    Inputs
    ------
    Historical data may be sourced from several statement tables:
  
    - income statement (hist_inc),
  
    - cash flow statement (hist_cf),
  
    - balance sheet (hist_bal), and
  
    - ratios (hist_ratios).

    Additionally, a special "nd" pseudo-sheet is supported to derive net debt from the balance
    sheet using ``_hist_net_debt_debt_minus_cash``.

    Annualisation
    -------------
    When the source series is quarterly and represents a flow (income statement and cash flow),
    annual totals are obtained as either:

    - trailing twelve months (TTM) when cashflow_is_ttm is True (treating the series as already
      annualised), or
   
    - a 4-quarter rolling sum when cashflow_is_ttm is False.

    Stock variables (balance sheet and ratios) are treated as levels and not summed.

    Fiscal year-end selection
    -------------------------
    The extracted series is reduced to a fiscal-year series by selecting one observation per year
    using ``_pick_fy_points`` with (fy_m, fy_d).

    Parameters
    ----------
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical tables.
    row_candidates:
        Ordered list of (sheet_key, row_label) candidates.
        sheet_key must be one of {"inc", "cf", "bal", "rat", "nd"}.
    fy_m, fy_d:
        Fiscal year-end month and day.
    cashflow_is_ttm:
        Whether income/cash-flow series are already TTM totals.

    Returns
    -------
    pandas.Series | None
        Annualised fiscal-year series, or None when no candidate can be extracted.
 
    """
 
    for sheet, row in row_candidates:
 
        if sheet == 'nd':

            if hist_bal is None:

                continue

            s = _hist_net_debt_debt_minus_cash(
                hist_bal = hist_bal
            )

            if s is None or s.empty:

                continue

            s.index = pd.to_datetime(s.index)

            s = s.sort_index()

            fy_idx = _pick_fy_points(
                idx = pd.DatetimeIndex(s.index),
                fy_d = fy_d,
                fy_m = fy_m
            )

            out = s.reindex(fy_idx).dropna()

            return out.dropna() if len(out.dropna()) else None

        df = {'inc': hist_inc, 'cf': hist_cf, 'bal': hist_bal, 'rat': hist_ratios}.get(sheet)

        if df is None or df.empty or row not in df.index:

            continue

        s = _as_numeric_series(
            s = df.loc[row]
        )

        s.index = pd.to_datetime(s.index, errors = 'coerce')

        s = s.dropna()

        if s.empty:

            continue

        s = s.sort_index()

        if sheet in {'inc', 'cf'}:

            s_a = s if cashflow_is_ttm else s.rolling(4, min_periods = 4).sum()

        else:

            s_a = s

        fy_idx = _pick_fy_points(
            idx = pd.DatetimeIndex(s_a.index),
            fy_d = fy_d,
            fy_m = fy_m
        )

        out = s_a.reindex(fy_idx).dropna()

        if len(out):

            return out

    return None


def _build_hist_panel(
    *,
    hist_inc: pd.DataFrame | None,
    hist_cf: pd.DataFrame | None,
    hist_bal: pd.DataFrame | None,
    hist_ratios: pd.DataFrame | None,
    fy_m: int,
    fy_d: int,
    history_spec: dict[str, list[tuple[str, str]]] | None = None
) -> pd.DataFrame:
    """
    Build a historical annual panel of core drivers aligned to fiscal year ends.

    The panel is constructed by extracting individual annual series for each driver key specified by
    ``history_spec`` (or the default ``_HIST_PANEL_KEY_CANDIDATES``) using
    ``_extract_hist_annual_series``. The resulting series are concatenated into a single DataFrame,
    indexed by fiscal year-end timestamps, and forward-filled to reduce sparse coverage.

    The panel is used for:
   
    - coherence checks (interest vs debt, DNWC vs revenue),
   
    - dependence estimation (rank correlation on innovations),
   
    - imputation priors and predictor selection.

    Parameters
    ----------
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical statement tables.
    fy_m, fy_d:
        Fiscal year-end month/day.
    history_spec:
        Optional mapping ``driver_key -> list[(sheet_key, row_label)]``.

    Returns
    -------
    pandas.DataFrame
        Historical panel indexed by fiscal year ends. Returns an empty DataFrame when no series can
        be extracted.
   
    """
   
    if all((df is None for df in (hist_inc, hist_cf, hist_bal, hist_ratios))):

        return pd.DataFrame()

    spec = history_spec if history_spec is not None else _HIST_PANEL_KEY_CANDIDATES

    cols: dict[str, pd.Series] = {}

    for key, candidates in spec.items():
      
        s = _extract_hist_annual_series(
            hist_inc = hist_inc,
            hist_cf = hist_cf,
            hist_bal = hist_bal,
            hist_ratios = hist_ratios,
            row_candidates = candidates,
            fy_m = fy_m,
            fy_d = fy_d,
            cashflow_is_ttm = True
        )

        if s is None or s.empty:

            continue

        s = pd.to_numeric(s, errors = 'coerce')

        s.index = pd.to_datetime(s.index, errors = 'coerce')

        s = s.dropna().sort_index()

        if not s.empty:

            cols[key] = s

    if not cols:

        return pd.DataFrame()

    df = pd.concat(cols, axis = 1).sort_index()

    df.index = pd.DatetimeIndex(df.index).normalize()

    return df.ffill().fillna(0.0)


def _infer_quarterly_flow_from_ttm(
    ttm: pd.Series
) -> pd.Series:
    """
    Approximate a quarterly flow series from a trailing-twelve-months (TTM) series.

    Relation
    --------
    For a quarterly flow q_t, the trailing twelve months total is:

        TTM_t = q_t + q_{t-1} + q_{t-2} + q_{t-3}

    This implies the recursion:

        q_t = TTM_t - TTM_{t-1} + q_{t-4}

    The first four quarters are seeded using a simple allocation:

        q_0..q_3 = TTM_3 / 4

    This is a pragmatic inversion used for seasonality detection when only TTM quarterly statement
    series are available.

    Parameters
    ----------
    ttm:
        TTM series indexed by quarter-end dates.

    Returns
    -------
    pandas.Series
        Approximate quarterly flow series aligned to the input index, or an empty series when
        insufficient data exist.
   
    """
   
    s = pd.to_numeric(ttm, errors = 'coerce').dropna().sort_index()

    if len(s) < 8:

        return pd.Series(dtype = float)

    s = s.copy()

    idx = pd.DatetimeIndex(s.index).sort_values()

    s = s.reindex(idx)

    q = pd.Series(index = idx, dtype = float)

    seed = s.iloc[3] / 4.0 if np.isfinite(s.iloc[3]) else 0.0

    q.iloc[0:4] = seed

    for i in range(4, len(s)):
    
        q.iloc[i] = s.iloc[i] - s.iloc[i - 1] + q.iloc[i - 4]

    return q


def _detect_capex_seasonality_weights_from_ttm_quarters(
    *,
    hist_cf_q_ttm: pd.DataFrame | None,
    hist_inc_q_ttm: pd.DataFrame | None,
    fy_m: int,
    min_years: int = 3
) -> np.ndarray | None:
    """
    Detect CapEx seasonality weights across fiscal quarters using quarterly TTM statement data.

    Objective
    ---------
    When quarterly flow values are missing but annual totals are available, quarterly flows may be
    allocated uniformly. For CapEx, uniform allocation can be unrealistic due to seasonal investment
    patterns. This function attempts to infer a stable quarter weight vector w[Q1..Q4] from history.

    Method
    ------
    1. Extract quarterly TTM series for CapEx and Revenue from statement tables.
   
    2. Convert TTM series into approximate quarterly flows using ``_infer_quarterly_flow_from_ttm``.
   
    3. Compute a scale-free ratio series:

           r_t = abs(CapEx_q_t) / abs(Revenue_q_t)

    4. Group r_t by fiscal quarter number and compute per-quarter medians and within-quarter MAD.
   
    5. Reject seasonality detection when:
   
       - insufficient data exist, or
   
       - between-quarter spread is not materially larger than within-quarter noise.
    
    6. Convert per-quarter medians into weights:

           w_q = median_q / sum(median_q)

       then clip to [0.05, 0.70] and re-normalise.

    The resulting weights can be used to allocate annual residual CapEx across missing quarters in
    ``_future_to_period_aligned`` and related imputation steps.

    Parameters
    ----------
    hist_cf_q_ttm:
        Quarterly cash flow statement table with TTM-like series.
    hist_inc_q_ttm:
        Quarterly income statement table with TTM-like series (used for revenue).
    fy_m:
        Fiscal year-end month.
    min_years:
        Minimum years of quarterly history required.

    Returns
    -------
    numpy.ndarray | None
        Array of length 4 of quarter weights (Q1..Q4) summing to 1, or None when seasonality cannot
        be inferred robustly.
   
    """
   
    if hist_cf_q_ttm is None or hist_cf_q_ttm.empty:

        return None

    if hist_inc_q_ttm is None or hist_inc_q_ttm.empty:

        return None

    cap_row = _first_existing_row(
        df = hist_cf_q_ttm,
        candidates = ('Capital Expenditure', 'Capital Expenditures')
    )

    rev_row = _first_existing_row(
        df = hist_inc_q_ttm,
        candidates = ('Revenue', 'Total Revenue')
    )

    if cap_row is None or rev_row is None:

        return None

    cap_ttm = _as_numeric_series(
        s = hist_cf_q_ttm.loc[cap_row]
    ).dropna()

    rev_ttm = _as_numeric_series(
        s = hist_inc_q_ttm.loc[rev_row]
    ).dropna()

    cap_ttm.index = pd.to_datetime(cap_ttm.index, errors = 'coerce')

    rev_ttm.index = pd.to_datetime(rev_ttm.index, errors = 'coerce')

    cap_ttm = cap_ttm.dropna().sort_index()

    rev_ttm = rev_ttm.dropna().sort_index()

    idx = cap_ttm.index.intersection(rev_ttm.index)

    cap_ttm = cap_ttm.reindex(idx)

    rev_ttm = rev_ttm.reindex(idx)

    if len(idx) < 4 * min_years:

        return None

    cap_q = _infer_quarterly_flow_from_ttm(
        ttm = cap_ttm
    )

    rev_q = _infer_quarterly_flow_from_ttm(
        ttm = rev_ttm
    )

    idx2 = cap_q.index.intersection(rev_q.index)

    cap_q = cap_q.reindex(idx2)

    rev_q = rev_q.reindex(idx2)

    m = np.isfinite(cap_q.to_numpy()) & np.isfinite(rev_q.to_numpy()) & (np.abs(rev_q.to_numpy()) > e12)

    if m.sum() < 4 * min_years:

        return None

    cap_q = cap_q.iloc[np.where(m)[0]]

    rev_q = rev_q.iloc[np.where(m)[0]]

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ratio = (cap_q.abs() / np.maximum(rev_q.abs(), e12)).replace([np.inf, -np.inf], np.nan).dropna()

    if len(ratio) < 4 * min_years:

        return None

    fq = np.array([_fiscal_quarter_num(
        d = d,
        M = fy_m
    ) for d in ratio.index], dtype = int)

    meds = []

    within = []

    for q in (1, 2, 3, 4):
        r = ratio.iloc[np.where(fq == q)[0]].to_numpy(dtype = float)

        r = r[np.isfinite(r)]

        if r.size < min_years * 2:

            return None

        med = np.median(r)

        mad = 1.4826 * np.median(np.abs(r - med)) if r.size >= 2 else 0.0

        meds.append(max(med, 0.0))

        within.append(max(mad, e6))

    meds = np.asarray(meds, float)

    within = np.asarray(within, float)

    if not np.isfinite(meds).all() or meds.sum() <= e12:

        return None

    spread = np.max(meds) - np.min(meds)

    noise = np.median(within)

    if spread < 2.5 * noise:

        return None

    w = meds / meds.sum()

    w = np.clip(w, 0.05, 0.7)

    w = w / w.sum()

    return w


def _fit_dnwc_model(
    hist_dnwc: pd.Series,
    hist_rev: pd.Series
):
    """
    Fit a simple model for DNWC (change in net working capital) conditional on revenue activity.

    Two model forms are supported:

    1. Ratio model (small-sample fallback):

           DNWC_t ~= Revenue_t * eps_t

       where eps_t is a heavy-tailed ratio draw calibrated from historical DNWC/Revenue.

    2. Linear activity model:

           DNWC_t = a + b1 * Revenue_t + b2 * dRevenue_t + eps_t

       where dRevenue_t is the first difference of revenue and eps_t is a heavy-tailed residual.

    The linear model captures both scale (b1 * Revenue) and activity (b2 * change in revenue)
    effects, which is often a better approximation for working capital dynamics than a pure ratio.

    Parameters
    ----------
    hist_dnwc:
        Historical DNWC series.
    hist_rev:
        Historical revenue series aligned to the same annual grid.

    Returns
    -------
    object
        Model object suitable for ``_simulate_dnwc``. Either:
        - ("ratio", loc, sc, nu), or
        - (coef, loc, sc, nu) where coef = [a, b1, b2].
   
    """
   
    dnwc = hist_dnwc.dropna()

    rev = hist_rev.dropna()

    idx = dnwc.index.intersection(rev.index)

    dnwc = dnwc.reindex(idx)

    rev = rev.reindex(idx)

    if len(idx) < 8:

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
         
            ratio = (dnwc / rev).replace([np.inf, -np.inf], np.nan).dropna()

        loc, sc = _robust_loc_scale(
            x = ratio.to_numpy()
        )

        return ('ratio', loc, sc, 8.0)

    revv = rev.to_numpy(dtype = float)

    d_rev = np.concatenate([[np.nan], np.diff(revv)])

    m = np.isfinite(dnwc.to_numpy()) & np.isfinite(revv) & np.isfinite(d_rev)

    if m.sum() < 8:

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
         
            ratio = (dnwc / rev).replace([np.inf, -np.inf], np.nan).dropna()

        loc, sc = _robust_loc_scale(
            x = ratio.to_numpy()
        )

        return ('ratio', loc, sc, 8.0)

    y = dnwc.to_numpy(dtype = float)[m]

    X = np.column_stack([np.ones(m.sum()), revv[m], d_rev[m]])

    coef, *_ = np.linalg.lstsq(X, y, rcond = None)

    resid = y - X @ coef

    loc, sc = _robust_loc_scale(
        x = resid
    )

    nu = np.clip(m.sum() - 3, 5, 30)

    return (coef, loc, sc, nu)


def _simulate_dnwc(
    model,
    revenue_draws: np.ndarray,
    *,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate DNWC draws from a fitted DNWC model and simulated revenue draws.

    For the ratio model:

        eps ~ t_nu(loc=loc, scale=sc)
    
        DNWC = Revenue * eps

    For the linear activity model:

        dRevenue_t = Revenue_t - Revenue_{t-1}
     
        eps ~ t_nu(loc=loc, scale=sc)
     
        DNWC = a + b1 * Revenue + b2 * dRevenue + eps

    Revenue draw matrices are sanitised for non-finite entries by forward filling along time within
    each simulation column and using 0.0 as an initial fallback.

    Parameters
    ----------
    model:
        Model object returned by ``_fit_dnwc_model``.
    revenue_draws:
        Revenue draw matrix of shape (T, n_sims).
    rng:
        Random generator.

    Returns
    -------
    numpy.ndarray
        DNWC draw matrix of shape (T, n_sims).
   
    """
  
    T, n = revenue_draws.shape

    if isinstance(model, (tuple, list)) and len(model) > 0 and isinstance(model[0], str) and (model[0] == 'ratio'):

        _, loc, sc, nu = model

        eps = rng.standard_t(df = nu, size = (T, n)) * sc + loc

        return revenue_draws * eps

    coef, loc, sc, nu = model

    rev = np.asarray(revenue_draws, dtype = float).copy()

    if np.any(~np.isfinite(rev)):

        for t in range(1, rev.shape[0]):
        
            bad = ~np.isfinite(rev[t])

            if np.any(bad):

                rev[t, bad] = rev[t - 1, bad]

        bad0 = ~np.isfinite(rev[0])

        if np.any(bad0):

            rev[0, bad0] = 0.0

    d_rev = np.vstack([np.full((1, n), np.nan), np.diff(rev, axis = 0)])

    d_rev[0, :] = 0.0

    eps = rng.standard_t(df = nu, size = (T, n)) * sc + loc

    a, b1, b2 = coef

    return a + b1 * rev + b2 * d_rev + eps


def _gross_debt_from_balance(
    hist_bal: pd.DataFrame
) -> float | None:
    """
    Extract a gross debt proxy from a historical balance-sheet table.

    The function searches for the first available debt row label from the configured ``_DEBT_ROWS``
    candidates. The most recent numeric value is returned in absolute value, reflecting that balance
    sheet conventions may record liabilities as negative numbers.

    Parameters
    ----------
    hist_bal:
        Historical balance-sheet table.

    Returns
    -------
    float | None
        Gross debt proxy (unscaled), or None when a debt row cannot be extracted.
    """
  
    row = _first_existing_row(
        df = hist_bal,
        candidates = _DEBT_ROWS
    )

    if row is None:

        return None

    s = _as_numeric_series(
        s = hist_bal.loc[row]
    ).dropna()

    if s.empty:

        return None

    val = s.iloc[-1]

    return abs(val)


def _align_draws_to_periods(
    draws: np.ndarray,
    src_cols: pd.DatetimeIndex,
    tgt_cols: pd.DatetimeIndex
) -> np.ndarray:
    """
    Align a draw matrix from a source period grid to a target grid using nearest-neighbour mapping.

    The function assumes draws are indexed by time along axis 0. For each target period date, the
    closest source period date (by absolute day difference) is selected and the corresponding draw
    row is copied.

    This is used as a pragmatic alignment step when:
 
    - the forecast table and the valuation grid do not match exactly, and
 
    - interpolation is not appropriate (for example, for discrete period-end forecasts).

    Parameters
    ----------
    draws:
        Draw matrix of shape (T_src, n_sims).
    src_cols:
        Source period end timestamps aligned to draws rows.
    tgt_cols:
        Target period end timestamps.

    Returns
    -------
    numpy.ndarray
        Draw matrix of shape (T_tgt, n_sims) aligned to the target periods.

    """

    draws = np.asarray(draws, dtype = float)

    src = pd.DatetimeIndex(pd.to_datetime(src_cols, errors = 'coerce')).normalize()

    tgt = pd.DatetimeIndex(pd.to_datetime(tgt_cols, errors = 'coerce')).normalize()

    src = src[pd.notna(src)]

    tgt = tgt[pd.notna(tgt)]

    if len(src) == 0 or len(tgt) == 0:

        return draws

    order = np.argsort(src.values)

    src = src[order]

    draws_s = draws[order, :]

    srcD = src.values.astype('datetime64[D]')

    tgtD = tgt.values.astype('datetime64[D]')

    pos = np.searchsorted(srcD, tgtD, side = 'left')

    pos = np.clip(pos, 0, len(srcD) - 1)

    left = np.clip(pos - 1, 0, len(srcD) - 1)

    right = pos

    d_left = np.abs((tgtD - srcD[left]).astype('timedelta64[D]').astype(int))

    d_right = np.abs((tgtD - srcD[right]).astype('timedelta64[D]').astype(int))

    idx = np.where(d_left <= d_right, left, right).astype(int)

    return draws_s[idx, :]


def _nearest_leq_index(
    periods: pd.DatetimeIndex,
    dt: pd.Timestamp
) -> int:
    """
    Find the index of the last period end less than or equal to a given timestamp.

    When no period end is <= dt, the index of the closest period end by absolute day difference is
    returned. This behaviour supports extension logic where a "native last" date may precede the
    grid or may be missing.

    Parameters
    ----------
    periods:
        Period grid.
    dt:
        Target date.

    Returns
    -------
    int
        Index into periods.
 
    """
 
    periods = pd.DatetimeIndex(pd.to_datetime(periods, errors = 'coerce')).normalize().sort_values()

    dt = pd.Timestamp(dt).normalize()

    leq = np.where(periods <= dt)[0]

    if leq.size:

        return leq[-1]

    return np.argmin(np.abs((periods - dt).days))


def _robust_time_mu_sd(
    M: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute robust per-simulation location and scale across a time axis.

    Given a matrix M of shape (T, n_sims), the function returns:

        mu_j = median_t M[t, j]
      
        sd_j = 1.4826 * median_t |M[t, j] - mu_j|

    for each simulation column j. The scale is floored at ``e6``.

    These statistics are used when extending margins or ratios beyond the native forecast horizon,
    allowing each simulation path to mean revert towards its own robust centre with a robust
    dispersion estimate.

    Parameters
    ----------
    M:
        Time-by-simulation matrix.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple (mu, sd) of length n_sims arrays.
   
    """
   
    M = np.asarray(M, dtype = float)

    mu = np.nanmedian(M, axis = 0)

    mad = 1.4826 * np.nanmedian(np.abs(M - mu[None, :]), axis = 0)

    mad = np.where(np.isfinite(mad), mad, 0.0)

    sd = np.maximum(mad, e6)

    return (mu, sd)


def _mean_revert_extend(
    *,
    last: np.ndarray,
    target: np.ndarray,
    sd: np.ndarray,
    T_ext: int,
    rng: np.random.Generator,
    phi: float,
    noise_scale: float,
    df_t: float
) -> np.ndarray:
    """
    Extend a per-simulation level vector forward using a mean-reverting AR(1) process with t noise.

    Model
    -----
    For each simulation column j:

        x_t(j) = target(j) + phi * (x_{t-1}(j) - target(j)) + eps_t(j)

    where eps_t(j) are i.i.d. Student-t innovations:

        eps_t(j) ~ t_df_t(scale = noise_scale * sd(j))

    The function returns the extension block (without the initial last vector).

    Parameters
    ----------
    last:
        Last observed level vector of shape (n_sims,).
    target:
        Mean reversion target vector of shape (n_sims,).
    sd:
        Scale vector of shape (n_sims,).
    T_ext:
        Number of extension steps.
    rng:
        Random generator.
    phi:
        AR(1) persistence parameter in [0, 1).
    noise_scale:
        Multiplier applied to sd.
    df_t:
        Student-t degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Extension matrix of shape (T_ext, n_sims).
   
    """
   
    last = np.asarray(last, float)

    target = np.asarray(target, float)

    sd = np.asarray(sd, float)

    out = np.empty((T_ext, last.size), dtype = float)

    x = last.copy()

    eps = rng.standard_t(df = df_t, size = (T_ext, last.size)) * (noise_scale * sd[None, :])

    for t in range(T_ext):
       
        x = target + phi * (x - target) + eps[t]

        out[t, :] = x

    return out


def _native_end_idx(
    key: str,
    native_last,
    periods
) -> int:
    """
    Convert a native-last timestamp for a given key into an index on the valuation period grid.

    The ``native_last`` mapping records, for each driver key, the last period end date for which
    the forecast was "native" (directly observed or reliably aligned) rather than extended or
    imputed. This helper converts that timestamp into an index into the supplied ``periods`` grid,
    selecting the last grid element less than or equal to the timestamp.

    Parameters
    ----------
    key:
        Driver key.
    native_last:
        Mapping ``driver_key -> pandas.Timestamp`` (or None/NaN).
    periods:
        Period grid.

    Returns
    -------
    int
        Index into periods, or -1 when the native last date is missing.
  
    """
  
    dt = native_last.get(key, None)

    if dt is None or (not isinstance(dt, pd.Timestamp) and pd.isna(dt)):

        return -1

    return _nearest_leq_index(
        periods = periods,
        dt = pd.Timestamp(dt)
    )


def _extend_operating_profit_with_margins(
    *,
    sim: dict[str, np.ndarray],
    periods: pd.DatetimeIndex,
    native_last: dict[str, pd.Timestamp | None],
    hist_annual: pd.DataFrame | None,
    phi: float | None = None,
    noise_scale: float | None = None,
    df_t: float | None = None,
    min_margin: float = -0.5,
    max_margin: float = 0.8,
    rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    Extend EBIT and EBITDA beyond the native forecast horizon using margin dynamics and guardrails.

    Motivation
    ----------
    Forecast tables frequently contain EBIT and/or EBITDA only for a limited horizon. Downstream
    cash-flow formulas (FCFF and FCFE variants) may require operating profit across the full valuation
    period grid, especially when a terminal value is computed at the horizon. Naively holding the
    last forecast constant can produce implausible long-run behaviour and can violate simple
    accounting constraints (for example, EBIT exceeding gross profit).

    This helper extends operating profit paths in a way that is:
   
    - anchored to simulated revenue (required input),
   
    - optionally constrained by gross margin (when available), and
   
    - calibrated using historical annual ratios when available.

    Definitions
    -----------
    For each period t and simulation column j:

        EBIT_margin_t(j)   = EBIT_t(j) / Revenue_t(j)
    
        EBITDA_margin_t(j) = EBITDA_t(j) / Revenue_t(j)

    When gross margin GM_t(j) is available (in decimals), a derived operating expense ratio can be
    defined as:

        OPEX_ratio_t(j) = GM_t(j) - EBIT_margin_t(j)

    since gross profit margin minus operating profit margin approximates operating expenses as a
    fraction of revenue (subject to classification differences).

    Native horizon
    --------------
    The ``native_last`` mapping indicates, for each key, the last period end for which values are
    considered native. The extension is applied only to periods strictly after:

        end_ebit   = _native_end_idx("ebit", native_last, periods)
      
        end_ebitda = _native_end_idx("ebitda", native_last, periods)

    Extension model
    ---------------
    The extension uses a mean-reverting AR(1) process with heavy-tailed innovations implemented by
    ``_mean_revert_extend``:

        x_t = target + phi * (x_{t-1} - target) + eps_t
      
        eps_t ~ Student-t(df_t) scaled by noise_scale * sd

    Two extension paths are used for EBIT:

    1. Gross-margin constrained path (preferred when GM exists and native points exist):
      
       - Compute historical EBIT margins and derived OPEX ratios on the native segment.
      
       - Estimate per-simulation robust targets and scales using ``_robust_time_mu_sd``.
      
       - Extend OPEX_ratio forward using mean reversion, then compute:

             EBIT_margin_out = GM_out - OPEX_ratio_ext

       - Enforce:
      
         - EBIT_margin_out <= GM_out (cannot exceed gross profit margin),
      
         - EBIT_margin_out clipped to [min_margin, max_margin].
      
       - Set EBIT_out = Revenue_out * EBIT_margin_out.

       This construction helps maintain a coherent decomposition between gross margin, operating
       expenses, and operating profit.

    2. Margin-only path (used when GM is unavailable):
      
       - Compute native EBIT margins when available, otherwise use historical medians or defaults.
      
       - Extend EBIT_margin forward using mean reversion and clip to [min_margin, max_margin].
      
       - When GM exists, cap EBIT_margin_out to GM_out.
      
       - Compute EBIT_out = Revenue_out * EBIT_margin_out.

    EBITDA extension:
    -----------------
    - When EBIT and DA are available, EBITDA is extended deterministically as:

          EBITDA_out = EBIT_out + DA_out

    - Otherwise, EBITDA margin is extended using the same mean reversion mechanism applied to a
      native EBITDA margin history (or historical/default anchors), then:

          EBITDA_out = Revenue_out * EBITDA_margin_out

    Parameter calibration
    ---------------------
    The AR(1) parameters (phi, noise_scale, df_t) may be supplied explicitly. When not supplied,
    they are estimated from historical annual data using ``_estimate_ar1_params_from_levels``:

    - For OPEX_ratio (when revenue, EBIT, and gross margin history exist).
    
    - For EBITDA_margin (when revenue and EBITDA history exist).

    Parameters are clipped to conservative ranges to avoid near-unit-root behaviour and excessively
    volatile extensions.

    Advantages
    ----------
   
    - Provides smooth, mean-reverting long-run behaviour rather than hard extrapolation.
   
    - Uses robust statistics (median/MAD) and heavy-tailed innovations to reduce sensitivity to
      outliers while retaining realistic tail risk.
   
    - Respects gross margin as an upper bound on operating profit margins when available.
   
    - Preserves internal consistency between EBIT, EBITDA, and depreciation when possible.

    Parameters
    ----------
    sim:
        Simulation mapping containing at least "revenue" and optionally "ebit", "ebitda",
        "gross_margin", and "da".
    periods:
        Valuation period grid aligned to the draw matrices.
    native_last:
        Mapping indicating the last native forecast period per key.
    hist_annual:
        Historical annual panel used to estimate margin dynamics.
    phi, noise_scale, df_t:
        Optional overrides for the mean reversion dynamics.
    min_margin, max_margin:
        Bounds applied to extended margins for stability and plausibility.
    rng:
        Random generator used for extension innovations.

    Returns
    -------
    dict[str, numpy.ndarray]
        Updated simulation mapping containing extended "ebit" and "ebitda" arrays.
   
    """
   
    if 'revenue' not in sim:

        return sim

    periods = pd.DatetimeIndex(pd.to_datetime(periods, errors = 'coerce')).normalize().sort_values()

    rev = np.asarray(sim['revenue'], float)

    T, n = rev.shape

    rev_safe = np.maximum(np.abs(rev), e12)

    gm = sim.get('gross_margin', None)

    if gm is not None:

        gm = np.asarray(gm, float)

        gm = _pct_to_dec_if_needed(
            x = gm
        )

        gm = np.clip(gm, 0.0, 1.0)

    da = sim.get('da', None)

    if da is not None:

        da = np.asarray(da, float)

        da = np.maximum(da, 0.0)

    hist_opex_mu = None

    hist_opex_sd = None

    if hist_annual is not None and (not hist_annual.empty):

        if all((c in hist_annual.columns for c in ('revenue', 'ebit', 'gross_margin'))):

            h_rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').to_numpy(dtype = float)

            h_ebit = pd.to_numeric(hist_annual['ebit'], errors = 'coerce').to_numpy(dtype = float)

            h_gm = pd.to_numeric(hist_annual['gross_margin'], errors = 'coerce').to_numpy(dtype = float)

            h_gm = _pct_to_dec_if_needed(
                x = h_gm
            )

            m = np.isfinite(h_rev) & np.isfinite(h_ebit) & np.isfinite(h_gm) & (np.abs(h_rev) > e12)

            if m.sum() >= MIN_POINTS:

                ebit_m = h_ebit[m] / np.maximum(np.abs(h_rev[m]), e12)

                opex_r = h_gm[m] - ebit_m

                opex_r = opex_r[np.isfinite(opex_r)]

                if opex_r.size >= MIN_POINTS:

                    hist_opex_mu = np.median(opex_r)

                    hist_opex_sd = 1.4826 * np.median(np.abs(opex_r - hist_opex_mu))

                    hist_opex_sd = max(hist_opex_sd, e6)

    phi_opex = float(phi) if phi is not None and np.isfinite(phi) else None

    noise_opex = float(noise_scale) if noise_scale is not None and np.isfinite(noise_scale) else None

    df_opex = float(df_t) if df_t is not None and np.isfinite(df_t) else None

    phi_m = phi_opex

    noise_m = noise_opex

    df_m = df_opex

    if hist_annual is not None and (not hist_annual.empty):

        if all((c in hist_annual.columns for c in ('revenue', 'ebit', 'gross_margin'))):

            h_rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').to_numpy(dtype = float)

            h_ebit = pd.to_numeric(hist_annual['ebit'], errors = 'coerce').to_numpy(dtype = float)

            h_gm = pd.to_numeric(hist_annual['gross_margin'], errors = 'coerce').to_numpy(dtype = float)

            h_gm = _pct_to_dec_if_needed(
                x = h_gm
            )

            m0 = np.isfinite(h_rev) & np.isfinite(h_ebit) & np.isfinite(h_gm) & (np.abs(h_rev) > e12)

            if m0.sum() >= MIN_POINTS:

                ebit_m = h_ebit[m0] / np.maximum(np.abs(h_rev[m0]), e12)

                opex_r = h_gm[m0] - ebit_m

                opex_r = opex_r[np.isfinite(opex_r)]

                if opex_r.size >= MIN_POINTS + 2:

                    tgt = hist_opex_mu if hist_opex_mu is not None and np.isfinite(hist_opex_mu) else float(np.nanmedian(opex_r))

                    phi_hat, noise_hat, df_hat = _estimate_ar1_params_from_levels(
                        x = opex_r,
                        target = tgt
                    )

                    if phi_opex is None:

                        phi_opex = phi_hat

                    if noise_opex is None:

                        noise_opex = noise_hat

                    if df_opex is None:

                        df_opex = df_hat

        if all((c in hist_annual.columns for c in ('revenue', 'ebitda'))):

            h_rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').to_numpy(dtype = float)

            h_ebitda = pd.to_numeric(hist_annual['ebitda'], errors = 'coerce').to_numpy(dtype = float)

            m1 = np.isfinite(h_rev) & np.isfinite(h_ebitda) & (np.abs(h_rev) > e12)

            if m1.sum() >= MIN_POINTS:

                m_hist = h_ebitda[m1] / np.maximum(np.abs(h_rev[m1]), e12)

                m_hist = m_hist[np.isfinite(m_hist)]

                if m_hist.size >= MIN_POINTS + 2:

                    tgt = float(np.nanmedian(m_hist))

                    phi_hat, noise_hat, df_hat = _estimate_ar1_params_from_levels(
                        x = m_hist,
                        target = tgt
                    )

                    if phi_m is None:

                        phi_m = phi_hat

                    if noise_m is None:

                        noise_m = noise_hat

                    if df_m is None:

                        df_m = df_hat

    if phi_opex is None:

        phi_opex = 0.0

    if noise_opex is None:

        noise_opex = 1.0

    if df_opex is None:

        df_opex = 8.0

    if phi_m is None:

        phi_m = float(phi_opex)

    if noise_m is None:

        noise_m = float(noise_opex)

    if df_m is None:

        df_m = float(df_opex)

    phi_opex = float(np.clip(phi_opex, 0.0, 0.97))

    phi_m = float(np.clip(phi_m, 0.0, 0.97))

    noise_opex = float(np.clip(noise_opex, 0.05, 3.0))

    noise_m = float(np.clip(noise_m, 0.05, 3.0))

    df_opex = float(np.clip(df_opex, 4.5, 60.0))

    df_m = float(np.clip(df_m, 4.5, 60.0))

    end_ebit = _native_end_idx(
        key = 'ebit',
        native_last = native_last,
        periods = periods
    )

    end_ebitda = _native_end_idx(
        key = 'ebitda',
        native_last = native_last,
        periods = periods
    )

    if end_ebit >= T - 1 and end_ebitda >= T - 1:

        return sim

    need_ebit_extend = end_ebit < T - 1

    if need_ebit_extend:

        if 'ebit' not in sim:

            if 'ebitda' in sim and da is not None:

                sim['ebit'] = np.asarray(sim['ebitda'], float) - da

            else:

                sim['ebit'] = np.full((T, n), np.nan, dtype = float)

        ebit = np.asarray(sim['ebit'], float)

        t0 = end_ebit if end_ebit >= 0 else min(T - 1, 4)

        t_native = np.arange(0, t0 + 1) if end_ebit >= 0 else np.array([], dtype = int)

        if gm is not None and t_native.size >= 2:

            ebit_m_hist = ebit[t_native, :] / rev_safe[t_native, :]

            ebit_m_hist = np.clip(ebit_m_hist, min_margin, max_margin)

            opex_r_hist = gm[t_native, :] - ebit_m_hist

            opex_r_hist = np.clip(opex_r_hist, -0.25, 1.25)

            opex_mu, opex_sd = _robust_time_mu_sd(
                M = opex_r_hist
            )

            last_opex = opex_r_hist[-1, :]

            last_opex = np.where(np.isfinite(last_opex), last_opex, opex_mu)

            T_ext = T - 1 - end_ebit

            opex_ext = _mean_revert_extend(
                last = last_opex,
                target = opex_mu,
                sd = opex_sd,
                T_ext = T_ext,
                rng = rng,
                phi = phi_opex,
                noise_scale = noise_opex,
                df_t = df_opex
            )

            gm_out = gm[end_ebit + 1:, :]

            ebit_m_out = gm_out - opex_ext

            ebit_m_out = np.minimum(ebit_m_out, gm_out)

            ebit_m_out = np.clip(ebit_m_out, min_margin, max_margin)

            ebit_out = rev[end_ebit + 1:, :] * ebit_m_out

            ebit[end_ebit + 1:, :] = ebit_out

            if 'opex_ratio' not in sim:

                sim['opex_ratio'] = np.full((T, n), np.nan, dtype = float)

            sim['opex_ratio'][t_native, :] = opex_r_hist

            sim['opex_ratio'][end_ebit + 1:, :] = opex_ext

        else:

            if t_native.size >= 2:

                ebit_m_hist = ebit[t_native, :] / rev_safe[t_native, :]

                ebit_m_hist = np.clip(ebit_m_hist, min_margin, max_margin)

                m_mu, m_sd = _robust_time_mu_sd(
                    M = ebit_m_hist
                )

                last_m = ebit_m_hist[-1, :]

                last_m = np.where(np.isfinite(last_m), last_m, m_mu)

            else:

                m_mu = np.full(n, 0.1, dtype = float)

                m_sd = np.full(n, 0.03, dtype = float)

                if hist_annual is not None and (not hist_annual.empty) and all((c in hist_annual.columns for c in ('revenue', 'ebit'))):

                    h_rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').to_numpy(dtype = float)

                    h_ebit = pd.to_numeric(hist_annual['ebit'], errors = 'coerce').to_numpy(dtype = float)

                    m = np.isfinite(h_rev) & np.isfinite(h_ebit) & (np.abs(h_rev) > e12)

                    if m.sum() >= MIN_POINTS:

                        mr = h_ebit[m] / np.maximum(np.abs(h_rev[m]), e12)

                        mr = mr[np.isfinite(mr)]

                        if mr.size >= MIN_POINTS:

                            mu0 = np.median(mr)

                            sd0 = 1.4826 * np.median(np.abs(mr - mu0))

                            m_mu[:] = mu0

                            m_sd[:] = max(sd0, e6)

                last_m = m_mu.copy()

            T_ext = T - 1 - end_ebit

            m_ext = _mean_revert_extend(
                last = last_m,
                target = m_mu,
                sd = m_sd,
                T_ext = T_ext,
                rng = rng,
                phi = phi_m,
                noise_scale = noise_m,
                df_t = df_m
            )

            if gm is not None:

                gm_out = gm[end_ebit + 1:, :]

                m_ext = np.minimum(m_ext, gm_out)

            m_ext = np.clip(m_ext, min_margin, max_margin)

            ebit[end_ebit + 1:, :] = rev[end_ebit + 1:, :] * m_ext

        sim['ebit'] = ebit

    need_ebitda_extend = end_ebitda < T - 1

    if 'ebitda' not in sim:

        if 'ebit' in sim and da is not None:

            sim['ebitda'] = np.asarray(sim['ebit'], float) + da

        else:

            sim['ebitda'] = np.full((T, n), np.nan, dtype = float)

    ebitda = np.asarray(sim['ebitda'], float)

    if need_ebitda_extend:

        if 'ebit' in sim and da is not None:

            ebitda[end_ebitda + 1:, :] = np.asarray(sim['ebit'], float)[end_ebitda + 1:, :] + da[end_ebitda + 1:, :]

        else:

            t0 = end_ebitda if end_ebitda >= 0 else min(T - 1, 4)

            t_native = np.arange(0, t0 + 1) if end_ebitda >= 0 else np.array([], dtype = int)

            if t_native.size >= 2:

                m_hist = ebitda[t_native, :] / rev_safe[t_native, :]

                m_hist = np.clip(m_hist, min_margin, max_margin)

                m_mu, m_sd = _robust_time_mu_sd(
                    M = m_hist
                )

                last_m = m_hist[-1, :]

                last_m = np.where(np.isfinite(last_m), last_m, m_mu)

            else:

                m_mu = np.full(n, 0.15, dtype = float)

                m_sd = np.full(n, 0.04, dtype = float)

                if hist_annual is not None and (not hist_annual.empty) and all((c in hist_annual.columns for c in ('revenue', 'ebitda'))):

                    h_rev = pd.to_numeric(hist_annual['revenue'], errors = 'coerce').to_numpy(dtype = float)

                    h_e = pd.to_numeric(hist_annual['ebitda'], errors = 'coerce').to_numpy(dtype = float)

                    m = np.isfinite(h_rev) & np.isfinite(h_e) & (np.abs(h_rev) > e12)

                    if m.sum() >= MIN_POINTS:

                        mr = h_e[m] / np.maximum(np.abs(h_rev[m]), e12)

                        mr = mr[np.isfinite(mr)]

                        if mr.size >= MIN_POINTS:

                            mu0 = np.median(mr)

                            sd0 = 1.4826 * np.median(np.abs(mr - mu0))

                            m_mu[:] = mu0

                            m_sd[:] = max(sd0, e6)

                last_m = m_mu.copy()

            T_ext = T - 1 - end_ebitda

            m_ext = _mean_revert_extend(
                last = last_m,
                target = m_mu,
                sd = m_sd,
                T_ext = T_ext,
                rng = rng,
                phi = phi_m,
                noise_scale = noise_m,
                df_t = df_m
            )

            if 'ebit' in sim:

                ebit_m_out = np.asarray(sim['ebit'], float)[end_ebitda + 1:, :] / rev_safe[end_ebitda + 1:, :]

                m_ext = np.maximum(m_ext, ebit_m_out)

            m_ext = np.clip(m_ext, min_margin, max_margin)

            ebitda[end_ebitda + 1:, :] = rev[end_ebitda + 1:, :] * m_ext

    sim['ebitda'] = ebitda

    if 'opex_ratio' not in sim and hist_opex_mu is not None and (gm is not None):

        sim['opex_ratio'] = np.full((T, n), hist_opex_mu, dtype = float)

    return sim


@dataclass
class TickerMetricBundle:
    """
    Per-ticker container of future consensus tables and (optional) historical statements.

    The bundle is created by ``preload_ticker_bundles_for_fcff`` and is used by ``run_valuation`` to
    route a ticker’s data into each valuation engine.

    Attributes
    ----------
    future:
        Mapping from internal driver keys (for example, ``"fcf"``, ``"capex"``, ``"ebitda"``) to future
        consensus tables. Each table is expected to contain metric rows (median/high/low/standard
        deviation/estimate counts) with date-like columns.
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical financial statements (income statement, cashflow statement, balance sheet, and
        ratios) used for coherence checks, imputation priors, and dependence modelling.
    capex_seasonality_w_q1_q4:
        Optional quarterly weights (Q1..Q4) inferred from TTM quarters for allocating annual CapEx into
        quarterly components when quarterly forecasts are incomplete.
  
    """
  
    ticker: str

    future: dict[str, pd.DataFrame]

    hist_inc: pd.DataFrame | None = None

    hist_cf: pd.DataFrame | None = None

    hist_bal: pd.DataFrame | None = None

    hist_ratios: pd.DataFrame | None = None

    hist_inc_q_ttm: pd.DataFrame | None = None

    hist_cf_q_ttm: pd.DataFrame | None = None

    capex_seasonality_w_q1_q4: np.ndarray | None = None

_FCFF_FUTURE_SPECS: dict[str, tuple[list[str], str]] = {
    'fcf': (['Free Cash Flow', 'Free Cash Flow (FCF)'], 'Free_Cash_Flow'), 
    'net_debt': (['Net Debt', 'Net Debt (incl. Leases)', 'Net Debt (Including Leases)'], 'Net_Debt'), 
    'revenue': (['Revenue', 'Total Revenue'], 'Revenue'), 
    'tax': (['Effective Tax Rate %', 'Tax Rate', 'Effective Tax Rate'], 'Tax_Rate_Pct'), 
    'capex': (['Capital Expenditure', 'CapEx'], 'CapEx'), 
    'maint_capex': (['Maintenance Capital Expenditure', 'Maintenance Capital Expenditures', 'Maintenance CapEx'], 'Maint_CapEx'), 
    'da': (['Depreciation & Amortization', 'Depreciation & Amort.'], 'DA'), 
    'interest': (['Interest Expense', 'Net Interest Expense'], 'Interest_Expense'), 
    'gross_margin': (['Gross Margin %', 'Gross Margin'], 'Gross_Margin_Pct'), 
    'ebit': (['EBIT', 'Operating Income'], 'EBIT'), 
    'ebitda': (['EBITDA'], 'EBITDA'), 
    'cfo': (['Cash From Operations', 'Operating Cash Flow'], 'CFO'), 
    'net_income': (['Net Income (Excl. Excep)', 'Net Income (Excl. Excep, GW)', 'Net Income (GAAP)', 'Net Income'], 'Net_Income'), 
    'eps': (['EPS Normalized', 'EPS (Excl. Excep, GW)', 'EPS (GAAP)'], 'EPS_Normalized'), 
    'roe': (['ROE %', 'ROE'], 'ROE_pct'), 
    'dps': (['DPS'], 'DPS'), 
    'bvps': (['Book Value / Share', 'Book Value Per Share', 'Book Value/Share', 'Net Asset Value Per Share'], 'BVPS'), 
    'ebt': (['EBT (GAAP)', 'EBT (Excl. Excep)', 'EBT (Excl. Excep, GW)'], 'EBT')
}


def _infer_fy_md_from_future_tables(
    future: dict[str, pd.DataFrame],
    default_md: tuple[int, int] = (12, 31)
) -> tuple[int, int]:
    """
    Infer fiscal year-end month/day from a set of future consensus tables.

    CapIQ forecast tables are often labelled with fiscal year-end dates. When the fiscal year-end
    month/day is not known a priori, it can be inferred by examining the date-like column labels in
    the consensus tables.

    Inference strategy
    ------------------
   
    1. Prefer explicit annual period columns when a ``period_type`` row is available:
   
       - For a small set of anchor keys ("fcf", "revenue", "eps", "dps", "net_debt"), extract the
         columns labelled as annual and compute the mode of (month, day).
   
    2. Fallback to the overall column timestamps:
   
       - For the same key order, parse all columns to datetimes, ignore tables that look quarterly,
         and compute the mode of (month, day).
   
    3. If no inference is possible, return ``default_md``.

    Parameters
    ----------
    future:
        Mapping of driver keys to future consensus tables.
    default_md:
        Default month/day to return when inference fails.

    Returns
    -------
    (int, int)
        Tuple (fy_month, fy_day).
    """


    def _mode_month_day(
        idx: pd.DatetimeIndex
    ) -> tuple[int, int] | None:
        """
        Compute the mode of (month, day) pairs for a DatetimeIndex.

        Parameters
        ----------
        idx:
            Datetime index.

        Returns
        -------
        (int, int) | None
            Most common (month, day) pair, or None when not well-defined.
        """
        if len(idx) == 0:

            return None

        md = pd.Series(list(zip(idx.month, idx.day)))

        if md.empty:

            return None

        m, d = md.mode().iat[0]

        m, d = (int(m), int(d))

        if 1 <= m <= 12 and 1 <= d <= 31:

            return (m, d)

        return None


    key_order = ('fcf', 'revenue', 'eps', 'dps', 'net_debt')

    for k in key_order:
   
        df = future.get(k, None)

        if df is None or df.empty:

            continue

        if 'period_type' not in df.index:

            continue

        try:

            pt = df.loc['period_type'].astype(str).str.strip().str.lower().reindex(df.columns)

        except (TypeError, ValueError, KeyError):

            continue

        ann_cols = pd.to_datetime(pt[pt.eq('annual')].index, errors = 'coerce')

        ann_cols = pd.DatetimeIndex(ann_cols[pd.notna(ann_cols)]).normalize()

        md = _mode_month_day(
            idx = ann_cols
        )

        if md is not None:

            return md

    for k in key_order:
     
        df = future.get(k, None)

        if df is None or df.empty:

            continue

        cols = pd.to_datetime(df.columns, errors = 'coerce')

        cols = pd.DatetimeIndex(cols[pd.notna(cols)]).normalize()

        if len(cols) == 0:

            continue

        if _is_quarterly_like(
            cols = cols
        ):

            continue

        md = _mode_month_day(
            idx = cols
        )

        if md is not None:

            return md

    return default_md


def _infer_fy_md_from_history_tables(
    *,
    hist_inc: pd.DataFrame | None,
    hist_cf: pd.DataFrame | None,
    hist_bal: pd.DataFrame | None,
    hist_ratios: pd.DataFrame | None,
    default_md: tuple[int, int] = (12, 31)
) -> tuple[int, int]:
    """
    Infer fiscal year-end month/day from historical statement table column timestamps.

    For each historical table, the column timestamps are coerced to datetimes. For each calendar
    year present, the last available timestamp in that year is taken as a proxy for the fiscal
    year-end reporting date. The (month, day) pairs are counted across all tables and years, and
    the most common pair is returned.

    This heuristic is designed to be robust when:
   
    - some statements are missing for some years,
   
    - reporting dates vary slightly around the fiscal year-end (for example, weekends), and
   
    - different statement types are available.

    Parameters
    ----------
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical statement tables.
    default_md:
        Default month/day when inference fails.

    Returns
    -------
    (int, int)
        Tuple (fy_month, fy_day).
 
    """
 
    md_counts: Counter[tuple[int, int]] = Counter()

    for df in (hist_inc, hist_cf, hist_bal, hist_ratios):
        if df is None or df.empty:

            continue

        cols = pd.to_datetime(df.columns, errors = 'coerce')

        cols = pd.DatetimeIndex(cols[pd.notna(cols)]).normalize().sort_values()

        if len(cols) == 0:

            continue

        years = sorted(set(cols.year))

        for y in years:
            ycols = cols[cols.year == y]

            if len(ycols) == 0:

                continue

            y_last = pd.Timestamp(ycols.max())

            md_counts[int(y_last.month), int(y_last.day)] += 1

    if not md_counts:

        return default_md

    m, d = md_counts.most_common(1)[0][0]

    if 1 <= int(m) <= 12 and 1 <= int(d) <= 31:

        return (int(m), int(d))

    return default_md


def _pick_hist_annual_cols(
    df: pd.DataFrame,
    *,
    fy_m: int,
    keep_last_n: int = 100
) -> pd.Index:
    """
    Select a plausible set of annual columns from a historical statement table.

    Historical statement exports may include quarterly columns, annual columns, or both. For many
    downstream steps only annual points are required. This helper selects columns as follows:

    - Convert columns to datetimes and normalise to midnight.
  
    - If at least two columns fall in the fiscal year-end month fy_m, select only those columns.
      This tends to isolate fiscal year-end columns even when quarterly data exist.
  
    - Otherwise, keep all date-like columns.
  
    - Sort selected columns by date and keep only the last ``keep_last_n`` columns.

    Parameters
    ----------
    df:
        Historical statement table.
    fy_m:
        Fiscal year-end month.
    keep_last_n:
        Maximum number of columns to keep, preserving the most recent observations.

    Returns
    -------
    pandas.Index
        Selected column labels in chronological order.
   
    """
   
    if df is None or df.empty:

        return pd.Index([])

    col_dt = pd.to_datetime(df.columns, errors = 'coerce')

    ok = pd.notna(col_dt)

    if not ok.any():

        return pd.Index([])

    cols = df.columns[ok]

    dt = pd.DatetimeIndex(col_dt[ok]).normalize()

    mask_fy_month = dt.month == fy_m

    if mask_fy_month.sum() >= 2:

        cols2 = cols[mask_fy_month]

        dt2 = dt[mask_fy_month]

    else:

        cols2 = cols

        dt2 = dt

    order = np.argsort(dt2.values)

    cols2 = cols2[order]

    if keep_last_n is not None and keep_last_n > 0 and (len(cols2) > keep_last_n):

        cols2 = cols2[-keep_last_n:]

    return cols2


def _subset_rows_if_present(
    df: pd.DataFrame,
    keep_rows: Iterable[str]
) -> pd.DataFrame:
    """
    Subset a DataFrame to a set of rows when any of those rows are present.

    This helper is used to reduce large statement tables to a smaller set of economically relevant
    rows for performance and memory reasons, without failing when expected rows are missing.

    Parameters
    ----------
    df:
        Input DataFrame.
    keep_rows:
        Iterable of row labels to keep.

    Returns
    -------
    pandas.DataFrame
        Row-subset DataFrame when at least one requested row exists, otherwise the original df.
  
    """
  
    if df is None or df.empty:

        return df

    idx = df.index.intersection(pd.Index(list(keep_rows)))

    if len(idx) == 0:

        return df

    return df.loc[idx]


def _reduce_history_tables_for_bundle(
    *,
    inc: pd.DataFrame,
    bal: pd.DataFrame,
    cf: pd.DataFrame,
    ratios: pd.DataFrame,
    fy_m: int,
    strict_rows: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reduce historical statement tables to an annual subset suitable for per-ticker bundling.

    The function performs:
 
    - annual column selection for each statement table using ``_pick_hist_annual_cols``, and
 
    - optional strict row subsetting to keep only a pre-defined row set for each statement type.

    This reduction improves performance of subsequent modelling steps by limiting the size of the
    statement tables carried in memory while retaining the rows required for coherence diagnostics
    and historical panel construction.

    Parameters
    ----------
    inc, bal, cf, ratios:
        Historical statement tables.
    fy_m:
        Fiscal year-end month used to prefer fiscal year-end columns.
    strict_rows:
        When True, subset to the configured keep row sets for each statement type.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        Reduced income, balance, cash flow, and ratios tables.
  
    """
  
    inc_cols = _pick_hist_annual_cols(
        df = inc,
        fy_m = fy_m
    )

    cf_cols = _pick_hist_annual_cols(
        df = cf,
        fy_m = fy_m
    )

    bal_cols = _pick_hist_annual_cols(
        df = bal,
        fy_m = fy_m
    )

    rat_cols = _pick_hist_annual_cols(
        df = ratios,
        fy_m = fy_m
    )

    if len(inc_cols):

        inc = inc.loc[:, inc_cols]

    if len(cf_cols):

        cf = cf.loc[:, cf_cols]

    if len(bal_cols):

        bal = bal.loc[:, bal_cols]

    if len(rat_cols):

        ratios = ratios.loc[:, rat_cols]

    if strict_rows:

        inc = _subset_rows_if_present(
            df = inc,
            keep_rows = _HIST_INC_ROWS_KEEP
        )

        cf = _subset_rows_if_present(
            df = cf,
            keep_rows = _HIST_CF_ROWS_KEEP
        )

        ratios = _subset_rows_if_present(
            df = ratios,
            keep_rows = _HIST_RAT_ROWS_KEEP
        )

    return (inc, bal, cf, ratios)


def _reduce_future_tables_for_bundle(
    future: dict[str, pd.DataFrame],
    *,
    keep_only_annual: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Reduce future consensus tables for bundling, optionally keeping only annual periods.

    Parameters
    ----------
    future:
        Mapping of driver keys to forecast tables.
    keep_only_annual:
        If True, each table is reduced to its inferred annual period columns using ``_annual_periods``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Reduced mapping of future tables.
 
    """
 
    out: dict[str, pd.DataFrame] = {}

    for k, df in future.items():
 
        if df is None or df.empty:

            continue

        if keep_only_annual:

            ann = _annual_periods(
                dfT = df
            )

            if len(ann) > 0:

                out[k] = df.reindex(columns = ann).copy()

            else:

                out[k] = df.copy()

        else:

            out[k] = df.copy()

    return out


def _drop_nonfuture_columns(
    df: pd.DataFrame | None,
    *,
    today: pd.Timestamp = TODAY_TS
) -> pd.DataFrame | None:
    """
    Drop forecast columns that are not strictly in the future relative to a valuation date.

    Consensus workbooks may include the current period or stale periods. The valuation engines are
    designed to operate on periods strictly after the valuation date. This helper:

    - parses columns to datetimes,
  
    - keeps only columns where period_end > today, and
  
    - normalises columns to midnight and removes duplicates.

    When no columns remain, an empty DataFrame with the same index and zero columns is returned.

    Parameters
    ----------
    df:
        Forecast table.
    today:
        Valuation date.

    Returns
    -------
    pandas.DataFrame | None
        Filtered table, or the original when df is None/empty.
  
    """
  
    if df is None or df.empty:

        return df

    cols_dt = pd.to_datetime(df.columns, errors = 'coerce')

    ok = pd.notna(cols_dt)

    if not bool(np.any(ok)):

        return df.iloc[:, :0].copy()

    cols_norm = pd.DatetimeIndex(cols_dt[ok]).normalize()

    keep = cols_norm > pd.Timestamp(today).normalize()

    if not bool(np.any(keep)):

        return df.iloc[:, :0].copy()

    kept_orig = pd.Index(df.columns)[ok][keep]

    out = df.loc[:, kept_orig].copy()

    if len(out.columns):

        out.columns = pd.DatetimeIndex(pd.to_datetime(out.columns, errors = 'coerce')).normalize()

        if out.columns.has_duplicates:

            out = out.loc[:, ~out.columns.duplicated(keep = 'last')]

    return out


def _drop_nonfuture_columns_from_tables(
    future: dict[str, pd.DataFrame],
    *,
    today: pd.Timestamp = TODAY_TS
) -> dict[str, pd.DataFrame]:
    """
    Apply ``_drop_nonfuture_columns`` to a mapping of forecast tables, dropping empty results.

    Parameters
    ----------
    future:
        Mapping of driver keys to forecast tables.
    today:
        Valuation date.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Filtered mapping containing only non-empty filtered tables.
   
    """
   
    out: dict[str, pd.DataFrame] = {}

    for k, df in future.items():
        
        dff = _drop_nonfuture_columns(
            df = df,
            today = today
        )

        if dff is None or dff.empty:

            continue

        out[k] = dff

    return out


def preload_ticker_bundles_for_fcff(
    *,
    tickers: list[str],
    pred_files: dict[str, str],
    hist_files: dict[str, Path],
    fy_freq: str = FY_FREQ,
    strict_history_rows: bool = True,
    keep_only_annual_future: bool = False
) -> dict[str, TickerMetricBundle]:
    """
    Preload per-ticker future consensus tables and optional historical statements into bundles.

    Purpose
    -------
    ``run_valuation`` can be called for many tickers. Reading and parsing workbooks repeatedly
    inside each valuation engine is expensive. This helper performs the I/O and parsing once per
    ticker and returns a mapping of ``TickerMetricBundle`` objects containing:

    - future consensus tables for all configured driver keys,
 
    - historical statement tables (when available), and
 
    - optional quarterly CapEx seasonality weights inferred from TTM statement data.

    FX scaling
    ----------
    Consensus workbooks may be denominated in a currency different from the quote currency inferred
    from the ticker symbol. When a conversion factor is available, both future and historical tables
    are scaled to the target currency using:

        scaled_value = raw_value * fx_scale

    Ratio-like rows are excluded from scaling where appropriate.

    Data reduction
    --------------
  
    - Future tables can be reduced to annual-only columns when ``keep_only_annual_future`` is True.
  
    - Future tables are filtered to keep only strictly future columns.
  
    - Historical tables are reduced to annual columns and optionally to a strict set of rows.

    Parameters
    ----------
    tickers:
        List of tickers to load.
    pred_files:
        Mapping ``ticker -> consensus workbook path``.
    hist_files:
        Mapping ``ticker -> historical statement workbook path``.
    fy_freq:
        Fiscal-year frequency label passed through to consensus parsing.
    strict_history_rows:
        Whether to reduce historical statement tables to a strict row subset.
    keep_only_annual_future:
        Whether to reduce future consensus tables to annual-only periods.

    Returns
    -------
    dict[str, TickerMetricBundle]
        Mapping ``ticker -> bundle``.
    """
   
    bundles: dict[str, TickerMetricBundle] = {}

    for tkr in tickers:
      
        logger.info('Loading metric bundle for %s...', tkr)

        pred_file = pred_files[tkr]

        src_ccy = _infer_consensus_currency_from_pred_file(
            pred_file = pred_file
        )

        tgt_ccy = _infer_quote_currency_from_ticker(
            ticker = tkr
        )

        fx_scale = _fx_target_per_source(
            source_ccy = src_ccy,
            target_ccy = tgt_ccy
        )

        if src_ccy is not None and tgt_ccy is not None and (src_ccy != tgt_ccy) and (fx_scale is None or not np.isfinite(fx_scale)):

            warnings.warn(f"{tkr}: couldn't compute FX conversion {src_ccy}->{tgt_ccy}; leaving forecast/history in source currency.")

        future = extract_many_future_metric_estimates(
            pred_file = pred_file,
            specs = _FCFF_FUTURE_SPECS,
            fy_freq = fy_freq,
            ticker = tkr
        )

        if fx_scale is not None and np.isfinite(fx_scale) and (abs(float(fx_scale) - 1.0) > 1e-12):

            logger.info('[FX] %s: converting financials %s->%s @ %.8f', tkr, src_ccy, tgt_ccy, float(fx_scale))

            future = _scale_future_tables_currency(
                future = future,
                factor = float(fx_scale)
            )

        future = _reduce_future_tables_for_bundle(
            future = future,
            keep_only_annual = keep_only_annual_future
        )

        future = _drop_nonfuture_columns_from_tables(
            future = future,
            today = TODAY_TS
        )

        hist_inc = hist_cf = hist_bal = hist_ratios = None

        xls_path = hist_files.get(tkr, None)

        if xls_path is not None:

            inc, bal, cf, ratios = load_statements_named(
                xls_path = xls_path
            )

            inc_q_ttm = inc.copy()

            cf_q_ttm = cf.copy()

            if fx_scale is not None and np.isfinite(fx_scale) and (abs(float(fx_scale) - 1.0) > 1e-12):

                inc = _scale_numeric_table_rows(
                    df = inc,
                    factor = float(fx_scale),
                    skip_rows = set(),
                    skip_ratio_like_rows = True
                )

                bal = _scale_numeric_table_rows(
                    df = bal,
                    factor = float(fx_scale),
                    skip_rows = set(),
                    skip_ratio_like_rows = False
                )

                cf = _scale_numeric_table_rows(
                    df = cf,
                    factor = float(fx_scale),
                    skip_rows = set(),
                    skip_ratio_like_rows = False
                )

                inc_q_ttm = _scale_numeric_table_rows(
                    df = inc_q_ttm,
                    factor = float(fx_scale),
                    skip_rows = set(),
                    skip_ratio_like_rows = True
                )

                cf_q_ttm = _scale_numeric_table_rows(
                    df = cf_q_ttm,
                    factor = float(fx_scale),
                    skip_rows = set(),
                    skip_ratio_like_rows = False
                )

            fy_m, fy_d = _infer_fy_md_from_future_tables(
                future = future
            )

            capex_w = _detect_capex_seasonality_weights_from_ttm_quarters(
                hist_cf_q_ttm = cf_q_ttm,
                hist_inc_q_ttm = inc_q_ttm,
                fy_m = fy_m
            )

            inc, bal, cf, ratios = _reduce_history_tables_for_bundle(
                inc = inc,
                bal = bal,
                cf = cf,
                ratios = ratios,
                fy_m = fy_m,
                strict_rows = strict_history_rows
            )

            hist_inc, hist_cf, hist_bal, hist_ratios = (inc, cf, bal, ratios)

        bundles[tkr] = TickerMetricBundle(
            ticker = tkr,
            future = future,  
            hist_inc = hist_inc,  
            hist_cf = hist_cf,  
            hist_bal = hist_bal,  
            hist_ratios = hist_ratios,  
            hist_inc_q_ttm = inc_q_ttm if xls_path is not None else None,  
            hist_cf_q_ttm = cf_q_ttm if xls_path is not None else None, 
            capex_seasonality_w_q1_q4 = capex_w if xls_path is not None else None
        )

    gc.collect()

    return bundles


@dataclass
class CashflowContext:
    """
    Shared per-ticker cashflow preparation context used by FCFF and FCFE engines.

    The context is created once per ticker and is intended to be reused by:
  
    - ``mc_equity_value_per_share_multi_fcff`` (enterprise DCF discounted at WACC), and
 
    - ``mc_equity_value_per_share_multi_fcfe`` (equity DCF discounted at CoE).

    It encapsulates:
  
    - the aligned valuation period grid (annual and quarterly),
  
    - simulated driver panels for all available drivers,
  
    - simulated and aligned net-debt paths,
  
    - historical annual panels used for coherence and dependence modelling,
  
    - caches that avoid repeated alignment and quarterly-override recomputation, and
  
    - shared terminal-growth uniforms (``terminal_u``) that can be transformed under model-specific
      caps without resampling.

    Advantages:
  
    - substantial reduction in duplicated alignment, simulation, and imputation work,
  
    - consistent simulated driver panels across FCFF and FCFE evaluations within a ticker,
  
    - improved performance without intentionally changing valuation mathematics.
  
    """
  
    ticker_label: str

    policy: SectorPolicy

    run_ctx: RunContext

    fcf_periods: pd.DatetimeIndex

    fcf_period_types: list[str]

    annual_pos: np.ndarray

    years_frac: np.ndarray

    hist_panel: pd.DataFrame | None

    sim: dict[str, np.ndarray]

    sim_components_full: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]]

    fcf_draws: np.ndarray | None

    nd_draws: np.ndarray

    nd_by_period: np.ndarray

    nd_val: float | None

    native_last: dict[str, pd.Timestamp | None]

    fy_m: int

    fy_d: int

    fcf_is_stub: bool

    _period_idx_cache: dict[frozenset[str], tuple[np.ndarray, list[str], bool]] = field(default_factory = dict)

    _aligned_cache: dict[tuple[int, tuple[str, ...], str], pd.DataFrame] = field(default_factory = dict)

    _draws_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = field(default_factory = dict)

    fcf_future: pd.DataFrame | None = None

    net_debt_future: pd.DataFrame | None = None

    driver_futures: dict[str, pd.DataFrame | None] = field(default_factory = dict)

    src_a: dict[str, pd.DatetimeIndex] = field(default_factory = dict)

    seasonal_flow_weights_q1_q4: np.ndarray | None = None

    tax_is_percent: bool = True

    terminal_u: np.ndarray | None = None


def _default_cashflow_method_needs() -> list[set[str]]:
    """
    Return the default driver requirement sets used by the FCFF/FCFE cashflow engines.

    The cashflow engines in this module support multiple "methods" (formula variants) for the same
    underlying valuation model. Each method requires a specific subset of simulated drivers. This
    helper provides a baseline list of such requirement sets so that the shared preparation pipeline
    can:
   
    - detect missing drivers early, and
   
    - trigger imputation or proxy construction for the union of drivers needed by the methods.

    Each element in the returned list is a set of driver keys. Typical groups correspond to:
   
    - direct FCFF ("fcf"),
   
    - CFO minus CapEx unlevering bridges ("cfo", "capex", "interest", "tax"),
   
    - CFO minus maintenance CapEx variants ("cfo", "maint_capex", "interest", "tax"),
   
    - EBIT / EBITDA operating bridges ("ebit"/"ebitda", "tax", "da", "capex", "dnwc"),
   
    - net income bridges ("net_income", "da", "interest", "tax", "capex", "dnwc").

    Returns
    -------
    list[set[str]]
        A list of driver-key sets. Each set is intended to be interpretable by the method-definition
        builders and the generic evaluator.
   
    """
   
    return [
        {'cfo', 'capex', 'interest', 'tax'}, 
        {'cfo', 'maint_capex', 'interest', 'tax'}, 
        {'ebit', 'tax', 'da', 'capex', 'dnwc'}, 
        {'ebitda', 'da', 'tax', 'capex', 'dnwc'}, 
        {'net_income', 'da', 'interest', 'tax', 'capex', 'dnwc'}, 
        {'fcf', 'interest', 'tax', 'capex', 'da', 'dnwc'}
    ]


def _zero_cashflow_result() -> dict[str, float | list[str]]:
    """
    Construct a zero-valued result dictionary for cashflow-based valuation engines.

    Several call sites treat valuation as a best-effort process: if data are missing, period grids
    cannot be formed, or numerical failures occur, the engine returns a structurally valid object
    with neutral numerical outputs rather than raising. This helper centralises that structure.

    The returned mapping follows the common schema used by the FCFF and FCFE Monte Carlo engines:
  
    - per-share value distribution summary (mean/median/5th/95th/std),
  
    - implied return distribution summary (mean/median/5th/95th/std),
  
    - method labels indicating which formula variants contributed.

    Returns
    -------
    dict[str, float | list[str]]
        Mapping with the keys:
  
        - "per_share_mean", "per_share_median", "per_share_p05", "per_share_p95", "per_share_std"
  
        - "returns_mean", "returns_median", "returns_p05", "returns_p95", "returns_std"
  
        - "methods_used" (empty list)
  
    """
  
    return {
        'per_share_mean': 0.0, 
        'per_share_median': 0.0, 
        'per_share_p05': 0.0, 
        'per_share_p95': 0.0,
        'per_share_std': 0.0, 
        'returns_mean': 0.0,
        'returns_median': 0.0, 
        'returns_p05': 0.0, 
        'returns_p95': 0.0, 
        'returns_std': 0.0, 
        'methods_used': []
    }


def _coherence_flags_from_history_panel(
    *,
    hist_panel: pd.DataFrame | None,
    policy: SectorPolicy
) -> tuple[bool, bool]:
    """
    Infer simple "coherence" flags from historical statement data to gate fragile method variants.

    Certain cashflow bridges implicitly assume that:
   
    1) interest expense covaries with the magnitude of debt (at least directionally), and
   
    2) working-capital investment covaries with the scale of operations (often proxied by revenue).

    When these relationships are absent in the historical panel, forecast-imputation and method
    selection become materially less reliable. This helper computes two boolean flags used by method
    definition builders:
   
    - `has_interest_debt_coherence`: based on a rank correlation check between interest and absolute
      net debt, subject to minimum-point and minimum-absolute-correlation thresholds defined by the
      sector policy.
   
    - `has_wc_coherence`: based on a rank correlation check between dnwc and revenue. The absolute
      correlation is evaluated (sign is not required) because the sign convention of dnwc is
      dataset-dependent, whilst the magnitude relationship is typically more stable.

    Parameters
    ----------
    hist_panel:
        Historical annual panel produced by `_build_hist_panel(...)`. Expected to contain columns such
        as "interest", "net_debt", "dnwc", and "revenue" when available. If `None` or empty, both
        flags are returned as `False`.
    policy:
        Sector-specific thresholds controlling minimum sample sizes and minimum coherence strength.

    Returns
    -------
    tuple[bool, bool]
        `(has_interest_debt_coherence, has_wc_coherence)`.

    Notes
    -----
   
    - The debt coherence check uses absolute net debt to avoid sign ambiguity (net cash vs net debt).
  
    - If typical absolute net debt is effectively zero, the interest/debt coherence flag is forced to
      `False` because any observed correlation is likely to be a numerical artefact.
  
    """
 
    has_interest_debt_coherence = False

    has_wc_coherence = False

    if hist_panel is not None and (not hist_panel.empty):

        if {'interest', 'net_debt'}.issubset(hist_panel.columns):

            s_i = pd.to_numeric(hist_panel['interest'], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

            s_nd = pd.to_numeric(hist_panel['net_debt'], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

            nd_abs = s_nd.abs()

            has_interest_debt_coherence, _, _ = _coherence_flag_from_history(
                x = s_i,
                y = nd_abs,
                min_points = policy.min_points_interest_debt,
                min_abs_corr = policy.min_abs_corr_interest_debt,
                use_abs = False
            )

            if np.nanmedian(np.abs(nd_abs.to_numpy(dtype = float))) <= e12:

                has_interest_debt_coherence = False

        if {'dnwc', 'revenue'}.issubset(hist_panel.columns):

            s_d = pd.to_numeric(hist_panel['dnwc'], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

            s_r = pd.to_numeric(hist_panel['revenue'], errors = 'coerce').replace([np.inf, -np.inf], np.nan)

            has_wc_coherence, _, _ = _coherence_flag_from_history(
                x = s_d,
                y = s_r,
                min_points = policy.min_points_wc,
                min_abs_corr = policy.min_abs_corr_wc,
                use_abs = True
            )

    return (has_interest_debt_coherence, has_wc_coherence)


def _fcff_formula_key_from_required(
    required: set[str]
) -> str | None:
    """
    Map an FCFF driver requirement set to a canonical formula-dispatch key.

    The FCFF engine represents each method definition as:
   
    - a human-readable method name,
   
    - a required driver-key set, and
   
    - a formula key that selects the appropriate cashflow construction function.

    This helper converts the requirement set into that formula key. Returning `None` indicates that
    the requirement set is not recognised by the current FCFF formula dispatch table.

    Parameters
    ----------
    required:
        Set of driver keys required for a method.

    Returns
    -------
    str | None
        Formula key understood by `mc_equity_value_per_share_multi_fcff(...)`, or `None` if no mapping
        exists.
  
    """
  
    mapping: dict[frozenset[str], str] = {
        frozenset({'fcf'}): 'fcf',
        frozenset({'cfo', 'capex', 'interest', 'tax'}): 'cfo_capex', 
        frozenset({'cfo', 'maint_capex', 'interest', 'tax'}): 'cfo_maint', 
        frozenset({'ebit', 'tax', 'da', 'capex', 'dnwc'}): 'ebit',
        frozenset({'ebitda', 'da', 'tax', 'capex', 'dnwc'}): 'ebitda', 
        frozenset({'net_income', 'da', 'interest', 'tax', 'capex', 'dnwc'}): 'ni'
    }

    return mapping.get(frozenset(required))


def _prepare_cashflow_context(
    *,
    ctx: RunContext | None = None,
    ticker: str | None = None,
    sector_label: str | None = None,
    sector_policy: SectorPolicy | None = None,
    fcf_future: pd.DataFrame | None,
    net_debt_future: pd.DataFrame | None,
    driver_futures: dict[str, pd.DataFrame | None],
    hist_inc: pd.DataFrame,
    hist_cf: pd.DataFrame,
    hist_bal: pd.DataFrame,
    hist_ratios: pd.DataFrame,
    src_a: dict[str, pd.DatetimeIndex],
    cost_of_debt: float | None = None,
    tax_is_percent: bool = True,
    net_debt_use: str = 'first',
    seasonal_flow_weights_q1_q4: np.ndarray | None = None,
    method_needs: Sequence[set[str]] | None = None,
    rng_labels: dict[str, str] | None = None
) -> CashflowContext:
    """
    Build a shared per-ticker cashflow simulation context for FCFF and FCFE valuation.

    This function centralises the expensive, data-dependent steps that are common to the free-cashflow
    models:
   
    - construction and filtering of the valuation period grid (annual years plus optional stub
      quarters),
   
    - alignment of forecast tables to that grid (including flow/stock/ratio conventions),
   
    - Monte Carlo simulation of forecast uncertainty from consensus "median/high/low/std" inputs
      using a skewed Student-t distribution,
   
    - imputation of missing drivers and repair of partial forecast tables,
   
    - optional working-capital modelling (dnwc) as a stochastic function of revenue,
   
    - optional dependence modelling via a rank-preserving copula-like reorder step, and
   
    - practical bounds and accounting coherence constraints (for example, ensuring EBIT does not
      exceed EBITDA minus non-negative depreciation).

    The returned `CashflowContext` is designed to be passed into both the FCFF and FCFE engines so
    that alignment, simulation, imputation, and dependence work are performed once per ticker.

    Parameters
    ----------
    ctx:
        Deterministic random-number context. If `None`, a default context is created.
    ticker:
        Ticker label used for logging and deterministic seeding.
    sector_label:
        Sector label used only for selecting the default policy when `sector_policy` is not provided.
    sector_policy:
        Sector-specific modelling policy (bounds, coherence thresholds, and method-gating settings).
        If `None`, a policy is inferred from `sector_label`.
    fcf_future:
        Forecast table for free cash flow (interpreted as FCFF in this module), in the standard
        CapIQ-style consensus layout (rows such as "Median/High/Low/Std_Dev/No_of_Estimates" and
        columns as period end dates).
    net_debt_future:
        Forecast table for net debt, when available.
    driver_futures:
        Mapping of driver keys (for example, "revenue", "ebit", "capex", "dnwc") to future consensus
        tables. Entries may be `None` when a driver is unavailable.
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical income statement, cashflow statement, balance sheet, and ratio tables. These are
        used to build an annual historical panel for:
        - sign conventions (for example, CapEx recorded as a negative outflow),
        - plausibility checks and bounds,
        - imputation anchors, and
        - dependence calibration (Spearman correlations, AR(1) persistence, and tail thickness).
    src_a:
        Mapping from driver key to the native annual periods available in the source forecast table.
        This is used to infer `native_last` dates for operating-profit extension logic.
    cost_of_debt:
        Optional cost of debt estimate, used when estimating net debt at the valuation date.
    tax_is_percent:
        If `True`, tax-rate values in forecasts are treated as percentages and converted to fractions
        when they appear to be in the range 0-100.
    net_debt_use:
        Net debt selection convention in downstream enterprise-to-equity conversion ("first" selects
        the earliest valuation period, "last" selects the latest).
    seasonal_flow_weights_q1_q4:
        Optional seasonal weights used when converting annual flow forecasts into quarterly stubs
        during alignment.
    method_needs:
        Optional list of driver requirement sets. If `None`, `_default_cashflow_method_needs()` is
        used. The union of these sets influences which drivers are imputed when missing.
    rng_labels:
        Optional mapping that customises stochastic stream labels for key stages ("dnwc", "copula",
        "op_extend", "terminal_u"). This allows FCFF and FCFE engines to share the same context while
        still using distinct random streams when desired.

    Returns
    -------
    CashflowContext
  
        A dataclass containing:
  
        - the valuation period grid and types,
  
        - a historical annual panel (when constructible),
  
        - simulated driver arrays (shape: n_periods x N_SIMS),
  
        - net debt draws aligned to the valuation grid,
  
        - caches for aligned forecast tables, simulated draws, and method-specific period indices,
  
        - fiscal year end metadata, and
  
        - a pre-sampled terminal-growth uniform vector (`terminal_u`) for reuse across models.

    Modelling Notes
    --------------
    Period grid construction
      
        Periods are built from the available future tables using `_build_mixed_valuation_periods(...)`,
        then filtered to exclude valuation dates not strictly after `TODAY_TS`. Time-to-cashflow is
        represented as a fractional year:
    
        - years_frac_t = (period_t - today) / 365.

    Forecast simulation
   
        Each forecast table is aligned to the period grid and then simulated using
        `_simulate_skewt_from_rows(...)`, which fits a skewed Student-t distribution from consensus
        summary statistics. This choice provides:
   
        - heavy tails (robustness to analyst disagreement and non-Gaussian uncertainty),
   
        - asymmetric distributions (skew), and
   
        - resilience when only high/low ranges are available (variance is approximated).

    Imputation and proxy construction
    
        Missing or partially non-finite driver arrays are imputed via `_impute_missing_driver_draws(...)`
        using historical medians, cross-driver relations, and sector-aware bounds. This enables method
        fallback rather than outright failure when only a subset of drivers exists.

    Dependence (copula-like reorder)
    
        When a historical panel is available, simulated marginal draws are reordered to better match
        historical rank dependence using `_reorder_sim_and_netdebt_by_history(...)`. The approach can
        be viewed as a pragmatic t-copula approximation:
    
        - marginals are simulated independently (skew-t),
    
        - a joint latent correlation structure is estimated from historical changes (Spearman-to-Pearson),
    
        - multivariate t innovations (with AR(1) persistence) are drawn, and
    
        - simulated paths are rank-reordered to match the joint ordering implied by those innovations.

        Advantages include preservation of the marginal forecast distributions whilst injecting
        realistic co-movement, without requiring a full parametric multivariate skew-t fit.

    Practical bounds and coherence checks
        `_apply_practical_checks_and_bounds(...)` enforces sector-aware plausibility constraints (for
        example, non-negative depreciation, reasonable tax ranges, and revenue consistency) and
        `_extend_operating_profit_with_margins(...)` extends operating profit series using margins when
        direct forecasts are missing or too short.
 
    """
 
    ctx = _ensure_ctx(
        ctx = ctx
    )

    ticker_label = str(ticker or ctx.ticker or 'UNKNOWN')

    policy = sector_policy if sector_policy is not None else _policy_for_sector(
        sector_label = sector_label
    )

    if net_debt_use not in {'first', 'last'}:

        raise ValueError("net_debt_use must be 'first' or 'last'")

    rng_cfg = {'dnwc': 'dnwc_shared', 'copula': 'copula:shared', 'op_extend': 'op_extend_shared', 'terminal_u': 'terminal_g:shared_u'}

    if rng_labels:

        rng_cfg.update({str(k): str(v) for k, v in rng_labels.items()})

    method_needs_use = [set(s) for s in (method_needs if method_needs is not None else _default_cashflow_method_needs())]

    base_future = fcf_future

    if base_future is None:

        for _df in driver_futures.values():
       
            if isinstance(_df, pd.DataFrame) and _df.shape[1] > 0:

                base_future = _df

                break

    if base_future is None:

        raise ValueError('No usable future tables provided')

    fy_hist_default = _infer_fy_md_from_history_tables(
        hist_inc = hist_inc,
        hist_cf = hist_cf,
        hist_bal = hist_bal,
        hist_ratios = hist_ratios,
        default_md = (12, 31)
    )

    drivers_for_period_union = dict(driver_futures) if driver_futures is not None else {}

    drivers_for_period_union['fcf'] = fcf_future

    drivers_for_period_union['net_debt'] = net_debt_future

    fy_m, fy_d = _infer_fy_md_from_future_tables(
        future = drivers_for_period_union,
        default_md = fy_hist_default
    )

    if not (1 <= fy_m <= 12 and 1 <= fy_d <= 31):

        fy_m, fy_d = fy_hist_default

    ann_periods = _annual_periods(
        dfT = base_future,
        fy_m = fy_m
    )

    if len(ann_periods) > 0:

        fy_days = pd.Series(pd.DatetimeIndex(ann_periods).day)

        if not fy_days.empty:

            fy_d = int(fy_days.mode().iat[0])

    else:

        ann_periods = _annual_periods(
            dfT = base_future
        )

    if len(ann_periods) == 0:

        raise ValueError('No annual periods found in future table.')

    hist_panel = _build_hist_panel(
        hist_inc = hist_inc,
        hist_cf = hist_cf,
        hist_bal = hist_bal,
        hist_ratios = hist_ratios,
        fy_m = fy_m,
        fy_d = fy_d
    )

    fcf_periods, fcf_period_types, _q_stub = _build_mixed_valuation_periods(
        base_future = base_future,
        driver_futures = drivers_for_period_union,
        include_stub_quarters = True,
        fy_m = fy_m,
        fy_d = fy_d
    )

    fcf_periods, fcf_period_types = _filter_future_periods(
        periods = fcf_periods,
        period_types = fcf_period_types,
        today = TODAY_TS
    )

    if len(fcf_periods) == 0:

        raise ValueError('No future valuation periods after filtering.')

    pt_global = [str(x).lower() for x in fcf_period_types]

    days = (fcf_periods.to_numpy() - np.datetime64(TODAY_TS)) / np.timedelta64(1, 'D')

    years_frac = days / 365.0

    annual_pos = np.array([i for i, t in enumerate(pt_global) if t == 'annual'], dtype = int)

    if annual_pos.size == 0:

        annual_pos = np.arange(len(fcf_periods), dtype = int)

    sim: dict[str, np.ndarray] = {}

    sim_components_full: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}

    period_idx_cache: dict[frozenset[str], tuple[np.ndarray, list[str], bool]] = {}

    aligned_cache: dict[tuple[int, tuple[str, ...], str], pd.DataFrame] = {}

    draws_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}


    def _simulate_driver_draws_for_period_types(
        key: str,
        dfF: pd.DataFrame,
        pt_local: list[str]
    ) -> np.ndarray:
        """
        Simulate and cache driver draws for a specific period-type configuration.

        This nested helper supports method-level quarterly/annual fallback without repeating the
        expensive "align future table -> simulate skew-t draws" sequence. It is keyed by:
        - driver name (`key`), and
        - the local period-type vector (`pt_local`) describing whether each `fcf_periods` element is
          treated as "annual" or "quarterly" for the purposes of this driver.

        The function performs:
     
        1) alignment via `_future_to_period_aligned(...)` using a mode determined by the driver:
     
           - "flow" for most statement flows,
     
           - "stock" for level quantities such as net debt, and
     
           - "ratio" for bounded rates such as tax rate and margins;
     
        2) sign sanitation for commonly negative-reporting flows (CapEx, interest, depreciation);
     
        3) skewed Student-t simulation from consensus summary rows via `_simulate_skewt_from_rows(...)`;
     
        4) unit normalisation for percentage-style series (divide by 100 when medians indicate 0-100);
     
        5) clipping of bounded series (for example, 0 <= tax <= 0.40, 0 <= gross_margin <= 1.0).

        Parameters
        ----------
        key:
            Driver key (for example, "capex", "tax", "net_debt").
        dfF:
            Forecast DataFrame in the standard consensus layout.
        pt_local:
            Period-type vector aligned to `fcf_periods`. Values are lower-cased internally.

        Returns
        -------
        numpy.ndarray
            Simulated driver array of shape (n_periods, N_SIMS) aligned to `fcf_periods`.

        Notes
        -----
        Caching strategy:
    
        - Alignment is cached in `aligned_cache` keyed by `(id(dfF), period_types, mode)`.
    
        - Draws are cached in `draws_cache` keyed by `(key, period_types)`.
        This design avoids hashing the full DataFrame content whilst remaining effective in a
        single-ticker evaluation pass where the DataFrame objects are stable.
    
        """
    
        pt_key = tuple((str(x).lower() for x in pt_local))

        cache_key = (key, pt_key)

        cached = draws_cache.get(cache_key)

        if cached is not None:

            return cached

        mode = 'ratio' if key in {'tax', 'gross_margin', 'roe', 'roa', 'roe_pct', 'roa_pct'} else 'stock' if key in {'net_debt'} else 'flow'

        align_key = (id(dfF), pt_key, mode)

        dfA = aligned_cache.get(align_key)

        if dfA is None:

            dfA = _future_to_period_aligned(
                dfT = dfF,
                periods = fcf_periods,
                period_types = list(pt_key),
                mode = mode,
                fy_m = fy_m,
                fy_d = fy_d,
                seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
            )

            aligned_cache[align_key] = dfA

        if key in {'capex', 'maint_capex', 'interest', 'da'} and len(dfA.index):

            row0 = dfA.index[0]

            vals = pd.to_numeric(dfA.loc[row0], errors = 'coerce').to_numpy(dtype = float, copy = False)

            med = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan

            if np.isfinite(med) and med < 0:

                dfA = _flip_future_metric_sign(
                    dfT = dfA,
                    value_row = row0
                )

        unit = 1.0 if key in {'tax', 'gross_margin'} else UNIT_MULT

        floor0 = key in {'capex', 'maint_capex', 'da', 'interest'}

        seed_label = f"skewt_q:{key}|pt={','.join(pt_key)}|mode={mode}"

        draws, _, mu_sims, sigma_sims, x_std = _simulate_skewt_from_rows(
            dfT = dfA,
            value_row = dfA.index[0],
            unit_mult = unit,
            floor_at_zero = floor0,
            rng = ctx.rng(seed_label),
            return_components = True
        )

        sim[key] = draws

        sim_components_full[key] = (mu_sims, sigma_sims, x_std, floor0)

        if key in {'gross_margin', 'tax', 'roe', 'roe_pct', 'roa', 'roa_pct'}:

            med0 = np.nanmedian(draws) if np.isfinite(draws).any() else 0.0

            if abs(med0) > 1.5:

                draws = draws / 100.0

        if key == 'tax':

            if tax_is_percent:

                med0 = np.nanmedian(draws) if np.isfinite(draws).any() else 0.0

                if abs(med0) > 1.5:

                    draws = draws / 100.0

            draws = np.clip(draws, 0.0, 0.4)

        if key == 'gross_margin':

            draws = np.clip(draws, 0.0, 1.0)

        draws_cache[cache_key] = draws

        return draws


    if fcf_future is not None and (not fcf_future.empty):

        fcfT = _future_to_period_aligned(
            dfT = fcf_future,
            periods = fcf_periods,
            period_types = pt_global,
            fy_m = fy_m,
            fy_d = fy_d,
            seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
        )

        fcf_is_stub = _is_zero_like_future_table(
            dfT = fcfT,
            value_row = 'Free_Cash_Flow'
        )

    else:

        fcfT = None

        fcf_is_stub = True

    if fcf_is_stub or fcfT is None:

        fcf_draws = None

    else:

        fcf_draws, _ = _simulate_skewt_from_rows(
            dfT = fcfT,
            value_row = 'Free_Cash_Flow',
            unit_mult = UNIT_MULT,
            floor_at_zero = False,
            rng = ctx.rng('skewt:fcf')
        )

    nd_is_stub = True if net_debt_future is None else _is_zero_like_future_table(
        dfT = net_debt_future,
        value_row = 'Net_Debt'
    )

    if nd_is_stub:

        nd_use_cols = pd.DatetimeIndex(ann_periods).sort_values()

        hist_nd = _hist_net_debt_debt_minus_cash(
            hist_bal = hist_bal
        ) if hist_bal is not None else None

        if hist_nd is not None and len(hist_nd.dropna()) >= 3:

            hist_nd_scaled = pd.to_numeric(hist_nd, errors = 'coerce').dropna() * UNIT_MULT

            nd_draws = _simulate_rw_from_history_levels(
                hist = hist_nd_scaled,
                T = len(nd_use_cols),
                floor_at_zero = False,
                rng = ctx.rng('net_debt:rw')
            )

            nd_components = None

        else:

            nd_draws = np.zeros((len(nd_use_cols), N_SIMS), dtype = float)

            nd_components = None

    else:

        ndT = net_debt_future.copy()

        nd_periods_all = pd.to_datetime(ndT.columns, errors = 'coerce')

        ok = pd.notna(nd_periods_all)

        ndT = ndT.loc[:, ok]

        nd_periods_all = pd.DatetimeIndex(nd_periods_all[ok]).sort_values()

        ndT = ndT.reindex(columns = nd_periods_all)

        nd_cols = list(nd_periods_all)

        if len(nd_cols) > 1:

            nd_mask_exact = [d.month == fy_m and d.day == fy_d for d in nd_cols]

            if sum(nd_mask_exact) >= 2:

                nd_use_cols = [c for c, okk in zip(nd_cols, nd_mask_exact) if okk]

            else:

                nd_mask_me = [d.month == fy_m and d.is_month_end for d in nd_cols]

                nd_use_cols = [c for c, okk in zip(nd_cols, nd_mask_me) if okk]

                if len(nd_use_cols) < 1:

                    nd_use_cols = nd_cols

        else:

            nd_use_cols = nd_cols

        nd_use_cols = pd.DatetimeIndex(nd_use_cols).sort_values()

        nd_sim = ndT.reindex(columns = nd_use_cols)

        nd_draws, _, nd_mu, nd_sigma, nd_x = _simulate_skewt_from_rows(
            dfT = nd_sim,
            value_row = 'Net_Debt',
            unit_mult = UNIT_MULT,
            floor_at_zero = False,
            rng = ctx.rng('skewt:net_debt'),
            return_components = True
        )

        nd_components = (nd_mu, nd_sigma, nd_x, False)

    nd_by_period = _align_draws_to_periods(
        draws = nd_draws,
        src_cols = pd.DatetimeIndex(nd_use_cols),
        tgt_cols = pd.DatetimeIndex(fcf_periods)
    )

    nd_val = _net_debt_at_valuation_date(
        hist_bal = hist_bal,
        net_debt_future = net_debt_future,
        today = TODAY_TS,
        cost_of_debt = cost_of_debt
    )

    src_a_local = src_a if src_a is not None else {}

    native_last: dict[str, pd.Timestamp | None] = {}

    for k, dfF in driver_futures.items():
     
        if dfF is None:

            continue

        if k not in {'dps'}:

            try:

                _row0 = dfF.index[0] if len(dfF.index) else None

                if _row0 is not None:

                    want = ['Median', 'High', 'Low', 'Std_Dev', 'No_of_Estimates']

                    if all((r in dfF.index for r in want)) and _is_zero_like_future_table(dfT = dfF, value_row = str(_row0)):

                        ne = pd.to_numeric(dfF.loc['No_of_Estimates'], errors = 'coerce')

                        sd = pd.to_numeric(dfF.loc['Std_Dev'], errors = 'coerce')

                        ne_ok = not np.isfinite(ne.to_numpy(dtype = float)).any() or float(np.nanmax(np.abs(ne.to_numpy(dtype = float)))) == 0.0

                        sd_ok = not np.isfinite(sd.to_numpy(dtype = float)).any() or float(np.nanmax(np.abs(sd.to_numpy(dtype = float)))) <= 1e-12

                        if ne_ok and sd_ok:

                            logger.warning("%s '%s' forecast not found in workbook.", ticker_label, k)

                            continue

            except (TypeError, ValueError, KeyError):

                pass

        src_ann = src_a_local.get(k, pd.DatetimeIndex([]))

        native_last[k] = pd.Timestamp(src_ann.max()).normalize() if len(src_ann) else None

        mode = 'ratio' if k in {'tax', 'gross_margin', 'roe', 'roa', 'roe_pct', 'roa_pct'} else 'stock' if k in {'net_debt'} else 'flow'

        dfA = _future_to_period_aligned(
            dfT = dfF,
            periods = fcf_periods,
            period_types = pt_global,
            mode = mode,
            fy_m = fy_m,
            fy_d = fy_d,
            seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
        )

        if k in {'capex', 'maint_capex', 'interest', 'da'}:

            row0 = dfA.index[0]

            vals = pd.to_numeric(dfA.loc[row0], errors = 'coerce').to_numpy(dtype = float)

            med = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan

            if np.isfinite(med) and med < 0:

                dfA = _flip_future_metric_sign(
                    dfT = dfA,
                    value_row = row0
                )

        unit = 1.0 if k in {'tax', 'gross_margin'} else UNIT_MULT

        floor0 = k in {'capex', 'maint_capex', 'da'}

        draws, _ = _simulate_skewt_from_rows(
            dfT = dfA,
            value_row = dfA.index[0],
            unit_mult = unit,
            floor_at_zero = floor0,
            rng = ctx.rng(f'skewt:{k}')
        )

        sim[k] = draws

    for key in ['gross_margin', 'tax', 'roe', 'roe_pct', 'roa', 'roa_pct']:
       
        if key in sim:

            vals = sim[key]

            median_val = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0

            if abs(median_val) > 1.5:

                sim[key] = vals / 100.0

    if hist_panel is not None and (not hist_panel.empty):

        for k in ('capex', 'maint_capex', 'interest', 'da'):
       
            if k in hist_panel.columns:

                s = pd.to_numeric(hist_panel[k], errors = 'coerce')

                med = np.nanmedian(s.to_numpy(dtype = float))

                if np.isfinite(med) and med < 0:

                    hist_panel[k] = -s

    to_impute: set[str] = set()

    for k, v in sim.items():
       
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.any(~np.isfinite(v)):

            to_impute.add(k)

    sim_keys = set(sim.keys())

    for need in method_needs_use:
     
        missing = need - sim_keys

        if missing:

            to_impute |= missing

    if fcf_draws is None or (isinstance(fcf_draws, np.ndarray) and np.any(~np.isfinite(fcf_draws))):

        to_impute.add('fcf')

    if 'revenue' in sim and isinstance(sim['revenue'], np.ndarray) and np.any(~np.isfinite(sim['revenue'])):

        to_impute.add('revenue')

    if to_impute:

        nd_for_interest = _align_draws_to_periods(
            draws = nd_draws,
            src_cols = pd.DatetimeIndex(nd_use_cols),
            tgt_cols = pd.DatetimeIndex(fcf_periods)
        )

        sim, _imputed = _impute_missing_driver_draws(
            sim = sim,
            hist_panel = hist_panel,
            missing = to_impute,
            ctx = ctx,
            net_debt_draws = nd_for_interest,
            fy_m = fy_m,
            fy_d = fy_d,
            seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4,
            periods = fcf_periods,
            period_types = pt_global
        )

    if fcf_draws is None and 'fcf' in sim:

        fcf_draws = sim['fcf']

    if 'revenue' in sim and hist_panel is not None and (not hist_panel.empty) and ('dnwc' in hist_panel.columns) and ('revenue' in hist_panel.columns):

        s_dnwc = pd.to_numeric(hist_panel['dnwc'], errors = 'coerce').dropna()

        s_rev = pd.to_numeric(hist_panel['revenue'], errors = 'coerce').dropna()

        if len(s_dnwc) >= MIN_POINTS and len(s_rev) >= MIN_POINTS:

            model = _fit_dnwc_model(
                hist_dnwc = s_dnwc * UNIT_MULT,
                hist_rev = s_rev * UNIT_MULT
            )

            sim['dnwc'] = _simulate_dnwc(
                model = model,
                revenue_draws = sim['revenue'],
                rng = ctx.rng(rng_cfg['dnwc'])
            )

    if hist_panel is not None and (not hist_panel.empty):

        if fcf_draws is not None and 'fcf' not in sim:

            sim['fcf'] = fcf_draws

        for k, dfF in sorted(driver_futures.items()):
            if dfF is None or dfF.empty:

                continue

            _simulate_driver_draws_for_period_types(
                key = k,
                dfF = dfF,
                pt_local = list(pt_global)
            )

        sim, nd_draws, corr_status = _reorder_sim_and_netdebt_by_history(
            sim = sim,
            sim_components = sim_components_full,
            nd_draws = nd_draws,
            nd_components = nd_components,
            fcf_periods = fcf_periods,
            nd_use_cols = nd_use_cols,
            hist_annual = hist_panel,
            rng = ctx.rng(rng_cfg['copula']),
            tax_is_percent = tax_is_percent
        )

        if 'fcf' in sim:

            fcf_draws = sim['fcf']

    else:

        corr_status = {'attempted': False, 'used': False, 'reason': 'hist_panel_missing'}

    logger.debug('[CASHFLOW-CORR] %s: attempted=%s used=%s reason=%s', ticker_label, corr_status['attempted'], corr_status['used'], corr_status['reason'])

    nd_by_period = _align_draws_to_periods(
        draws = nd_draws,
        src_cols = pd.DatetimeIndex(nd_use_cols),
        tgt_cols = pd.DatetimeIndex(fcf_periods)
    )

    sim = _apply_practical_checks_and_bounds(
        sim = sim,
        hist_annual = hist_panel,
        sector_policy = policy
    )

    sim = _extend_operating_profit_with_margins(
        sim = sim,
        periods = fcf_periods,
        native_last = native_last,
        hist_annual = hist_panel,
        rng = ctx.rng(rng_cfg['op_extend'])
    )

    _unit_sanity_warning(
        sim = sim,
        cash_unit_mult = UNIT_MULT,
        market_cap = None
    )

    if 'ebit' in sim and 'ebitda' in sim:

        da_vec = sim.get('da', np.zeros_like(sim['ebit']))

        da_safe = np.maximum(da_vec, 0.0)

        ebit_ceiling = sim['ebitda'] - da_safe

        sim['ebit'] = np.minimum(sim['ebit'], ebit_ceiling)

    terminal_u = ctx.rng(rng_cfg['terminal_u']).random(N_SIMS)

    return CashflowContext(
        ticker_label = ticker_label,
        policy = policy, 
        run_ctx = ctx, 
        fcf_periods = pd.DatetimeIndex(fcf_periods), 
        fcf_period_types = pt_global, 
        annual_pos = annual_pos, 
        years_frac = years_frac, 
        hist_panel = hist_panel, 
        sim = sim, 
        sim_components_full = sim_components_full, 
        fcf_draws = fcf_draws, 
        nd_draws = nd_draws, 
        nd_by_period = nd_by_period, 
        nd_val = float(nd_val) if nd_val is not None and np.isfinite(nd_val) else None, 
        native_last = native_last, 
        fy_m = fy_m, 
        fy_d = fy_d, 
        fcf_is_stub = fcf_is_stub, 
        _period_idx_cache = period_idx_cache, 
        _aligned_cache = aligned_cache, 
        _draws_cache = draws_cache, 
        fcf_future = fcf_future, 
        net_debt_future = net_debt_future, 
        driver_futures = dict(driver_futures), 
        src_a = dict(src_a_local), 
        seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4, 
        tax_is_percent = tax_is_percent, 
        terminal_u = terminal_u
    )


def _build_shared_cashflow_terms(
    *,
    sim_use: dict[str, np.ndarray],
    policy: SectorPolicy
) -> dict[str, np.ndarray]:
    """
    Pre-compute reusable cashflow algebra terms from a selected driver dictionary.

    Both FCFF and FCFE engines construct cashflows by combining a small set of recurring derived
    quantities (for example, "interest after tax" and "CapEx minus depreciation"). Computing these
    once per method evaluation:
  
    - reduces duplicated arithmetic inside formula branches,
  
    - ensures sign conventions are applied consistently across methods, and
  
    - improves readability of method definitions by factoring out common sub-expressions.

    Parameters
    ----------
    sim_use:
        Mapping from driver key to a 2D array of simulated values with shape (T, N_SIMS), where T is
        the number of periods used by the current method evaluation pass. The mapping typically
        contains a subset of the full simulation dictionary sliced to the method's period indices.
    policy:
        Sector policy providing bounds for tax-rate clipping.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary of derived arrays, each of shape (T, N_SIMS):
  
        - "tax_rate": clipped to [policy.tax_lo, policy.tax_hi]
  
        - "one_minus_tax": 1 - tax_rate
  
        - "interest_after_tax": interest * (1 - tax_rate)
  
        - "capex_minus_depr": capex - da
  
        - "delta_wc_outflow": -dnwc
  
        - "ebit_proxy": ebitda - da
  
        - "tax_amt_from_ebt_proxy": max(ebit_proxy - interest, 0) * tax_rate

    Notes
    -----
    Working capital sign convention
        The historical/future driver "dnwc" is treated as a cashflow add-back term in the FCFF
        bridges in this module. That is, dnwc is typically negative when working capital consumes
        cash. To express "working capital outflow" as a positive quantity, this helper defines:

        - delta_wc_outflow = -dnwc

        Using delta_wc_outflow in formula text makes the cash direction explicit.

    Depreciation proxy
        The driver "da" is treated as a combined depreciation and amortisation proxy and is used as
        the "Depreciation" term in CapEx minus depreciation and in EBITDA-to-EBIT conversions.
   
    """
   
    first_arr = None

    for arr in sim_use.values():
     
        arr_np = np.asarray(arr, dtype = float)

        if arr_np.ndim == 2:

            first_arr = arr_np

            break

    if first_arr is None:

        raise ValueError('sim_use must contain at least one 2D array.')


    def _arr(
        key: str,
        default: np.ndarray
    ) -> np.ndarray:
        """
        Fetch a driver array from `sim_use`, falling back to a caller-provided default.

        Parameters
        ----------
        key:
            Driver name to retrieve.
        default:
            Array used when `key` is absent. This is typically a zeros array of the correct shape.

        Returns
        -------
        numpy.ndarray
            A float64 array suitable for vectorised arithmetic.
     
        """
     
        if key not in sim_use:

            return default

        return np.asarray(sim_use[key], dtype = float)


    zeros = np.zeros_like(first_arr, dtype = float)

    tax_rate = np.clip(_arr(
        key = 'tax',
        default = zeros
    ), policy.tax_lo, policy.tax_hi)

    one_minus_tax = 1.0 - tax_rate

    interest = _arr(
        key = 'interest',
        default = zeros
    )

    da = _arr(
        key = 'da',
        default = zeros
    )

    capex = _arr(
        key = 'capex',
        default = zeros
    )

    dnwc = _arr(
        key = 'dnwc',
        default = zeros
    )

    ebit = _arr(
        key = 'ebit',
        default = _arr(
            key = 'ebitda',
            default = zeros
        ) - da
    )

    ebitda = _arr(
        key = 'ebitda',
        default = ebit + da
    )

    capex_minus_depr = capex - da

    delta_wc_outflow = -dnwc

    ebit_proxy = ebitda - da

    tax_amt_from_ebt_proxy = np.maximum(ebit_proxy - interest, 0.0) * tax_rate

    interest_after_tax = interest * one_minus_tax

    return {
        'tax_rate': tax_rate, 
        'one_minus_tax': one_minus_tax, 
        'interest_after_tax': interest_after_tax, 
        'capex_minus_depr': capex_minus_depr, 
        'delta_wc_outflow': delta_wc_outflow, 
        'ebit_proxy': ebit_proxy, 
        'tax_amt_from_ebt_proxy': tax_amt_from_ebt_proxy
    }


def _evaluate_cashflow_methods(
    *,
    context: CashflowContext,
    method_defs: Sequence[tuple[str, set[str], str]],
    required_keys: Callable[[tuple[str, set[str], str]], set[str]] | None,
    formula_dispatch: dict[str, Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray]],
    discount_rate: float,
    g_draw: np.ndarray,
    shares_outstanding: float,
    last_price: float,
    lb: float,
    ub: float,
    valuation_mode: str,
    net_debt_use: str = 'first',
    sim_data: dict[str, np.ndarray] | None = None,
    fcf_draws: np.ndarray | None = None,
    nd_by_period: np.ndarray | None = None
) -> dict[str, float | list[str]]:
    """
    Evaluate multiple cashflow formula variants and return a pooled Monte Carlo valuation result.

    This function is the core "method evaluator" shared by FCFF and FCFE engines. Its role is to
    take:
 
    - a prepared simulation context (period grid, simulated drivers, net debt paths, caches),
 
    - a list of method definitions describing alternative cashflow formula variants, and
 
    - a mapping from formula keys to vectorised cashflow construction callbacks,
    then produce a single per-share valuation distribution by pooling the per-method results.

    High-level algorithm
    --------------------
    For each method definition:
 
    1) Determine whether a quarterly override is feasible for the required drivers, and select the
       evaluation period indices accordingly.
 
    2) Assemble the required driver arrays for those periods, using cached alignment and simulation
       when quarterly series are needed.
 
    3) Compute shared algebra terms (tax, interest after tax, CapEx minus depreciation, working
       capital outflow).
 
    4) Construct the cashflow matrix CF[t, i] using the method's formula callback.
 
    5) Discount cashflows and compute terminal value using a perpetuity growth model.
 
    6) Convert enterprise value to equity value when requested (net debt subtraction).
 
    7) Convert to per-share values and add to the pooled mixture distribution.

    Discounting and terminal value
    ------------------------------
    Discount factors are constructed using discrete compounding on a continuous time axis:

    - DF_t = 1 / exp( years_frac_t * log(1 + r) )

    where `r` is `discount_rate` and `years_frac_t` is the time from valuation date to period end in
    fractional years. The present value of a cashflow path is:

    - PV = sum_t DF_t * CF_t

    Terminal value uses a discrete-compounding perpetuity formula with time step dt between the last
    two evaluated periods:

    - TV_T = CF_T * (1 + g)^dt / ( (1 + r)^dt - (1 + g)^dt )

    and is discounted back as `TV_T * DF_T`.

    Valuation modes
    ---------------
 
    - valuation_mode == "equity":
 
        The discounted cashflow stream is treated as an equity cashflow (FCFE); no net debt
        adjustment is applied.
 
    - valuation_mode == "enterprise":
 
        The discounted cashflow stream is treated as an unlevered cashflow (FCFF). Equity value is
        obtained by subtracting net debt. If `context.nd_val` is available, it is used as a scalar.
        Otherwise, net debt is taken from the simulated net debt path at either the first or last
        evaluated period, controlled by `net_debt_use`.

    Pooling across methods
    ----------------------
    Each method produces a vector of per-share values across simulations. These vectors are
    concatenated and treated as a single mixture distribution. This design:
 
    - reflects model uncertainty in the cashflow construction itself, and
 
    - provides graceful degradation when only a subset of methods is feasible for a ticker.

    Parameters
    ----------
    context:
        Shared per-ticker `CashflowContext` built by `_prepare_cashflow_context(...)`.
    method_defs:
        Sequence of method tuples `(name, required_set, formula_key)`. `required_set` is interpreted
        by `required_keys` (or by default the second tuple element).
    required_keys:
        Optional accessor that extracts the required driver set from a method definition tuple.
        When `None`, `item[1]` is used.
    formula_dispatch:
        Mapping from `formula_key` to a callback:

        - cashflow = callback(sim_selected, terms)

        where `sim_selected` contains the required driver arrays sliced to the selected period
        indices and `terms` is the output of `_build_shared_cashflow_terms(...)`.
    discount_rate:
        Discount rate applied to the cashflow stream. For FCFF this is typically WACC; for FCFE this
        is typically COE.
    g_draw:
        Terminal-growth draw vector of length N_SIMS. Elements are selected using the finite-mask of
        the cashflow matrix for each method.
    shares_outstanding:
        Share count used for enterprise/equity to per-share conversion.
    last_price:
        Observed last price used to compute implied returns: return = price_model / last_price - 1.
    lb, ub:
        Per-share value clipping bounds used to limit the influence of extreme tails.
    valuation_mode:
        Either "enterprise" or "equity".
    net_debt_use:
        When `valuation_mode == "enterprise"` and no scalar net debt estimate exists, selects
        whether simulated net debt is taken from the first or last evaluated period.
    sim_data, fcf_draws, nd_by_period:
        Optional overrides allowing evaluation against modified simulation dictionaries (for example,
        primitive-derived FCFF) without mutating the underlying context.

    Returns
    -------
    dict[str, float | list[str]]
        Mapping containing per-share and return summary statistics and a list of method labels used.

    Advantages of this refactor
    ---------------------------
    The evaluator consolidates logic that was historically duplicated across FCFF and FCFE engines:
    alignment decisions, quarterly/annual fallback, discounting, terminal value computation, net debt
    handling, and result pooling. This:
  
    - reduces time complexity by reusing caches and precomputed arrays, and
  
    - improves numerical consistency by ensuring all methods share the same valuation arithmetic.
  
    """
  
    if valuation_mode not in {'enterprise', 'equity'}:

        raise ValueError("valuation_mode must be 'enterprise' or 'equity'")

    sim_use_base = context.sim if sim_data is None else sim_data

    fcf_draws_use = context.fcf_draws if fcf_draws is None else fcf_draws

    nd_by_period_use = context.nd_by_period if nd_by_period is None else nd_by_period

    req_getter = required_keys if required_keys is not None else lambda item: item[1]

    rng_owner = context.run_ctx if isinstance(getattr(context, 'run_ctx', None), RunContext) else RunContext(seed = SEED, ticker = context.ticker_label)


    def _method_period_idx(
        required: set[str]
    ) -> tuple[np.ndarray, list[str], bool]:
        """
        Select period indices and local period types for a method, with caching.

        A method may be evaluated either:
     
        - on an annual-only grid (more stable when quarterly information is sparse), or
     
        - on a mixed grid including quarterly stubs (higher temporal resolution when quarterly
          forecasts are available for the required drivers).

        The selection is delegated to `_build_quarterly_override_period_types(...)`, which inspects
        the availability of quarterly forecasts in the required future tables. The result is cached
        on the `CashflowContext` keyed by the required driver set.

        Parameters
        ----------
        required:
            Driver-key set required by the method.

        Returns
        -------
        tuple[numpy.ndarray, list[str], bool]
            `(period_idx, pt_local, allow_quarterly)` where:
     
            - period_idx are integer positions into `context.fcf_periods`,
     
            - pt_local is the period-type vector aligned to `context.fcf_periods`, and
     
            - allow_quarterly indicates whether quarterly evaluation is permitted.
     
        """
     
        fk = frozenset(required)

        cached = context._period_idx_cache.get(fk)

        if cached is not None:

            return cached

        required_futs: dict[str, pd.DataFrame | None] = {}

        for k in sorted(required):
            if k == 'fcf':

                required_futs[k] = context.fcf_future
            elif k == 'net_debt':

                required_futs[k] = context.net_debt_future

            else:

                required_futs[k] = context.driver_futures.get(k)

        pt_local, allow, _missing = _build_quarterly_override_period_types(
            periods = context.fcf_periods,
            period_types_global = context.fcf_period_types,
            required_futures = required_futs,
            fy_m = context.fy_m,
            fy_d = context.fy_d
        )

        if allow:

            res = (np.arange(len(context.fcf_periods), dtype = int), [str(x).lower() for x in pt_local], True)

        else:

            res = (context.annual_pos.copy(), list(context.fcf_period_types), False)

        context._period_idx_cache[fk] = res

        return res


    def _simulate_driver_draws_for_period_types(
        key: str,
        dfF: pd.DataFrame,
        pt_local: list[str]
    ) -> np.ndarray:
        """
        Simulate and cache a driver under a method-specific period-type vector.

        This helper mirrors the simulation logic used in `_prepare_cashflow_context(...)` but is
        scoped to the method evaluation stage. It exists because quarterly override decisions are
        method-specific: some methods may require quarterly alignment for certain drivers even when
        other methods are evaluated annually.

        The function:
  
        - aligns `dfF` to `context.fcf_periods` under the requested `pt_local`,
  
        - simulates skewed Student-t draws from consensus rows,
  
        - normalises percentage-like series when they appear to be in 0-100 units,
  
        - clips bounded ratios (tax, gross margin), and
  
        - caches both alignment and draws on the context for reuse across methods.

        Parameters
        ----------
        key:
            Driver key.
        dfF:
            Forecast consensus table for the driver.
        pt_local:
            Period-type vector aligned to `context.fcf_periods` for this method.

        Returns
        -------
        numpy.ndarray
            Simulated array of shape (n_periods, N_SIMS).
    
        """
    
        pt_key = tuple((str(x).lower() for x in pt_local))

        cache_key = (key, pt_key)

        cached = context._draws_cache.get(cache_key)

        if cached is not None:

            return cached

        mode = 'ratio' if key in {'tax', 'gross_margin', 'roe', 'roa', 'roe_pct', 'roa_pct'} else 'stock' if key in {'net_debt'} else 'flow'

        align_key = (id(dfF), pt_key, mode)

        dfA = context._aligned_cache.get(align_key)

        if dfA is None:

            dfA = _future_to_period_aligned(
                dfT = dfF,
                periods = context.fcf_periods,
                period_types = list(pt_key),
                mode = mode,
                fy_m = context.fy_m,
                fy_d = context.fy_d,
                seasonal_flow_weights_q1_q4 = context.seasonal_flow_weights_q1_q4
            )

            context._aligned_cache[align_key] = dfA

        if key in {'capex', 'maint_capex', 'interest', 'da'} and len(dfA.index):

            row0 = dfA.index[0]

            vals = pd.to_numeric(dfA.loc[row0], errors = 'coerce').to_numpy(dtype = float, copy = False)

            med = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan

            if np.isfinite(med) and med < 0:

                dfA = _flip_future_metric_sign(
                    dfT = dfA,
                    value_row = row0
                )

        unit = 1.0 if key in {'tax', 'gross_margin'} else UNIT_MULT

        floor0 = key in {'capex', 'maint_capex', 'da', 'interest'}

        seed_label = f"skewt_q:{key}|pt={','.join(pt_key)}|mode={mode}"

        draws, _, mu_sims, sigma_sims, x_std = _simulate_skewt_from_rows(
            dfT = dfA,
            value_row = dfA.index[0],
            unit_mult = unit,
            floor_at_zero = floor0,
            rng = rng_owner.rng(seed_label),
            return_components = True
        )

        sim_use_base[key] = draws

        context.sim_components_full[key] = (mu_sims, sigma_sims, x_std, floor0)

        if key in {'gross_margin', 'tax', 'roe', 'roe_pct', 'roa', 'roa_pct'}:

            med0 = np.nanmedian(draws) if np.isfinite(draws).any() else 0.0

            if abs(med0) > 1.5:

                draws = draws / 100.0

        if key == 'tax':

            if context.tax_is_percent:

                med0 = np.nanmedian(draws) if np.isfinite(draws).any() else 0.0

                if abs(med0) > 1.5:

                    draws = draws / 100.0

            draws = np.clip(draws, 0.0, 0.4)

        if key == 'gross_margin':

            draws = np.clip(draws, 0.0, 1.0)

        context._draws_cache[cache_key] = draws

        return draws


    def _annual_arrays_for(
        required: set[str]
    ) -> dict[str, np.ndarray]:
        """
        Retrieve required driver arrays for annual evaluation.

        For annual evaluation the engine operates on the subset of periods indexed by
        `context.annual_pos`. Drivers are taken directly from the base simulation dictionary (or from
        `fcf_draws_use` for the special key "fcf").

        Parameters
        ----------
        required:
            Required driver keys.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from key to 2D array (n_periods_all, N_SIMS). An empty dict indicates that at
            least one required key is unavailable.
     
        """
     
        out: dict[str, np.ndarray] = {}

        for k in required:
            
            if k == 'fcf':

                if fcf_draws_use is None:

                    return {}

                out['fcf'] = fcf_draws_use

            else:

                a = sim_use_base.get(k, None)

                if a is None:

                    return {}

                out[k] = a

        return out


    def _quarterly_arrays_for(
        required: set[str],
        pt_local: list[str]
    ) -> dict[str, np.ndarray]:
        """
        Retrieve required driver arrays for quarterly (mixed-grid) evaluation.

        Quarterly evaluation requires that each requested driver be represented on the full mixed
        period grid `context.fcf_periods`. When a forecast table exists for a driver, the driver is
        re-aligned and re-simulated under `pt_local` using `_simulate_driver_draws_for_period_types`.
        When no suitable future table exists, an existing simulated array is copied from
        `sim_use_base`.

        A key complication arises when a driver is only available as an annual series but the method
        is being evaluated on a mixed grid. In such cases, the annual total is typically stored on
        the fiscal year-end quarter. The procedure below adjusts the fiscal year-end quarter so that:

        - annual_total = sum(quarter_1, quarter_2, quarter_3, quarter_4)

        This avoids double counting and ensures that the quarterly path aggregates coherently to the
        annual total implied by the annual simulation.

        Parameters
        ----------
        required:
            Required driver keys.
        pt_local:
            Period-type vector aligned to `context.fcf_periods`.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from key to 2D array (n_periods_all, N_SIMS). An empty dict indicates that at
            least one required key is unavailable.
     
        """
     
        out: dict[str, np.ndarray] = {}

        from_sim: set[str] = set()


        def _mode_for_key(
            k: str
        ) -> str:
            """
            Classify a driver as a "flow", "stock", or "ratio" for alignment and adjustment logic.

            Returns
            -------
            str
                "ratio" for bounded rates (tax, margins, ROE/ROA), "stock" for level series such as
                net debt, and "flow" otherwise.
     
            """
     
            return 'ratio' if k in {'tax', 'gross_margin', 'roe', 'roa', 'roe_pct', 'roa_pct'} else 'stock' if k in {'net_debt'} else 'flow'


        for k in sorted(required):
          
            if k == 'fcf':

                dfF = context.fcf_future
         
            elif k == 'net_debt':

                dfF = context.net_debt_future

            else:

                dfF = context.driver_futures.get(k)

            if dfF is not None and (not dfF.empty):

                out[k] = _simulate_driver_draws_for_period_types(
                    key = k,
                    dfF = dfF,
                    pt_local = pt_local
                )

                continue

            a = sim_use_base.get(k)

            if a is None:

                return {}

            out[k] = np.asarray(a, dtype = float).copy()

            from_sim.add(k)

        if from_sim:

            fy_end_idx = [i for i, p in enumerate(context.fcf_periods) if pt_local[i] == 'quarterly' and p.month == context.fy_m and (p.day == context.fy_d)]

            if fy_end_idx:

                fy_end_for = np.array([_fiscal_year_end_for_date(
                    d = p,
                    fy_m = context.fy_m,
                    fy_d = context.fy_d
                ) for p in context.fcf_periods], dtype = 'datetime64[ns]')

                for k in list(from_sim):
              
                    if _mode_for_key(
                        k = k
                    ) != 'flow':

                        continue

                    a = out[k]

                    for i in fy_end_idx:
                        fy = np.datetime64(context.fcf_periods[i])

                        q_idx = [j for j in range(len(context.fcf_periods)) if pt_local[j] == 'quarterly' and fy_end_for[j] == fy]

                        q_others = [j for j in q_idx if j != i]

                        if not q_others:

                            continue

                        qsum = np.nansum(a[q_others, :], axis = 0)

                        total = a[i, :]

                        total2 = np.where(np.isfinite(total), total, qsum)

                        a[i, :] = total2 - qsum

                    out[k] = a

        return out


    pooled_per_share: list[np.ndarray] = []

    methods_used: list[str] = []

    full_df = 1.0 / np.exp(context.years_frac * np.log1p(discount_rate))

    for method_item in method_defs:
     
        name, _req_unused, formula_key = method_item

        req = req_getter(method_item)

        period_idx, pt_local, allow_q = _method_period_idx(
            required = req
        )

        sim_use = {}

        used_quarterly = False

        if allow_q:

            sim_use = _quarterly_arrays_for(
                required = req,
                pt_local = pt_local
            )

            if sim_use and all((k in sim_use for k in req)):

                bad = False

                for k in req:
                    a = np.asarray(sim_use[k], dtype = float)

                    if a.shape[0] != len(context.fcf_periods):

                        bad = True

                        break

                    if np.isnan(a[period_idx, :]).any():

                        bad = True

                        break

                if not bad:

                    used_quarterly = True

        if not used_quarterly:

            period_idx = context.annual_pos.copy()

            sim_use = _annual_arrays_for(
                required = req
            )

            if not sim_use or any((k not in sim_use for k in req)):

                continue

        sim_selected = {k: np.asarray(v, dtype = float)[period_idx, :] for k, v in sim_use.items()}

        terms = _build_shared_cashflow_terms(
            sim_use = sim_selected,
            policy = context.policy
        )

        formula_fn = formula_dispatch.get(formula_key)

        if formula_fn is None:

            continue

        cashflow = np.asarray(formula_fn(sim_selected, terms), dtype = float)

        if cashflow.ndim != 2 or cashflow.shape[0] != len(period_idx):

            continue

        mask = np.all(np.isfinite(cashflow), axis = 0)

        if mask.sum() < 50:

            continue

        df_m = full_df[period_idx]

        years_m = context.years_frac[period_idx]

        dt_last_m = float(years_m[-1] - years_m[-2]) if len(years_m) >= 2 else 1.0

        pv = df_m @ cashflow[:, mask]

        cf_T = cashflow[-1, mask]

        tv = _terminal_value_perpetuity(
            cf_T = cf_T,
            r = discount_rate,
            g = g_draw[mask],
            dt_years = dt_last_m
        )

        base_val = pv + tv * df_m[-1]

        m_val = np.isfinite(base_val)

        if m_val.sum() < 50:

            continue

        base_val = base_val[m_val]

        if valuation_mode == 'enterprise':

            if context.nd_val is not None and np.isfinite(context.nd_val):

                eq = base_val - float(context.nd_val)

            else:

                nd_idx = int(period_idx[0]) if net_debt_use == 'first' else int(period_idx[-1])

                nd_paths = np.asarray(nd_by_period_use[nd_idx, :], float)

                nd_sel = nd_paths[mask]

                nd_sel = nd_sel[m_val]

                eq = base_val - nd_sel

        else:

            eq = base_val

        ps = eq / shares_outstanding

        pooled_per_share.append(ps)

        methods_used.append(name)

    if len(pooled_per_share) == 0:

        return _zero_cashflow_result()

    pooled = np.concatenate(pooled_per_share)

    pooled = np.clip(pooled, lb, ub)

    rets = pooled / last_price - 1.0

    return {
        'per_share_mean': float(np.mean(pooled)), 
        'per_share_median': float(np.median(pooled)),
        'per_share_p05': float(np.percentile(pooled, 5)),
        'per_share_p95': float(np.percentile(pooled, 95)),
        'per_share_std': float(np.std(pooled)), 
        'returns_mean': float(np.mean(rets)), 
        'returns_median': float(np.median(rets)), 
        'returns_p05': float(np.percentile(rets, 5)),
        'returns_p95': float(np.percentile(rets, 95)),
        'returns_std': float(np.std(rets)), 
        'methods_used': sorted(set(methods_used))
    }


def mc_equity_value_per_share_multi_fcff(
    *,
    ctx: RunContext | None = None,
    ticker: str | None = None,
    sector_label: str | None = None,
    sector_policy: SectorPolicy | None = None,
    fcf_future: pd.DataFrame | None,
    net_debt_future: pd.DataFrame | None,
    driver_futures: dict[str, pd.DataFrame | None],
    wacc: float,
    cost_of_debt: float | None = None,
    shares_outstanding: float,
    last_price: float,
    tax_is_percent: bool = True,
    g_mu: float = config.RF,
    g_sd: float = 0.01,
    net_debt_use: str = 'first',
    hist_inc: pd.DataFrame,
    hist_cf: pd.DataFrame,
    hist_bal: pd.DataFrame,
    hist_ratios: pd.DataFrame,
    ub: float,
    lb: float,
    g_cap: float,
    src_a: dict[str, pd.DatetimeIndex],
    seasonal_flow_weights_q1_q4: np.ndarray | None = None,
    cashflow_context: CashflowContext | None = None
):
    """
    Monte Carlo FCFF discounted cashflow valuation (enterprise value, converted to equity per share).

    This engine estimates equity value per share by:
 
    1) simulating future free cash flows to the firm (FCFF) under several alternative cashflow
       construction methods,
 
    2) discounting those cashflows at WACC,
 
    3) adding a terminal value computed from a perpetuity growth model, and
 
    4) converting enterprise value to equity value by subtracting net debt.

    Multiple cashflow construction methods are evaluated and pooled into a single mixture
    distribution. Method availability is gated by forecast completeness and basic historical
    coherence checks.

    Cashflow methods (formula variants)
    -----------------------------------
    The following FCFF constructions are supported via `formula_dispatch`:

    1) Direct FCFF
 
       - FCFF_t = FCF_t

    2) CFO to FCFF unlevering bridge
 
       - FCFF_t = CFO_t - CapEx_t + Interest_t * (1 - tax_rate_t)

       and a maintenance CapEx variant:
 
       - FCFF_t = CFO_t - MaintCapEx_t + Interest_t * (1 - tax_rate_t)

    3) EBIT bridge
 
       - FCFF_t = EBIT_t * (1 - tax_rate_t) + Depreciation_t - CapEx_t + dNWC_t

    4) EBITDA bridge (implemented via EBIT proxy)
 
       - FCFF_t = (EBITDA_t - Depreciation_t) * (1 - tax_rate_t) + Depreciation_t - CapEx_t + dNWC_t

    5) Net income bridge (re-levered to unlevered)
 
       - FCFF_t = NetIncome_t + Depreciation_t + Interest_t * (1 - tax_rate_t) - CapEx_t + dNWC_t

    Working capital sign convention
 
        The driver "dnwc" follows the internal convention that a working-capital investment is
        typically negative (cash outflow). Accordingly, the bridges add dnwc:
 
        - adding a negative number reduces FCFF, consistent with an outflow.

    Valuation mathematics
    ---------------------
    Let DF_t denote the discount factor at time t in fractional years. With discount rate r = WACC:

    - DF_t = 1 / exp( years_frac_t * log(1 + r) )
 
    - PV = sum_t DF_t * FCFF_t

    Terminal value uses a discrete-compounding perpetuity:

    - TV_T = FCFF_T * (1 + g)^dt / ( (1 + r)^dt - (1 + g)^dt )

    where dt is the time step between the last two evaluation dates. Enterprise value is:

    - EV = PV + DF_T * TV_T

    Equity value is obtained by subtracting net debt:

    - EquityValue = EV - NetDebt

    and per-share value is EquityValue / shares_outstanding.

    Parameters
    ----------
    ctx, ticker, sector_label, sector_policy:
        Control deterministic simulation and sector-specific method gating.
    fcf_future, net_debt_future, driver_futures:
        Forecast tables for FCFF, net debt, and supporting drivers.
    wacc:
        Weighted average cost of capital used for discounting.
    cost_of_debt:
        Optional cost of debt estimate used when net debt at valuation date is estimated from history.
    shares_outstanding, last_price:
        Per-share conversion and return computation.
    tax_is_percent:
        Controls normalisation of the tax-rate driver when it appears to be quoted as 0-100.
    g_mu, g_sd, g_cap:
        Parameters for the terminal growth distribution. The growth draw is truncated to lie between
        `FLOOR` and `g_cap`, and callers are expected to set `g_cap <= wacc - SAFETY_SPREAD` to avoid
        singular terminal values.
    net_debt_use:
        If a scalar net debt estimate is unavailable, selects whether simulated net debt is taken
        from the first or last evaluation period.
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical tables used to build the shared context (simulation, imputation, dependence).
    ub, lb:
        Per-share clipping bounds.
    src_a:
        Mapping of driver keys to their native annual periods, used for operating-profit extension.
    seasonal_flow_weights_q1_q4:
        Seasonal allocation weights used when building quarterly stubs from annual flows.
    cashflow_context:
        Optional precomputed `CashflowContext`. When provided, the expensive preparation pipeline is
        reused across FCFF and FCFE engines.

    Returns
    -------
    dict[str, float | list[str]]
        Summary statistics for per-share values and returns, plus the set of methods used.

    Advantages of the approach
    --------------------------
    - Multiple method variants reduce sensitivity to any single accounting bridge and provide
      graceful degradation when forecasts are incomplete.
 
    - Skewed Student-t simulation captures asymmetric and heavy-tailed uncertainty typical of
      consensus forecasts.
 
    - Optional dependence calibration injects realistic co-movement across drivers without forcing a
      fully parametric multivariate forecast model.
 
    """
 
    ctx = _ensure_ctx(
        ctx = ctx
    )

    policy = sector_policy if sector_policy is not None else _policy_for_sector(
        sector_label = sector_label
    )

    if cashflow_context is None:

        try:

            cashflow_context = _prepare_cashflow_context(
                ctx = ctx,
                ticker = ticker,
                sector_label = sector_label,
                sector_policy = policy,
                fcf_future = fcf_future,
                net_debt_future = net_debt_future,
                driver_futures = driver_futures,
                hist_inc = hist_inc,
                hist_cf = hist_cf,
                hist_bal = hist_bal,
                hist_ratios = hist_ratios,
                src_a = src_a,
                tax_is_percent = tax_is_percent,
                net_debt_use = net_debt_use,
                seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4,
                cost_of_debt = cost_of_debt,
                rng_labels = {
                    'dnwc': 'dnwc',
                    'copula': 'copula:fcff', 
                    'op_extend': 'op_extend', 
                    'terminal_u': 'terminal_g:fcff_u'
                }
            )

        except (TypeError, ValueError):

            warnings.warn('FCFF: no future valuation periods after filtering; returning zeros.')

            return _zero_cashflow_result()

    sim_eval = cashflow_context.sim

    nd_eval = cashflow_context.nd_by_period

    fcf_eval = cashflow_context.fcf_draws

    fcf_is_stub = cashflow_context.fcf_is_stub

    primitives_used = False

    if USE_PRIMITIVE_FCFF:

        sim_eval = {k: np.asarray(v, dtype = float).copy() for k, v in cashflow_context.sim.items()}

        nd_eval = np.asarray(cashflow_context.nd_by_period, dtype = float).copy()

        sim_eval, nd_eval, primitives_used = _derive_fcff_from_primitives(
            sim = sim_eval,
            hist_annual = cashflow_context.hist_panel,
            fcf_periods = cashflow_context.fcf_periods,
            nd_draws = nd_eval,
            ctx = ctx,
            sector_policy = cashflow_context.policy
        )

        if primitives_used:

            fcf_eval = sim_eval.get('fcf', fcf_eval)

            fcf_is_stub = False

    has_interest_debt_coherence, has_wc_coherence = _coherence_flags_from_history_panel(
        hist_panel = cashflow_context.hist_panel,
        policy = cashflow_context.policy
    )

    if primitives_used:

        method_defs: list[tuple[str, set[str], str]] = [('Primitives_FCFF', {'fcf'}, 'fcf')]

    else:

        method_defs = []

        for name, req in _build_fcff_method_defs(
            policy = cashflow_context.policy,
            fcf_is_stub = fcf_is_stub,
            has_interest_debt_coherence = has_interest_debt_coherence,
            has_wc_coherence = has_wc_coherence
        ):
         
            formula_key = _fcff_formula_key_from_required(
                required = req
            )

            if formula_key is not None:

                method_defs.append((name, req, formula_key))

    if cashflow_context.terminal_u is not None:

        g_draw = _simulate_terminal_g_draws_from_u(
            u = cashflow_context.terminal_u,
            g_mu = g_mu,
            g_sd = g_sd,
            g_cap = g_cap
        )

    else:

        g_draw = _simulate_terminal_g_draws(
            g_mu = g_mu,
            g_sd = g_sd,
            g_cap = g_cap,
            rng = ctx.rng('terminal_g:fcff')
        )

    formula_dispatch: dict[str, Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray]] = {
        'fcf': lambda s, t: s['fcf'], 
        'cfo_capex': lambda s, t: s['cfo'] - s['capex'] + t['interest_after_tax'],
        'cfo_maint': lambda s, t: s['cfo'] - s['maint_capex'] + t['interest_after_tax'], 
        'ebit': lambda s, t: s['ebit'] * t['one_minus_tax'] + s['da'] - s['capex'] + s['dnwc'], 
        'ebitda': lambda s, t: (s['ebitda'] - s['da']) * t['one_minus_tax'] + s['da'] - s['capex'] + s['dnwc'], 
        'ni': lambda s, t: s['net_income'] + s['da'] + t['interest_after_tax'] - s['capex'] + s['dnwc']
    }

    return _evaluate_cashflow_methods(
        context = cashflow_context,
        method_defs = method_defs,
        required_keys = lambda item: item[1],
        formula_dispatch = formula_dispatch,
        discount_rate = wacc,
        g_draw = g_draw,
        shares_outstanding = shares_outstanding,
        last_price = last_price,
        lb = lb,
        ub = ub,
        valuation_mode = 'enterprise',
        net_debt_use = net_debt_use,
        sim_data = sim_eval,
        fcf_draws = fcf_eval,
        nd_by_period = nd_eval
    )


def mc_equity_value_per_share_multi_fcfe(
    *,
    ctx: RunContext | None = None,
    ticker: str | None = None,
    sector_label: str | None = None,
    sector_policy: SectorPolicy | None = None,
    fcf_future: pd.DataFrame | None,
    net_debt_future: pd.DataFrame | None,
    driver_futures: dict[str, pd.DataFrame | None],
    coe: float,
    debt_ratio: float,
    cost_of_debt: float | None = None,
    shares_outstanding: float,
    last_price: float,
    tax_is_percent: bool = True,
    g_mu: float = config.RF,
    g_sd: float = 0.01,
    net_debt_use: str = 'first',
    hist_inc: pd.DataFrame,
    hist_cf: pd.DataFrame,
    hist_bal: pd.DataFrame,
    hist_ratios: pd.DataFrame,
    ub: float,
    lb: float,
    g_cap: float,
    src_a: dict[str, pd.DatetimeIndex],
    seasonal_flow_weights_q1_q4: np.ndarray | None = None,
    cashflow_context: CashflowContext | None = None
):
    """
    Monte Carlo FCFE discounted cashflow valuation (equity value per share).

    This engine values equity directly by discounting free cash flow to equity (FCFE) at the cost of
    equity (COE). Unlike FCFF valuation, the output is an equity value per share without a net debt
    subtraction step.

    The FCFE series is constructed using DR-adjusted bridges that incorporate a debt ratio (DR)
    derived from the WACC capital structure weights:

    - DR = D / (E + D)

    In the surrounding pipeline, E is market capitalisation and D is the debt measure used for WACC
    construction. `debt_ratio` is expected to be this DR (often passed as `wD` from
    `_build_discount_factor_vector(..., return_components=True)`).

    FCFE formula variants
    ---------------------
    The supported variants correspond to the documented DR-based formulations:

    1) Net income bridge
   
       - FCFE = NetIncome - (1 - DR) * (CapEx - Depreciation) - (1 - DR) * DeltaWorkingCapitalOutflow

    2) EBITDA / interest / tax bridge
       
       - FCFE = (EBITDA - Interest - TaxAmount) - (1 - DR) * (CapEx - Depreciation) - (1 - DR) * DeltaWorkingCapitalOutflow

       Here TaxAmount is proxied as:
      
       - TaxAmount = max(EBIT_proxy - Interest, 0) * tax_rate
      
       with EBIT_proxy defined as (EBITDA - Depreciation).

    3) FCFF bridge
    
       - FCFE = FCFF - Interest * (1 - tax_rate) + DR * ( (CapEx - Depreciation) + DeltaWorkingCapitalOutflow )

       In this module, the driver key "fcf" is treated as an FCFF (unlevered free cash flow) series.

    4) EBIT / interest bridge
       
       - FCFE = (EBIT - Interest) * (1 - tax_rate) - (1 - DR) * (CapEx - Depreciation) - (1 - DR) * DeltaWorkingCapitalOutflow

    Working capital convention
        DeltaWorkingCapitalOutflow is defined as:
    
        - DeltaWorkingCapitalOutflow = -dnwc

        This provides a consistent interpretation that positive values represent cash outflows.

    Valuation mathematics
    ---------------------
    Discounting uses COE (r = coe) on the shared period grid:
  
    - DF_t = 1 / exp( years_frac_t * log(1 + r) )
  
    - PV = sum_t DF_t * FCFE_t

    Terminal value:
  
    - TV_T = FCFE_T * (1 + g)^dt / ( (1 + r)^dt - (1 + g)^dt )
  
    - EquityValue = PV + DF_T * TV_T

    Parameters
    ----------
    ctx, ticker, sector_label, sector_policy:
        Deterministic simulation context and method gating policy.
    fcf_future, net_debt_future, driver_futures:
        Forecast tables. Net debt is not subtracted in equity mode, but net debt series may be used
        during driver imputation and dependence modelling.
    coe:
        Cost of equity used for discounting.
    debt_ratio:
        Debt ratio DR used in DR-adjusted FCFE bridges. Values are clipped into [0, 0.99].
    cost_of_debt:
        Optional cost of debt estimate used in context preparation.
    shares_outstanding, last_price:
        Per-share conversion and return computation.
    tax_is_percent:
        Controls normalisation of tax-rate forecasts when they appear to be in 0-100 units.
    g_mu, g_sd, g_cap:
        Terminal growth distribution parameters (truncated to `g_cap`).
    net_debt_use:
        Passed through to context preparation for consistency; not used in equity valuation mode.
    hist_inc, hist_cf, hist_bal, hist_ratios:
        Historical tables used in the shared preparation pipeline.
    ub, lb:
        Per-share clipping bounds.
    src_a:
        Mapping of driver keys to their native annual periods, used for operating-profit extension.
    seasonal_flow_weights_q1_q4:
        Seasonal allocation weights used when building quarterly stubs from annual flows.
    cashflow_context:
        Optional precomputed `CashflowContext` reused across FCFF and FCFE engines.

    Returns
    -------
    dict[str, float | list[str]]
        Summary statistics for per-share values and implied returns, plus a list of methods used.

    Advantages of DR-adjusted FCFE
    ------------------------------
    - Provides an equity-anchored valuation that avoids explicit net debt subtraction.
  
    - The DR adjustment approximates financing of reinvestment through debt issuance in proportion to
      the observed capital structure, reducing sensitivity to noisy net borrowing forecasts.
  
    - Multiple bridge variants offer robustness when a subset of drivers (for example, net income vs
      EBITDA) is available.
 
    """
 
    ctx = _ensure_ctx(
        ctx = ctx
    )

    ticker_label = str(ticker or ctx.ticker or 'UNKNOWN')

    policy = sector_policy if sector_policy is not None else _policy_for_sector(
        sector_label = sector_label
    )

    dr_scalar = float(np.clip(np.nan_to_num(debt_ratio, nan = 0.0), 0.0, 0.99))

    one_minus_dr_scalar = 1.0 - dr_scalar

    if cashflow_context is None:

        try:

            cashflow_context = _prepare_cashflow_context(
                ctx = ctx,
                ticker = ticker,
                sector_label = sector_label,
                sector_policy = policy,
                fcf_future = fcf_future,
                net_debt_future = net_debt_future,
                driver_futures = driver_futures,
                hist_inc = hist_inc,
                hist_cf = hist_cf,
                hist_bal = hist_bal,
                hist_ratios = hist_ratios,
                src_a = src_a,
                tax_is_percent = tax_is_percent,
                net_debt_use = net_debt_use,
                seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4,
                cost_of_debt = cost_of_debt,
                rng_labels = {'dnwc': 'dnwc_fcfe', 'copula': 'copula:fcfe', 'op_extend': 'op_extend_fcfe', 'terminal_u': 'terminal_g:fcfe_u'}
            )

        except (TypeError, ValueError):

            warnings.warn('FCFE: no future valuation periods after filtering; returning zeros.')

            return _zero_cashflow_result()

    has_interest_debt_coherence, has_wc_coherence = _coherence_flags_from_history_panel(
        hist_panel = cashflow_context.hist_panel,
        policy = cashflow_context.policy
    )

    method_defs, skipped_methods = _build_fcfe_method_defs_dr(
        policy = cashflow_context.policy,
        fcf_is_stub = cashflow_context.fcf_is_stub,
        has_interest_debt_coherence = has_interest_debt_coherence,
        has_wc_coherence = has_wc_coherence
    )

    if skipped_methods:

        logger.debug('[FCFE-METHODS] %s skipped=%s', ticker_label, skipped_methods)

    if cashflow_context.terminal_u is not None:

        g_draw = _simulate_terminal_g_draws_from_u(
            u = cashflow_context.terminal_u,
            g_mu = g_mu,
            g_sd = g_sd,
            g_cap = g_cap
        )

    else:

        g_draw = _simulate_terminal_g_draws(
            g_mu = g_mu,
            g_sd = g_sd,
            g_cap = g_cap,
            rng = ctx.rng('terminal_g:fcfe')
        )

    formula_dispatch: dict[str, Callable[[dict[str, np.ndarray], dict[str, np.ndarray]], np.ndarray]] = {
        'ni_dr': lambda s, t: s['net_income'] - one_minus_dr_scalar * t['capex_minus_depr'] - one_minus_dr_scalar * t['delta_wc_outflow'],  
        'ebitda_int_tax_dr': lambda s, t: s['ebitda'] - s['interest'] - t['tax_amt_from_ebt_proxy'] - one_minus_dr_scalar * t['capex_minus_depr'] - one_minus_dr_scalar * t['delta_wc_outflow'],  
        'fcff_bridge_dr': lambda s, t: s['fcf'] - t['interest_after_tax'] + dr_scalar * (t['capex_minus_depr'] + t['delta_wc_outflow']), 'ebit_int_dr': lambda s, t: (s['ebit'] - s['interest']) * t['one_minus_tax'] - one_minus_dr_scalar * t['capex_minus_depr'] - one_minus_dr_scalar * t['delta_wc_outflow']
    }

    return _evaluate_cashflow_methods(
        context = cashflow_context,
        method_defs = method_defs,
        required_keys = lambda item: item[1],
        formula_dispatch = formula_dispatch,
        discount_rate = coe,
        g_draw = g_draw,
        shares_outstanding = shares_outstanding,
        last_price = last_price,
        lb = lb,
        ub = ub,
        valuation_mode = 'equity',
        net_debt_use = net_debt_use
    )


def _normalize_g_cap_array(
    g_cap: float | np.ndarray,
    n: int
) -> np.ndarray:
    """
    Normalise a terminal-growth cap into a length-n vector.

    Several models treat the terminal growth cap as simulation-dependent (for example, a cap path
    defined as `coe_s - SAFETY_SPREAD` in the DDM engine). This helper converts a scalar cap into a
    constant vector, or validates and passes through an already vectorised cap.

    Parameters
    ----------
    g_cap:
        Scalar cap or an array-like of caps.
    n:
        Required output length.

    Returns
    -------
    numpy.ndarray
        Array of shape (n,) containing per-simulation growth caps.

    Raises
    ------
    ValueError
        If `g_cap` is array-like with a length other than 1 or `n`.
 
    """
 
    g_cap_arr = np.asarray(g_cap, dtype = float)

    if g_cap_arr.ndim == 0:

        return np.full(n, float(g_cap_arr), dtype = float)

    g_cap_arr = g_cap_arr.reshape(-1)

    if g_cap_arr.size == 1:

        return np.full(n, float(g_cap_arr[0]), dtype = float)

    if g_cap_arr.size != n:

        raise ValueError(f'g_cap must be scalar or length {n}, got shape {g_cap_arr.shape}')

    return g_cap_arr.astype(float, copy = False)


def _simulate_terminal_g_draws(
    *,
    g_mu: float,
    g_sd: float,
    g_cap: float | np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate terminal growth rates using a truncated normal model.

    The terminal growth rate `g` is modelled as a normal distribution truncated to:
 
    - lower bound: `FLOOR` (typically 0),
 
    - upper bound: `g_cap` (scalar or per-simulation cap).

    Sampling is performed by generating uniforms `u` and transforming them via the inverse CDF
    (`ppf`) of the truncated normal distribution. This helper draws `u` internally and then delegates
    to `_simulate_terminal_g_draws_from_u(...)`.

    Parameters
    ----------
    g_mu:
        Mean of the (untruncated) normal distribution.
    g_sd:
        Standard deviation of the (untruncated) normal distribution.
    g_cap:
        Upper truncation cap. May be scalar or a length-N_SIMS vector.
    rng:
        NumPy random generator.

    Returns
    -------
    numpy.ndarray
        Simulated terminal-growth vector of length N_SIMS.

    Notes
    -----
    Truncation is essential for numerical stability in the perpetuity terminal value:
   
    - the denominator ( (1 + r)^dt - (1 + g)^dt ) must remain positive, typically enforced by setting
      g_cap <= r - SAFETY_SPREAD.
 
    """
 
    u = rng.random(N_SIMS)

    return _simulate_terminal_g_draws_from_u(
        u = u,
        g_mu = g_mu,
        g_sd = g_sd,
        g_cap = g_cap
    )


def _simulate_terminal_g_draws_from_u(
    *,
    u: np.ndarray,
    g_mu: float,
    g_sd: float,
    g_cap: float | np.ndarray
) -> np.ndarray:
    """
    Transform uniforms into terminal growth draws under a truncated normal specification.

    This function is identical in distribution to `_simulate_terminal_g_draws(...)` but accepts the
    uniform draws `u` as an explicit input. This enables safe reuse of the same random base across
    multiple valuation engines within a ticker, ensuring that terminal-growth randomness is shared
    rather than re-sampled independently.

    Distribution
    ------------
    Let Z ~ Normal(g_mu, g_sd). The terminal growth g is:
 
    - g = Z conditioned on FLOOR <= Z <= g_cap

    Sampling is performed using inverse transform sampling:
   
    - u ~ Uniform(0, 1)
   
    - g = TruncatedNormalPPF(u; lower=FLOOR, upper=g_cap, mean=g_mu, sd=g_sd)

    Parameters
    ----------
    u:
        1D array of uniforms in (0, 1). Values are clipped away from 0 and 1 for numerical stability.
    g_mu, g_sd:
        Mean and standard deviation of the underlying normal.
    g_cap:
        Upper truncation cap. May be scalar or a length-n vector, where n == len(u).

    Returns
    -------
    numpy.ndarray
        Array of terminal-growth draws of length n.

    Notes
    -----
 
    - If `g_sd` is non-positive or non-finite, the output degenerates to `min(max(g_mu, FLOOR), g_cap)`.
 
    - Any non-finite caps are replaced with FLOOR and then floored at FLOOR.
 
    """
 
    u = np.asarray(u, dtype = float)

    if u.ndim != 1:

        raise ValueError('u must be a 1D array')

    n = int(u.size)

    if n == 0:

        return np.array([], dtype = float)

    g_caps = _normalize_g_cap_array(
        g_cap = g_cap,
        n = n
    )

    g_caps = np.where(np.isfinite(g_caps), g_caps, FLOOR)

    g_caps = np.maximum(g_caps, FLOOR)

    g0 = float(np.nan_to_num(g_mu, nan = FLOOR))

    base = np.clip(np.full(n, g0, dtype = float), FLOOR, None)

    out = np.minimum(base, g_caps)

    if not np.isfinite(g_sd) or g_sd <= 0 or (not np.isfinite(g_mu)):

        return out

    u = np.clip(u, 1e-12, 1.0 - 1e-12)

    a = np.full(n, (FLOOR - g_mu) / g_sd, dtype = float)

    b = (g_caps - g_mu) / g_sd

    ok = np.isfinite(a) & np.isfinite(b) & (b > a)

    if not np.any(ok):

        return out

    sampled = truncnorm.ppf(u[ok], a[ok], b[ok], loc = g_mu, scale = g_sd)

    sampled = np.where(np.isfinite(sampled), sampled, out[ok])

    out[ok] = np.minimum(np.clip(sampled, FLOOR, None), g_caps[ok])

    return out


def _bvps_path_from_eps_dps(
    *,
    bvps0: float | np.ndarray,
    eps_draws: np.ndarray,
    dps_draws: np.ndarray,
    floor_bvps_at_zero: bool = False
) -> np.ndarray:
    """
    Build a beginning-of-period book value per share (BVPS) path from EPS and DPS draws.

    The residual income model relies on a book value trajectory. Under the clean surplus relation:

    - BVPS_{t+1} = BVPS_t + EPS_t - DPS_t

    This helper constructs the implied BVPS path given:
 
    - an initial BVPS (scalar or per-simulation vector), and
 
    - simulated EPS and DPS arrays for each period.

    The returned array represents beginning-of-period BVPS, i.e. BVPS_t for each period t, which is
    the appropriate book value for computing the equity charge term:

    - EquityCharge_t = r_t * BVPS_t

    Parameters
    ----------
    bvps0:
        Initial book value per share. May be scalar (broadcast) or a length-N_SIMS vector.
    eps_draws, dps_draws:
        Simulated per-share earnings and dividends with shape (T, N_SIMS).
    floor_bvps_at_zero:
        If `True`, floors BVPS at zero after each update step.

    Returns
    -------
    numpy.ndarray
        Beginning-of-period BVPS array of shape (T, N_SIMS).

    Raises
    ------
    ValueError
        If EPS and DPS shapes differ, or if `bvps0` cannot be broadcast to N_SIMS.
  
    """
  
    eps_draws = np.asarray(eps_draws, dtype = float)

    dps_draws = np.asarray(dps_draws, dtype = float)

    if eps_draws.shape != dps_draws.shape:

        raise ValueError(f'eps_draws shape {eps_draws.shape} != dps_draws shape {dps_draws.shape}')

    T, n = eps_draws.shape

    bv0 = np.asarray(bvps0, dtype = float)

    if bv0.ndim == 0:

        bv0_vec = np.full(n, float(bv0), dtype = float)
   
    elif bv0.shape == (n,):

        bv0_vec = bv0.astype(float, copy = False)

    else:

        raise ValueError(f'bvps0 must be scalar or length n_sims={n}, got shape {bv0.shape}')

    bvps = np.empty((T + 1, n), dtype = float)

    bvps[0, :] = bv0_vec

    if floor_bvps_at_zero:

        bvps[0, :] = np.maximum(bvps[0, :], 0.0)

    for t in range(T):
       
        bvps[t + 1, :] = bvps[t, :] + eps_draws[t, :] - dps_draws[t, :]

        if floor_bvps_at_zero:

            bvps[t + 1, :] = np.maximum(bvps[t + 1, :], 0.0)

    bvps_beg = bvps[:-1, :]

    return bvps_beg


def _future_to_period_aligned_cached(
    *,
    cache: dict[tuple[Any, ...], pd.DataFrame] | None,
    df_future: pd.DataFrame,
    periods: pd.DatetimeIndex | Sequence[pd.Timestamp],
    period_types: Sequence[str],
    fy_m: int,
    fy_d: int,
    mode: str,
    seasonal_flow_weights_q1_q4: np.ndarray | None
) -> pd.DataFrame:
    """
    Align a future consensus table to a mixed period grid, with optional memoisation.

    The DDM and residual income engines repeatedly align the same forecast tables to the same mixed
    period grid when multiple derived quantities are built (for example, DPS used directly and also
    used to infer payout ratios). Alignment can be non-trivial because it may involve:
 
    - selecting annual vs quarterly columns,
 
    - converting annual flows to quarterly stubs using seasonal weights, and
 
    - applying "flow" vs "stock" vs "ratio" aggregation conventions.

    This helper wraps `_future_to_period_aligned(...)` with a cache keyed by object identity and
    period-grid metadata.

    Parameters
    ----------
    cache:
        Mutable dictionary used for memoisation. If `None`, alignment is computed and returned
        directly without caching.
    df_future:
        Future consensus DataFrame.
    periods:
        Period end dates for the valuation grid.
    period_types:
        Period type labels aligned to `periods` (for example, "annual" or "quarterly").
    fy_m, fy_d:
        Fiscal year end month and day used for annual/quarter reconciliation.
    mode:
        Alignment mode: "flow" for additive statement flows, "stock" for levels, and "ratio" for
        bounded rates.
    seasonal_flow_weights_q1_q4:
        Optional seasonal weights for splitting annual flows into quarterly stubs.

    Returns
    -------
    pandas.DataFrame
        Aligned forecast table with columns matching `periods`.

    Notes
    -----
    The cache key uses `id(df_future)` and `id(seasonal_flow_weights_q1_q4)` rather than hashing the
    DataFrame contents. This is appropriate for a single-ticker execution pass where the DataFrame
    objects are stable and avoids expensive hashing of large tables.
  
    """
  
    if cache is None:

        return _future_to_period_aligned(
            dfT = df_future,
            periods = periods,
            period_types = period_types,
            fy_m = fy_m,
            fy_d = fy_d,
            mode = mode,
            seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
        )

    periods_idx = pd.DatetimeIndex(periods)

    pt_key = tuple((str(x).lower() for x in period_types))

    key = (id(df_future), tuple(periods_idx.asi8.tolist()), pt_key, int(fy_m), int(fy_d), str(mode), id(seasonal_flow_weights_q1_q4) if seasonal_flow_weights_q1_q4 is not None else None)

    cached = cache.get(key)

    if cached is not None:

        return cached

    aligned = _future_to_period_aligned(
        dfT = df_future,
        periods = periods_idx,
        period_types = list(pt_key),
        fy_m = fy_m,
        fy_d = fy_d,
        mode = mode,
        seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
    )

    cache[key] = aligned

    return aligned


def mc_equity_value_per_share_multi_ddm(
    *,
    dps_future: pd.DataFrame | None,
    eps_future: pd.DataFrame | None = None,
    hist_ratios: pd.DataFrame | None = None,
    sector_policy: SectorPolicy | None = None,
    coe: float,
    last_price: float,
    g_mu: float = config.RF,
    g_sd: float = 0.01,
    lb: float,
    ub: float,
    g_cap: float,
    seasonal_flow_weights_q1_q4: np.ndarray | None = None,
    ctx: RunContext,
    alignment_cache: dict[tuple[Any, ...], pd.DataFrame] | None = None
) -> dict[str, float]:
    """
    Monte Carlo dividend discount model (DDM) valuation with optional EPS and payout modelling.

    The DDM values equity as the present value of expected future dividends plus a terminal value.
    This implementation supports two dividend construction approaches:

    1) DPS-direct:
    
       Dividends per share (DPS) are simulated directly from the DPS consensus forecast table.

    2) EPS and payout ratio:
   
       Earnings per share (EPS) are simulated from the EPS consensus table and dividends are derived
       as:
    
       - DPS_t = max(EPS_t, 0) * PayoutRatio_t

       where the payout ratio is inferred from DPS/EPS consensus medians (or, if unavailable, from a
       historical median) and is itself simulated as a bounded stochastic series.

    Discount-rate and growth factor structure
    -----------------------------------------
    Discounting uses a simulation-specific cost of equity path `coe_s` constructed from a latent
    macro factor:
   
    - z_growth ~ Normal(0, 1)
   
    - z_rates = rho_rg * z_growth + sqrt(1 - rho_rg^2) * z_rates_idio
   
    - coe_s = clip(coe + coe_sd * z_rates, 0.01, 0.40)

    Terminal growth draws are correlated with this factor:
   
    - z_g = rho_g_div * z_growth + sqrt(1 - rho_g_div^2) * z_idio
   
    - u = Phi(z_g)
   
    - g_draw = TruncatedNormalPPF(u; lower=FLOOR, upper=g_cap_path, mean=g_mu, sd=g_sd)

    where g_cap_path is constrained by both an exogenous cap and the discount rate:
   
    - g_cap_path = min(g_cap, coe_s - SAFETY_SPREAD)

    This structure is intended to reflect that low-rate environments are typically associated with
    both lower discount rates and (to a degree) different growth expectations, whilst still
    preventing g >= coe_s which would destabilise the terminal value.

    Valuation mathematics
    ---------------------
    For each simulation i and period t:
   
    - DF_t,i = 1 / exp( years_frac_t * log(1 + coe_s_i) )
   
    - PV_div_i = sum_t DF_t,i * DPS_t,i

    Terminal value at horizon T uses the perpetuity model:
   
    - TV_T,i = DPS_T,i * (1 + g_i)^dt / ( (1 + coe_s_i)^dt - (1 + g_i)^dt )

    Price_i = PV_div_i + DF_T,i * TV_T,i

    Parameters
    ----------
    dps_future:
        DPS consensus forecast table. If provided, enables the "DPS_direct" method.
    eps_future:
        EPS consensus forecast table. If provided, enables the "EPS_payout" derived-dividend method.
    hist_ratios:
        Optional historical ratios table used to infer payout ratios and calibrate EPS/payout
        dependence when direct DPS inputs are sparse.
    sector_policy:
        Sector policy supplying payout ratio caps.
    coe:
        Base cost of equity used as the centre of the scenario distribution.
    last_price:
        Observed last price used to compute implied returns.
    g_mu, g_sd, g_cap:
        Terminal growth distribution parameters. `g_cap` is further capped by `coe_s - SAFETY_SPREAD`
        per simulation.
    lb, ub:
        Per-share clipping bounds.
    seasonal_flow_weights_q1_q4:
        Seasonal weights used if the period grid includes quarterly stubs.
    ctx:
        Deterministic random-number context.
    alignment_cache:
        Optional memoisation dictionary shared with the residual income engine to avoid repeated
        alignment of the same forecast tables in a single ticker pass.

    Returns
    -------
    dict[str, float]
  
        A mapping containing at least:
  
        - "per_share_mean"
  
        - "returns_mean", "returns_median", "returns_p05", "returns_p95", "returns_std"

        When at least one method succeeds, additional per-share distribution keys and a "methods_used"
        list are also included.

    Advantages
    ----------
    - Direct DPS simulation preserves analyst-provided dividend distributions when available.
  
    - EPS and payout modelling provides coverage for firms with sparse dividend forecasts, whilst
      enforcing non-negative dividends via a bounded payout ratio.
  
    - The factor structure induces coherent co-movement between discount rates and growth, reducing
      unrealistic combinations such as high growth paired with very high discount rates.
  
    """
  
    ctx = _ensure_ctx(
        ctx = ctx
    )

    payout_hi_cap = float(sector_policy.ddm_payout_hi) if sector_policy is not None else 1.5

    base_future = None

    driver_futures: dict[str, pd.DataFrame] = {}

    if dps_future is not None and (not dps_future.empty):

        base_future = dps_future

        driver_futures['dps'] = dps_future

    if eps_future is not None and (not eps_future.empty):

        if base_future is None:

            base_future = eps_future

        driver_futures['eps'] = eps_future

    if base_future is None or base_future.empty:

        raise ValueError('DDM: need at least dps_future or eps_future to build valuation periods.')

    ann_periods = _annual_periods(
        dfT = base_future
    )

    base_ann = _future_to_annual_aligned(
        dfT = base_future,
        periods = ann_periods,
        mode = 'flow'
    )

    fy_m, fy_d = _infer_fy_end_month_day_from_future(
        metric_future = base_ann
    )

    if not (1 <= fy_m <= 12 and 1 <= fy_d <= 31):

        fy_m, fy_d = (12, 31)

    periods, period_types_global, _q_stub = _build_mixed_valuation_periods(
        base_future = base_future,
        driver_futures = driver_futures,
        include_stub_quarters = True,
        fy_m = fy_m,
        fy_d = fy_d
    )

    periods, period_types_global = _filter_future_periods(
        periods = periods,
        period_types = period_types_global,
        today = TODAY_TS
    )

    if len(periods) == 0:

        warnings.warn('DDM: no future valuation periods after filtering; returning zeros.')

        return {
            'per_share_mean': 0.0, 
            'returns_mean': 0.0, 
            'returns_median': 0.0, 
            'returns_p05': 0.0, 
            'returns_p95': 0.0, 
            'returns_std': 0.0
        }

    pt_local, _allow_q, _missing = _build_quarterly_override_period_types(
        periods = periods,
        period_types_global = period_types_global,
        required_futures = driver_futures,
        fy_m = fy_m,
        fy_d = fy_d
    )

    period_idx = pd.DatetimeIndex(periods).normalize()

    days = (period_idx.to_numpy() - np.datetime64(TODAY_TS)) / np.timedelta64(1, 'D')

    years_frac = days / 365.0

    dt_last = float(years_frac[-1] - years_frac[-2]) if len(years_frac) >= 2 else 1.0

    z_growth = ctx.rng('ddm:factor:growth').standard_normal(N_SIMS)

    rho_g_div, rho_rg = _calibrate_ddm_rhos(
        hist_ratios = hist_ratios,
        macro_df = macro
    )

    coe_sd = 0.005

    z_rates_idio = ctx.rng('ddm:factor:rates_idio').standard_normal(N_SIMS)

    z_rates = rho_rg * z_growth + np.sqrt(max(0.0, 1.0 - rho_rg ** 2)) * z_rates_idio

    coe_s = np.clip(coe + coe_sd * z_rates, 0.01, 0.4)

    g_cap_global = float(g_cap) if np.isfinite(g_cap) else FLOOR

    g_cap_path = np.minimum(g_cap_global, coe_s - SAFETY_SPREAD)

    g_cap_path = np.where(np.isfinite(g_cap_path), g_cap_path, FLOOR)

    g_cap_path = np.maximum(g_cap_path, FLOOR)

    z_idio = ctx.rng('ddm:factor:g_idio').standard_normal(N_SIMS)

    z_g = rho_g_div * z_growth + np.sqrt(max(0.0, 1.0 - rho_g_div ** 2)) * z_idio

    u = norm.cdf(z_g)

    g_draw = _simulate_terminal_g_draws_from_u(
        u = u,
        g_mu = g_mu,
        g_sd = g_sd,
        g_cap = g_cap_path
    )

    g_draw = np.minimum(g_draw, np.maximum(coe_s - SAFETY_SPREAD, FLOOR))

    df = np.exp(-years_frac[:, None] * np.log1p(coe_s)[None, :])

    pooled_per_share: list[np.ndarray] = []

    methods_used: list[str] = []


    def _price_from_dps(
        dps_draws: np.ndarray
    ) -> np.ndarray:
        """
        Compute a DDM price vector from a DPS draw matrix.

        Parameters
        ----------
        dps_draws:
            Dividends per share matrix of shape (T, N_SIMS) aligned to the model's period grid.

        Returns
        -------
        numpy.ndarray
            Per-simulation price vector of length N_SIMS, clipped to [lb, ub]. Non-finite paths are
            returned as NaN.
    
        """
    
        dps = np.asarray(dps_draws, float)

        mask = np.all(np.isfinite(dps), axis = 0)

        price = np.full(dps.shape[1], np.nan, dtype = float)

        if mask.sum() == 0:

            return price

        dps_v = dps[:, mask]

        df_v = df[:, mask]

        pv_div = np.sum(df_v * dps_v, axis = 0)

        dps_T = dps_v[-1, :]

        tv = _terminal_value_perpetuity(
            cf_T = dps_T,
            r = coe_s[mask],
            g = g_draw[mask],
            dt_years = dt_last
        )

        price[mask] = pv_div + tv * df_v[-1, :]

        return np.clip(price, lb, ub)


    if dps_future is not None and (not dps_future.empty):

        dpsP = _future_to_period_aligned_cached(
            cache = alignment_cache,
            df_future = dps_future,
            periods = periods,
            period_types = pt_local,
            fy_m = fy_m,
            fy_d = fy_d,
            mode = 'flow',
            seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
        )

        dps_draws_direct, _ = _simulate_skewt_from_rows(
            dfT = dpsP,
            value_row = dpsP.index[0],
            unit_mult = 1.0,
            floor_at_zero = True,
            rng = ctx.rng('ddm:dps_direct')
        )

        price = _price_from_dps(
            dps_draws = dps_draws_direct
        )

        m = np.isfinite(price)

        if m.sum() >= 50:

            pooled_per_share.append(price[m])

            methods_used.append('DPS_direct')

    if eps_future is not None and (not eps_future.empty):

        epsP = _future_to_period_aligned_cached(
            cache = alignment_cache,
            df_future = eps_future,
            periods = periods,
            period_types = pt_local,
            fy_m = fy_m,
            fy_d = fy_d,
            mode = 'flow',
            seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
        )

        eps_draws, _, eps_mu, eps_sigma, eps_x = _simulate_skewt_from_rows(
            dfT = epsP,
            value_row = epsP.index[0],
            unit_mult = 1.0,
            floor_at_zero = False,
            rng = ctx.rng('ddm:eps'),
            return_components = True
        )

        payout_mu = None

        payout_hi = None

        payout_lo = None

        payout_sd = None

        if dps_future is not None and (not dps_future.empty):

            dpsP_for_pay = _future_to_period_aligned_cached(
                cache = alignment_cache,
                df_future = dps_future,
                periods = periods,
                period_types = pt_local,
                fy_m = fy_m,
                fy_d = fy_d,
                mode = 'flow',
                seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
            )

            eps_mu_row = pd.to_numeric(epsP.loc[epsP.index[0]], errors = 'coerce').to_numpy(dtype = float)

            dps_mu_row = pd.to_numeric(dpsP_for_pay.loc[dpsP_for_pay.index[0]], errors = 'coerce').to_numpy(dtype = float)

            payout_mu = np.divide(dps_mu_row, eps_mu_row, out = np.full_like(dps_mu_row, np.nan), where = np.abs(eps_mu_row) > e12)

            payout_mu = np.nan_to_num(payout_mu, nan = 0.0, posinf = 0.0, neginf = 0.0)

            payout_mu = np.clip(payout_mu, 0.0, payout_hi_cap)

            if 'High' in epsP.index and 'Low' in epsP.index and ('High' in dpsP_for_pay.index) and ('Low' in dpsP_for_pay.index):

                eps_hi = pd.to_numeric(epsP.loc['High'], errors = 'coerce').to_numpy(dtype = float)

                eps_lo = pd.to_numeric(epsP.loc['Low'], errors = 'coerce').to_numpy(dtype = float)

                dps_hi = pd.to_numeric(dpsP_for_pay.loc['High'], errors = 'coerce').to_numpy(dtype = float)

                dps_lo = pd.to_numeric(dpsP_for_pay.loc['Low'], errors = 'coerce').to_numpy(dtype = float)

                payout_hi = np.divide(dps_hi, eps_hi, out = np.full_like(dps_hi, np.nan), where = np.abs(eps_hi) > e12)

                payout_lo = np.divide(dps_lo, eps_lo, out = np.full_like(dps_lo, np.nan), where = np.abs(eps_lo) > e12)

                payout_hi = np.clip(np.nan_to_num(payout_hi, nan = 0.0, posinf = 0.0, neginf = 0.0), 0.0, payout_hi_cap)

                payout_lo = np.clip(np.nan_to_num(payout_lo, nan = 0.0, posinf = 0.0, neginf = 0.0), 0.0, payout_hi_cap)

            std_row = 'Std_Dev' if 'Std_Dev' in dpsP_for_pay.index else 'Std Dev.' if 'Std Dev.' in dpsP_for_pay.index else None

            if std_row is not None and std_row in dpsP_for_pay.index and (std_row in epsP.index):

                eps_sd = pd.to_numeric(epsP.loc[std_row], errors = 'coerce').to_numpy(dtype = float)

                dps_sd = pd.to_numeric(dpsP_for_pay.loc[std_row], errors = 'coerce').to_numpy(dtype = float)

                denom = np.maximum(np.abs(eps_mu_row), e12)

                payout_sd = np.sqrt((dps_sd / denom) ** 2 + (np.abs(dps_mu_row) * eps_sd / denom ** 2) ** 2)

                payout_sd = np.nan_to_num(payout_sd, nan = 0.0, posinf = 0.0, neginf = 0.0)

        if payout_mu is None and hist_ratios is not None and (not hist_ratios.empty):

            hist_eps = _extract_hist_ratios_series(
                df = hist_ratios,
                row_candidates = _EPS_HIST_ROWS
            )

            hist_dps = _extract_hist_ratios_series(
                df = hist_ratios,
                row_candidates = _DPS_HIST_ROWS
            )

            if hist_eps is not None and hist_dps is not None:

                payout_hist = (hist_dps / hist_eps).replace([np.inf, -np.inf], np.nan).dropna()

                if len(payout_hist) >= 6:

                    base = float(np.clip(np.nanmedian(payout_hist.to_numpy(dtype = float)), 0.0, payout_hi_cap))

                    payout_mu = np.full(len(periods), base, dtype = float)

        if payout_mu is None:

            payout_mu = np.zeros(len(periods), dtype = float)

        payoutP = epsP.copy()

        idx0 = list(payoutP.index)

        idx0[0] = 'Payout_Ratio'

        payoutP.index = idx0

        payoutP.loc[payoutP.index[0]] = payout_mu

        if 'Median' in payoutP.index:

            payoutP.loc['Median'] = payout_mu

        if payout_hi is not None and 'High' in payoutP.index:

            payoutP.loc['High'] = payout_hi

        if payout_lo is not None and 'Low' in payoutP.index:

            payoutP.loc['Low'] = payout_lo

        std_row_p = 'Std_Dev' if 'Std_Dev' in payoutP.index else 'Std Dev.' if 'Std Dev.' in payoutP.index else None

        if std_row_p is not None:

            if payout_sd is None:

                payout_sd = np.zeros_like(payout_mu)

            payoutP.loc[std_row_p] = np.maximum(payout_sd, 0.0)

        payout_draws, _, pay_mu_sims, pay_sigma_sims, pay_x = _simulate_skewt_from_rows(
            dfT = payoutP,
            value_row = payoutP.index[0],
            unit_mult = 1.0,
            floor_at_zero = False,
            rng = ctx.rng('ddm:payout'),
            return_components = True
        )

        hist_corr = None

        if hist_ratios is not None and (not hist_ratios.empty):

            hist_eps = _extract_hist_ratios_series(
                df = hist_ratios,
                row_candidates = _EPS_HIST_ROWS
            )

            hist_dps = _extract_hist_ratios_series(
                df = hist_ratios,
                row_candidates = _DPS_HIST_ROWS
            )

            if hist_eps is not None and hist_dps is not None:

                payout_hist = (hist_dps / hist_eps).replace([np.inf, -np.inf], np.nan)

                hist_corr = pd.concat([hist_eps.rename('eps'), payout_hist.rename('payout_ratio')], axis = 1).dropna()

        if hist_corr is not None and len(hist_corr) >= 10:

            cols = pd.DatetimeIndex(periods).normalize()

            var_periods = {'eps': cols, 'payout_ratio': cols}

            corr_res = _build_joint_corr_multi_from_history(
                var_periods = var_periods,
                hist_annual = hist_corr,
                min_points = 10
            )

            if corr_res is not None:

                R_pair, keep_vars, nu_pair = corr_res

                if set(keep_vars) >= {'eps', 'payout_ratio'}:

                    phi_map = _estimate_ar1_phi_multi_from_history(
                        hist_annual = hist_corr,
                        vars_ = keep_vars
                    )

                    idx = {v: i for i, v in enumerate(keep_vars)}

                    rng_joint = ctx.rng('ddm:joint_innov')

                    prev = {v: np.zeros(eps_x.shape[1], dtype = float) for v in keep_vars}

                    for t in range(eps_x.shape[0]):
                
                        z = _mv_t_innovations(
                            n_sims = eps_x.shape[1],
                            R = R_pair,
                            nu = nu_pair,
                            rng = rng_joint
                        )

                        for j, v in enumerate(keep_vars):
                            phi = float(phi_map.get(v, 0.0))

                            if phi > 0.0:

                                z[:, j] = phi * prev[v] + np.sqrt(max(1.0 - phi * phi, 0.0)) * z[:, j]

                            prev[v] = z[:, j]

                        eps_x[t, :] = z[:, idx['eps']]

                        pay_x[t, :] = z[:, idx['payout_ratio']]

        eps_draws = eps_mu + eps_sigma * eps_x

        payout_draws = np.clip(pay_mu_sims + pay_sigma_sims * pay_x, 0.0, payout_hi_cap)

        dps_draws_derived = np.maximum(eps_draws, 0.0) * payout_draws

        price = _price_from_dps(
            dps_draws = dps_draws_derived
        )

        m = np.isfinite(price)

        if m.sum() >= 50:

            pooled_per_share.append(price[m])

            methods_used.append('EPS_payout')

    if len(pooled_per_share) == 0:

        return {'per_share_mean': 0.0, 'returns_mean': 0.0, 'returns_median': 0.0, 'returns_p05': 0.0, 'returns_p95': 0.0, 'returns_std': 0.0}

    pooled = np.concatenate(pooled_per_share)

    pooled = np.clip(pooled, lb, ub)

    rets = pooled / last_price - 1.0

    out = {
        'per_share_mean': float(np.mean(pooled)), 
        'per_share_median': float(np.median(pooled)),
        'per_share_p05': float(np.percentile(pooled, 5)), 
        'per_share_p95': float(np.percentile(pooled, 95)), 
        'per_share_std': float(np.std(pooled)), 
        'returns_mean': float(np.mean(rets)), 
        'returns_median': float(np.median(rets)), 
        'returns_p05': float(np.percentile(rets, 5)), 
        'returns_p95': float(np.percentile(rets, 95)), 
        'returns_std': float(np.std(rets)), 
        'methods_used': methods_used
    }

    return out


def mc_equity_value_per_share_multi_residual_income(
    *,
    eps_future: pd.DataFrame,
    dps_future: pd.DataFrame,
    bvps_future: pd.DataFrame | None,
    hist_ratios: pd.DataFrame | None = None,
    hist_bal: pd.DataFrame | None = None,
    shares_outstanding: float | None = None,
    sector_policy: SectorPolicy | None = None,
    coe: float,
    last_price: float,
    g_mu: float = config.RF,
    g_sd: float = 0.01,
    lb: float,
    ub: float,
    g_cap: float,
    seasonal_flow_weights_q1_q4: np.ndarray | None = None,
    ctx: RunContext | None = None,
    alignment_cache: dict[tuple[Any, ...], pd.DataFrame] | None = None
) -> dict[str, float]:
    """
    Monte Carlo residual income (RI) valuation with clean-surplus book value dynamics.

    The residual income model expresses equity value per share as:

    - Price = BVPS_0 + PV(ResidualIncome) + PV(TerminalResidualIncome)

    where residual income for a period is defined as earnings in excess of the equity charge on
    beginning book value:

    - RI_t = EPS_t - r_t * BVPS_t

    and BVPS evolves under the clean surplus relation:

    - BVPS_{t+1} = BVPS_t + EPS_t - DPS_t

    This implementation supports two book value constructions:
  
    1) Derived BVPS from EPS and DPS draws (clean surplus implied),
  
    2) Analyst BVPS forecasts (when `bvps_future` is provided), used as an alternative beginning-book
       value series for the equity charge term.

    Period-length-aware equity charge
    --------------------------------
    The period grid may contain mixed annual and quarterly dates. The equity charge uses an
    effective rate over each period length dt:

    - r_dt = (1 + coe)^dt - 1

    such that the equity charge is consistent with discrete compounding on irregular time steps.

    Valuation mathematics
    ---------------------
    Let DF_t = 1 / exp( years_frac_t * log(1 + coe) ). Then for each simulation:
  
    - PV_RI = sum_t DF_t * RI_t
  
    - TV_T = RI_T * (1 + g)^dt / ( (1 + coe)^dt - (1 + g)^dt )
  
    - Price = BVPS_0 + PV_RI + DF_T * TV_T

    where g is a truncated-normal terminal growth draw.

    Stochastic components
    ---------------------
  
    - EPS is simulated via a skewed Student-t distribution fitted to consensus summary rows.
  
    - Dividends are derived using a simulated payout ratio:
  
      - PayoutRatio_t is inferred from DPS/EPS consensus medians (or bounded defaults) and simulated
        as a skew-t bounded series, clipped to a sector policy cap.
  
      - DPS_t = max(EPS_t, 0) * PayoutRatio_t
  
    - When sufficient history exists, dependence between EPS and payout ratio is injected using the
      same copula-like machinery used elsewhere in the module (rank dependence and AR(1) persistence).

    Parameters
    ----------
    eps_future, dps_future:
        Consensus forecast tables for EPS and DPS.
    bvps_future:
        Optional consensus forecast table for book value per share. If provided and usable, it
        enables an additional residual income method based on analyst-provided book values.
    hist_ratios, hist_bal:
        Optional history used to infer initial BVPS when no BVPS forecast is available.
    shares_outstanding:
        Optional share count used by the BVPS inference helper when book value must be inferred from
        balance sheet levels.
    sector_policy:
        Sector policy providing payout ratio caps.
    coe:
        Cost of equity used for discounting and equity charge computation.
    last_price:
        Observed last price used to compute implied returns.
    g_mu, g_sd, g_cap:
        Terminal growth distribution parameters for residual income.
    lb, ub:
        Per-share clipping bounds.
    seasonal_flow_weights_q1_q4:
        Seasonal weights used when constructing quarterly stubs.
    ctx:
        Deterministic random-number context.
    alignment_cache:
        Optional memoisation dictionary shared with the DDM engine to avoid repeated alignment of the
        same forecast tables.

    Returns
    -------
    dict[str, float]
        Summary statistics for the per-share value and implied return distributions.

    Advantages
    ----------
  
    - Anchors valuation in current book value, which can stabilise estimates when cashflows are
      volatile or negative.
  
    - The clean surplus relation enforces internal consistency between earnings, dividends, and book
      value evolution.
  
    - Period-length-aware equity charges support mixed annual/quarterly valuation grids without
      forcing an artificial uniform time step.
  
    """
  
    ctx = _ensure_ctx(
        ctx = ctx
    )

    payout_hi_cap = float(sector_policy.ri_payout_hi) if sector_policy is not None else 1.5

    if eps_future is None or eps_future.empty:

        raise ValueError('Residual Income requires eps_future.')

    if dps_future is None or dps_future.empty:

        raise ValueError('Residual Income requires dps_future.')

    ann_periods = _annual_periods(
        dfT = eps_future
    )

    if len(ann_periods) == 0:

        raise ValueError('RI requires at least one annual period in eps_future.')

    eps_ann = _future_to_annual_aligned(
        dfT = eps_future,
        periods = ann_periods,
        mode = 'flow'
    )

    fy_m, fy_d = _infer_fy_end_month_day_from_future(
        metric_future = eps_ann
    )

    if not (1 <= fy_m <= 12 and 1 <= fy_d <= 31):

        fy_m, fy_d = (12, 31)

    driver_union = {'eps': eps_future, 'dps': dps_future, 'bvps': bvps_future}

    periods, period_types_global, _q_stub = _build_mixed_valuation_periods(
        base_future = eps_future,
        driver_futures = driver_union,
        include_stub_quarters = True,
        fy_m = fy_m,
        fy_d = fy_d
    )

    periods, period_types_global = _filter_future_periods(
        periods = periods,
        period_types = period_types_global,
        today = TODAY_TS
    )

    if len(periods) == 0:

        warnings.warn('RI: no future valuation periods after filtering; returning zeros.')

        return {
            'model': 'RI',  
            'coe': float(coe),  
            'per_share_mean': 0.0,  
            'per_share_median': 0.0,  
            'per_share_p05': 0.0,  
            'per_share_p95': 0.0,  
            'per_share_std': 0.0,  
            'returns_mean': 0.0,  
            'returns_median': 0.0,  
            'returns_p05': 0.0,  
            'returns_p95': 0.0,  
            'returns_std': 0.0
        }

    pt_local, _allow_q, _missing = _build_quarterly_override_period_types(
        periods = periods,
        period_types_global = period_types_global,
        required_futures = {'eps': eps_future, 'dps': dps_future, 'bvps': bvps_future},
        fy_m = fy_m,
        fy_d = fy_d
    )

    epsP = _future_to_period_aligned_cached(
        cache = alignment_cache,
        df_future = eps_future,
        periods = periods,
        period_types = pt_local,
        fy_m = fy_m,
        fy_d = fy_d,
        mode = 'flow',
        seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
    )

    dpsP = _future_to_period_aligned_cached(
        cache = alignment_cache,
        df_future = dps_future,
        periods = periods,
        period_types = pt_local,
        fy_m = fy_m,
        fy_d = fy_d,
        mode = 'flow',
        seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
    )

    eps_draws, _, eps_mu, eps_sigma, eps_x = _simulate_skewt_from_rows(
        dfT = epsP,
        value_row = epsP.index[0],
        unit_mult = 1.0,
        floor_at_zero = False,
        rng = ctx.rng('ri:eps'),
        return_components = True
    )

    payoutP = epsP.copy()

    idx0 = list(payoutP.index)

    idx0[0] = 'Payout_Ratio'

    payoutP.index = idx0

    eps_mu_row = pd.to_numeric(epsP.loc[epsP.index[0]], errors = 'coerce').to_numpy(dtype = float)

    dps_mu_row = pd.to_numeric(dpsP.loc[dpsP.index[0]], errors = 'coerce').to_numpy(dtype = float)

    payout_mu = np.divide(dps_mu_row, eps_mu_row, out = np.full_like(dps_mu_row, np.nan), where = np.abs(eps_mu_row) > e12)

    payout_mu = np.clip(payout_mu, 0.0, payout_hi_cap)

    payoutP.loc['Payout_Ratio'] = payout_mu

    std_row = 'Std_Dev' if 'Std_Dev' in payoutP.index else 'Std Dev.' if 'Std Dev.' in payoutP.index else None

    if 'High' in payoutP.index and 'Low' in payoutP.index and ('Median' in payoutP.index):

        eps_hi = pd.to_numeric(epsP.loc['High'], errors = 'coerce').to_numpy(dtype = float)

        eps_lo = pd.to_numeric(epsP.loc['Low'], errors = 'coerce').to_numpy(dtype = float)

        dps_hi = pd.to_numeric(dpsP.loc['High'], errors = 'coerce').to_numpy(dtype = float)

        dps_lo = pd.to_numeric(dpsP.loc['Low'], errors = 'coerce').to_numpy(dtype = float)

        payout_hi = np.divide(dps_hi, eps_hi, out = np.full_like(dps_hi, np.nan), where = np.abs(eps_hi) > e12)

        payout_lo = np.divide(dps_lo, eps_lo, out = np.full_like(dps_lo, np.nan), where = np.abs(eps_lo) > e12)

        payout_hi = np.clip(payout_hi, 0.0, payout_hi_cap)

        payout_lo = np.clip(payout_lo, 0.0, payout_hi_cap)

        payoutP.loc['High'] = payout_hi

        payoutP.loc['Low'] = payout_lo

        payoutP.loc['Median'] = np.clip(payout_mu, 0.0, payout_hi_cap)

        if std_row is not None:

            payout_sd = (payout_hi - payout_lo) / 4.0

            payout_sd = np.where(np.isfinite(payout_sd) & (payout_sd > 0), payout_sd, 0.05)

            payoutP.loc[std_row] = payout_sd
   
    elif std_row is not None:

        payoutP.loc[std_row] = np.full_like(payout_mu, 0.05)

    payout_draws, _, pay_mu, pay_sigma, pay_x = _simulate_skewt_from_rows(
        dfT = payoutP,
        value_row = payoutP.index[0],
        unit_mult = 1.0,
        floor_at_zero = False,
        rng = ctx.rng('ri:payout'),
        return_components = True
    )

    payout_draws = np.clip(payout_draws, 0.0, payout_hi_cap)

    hist_corr = None

    if hist_ratios is not None and (not hist_ratios.empty):

        hist_eps = _extract_hist_ratios_series(
            df = hist_ratios,
            row_candidates = _EPS_HIST_ROWS
        )

        hist_dps = _extract_hist_ratios_series(
            df = hist_ratios,
            row_candidates = _DPS_HIST_ROWS
        )

        if hist_eps is not None and hist_dps is not None:

            payout_hist = (hist_dps / hist_eps).replace([np.inf, -np.inf], np.nan)

            hist_corr = pd.concat([hist_eps.rename('eps'), payout_hist.rename('payout_ratio')], axis = 1).dropna()

    cols = pd.DatetimeIndex(periods).normalize()

    if hist_corr is not None and len(hist_corr) >= 10:

        var_periods = {'eps': cols, 'payout_ratio': cols}

        corr_res = _build_joint_corr_multi_from_history(
            var_periods = var_periods,
            hist_annual = hist_corr,
            min_points = 10
        )

        if corr_res is not None:

            R_pair, keep_vars, nu_pair = corr_res

            if set(keep_vars) >= {'eps', 'payout_ratio'}:

                phi_map = _estimate_ar1_phi_multi_from_history(
                    hist_annual = hist_corr,
                    vars_ = keep_vars
                )

                idx = {v: i for i, v in enumerate(keep_vars)}

                rng_joint = ctx.rng('ri:joint_innov')

                prev = {v: np.zeros(eps_x.shape[1], dtype = float) for v in keep_vars}

                for t in range(eps_x.shape[0]):
          
                    z = _mv_t_innovations(
                        n_sims = eps_x.shape[1],
                        R = R_pair,
                        nu = nu_pair,
                        rng = rng_joint
                    )

                    for j, v in enumerate(keep_vars):
               
                        phi = float(phi_map.get(v, 0.0))

                        if phi > 0.0:

                            z[:, j] = phi * prev[v] + np.sqrt(max(1.0 - phi * phi, 0.0)) * z[:, j]

                        prev[v] = z[:, j]

                    eps_x[t, :] = z[:, idx['eps']]

                    pay_x[t, :] = z[:, idx['payout_ratio']]

    if np.ndim(eps_mu) == 1:

        eps_draws = eps_mu[:, None] + eps_sigma[:, None] * eps_x

    else:

        eps_draws = eps_mu + eps_sigma * eps_x

    payout_draws = np.clip(pay_mu + pay_sigma * pay_x, 0.0, payout_hi_cap)

    dps_draws = np.maximum(eps_draws, 0.0) * payout_draws

    bvps_draws = None

    if bvps_future is not None and (not bvps_future.empty):

        try:

            bvpsP = _future_to_period_aligned_cached(
                cache = alignment_cache,
                df_future = bvps_future,
                periods = periods,
                period_types = pt_local,
                fy_m = fy_m,
                fy_d = fy_d,
                mode = 'stock',
                seasonal_flow_weights_q1_q4 = seasonal_flow_weights_q1_q4
            )

            bvps_draws, _ = _simulate_skewt_from_rows(
                dfT = bvpsP,
                value_row = bvpsP.index[0],
                unit_mult = 1.0,
                floor_at_zero = True,
                rng = ctx.rng('ri:bvps')
            )

        except (TypeError, ValueError, KeyError):

            bvps_draws = None

    if bvps_draws is not None:

        bv1 = bvps_draws[0, :]

        bv0 = bv1 - (eps_draws[0, :] - dps_draws[0, :])

        bv0 = np.maximum(bv0, 0.0)

    else:

        bvps0 = _infer_bvps0_from_history(
            hist_bal = hist_bal,
            hist_ratios = hist_ratios,
            shares_outstanding = shares_outstanding
        )

        if bvps0 is not None and np.isfinite(bvps0):

            bv0 = np.full(N_SIMS, float(bvps0), dtype = float)

        else:

            bv0 = np.zeros(N_SIMS, dtype = float)

    period_idx = pd.DatetimeIndex(periods).normalize()

    days = (period_idx.to_numpy() - np.datetime64(TODAY_TS)) / np.timedelta64(1, 'D')

    years_frac = days / 365.0

    df = 1.0 / np.exp(years_frac * np.log1p(coe))

    dt = np.empty_like(years_frac, dtype = float)

    dt[0] = float(years_frac[0])

    if len(dt) > 1:

        dt[1:] = np.diff(years_frac)

    dt = np.maximum(dt, 1e-08)

    r_dt = np.exp(np.log1p(coe) * dt) - 1.0

    T = len(period_idx)

    bvps_beg_eps_dps = _bvps_path_from_eps_dps(
        bvps0 = bv0,
        eps_draws = eps_draws[:T, :],
        dps_draws = dps_draws[:T, :],
        floor_bvps_at_zero = False
    )

    RI_eps_dps = eps_draws[:T, :] - r_dt[:, None] * bvps_beg_eps_dps

    RI_analyst = None

    if bvps_draws is not None:

        bv_start_analyst = np.empty((T, N_SIMS), dtype = float)

        bv_start_analyst[0, :] = bv0

        if T > 1:

            bv_start_analyst[1:T, :] = np.maximum(bvps_draws[0:T - 1, :], 0.0)

        RI_analyst = eps_draws[:T, :] - r_dt[:, None] * bv_start_analyst

    g_draw = _simulate_terminal_g_draws(
        g_mu = g_mu,
        g_sd = g_sd,
        g_cap = g_cap,
        rng = ctx.rng('terminal_g:ri')
    )

    dt_last = float(years_frac[-1] - years_frac[-2]) if len(years_frac) >= 2 else 1.0

    methods: list[np.ndarray] = []

    mask_eps = np.all(np.isfinite(RI_eps_dps), axis = 0)

    if mask_eps.sum() >= 50:

        pv_RI_eps = df @ RI_eps_dps[:, mask_eps]

        RI_T_eps = RI_eps_dps[-1, mask_eps]

        tv_eps = _terminal_value_perpetuity(
            cf_T = RI_T_eps,
            r = coe,
            g = g_draw[mask_eps],
            dt_years = dt_last
        )

        price_eps = bv0[mask_eps] + pv_RI_eps + tv_eps * df[-1]

        methods.append(price_eps)

    if RI_analyst is not None:

        mask_a = np.all(np.isfinite(RI_analyst), axis = 0)

        if mask_a.sum() >= 50:

            pv_RI_a = df @ RI_analyst[:, mask_a]

            RI_T_a = RI_analyst[-1, mask_a]

            tv_a = _terminal_value_perpetuity(
                cf_T = RI_T_a,
                r = coe,
                g = g_draw[mask_a],
                dt_years = dt_last
            )

            price_a = bv0[mask_a] + pv_RI_a + tv_a * df[-1]

            methods.append(price_a)

    if len(methods) == 0:

        return {
            'model': 'RI',  
            'coe': float(coe),  
            'per_share_mean': 0.0,  
            'per_share_median': 0.0,  
            'per_share_p05': 0.0,  
            'per_share_p95': 0.0,  
            'per_share_std': 0.0,  
            'returns_mean': 0.0,  
            'returns_median': 0.0,  
            'returns_p05': 0.0,  
            'returns_p95': 0.0,  
            'returns_std': 0.0
        }

    price = np.concatenate(methods, axis = 0) if len(methods) > 1 else methods[0]

    price = np.clip(price, lb, ub)

    rets = price / last_price - 1.0

    return {
        'model': 'RI',  
        'coe': float(coe),  
        'per_share_mean': float(np.mean(price)),  
        'per_share_median': float(np.median(price)),  
        'per_share_p05': float(np.percentile(price, 5)),  
        'per_share_p95': float(np.percentile(price, 95)),  
        'per_share_std': float(np.std(price)),  
        'returns_mean': float(np.mean(rets)),  
        'returns_median': float(np.median(rets)),  
        'returns_p05': float(np.percentile(rets, 5)),  
        'returns_p95': float(np.percentile(rets,  
        95)), 'returns_std': float(np.std(rets))
    }


def _ticker_default_file_maps(
    tickers: list[str]
) -> tuple[dict[str, str], dict[str, Path]]:
    """
    Build default forecast and history file path mappings for a list of tickers.

    The valuation pipeline expects two per-ticker workbooks:
  
    - a "pred" (forecast) workbook containing CapIQ consensus forecast tables, and
  
    - a "fin" (history) workbook containing historical financial statements and ratios.

    This helper constructs the default paths under a root directory controlled by configuration.
    The conventional layout is:

    - {ROOT_FIN_DIR}/{TICKER}/{TICKER}_capIQ_pred.xls
  
    - {ROOT_FIN_DIR}/{TICKER}/{TICKER}_capIQ_fin.xls

    Parameters
    ----------
    tickers:
        Ticker symbols.

    Returns
    -------
    tuple[dict[str, str], dict[str, pathlib.Path]]
        `(pred_files, hist_files)` where:
  
        - `pred_files[ticker]` is a string path to the forecast workbook, and
  
        - `hist_files[ticker]` is a Path to the historical workbook.

    Notes
    -----
    If `config.ROOT_FIN_DIR` is not defined, a fallback rooted at:
    - config.BASE_DIR / "modelling" / "stock_analysis_data"
    is used.
  
    """
  
    base_dir_cfg = Path(getattr(config, 'BASE_DIR', Path.cwd()))

    fallback_root = base_dir_cfg / 'modelling' / 'stock_analysis_data'

    root_fin_dir = Path(getattr(config, 'ROOT_FIN_DIR', fallback_root))

    pred_files = {tkr: str(root_fin_dir / str(tkr) / f'{tkr}_capIQ_pred.xls') for tkr in tickers}

    hist_files = {tkr: root_fin_dir / str(tkr) / f'{tkr}_capIQ_fin.xls' for tkr in tickers}

    return (pred_files, hist_files)


def _resolve_file_maps(
    tickers: list[str],
    pred_files: dict[str, str] | None,
    hist_files: dict[str, Path | str] | None
) -> tuple[dict[str, str], dict[str, Path]]:
    """
    Resolve per-ticker forecast and history file maps from defaults and user overrides.

    Parameters
    ----------
    tickers:
        Ticker symbols.
    pred_files:
        Optional mapping overriding the default forecast workbook paths.
    hist_files:
        Optional mapping overriding the default history workbook paths.

    Returns
    -------
    tuple[dict[str, str], dict[str, pathlib.Path]]
        `(pred_out, hist_out)` where:
 
        - pred_out maps tickers to string paths, and
 
        - hist_out maps tickers to `Path` objects.

    Notes
    -----
    This helper is intentionally permissive: any missing override entries fall back to defaults,
    enabling partial overrides for ad hoc runs.
 
    """
 
    default_pred, default_hist = _ticker_default_file_maps(
        tickers = tickers
    )

    pred_out: dict[str, str] = {}

    hist_out: dict[str, Path] = {}

    for tkr in tickers:
        pred_out[tkr] = str(pred_files.get(tkr, default_pred[tkr])) if pred_files is not None else default_pred[tkr]

        if hist_files is not None:

            hist_out[tkr] = Path(hist_files.get(tkr, default_hist[tkr]))

        else:

            hist_out[tkr] = default_hist[tkr]

    return (pred_out, hist_out)


def _ticker_value(
    series_like,
    ticker: str
) -> float:
    """
    Extract a numeric scalar for a given ticker from a Series-like container.

    The runtime data sources used by `run_valuation(...)` may supply per-ticker scalars as:
  
    - pandas Series indexed by ticker,
  
    - dict-like mappings keyed by ticker, or
  
    - objects exposing `.loc[ticker]`.

    This helper attempts the common access patterns and returns NaN when extraction fails.

    Parameters
    ----------
    series_like:
        A pandas Series/DataFrame-like object, dict-like object, or `None`.
    ticker:
        Ticker key.

    Returns
    -------
    float
        Parsed numeric value, or NaN when unavailable or non-finite.
   
    """
   
    if series_like is None:

        return np.nan

    try:

        v = series_like.loc[ticker]

    except (AttributeError, KeyError, TypeError):

        try:

            v = series_like[ticker]

        except (KeyError, TypeError, IndexError):

            return np.nan

    v = _as_scalar(
        x = v
    )

    vv = pd.to_numeric(v, errors = 'coerce')

    return float(vv) if np.isfinite(vv) else np.nan


def _zero_result_entry(
    ticker: str
) -> dict[str, float | str]:
    """
    Construct a zero-valued output row for a single ticker.

    The orchestration layer uses this structure for:
 
    - missing file cases,
 
    - preflight failures (missing market inputs), and
 
    - model-level exceptions.

    Parameters
    ----------
    ticker:
        Ticker symbol.

    Returns
    -------
    dict[str, float | str]
        Row-like mapping with keys expected by the output DataFrames:
        - "Ticker", "Price", "Returns", "Returns Low", "Returns High", "SE".
 
    """
 
    return {'Ticker': ticker, 'Price': 0.0, 'Returns': 0.0, 'Returns Low': 0.0, 'Returns High': 0.0, 'SE': 0.0}


def run_valuation(
    tickers = None,
    pred_files = None,
    hist_files = None,
    *,
    output_excel_file = None,
    export: bool = True,
    verbose: bool = False
) -> dict[str, pd.DataFrame]:
    """
    Run the full CapIQ-based valuation pipeline for a list of tickers.

    This is the public orchestration entry point. It coordinates:
   
    - file resolution and bundle loading,
   
    - macro and market input retrieval,
   
    - per-ticker shared cashflow context preparation (for FCFF/FCFE reuse),
   
    - execution of four valuation engines:
   
      - FCFF DCF ("DCF_CapIQ"),
   
      - DR-adjusted FCFE DCF ("FCFE_CapIQ"),
   
      - dividend discount model ("DDM_CapIQ"),
   
      - residual income model ("RI_CapIQ"),
   
    - construction of output DataFrames, and
   
    - optional export to an Excel workbook.

    The function is designed to degrade gracefully. When a ticker lacks required files or market
    inputs, or when a model fails due to missing drivers, a structurally valid zero row is returned
    for the affected model/ticker combination.

    Inputs and preflight
    --------------------
    For each ticker, the following market inputs are required:
  
    - market capitalisation (E),
  
    - shares outstanding,
  
    - last price,
  
    - cost of debt, and
  
    - cost of equity (COE).

    COE is read from the "COE" sheet of `config.FORECAST_FILE` when available; values above 1.0 are
    interpreted as percentages and divided by 100.

    Capital structure and discounting
    ---------------------------------
    Capital structure weights and WACC are built using `_build_discount_factor_vector(...)` with
    `return_components=True`. For a given ticker:
 
    - E is market capitalisation,
 
    - D is `debt_for_wacc` derived from gross debt when available, else from non-negative net debt,
 
    - wD = D / (E + D) is used as the debt ratio DR for FCFE,

    - wacc is used as the FCFF discount rate.

    Terminal growth caps are then set to maintain numerical stability of the perpetuity terminal
    value:
   
    - g_cap <= wacc - SAFETY_SPREAD for FCFF,
   
    - g_coe_cap <= coe - SAFETY_SPREAD for equity-discounted models.

    Shared context preparation (performance refactor)
    -------------------------------------------------
    The most expensive steps for FCFF and FCFE are shared:
   
    - valuation period grid construction and filtering,
   
    - forecast alignment to that grid,
   
    - skewed Student-t simulation of forecast uncertainty,
   
    - imputation of missing drivers,
   
    - dependence calibration (optional rank-preserving copula-like reorder), and
   
    - practical bounds and operating-profit extension.

    These steps are performed once per ticker by `_prepare_cashflow_context(...)`. The resulting
    `CashflowContext` is passed into both FCFF and FCFE engines to avoid duplicated computation. If
    shared preparation fails, each model falls back to its own context build.

    DDM and residual income reuse
    -----------------------------
    DDM and residual income engines perform repeated alignment of per-share forecast tables. A
    shared alignment cache is provided per ticker to reduce repeated alignment work for the same
    tables and period grid.

    Parameters
    ----------
    tickers:
        Iterable of tickers. If `None`, defaults to `config.tickers`.
    pred_files:
        Optional mapping from ticker to forecast workbook path. Missing entries fall back to the
        default layout produced by `_ticker_default_file_maps(...)`.
    hist_files:
        Optional mapping from ticker to historical workbook path. Missing entries fall back to the
        default layout.
    output_excel_file:
        Output workbook path for `export_results(...)`. If `None`, defaults to `config.MODEL_FILE`.
    export:
        If `True`, results are exported via `export_results(...)`.
    verbose:
        If `True`, enables debug logging and prints simple timing diagnostics for the major stages.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of sheet names to DataFrames indexed by ticker. The keys are:
  
        - "DCF_CapIQ"
  
        - "FCFE_CapIQ"
  
        - "DDM_CapIQ"
  
        - "RI_CapIQ"

        Each DataFrame contains the columns:
  
        - "Price": model-implied per-share value (mean of the simulated distribution),
  
        - "Returns": mean implied return relative to `last_price`,
  
        - "Returns Low": 5th percentile implied return,
  
        - "Returns High": 95th percentile implied return,
  
        - "SE": standard deviation of implied returns.

    Notes
    -----
    The function prints the resulting tables to stdout for convenience. Production use typically
    relies on the returned DataFrames or exported workbook.
  
    """
  
    if verbose:

        logger.setLevel(logging.DEBUG)
        
    elif logger.level == logging.NOTSET:

        logger.setLevel(logging.INFO)

    _, macro_obj = _ensure_runtime_data()

    r_obj = macro_obj.r

    tickers_list = list(tickers) if tickers is not None else list(config.tickers)

    if len(tickers_list) == 0:

        raise ValueError('No tickers provided.')

    pred_files_map, hist_files_map = _resolve_file_maps(
        tickers = tickers_list,
        pred_files = pred_files,
        hist_files = hist_files
    )

    try:

        coe = pd.read_excel(config.FORECAST_FILE, sheet_name = 'COE', index_col = 0, usecols = ['Ticker', 'COE'], engine = 'openpyxl')

    except (FileNotFoundError, OSError, ValueError, ImportError) as err:

        logger.warning('Failed to read COE sheet from %s (%s).', config.FORECAST_FILE, err)

        coe = pd.DataFrame(columns = ['COE'])

    country = r_obj.country

    shares_outstanding = r_obj.shares_outstanding

    market_cap = r_obj.mcap

    last_price = r_obj.last_price

    cost_of_debt = cod(
        tickers = tickers_list,
        country = country,
        tax_rate_source = r_obj.tax_rate,
        macro_source = macro_obj
    )

    lb = config.lbp * last_price

    ub = config.ubp * last_price

    valid_tickers: list[str] = []

    for tkr in tickers_list:
        
        pred_exists = Path(pred_files_map[tkr]).exists()

        hist_exists = Path(hist_files_map[tkr]).exists()

        if pred_exists and hist_exists:

            valid_tickers.append(tkr)

        else:

            logger.warning('Missing files for %s (pred=%s, hist=%s). Returning zeros.', tkr, pred_exists, hist_exists)

    bundles = preload_ticker_bundles_for_fcff(
        tickers = valid_tickers,
        pred_files = pred_files_map,
        hist_files = hist_files_map
    )

    dcf: list[dict[str, float | str]] = []

    fcfe: list[dict[str, float | str]] = []

    ddm: list[dict[str, float | str]] = []

    ri: list[dict[str, float | str]] = []

    for ticker in tickers_list:
        
        logger.info('Processing %s...', ticker)

        ctx = RunContext(seed = SEED, ticker = str(ticker))

        if ticker not in bundles:

            zero_entry = _zero_result_entry(
                ticker = ticker
            )

            dcf.append(dict(zero_entry))

            fcfe.append(dict(zero_entry))

            ddm.append(dict(zero_entry))

            ri.append(dict(zero_entry))

            continue

        mcap_t = _ticker_value(
            series_like = market_cap,
            ticker = ticker
        )

        cost_of_debt_t = _ticker_value(
            series_like = cost_of_debt,
            ticker = ticker
        )

        shares_out_t = _ticker_value(
            series_like = shares_outstanding,
            ticker = ticker
        )

        lp = _ticker_value(
            series_like = last_price,
            ticker = ticker
        )

        lb_t = _ticker_value(
            series_like = lb,
            ticker = ticker
        )

        ub_t = _ticker_value(
            series_like = ub,
            ticker = ticker
        )

        try:

            coe_t = _as_scalar(
                x = coe.loc[ticker, 'COE']
            )

            coe_t = float(pd.to_numeric(coe_t, errors = 'coerce'))

        except (KeyError, TypeError, ValueError):

            coe_t = np.nan

        if np.isfinite(coe_t) and coe_t > 1.0:

            coe_t = coe_t / 100.0

        preflight_reasons = []

        if not np.isfinite(mcap_t) or mcap_t <= 0.0:

            preflight_reasons.append('market_cap_missing')

        if not np.isfinite(cost_of_debt_t):

            preflight_reasons.append('cost_of_debt_missing')

        if not np.isfinite(coe_t) or coe_t <= 0.0:

            preflight_reasons.append('coe_missing')

        if not np.isfinite(shares_out_t) or shares_out_t <= 0.0:

            preflight_reasons.append('shares_outstanding_missing')

        if not np.isfinite(lp) or lp <= 0.0:

            preflight_reasons.append('last_price_missing')

        if not np.isfinite(lb_t):

            preflight_reasons.append('lb_missing')

        if not np.isfinite(ub_t):

            preflight_reasons.append('ub_missing')

        if preflight_reasons:

            logger.warning('%s: preflight failed (%s). Returning zeros.', ticker, ','.join(preflight_reasons))

            zero_entry = _zero_result_entry(
                ticker = ticker
            )

            dcf.append(dict(zero_entry))

            fcfe.append(dict(zero_entry))

            ddm.append(dict(zero_entry))

            ri.append(dict(zero_entry))

            continue

        b = bundles[ticker]

        inc = b.hist_inc

        bal = b.hist_bal

        cf = b.hist_cf

        rat = b.hist_ratios

        fcf_future = b.future.get('fcf')

        net_debt_future = b.future.get('net_debt')

        if net_debt_future is not None:

            net_debt_future = align_future_net_debt_sign_to_history(
                net_debt_future = net_debt_future,
                hist_bal = bal,
                value_row = 'Net_Debt'
            )

        revenue_future = b.future.get('revenue')

        eps_future = b.future.get('eps')

        roe_future = b.future.get('roe')

        dps_future = b.future.get('dps')

        bvps_future = b.future.get('bvps')

        cfo_future = b.future.get('cfo')

        capex_future = b.future.get('capex')

        maint_capex_future = b.future.get('maint_capex')

        interest_future = b.future.get('interest')

        tax_future = b.future.get('tax')

        da_future = b.future.get('da')

        ebit_future = b.future.get('ebit')

        ebitda_future = b.future.get('ebitda')

        net_income_future = b.future.get('net_income')

        gross_margin_future = b.future.get('gross_margin')

        ebt_future = b.future.get('ebt')

        roe_future = _dedupe_cols(
            df = roe_future
        ) if roe_future is not None else None

        eps_future = _dedupe_cols(
            df = eps_future
        ) if eps_future is not None else None

        dps_future = _dedupe_cols(
            df = dps_future
        ) if dps_future is not None else None

        bvps_future = _dedupe_cols(
            df = bvps_future
        ) if bvps_future is not None else None

        driver_futures = {
            'fcf': fcf_future,  
            'cfo': cfo_future,  
            'capex': capex_future,  
            'maint_capex': maint_capex_future,  
            'interest': interest_future,  
            'tax': tax_future,  
            'da': da_future,  
            'ebit': ebit_future,  
            'ebitda': ebitda_future,  
            'net_income': net_income_future,  
            'revenue': revenue_future,  
            'gross_margin': gross_margin_future,  
            'ebt': ebt_future,  
            'dps': dps_future,  
            'roe': roe_future,  
            'eps': eps_future,  
            'bvps': bvps_future
        }

        if verbose:

            logger.debug('driver_futures for %s:\n%s', ticker, driver_futures)

        mv_debt_t = 0.0

        D_from_bal = _gross_debt_from_balance(
            hist_bal = bal
        )

        if D_from_bal is not None:

            mv_debt_t = D_from_bal * UNIT_MULT

        nd_val_for_wacc = _net_debt_at_valuation_date(
            hist_bal = bal,
            net_debt_future = net_debt_future,
            today = TODAY_TS,
            cost_of_debt = cost_of_debt_t
        )

        debt_for_wacc = mv_debt_t if D_from_bal is not None else max(float(nd_val_for_wacc), 0.0) if nd_val_for_wacc is not None and np.isfinite(nd_val_for_wacc) else 0.0

        cap_struct = _build_discount_factor_vector(
            coe = coe_t,
            E = mcap_t,
            cost_of_debt = cost_of_debt_t,
            D = debt_for_wacc,
            return_components = True
        )

        wacc = float(cap_struct['wacc'])

        dr_fcfe = float(cap_struct['wD'])

        g_cap = G_CAP

        g_coe_cap = G_CAP

        if np.isfinite(wacc):

            g_cap = wacc - SAFETY_SPREAD

        if not np.isfinite(g_cap) or g_cap <= FLOOR:

            g_cap = GF_CAP

        g_cap = min(G_CAP, g_cap)

        if np.isfinite(coe_t):

            g_coe_cap = coe_t - SAFETY_SPREAD

        if not np.isfinite(g_coe_cap) or g_coe_cap <= FLOOR:

            g_coe_cap = GF_CAP

        g_coe_cap = min(G_CAP, g_coe_cap)

        sector_raw = _infer_sector_label_for_ticker(
            r_data = r_obj,
            ticker = str(ticker)
        )

        sector_name = _normalize_sector_label(
            raw = sector_raw
        )

        sector_policy = _policy_for_sector(
            sector_label = sector_name
        )

        logger.info('[SECTOR] %s: raw=%s canonical=%s profile=%s', ticker, sector_raw if sector_raw is not None else 'None', sector_name, sector_policy.fcff_profile)

        g_hat, g_sig = estimate_terminal_growth_from_forecasts(
            fcf_future = fcf_future,
            value_row = 'Free_Cash_Flow',
            revenue_future = revenue_future,
            roe_future = roe_future,
            eps_future = eps_future,
            dps_future = dps_future,
            g_cap = g_cap,
            sector_policy = sector_policy
        )

        g_fcfe_hat, g_fcfe_sig = estimate_terminal_growth_from_forecasts(
            fcf_future = fcf_future,
            value_row = 'Free_Cash_Flow',
            revenue_future = revenue_future,
            roe_future = roe_future,
            eps_future = eps_future,
            dps_future = dps_future,
            g_cap = g_coe_cap,
            sector_policy = sector_policy
        )

        g_ddm_hat, g_ddm_sig = estimate_terminal_growth_for_ddm(
            dps_future = dps_future,
            g_cap = g_coe_cap,
            sector_policy = sector_policy
        )

        g_ri_hat, g_ri_sig = estimate_terminal_growth_for_residual_income(
            eps_future = eps_future,
            dps_future = dps_future,
            roe_future = roe_future,
            g_cap = g_coe_cap,
            sector_policy = sector_policy
        )

        src_a = {}

        for k in sorted(driver_futures.keys()):
            
            dfF = driver_futures[k]

            if dfF is None:

                continue

            src_a[k] = _annual_periods(
                dfT = dfF
            )

        shared_cashflow_context: CashflowContext | None = None

        prepare_t0 = time.perf_counter()

        try:

            shared_cashflow_context = _prepare_cashflow_context(
                ctx = ctx,
                ticker = str(ticker),
                sector_label = sector_name,
                sector_policy = sector_policy,
                fcf_future = fcf_future,
                net_debt_future = net_debt_future,
                driver_futures = driver_futures,
                hist_inc = inc,
                hist_cf = cf,
                hist_bal = bal,
                hist_ratios = rat,
                src_a = src_a,
                cost_of_debt = cost_of_debt_t,
                tax_is_percent = True,
                net_debt_use = 'first',
                seasonal_flow_weights_q1_q4 = b.capex_seasonality_w_q1_q4,
                rng_labels = {
                    'dnwc': 'dnwc_shared',  
                    'copula': 'copula:shared',  
                    'op_extend': 'op_extend_shared',  
                    'terminal_u': 'terminal_g:shared_u'
                }
            )

        except (TypeError, ValueError, ZeroDivisionError) as err:

            logger.warning('[CASHFLOW] %s shared context unavailable (%s). Falling back to per-model setup.', ticker, err)

            shared_cashflow_context = None

        if verbose:

            logger.debug('[TIMING] %s prepare_context %.3fs', ticker, time.perf_counter() - prepare_t0)

        shared_alignment_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

        try:

            fcff_t0 = time.perf_counter()

            res = mc_equity_value_per_share_multi_fcff(
                ctx = ctx,
                ticker = str(ticker),
                sector_label = sector_name,
                sector_policy = sector_policy,
                fcf_future = fcf_future,
                net_debt_future = net_debt_future,
                driver_futures = driver_futures,
                wacc = wacc,
                cost_of_debt = cost_of_debt_t,
                shares_outstanding = shares_out_t,
                last_price = lp,
                g_mu = g_hat,
                g_sd = g_sig,
                hist_inc = inc,
                hist_cf = cf,
                hist_bal = bal,
                hist_ratios = rat,
                ub = ub_t,
                lb = lb_t,
                g_cap = g_cap,
                src_a = src_a,
                seasonal_flow_weights_q1_q4 = b.capex_seasonality_w_q1_q4,
                cashflow_context = shared_cashflow_context
            )

            dcf.append({
                'Ticker': ticker,  
                'Price': res['per_share_mean'],  
                'Returns': res['returns_mean'],  
                'Returns Low': res['returns_p05'],  
                'Returns High': res['returns_p95'],  
                'SE': res['returns_std']
            })

            if verbose:

                logger.debug('[TIMING] %s fcff_eval %.3fs', ticker, time.perf_counter() - fcff_t0)

        except (TypeError, ValueError, ZeroDivisionError) as err:

            logger.warning('[DCF] %s failed (%s). Returning zeros.', ticker, err)

            dcf.append(_zero_result_entry(
                ticker = ticker
            ))

        try:

            fcfe_t0 = time.perf_counter()

            res_fcfe = mc_equity_value_per_share_multi_fcfe(
                ctx = ctx,
                ticker = str(ticker),
                sector_label = sector_name,
                sector_policy = sector_policy,
                fcf_future = fcf_future,
                net_debt_future = net_debt_future,
                driver_futures = driver_futures,
                coe = coe_t,
                debt_ratio = dr_fcfe,
                cost_of_debt = cost_of_debt_t,
                shares_outstanding = shares_out_t,
                last_price = lp,
                g_mu = g_fcfe_hat,
                g_sd = g_fcfe_sig,
                hist_inc = inc,
                hist_cf = cf,
                hist_bal = bal,
                hist_ratios = rat,
                ub = ub_t,
                lb = lb_t,
                g_cap = g_coe_cap,
                src_a = src_a,
                seasonal_flow_weights_q1_q4 = b.capex_seasonality_w_q1_q4,
                cashflow_context = shared_cashflow_context
            )

            fcfe.append({
                'Ticker': ticker,  
                'Price': res_fcfe['per_share_mean'],  
                'Returns': res_fcfe['returns_mean'],  
                'Returns Low': res_fcfe['returns_p05'],  
                'Returns High': res_fcfe['returns_p95'],  
                'SE': res_fcfe['returns_std']
            })

            if verbose:

                logger.debug('[TIMING] %s fcfe_eval %.3fs', ticker, time.perf_counter() - fcfe_t0)

        except (TypeError, ValueError, ZeroDivisionError) as err:

            logger.warning('[FCFE] %s failed (%s). Returning zeros.', ticker, err)

            fcfe.append(_zero_result_entry(
                ticker = ticker
            ))

        try:

            ddm_t0 = time.perf_counter()

            res_ddm = mc_equity_value_per_share_multi_ddm(
                ctx = ctx,
                dps_future = dps_future,
                eps_future = eps_future,
                hist_ratios = rat,
                sector_policy = sector_policy,
                coe = coe_t,
                last_price = lp,
                g_mu = g_ddm_hat,
                g_sd = g_ddm_sig,
                ub = ub_t,
                lb = lb_t,
                g_cap = g_coe_cap,
                seasonal_flow_weights_q1_q4 = None,
                alignment_cache = shared_alignment_cache
            )

            ddm.append({
                'Ticker': ticker,  
                'Price': res_ddm['per_share_mean'],  
                'Returns': res_ddm['returns_mean'],  
                'Returns Low': res_ddm['returns_p05'],  
                'Returns High': res_ddm['returns_p95'],  
                'SE': res_ddm['returns_std']
            })

            if verbose:

                logger.debug('[TIMING] %s ddm_eval %.3fs', ticker, time.perf_counter() - ddm_t0)

        except (TypeError, ValueError, ZeroDivisionError) as err:

            logger.warning('[DDM] %s failed (%s). Returning zeros.', ticker, err)

            ddm.append(_zero_result_entry(
                ticker = ticker
            ))

        if eps_future is None or eps_future.empty:

            logger.info('[RI] %s: skipping residual income valuation (eps_future unavailable).', ticker)

            ri.append(_zero_result_entry(
                ticker = ticker
            ))

        else:

            try:

                ri_t0 = time.perf_counter()

                res_ri = mc_equity_value_per_share_multi_residual_income(
                    ctx = ctx,
                    eps_future = eps_future,
                    dps_future = dps_future,
                    bvps_future = bvps_future,
                    hist_ratios = rat,
                    hist_bal = bal,
                    shares_outstanding = shares_out_t,
                    sector_policy = sector_policy,
                    coe = coe_t,
                    last_price = lp,
                    g_mu = g_ri_hat,
                    g_sd = g_ri_sig,
                    ub = ub_t,
                    lb = lb_t,
                    g_cap = g_coe_cap,
                    seasonal_flow_weights_q1_q4 = None,
                    alignment_cache = shared_alignment_cache
                )

                ri.append({
                    'Ticker': ticker, 
                    'Price': res_ri['per_share_mean'],  
                    'Returns': res_ri['returns_mean'],  
                    'Returns Low': res_ri['returns_p05'],  
                    'Returns High': res_ri['returns_p95'],  
                    'SE': res_ri['returns_std']
                })

                if verbose:

                    logger.debug('[TIMING] %s ri_eval %.3fs', ticker, time.perf_counter() - ri_t0)

            except ValueError as err:

                logger.warning('[RI] %s failed (%s). Returning zeros.', ticker, err)

                ri.append(_zero_result_entry(
                    ticker = ticker
                ))

        del b

        gc.collect()

    dcf_df = pd.DataFrame(dcf).set_index('Ticker')

    fcfe_df = pd.DataFrame(fcfe).set_index('Ticker')

    ddm_df = pd.DataFrame(ddm).set_index('Ticker')

    ri_df = pd.DataFrame(ri).set_index('Ticker')

    print('DCF Results:')

    print(dcf_df)

    print('\nFCFE Results:')

    print(fcfe_df)

    print('\nDDM Results:')

    print(ddm_df)

    print('\nRI Results:')

    print(ri_df)

    sheets = {
        'DCF_CapIQ': dcf_df, 
        'FCFE_CapIQ': fcfe_df,
        'DDM_CapIQ': ddm_df, 
        'RI_CapIQ': ri_df
    }

    if export:

        export_results(
            output_excel_file = output_excel_file or config.MODEL_FILE, 
            sheets = sheets
        )

    return sheets


if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s:%(name)s:%(message)s')

    run_valuation(
        export = True,
        verbose = False
    )
