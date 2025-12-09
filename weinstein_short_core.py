#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weinstein_short_core.py

Shared short-side core logic for:
- Weinstein Short Intraday Watcher (email + charts)
- Live logic backtests (weinstein_live_logic_backtest_yfinance.py)

This module owns:
- Short-side constants (price/volume thresholds, risk/targets)
- Core primitives:
    _short_price_break
    _short_near_zone
    _short_ready_to_close
    _short_entry_stop_targets
- A stateful eval_short_bar() helper for the intraday watcher, so the sim
  can use the *same* short trigger rules as the production watcher.

NEW:
- ShortRegimeContext + build_short_regime_from_spy_stage(spy_stage, as_of)
  to implement Weinstein Option 4:
    "Only allow new shorts when SPY itself is in Stage 4 (Downtrend)
     on the weekly chart."

- ShortEntryParams / ShortEntryResult / check_short_entry for DAILY bars
  (used by the live logic backtester), structurally similar to:
      weinstein_long_core.LongEntryParams / check_long_entry

- short_stop_level / should_exit_short shared between PROD/SIM so that
  risk logic (hard stop / ATR / MA guard) is defined in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd


# ---------------- Shared short-side constants ----------------

# Intraday config
INTRADAY_INTERVAL        = "60m"   # Default; caller may override via arg
PIVOT_LOOKBACK_WEEKS     = 10

# Short trigger thresholds (price/volume)
SHORT_BREAK_PCT          = 0.004   # 0.4% below pivot/MA for confirm
NEAR_ABOVE_PIVOT_PCT     = 0.02    # within +2% above pivot considered "near"
VOL_PACE_MIN             = 1.30    # full-day vol pace gate for TRIG shorts
NEAR_VOL_PACE_MIN        = 1.00    # for NEAR shorts
INTRADAY_AVG_VOL_WINDOW  = 20
INTRADAY_LASTBAR_MULT    = 1.20    # only for non-60m modes

# 60m intrabar confirmations
INTRABAR_CONFIRM_MIN_ELAPSED  = 40    # minutes into bar
INTRABAR_VOLPACE_MIN          = 1.20  # intrabar pace vs avg for confirm

# READY-TO-CLOSE threshold
READY_ABOVE_MA_PCT       = 0.005  # 0.5% above MA150 => consider short ready to close

# Stateful short triggers
SHORT_NEAR_HITS_WINDOW   = 6
SHORT_NEAR_HITS_MIN      = 3
SHORT_COOLDOWN_SCANS     = 24

# Short risk/profit mapping (used by both intraday + backtests)
SHORT_HARD_STOP_PCT      = 0.20   # 20% above entry
SHORT_TRAIL_ATR_MULT     = 2.0
SHORT_MA_GUARD_PCT       = 0.03   # 3% over MA (MA150 intraday, MA30 in SIM)
SHORT_TARGET1_PCT        = 0.15   # 15% downside
SHORT_TARGET2_PCT        = 0.20   # 20% downside


# -------------------------------------------------------------------
# Short regime context (Option 4: SPY Stage 4)
# -------------------------------------------------------------------

@dataclass
class ShortRegimeContext:
    """
    Encapsulates whether new SHORT entries are allowed for a given environment.

    We use SPY's weekly Weinstein stage as a simple regime proxy:
      - allow_shorts = True  iff SPY is in Stage 4
      - otherwise False

    Existing shorts are still managed (exits, READY-to-close) even if the
    regime disallows new entries; callers decide how strictly to apply it.
    """
    allow_shorts: bool
    spy_stage: Optional[str] = None   # e.g. "Stage 4 (Downtrend)"
    as_of: Optional[str] = None       # e.g. "2019-08-23"
    note: Optional[str] = None        # human-friendly debug string


def build_short_regime_from_spy_stage(
    spy_stage: Optional[str],
    as_of: Optional[str] = None,
) -> ShortRegimeContext:
    """
    Option 4 (pure Weinstein): shorts only when SPY is Stage 4 on the weekly chart.

    Parameters
    ----------
    spy_stage : str or None
        Value from weekly 'stage' column for SPY, e.g. "Stage 4 (Downtrend)".
        If None/NaN, we conservatively *disable* new shorts.
    as_of : str or None
        Optional as-of label (date) for debug logging.

    Returns
    -------
    ShortRegimeContext
    """
    if spy_stage is None or (isinstance(spy_stage, float) and pd.isna(spy_stage)):
        return ShortRegimeContext(
            allow_shorts=False,
            spy_stage=None,
            as_of=as_of,
            note="No SPY stage available; blocking new shorts by default.",
        )

    stage_str = str(spy_stage)
    allow = stage_str.startswith("Stage 4")
    return ShortRegimeContext(
        allow_shorts=allow,
        spy_stage=stage_str,
        as_of=as_of,
        note=f"SPY stage = {stage_str}, allow_shorts={allow}",
    )


# -------------------------------------------------------------------
# Price/zone primitives (shared with watcher + sim)
# -------------------------------------------------------------------

def _short_price_break(px: float, ma: float, pivot_low: float) -> bool:
    """
    True if price has broken below short zone (~pivot low/MA).

    Used both by:
      - Intraday watcher (ma is typically MA150 on daily)
      - Daily sim (ma is MA30 on daily)
    """
    conds = []
    if pd.notna(pivot_low):
        conds.append(px <= pivot_low * (1.0 - SHORT_BREAK_PCT))
    if pd.notna(ma):
        conds.append(px <= ma * (1.0 - SHORT_BREAK_PCT))
    return any(conds) if conds else False


def _short_near_zone(px: float, ma: float, pivot_low: float) -> bool:
    """
    Near-breakdown zone: under MA but not yet breaking pivot/MA too hard.

    Requirements:
    - price below MA (downtrend active)
    - price above pivot_low but within +NEAR_ABOVE_PIVOT_PCT
      OR hugging MA slightly below full SHORT_BREAK_PCT.
    """
    if pd.isna(px) or (pd.isna(ma) and pd.isna(pivot_low)):
        return False

    below_ma = (pd.notna(ma) and px < ma)
    if not below_ma:
        return False

    if pd.notna(pivot_low):
        if px <= pivot_low:
            return False
        if px <= pivot_low * (1.0 + NEAR_ABOVE_PIVOT_PCT):
            return True

    if pd.notna(ma):
        if (px <= ma) and (px >= ma * (1.0 - SHORT_BREAK_PCT)):
            return True

    return False


def _short_ready_to_close(px: float, ma: float) -> bool:
    """
    READY-TO-CLOSE short:
      - price has reclaimed MA by READY_ABOVE_MA_PCT (e.g. 0.5+% above MA),
        suggesting downtrend thesis is weakening.

    Intraday: MA = weekly MA proxy (MA150 on daily).
    SIM     : MA = MA30 on daily.
    """
    if pd.isna(px) or pd.isna(ma):
        return False
    return px >= ma * (1.0 + READY_ABOVE_MA_PCT)


def _short_entry_stop_targets(
    px: float,
    ma: float,
    pivot_low: float,
    atr: float,
) -> tuple[float, float, float, float]:
    """
    For shorts (generic; caller decides what MA means):
      entry ≈ px (current price)
      stop  = max(
                 entry * (1 + SHORT_HARD_STOP_PCT),
                 entry + SHORT_TRAIL_ATR_MULT * ATR,
                 ma * (1 + SHORT_MA_GUARD_PCT)
              )
      targets = [ -15%, -20% from entry ]

    Returns:
        (entry, stop, target1, target2)
    """
    if pd.isna(px):
        return np.nan, np.nan, np.nan, np.nan

    entry = float(px)

    hard = entry * (1.0 + SHORT_HARD_STOP_PCT)
    atr_stop = (entry + SHORT_TRAIL_ATR_MULT * atr) if pd.notna(atr) else np.nan
    ma_guard = (ma * (1.0 + SHORT_MA_GUARD_PCT)) if pd.notna(ma) else np.nan

    cand = [c for c in (hard, atr_stop, ma_guard) if pd.notna(c)]
    stop = max(cand) if cand else hard

    t1 = entry * (1.0 - SHORT_TARGET1_PCT)
    t2 = entry * (1.0 - SHORT_TARGET2_PCT)
    return entry, stop, t1, t2


# -------------------------------------------------------------------
# DAILY short entry core (shared with backtester)
# -------------------------------------------------------------------

@dataclass
class ShortEntryParams:
    """
    Tunable thresholds for the short entry decision (DAILY approximation).

    These are generic enough that both:
      - intraday logic (which might call this on end-of-day),
      - daily SIM logic (daily bars)

    can share them.

    Attributes:
        min_break_pct:
            Required % breakdown under pivot low (e.g. 0.004 = 0.4%).
        vol_min:
            Minimum volume multiple vs 50dma (e.g. 1.3).
    """
    min_break_pct: float = SHORT_BREAK_PCT
    vol_min: float = VOL_PACE_MIN


@dataclass
class ShortEntryResult:
    """
    Outcome of the short entry check for DAILY bars.

    Attributes:
        can_enter:
            True when all gates (RS, MA, pivot, vol) pass.
        reason:
            Short diagnostic string explaining the first reason for rejection
            (or "ok" if can_enter=True).
    """
    can_enter: bool
    reason: str


def _is_nan(x: float) -> bool:
    try:
        return bool(np.isnan(x))
    except Exception:
        return False


def check_short_entry(
    *,
    price: float,
    ma_val: float,
    pivot_low: float,
    rs_above_ma: bool,
    vol_mult: float,
    params: Optional[ShortEntryParams] = None,
    # Optional overrides (if a caller wants to tweak thresholds ad-hoc)
    min_break_pct: Optional[float] = None,
    vol_min: Optional[float] = None,
) -> ShortEntryResult:
    """
    Shared Weinstein Stage 4 SHORT entry filter for DAILY bars.

    This is the short-side mirror of weinstein_long_core.check_long_entry,
    tuned to match your current live-logic backtest's `should_enter_short`
    behavior.

    Inputs:
        price:
            Current price (close / last).
        ma_val:
            MA(30) (or your equivalent trend MA).
        pivot_low:
            Lowest close in lookback window (e.g. 50d).
        rs_above_ma:
            True if RS line is above its MA (strong RS).
            For shorts we require *weak* RS, i.e. rs_above_ma must be False.
        vol_mult:
            Volume multiple vs 50dma (e.g. 1.3 means 30% above).
        params:
            ShortEntryParams object with thresholds.
        min_break_pct / vol_min:
            Optional explicit overrides. If provided, they win over params.

    Returns:
        ShortEntryResult(can_enter, reason)
    """
    if params is None:
        params = ShortEntryParams()

    thr_break = min_break_pct if min_break_pct is not None else params.min_break_pct
    thr_vol = vol_min if vol_min is not None else params.vol_min

    # --- Basic NaN guards ---
    if _is_nan(price) or _is_nan(ma_val) or _is_nan(pivot_low):
        return ShortEntryResult(
            can_enter=False,
            reason="nan_input",
        )

    # --- RS must NOT be above its MA (weak RS required for shorts) ---
    if rs_above_ma:
        return ShortEntryResult(
            can_enter=False,
            reason="rs_too_strong_for_short",
        )

    # --- Price must be below MA ---
    if price > ma_val:
        return ShortEntryResult(
            can_enter=False,
            reason="price_not_below_ma",
        )

    # --- Breakdown vs pivot low (e.g. 0.4% under 50-day low) ---
    required_pivot_level = pivot_low * (1.0 - thr_break)
    if price > required_pivot_level:
        return ShortEntryResult(
            can_enter=False,
            reason="no_breakdown_vs_pivot",
        )

    # --- Volume pace filter (e.g. ≥ 1.3× 50dma) ---
    if _is_nan(vol_mult) or vol_mult < thr_vol:
        return ShortEntryResult(
            can_enter=False,
            reason="volume_too_low",
        )

    return ShortEntryResult(
        can_enter=True,
        reason="ok",
    )


# -------------------------------------------------------------------
# Shared short-side stop / exit helpers (used by SIM + optionally PROD)
# -------------------------------------------------------------------

def short_stop_level(entry: float, atr: float, ma_val: float) -> float:
    """
    Compute an initial stop for a short position.

    Generic form used by:
      - SIM: ma_val = MA30
      - Intraday PROD: ma_val = MA150 (if you choose to reuse it)

    stop = min(
        entry * (1 + SHORT_HARD_STOP_PCT),
        entry + SHORT_TRAIL_ATR_MULT * ATR,
        ma_val * (1 + SHORT_MA_GUARD_PCT)
    )
    """
    if _is_nan(entry):
        return np.nan

    hard = entry * (1.0 + SHORT_HARD_STOP_PCT)
    atr_stop = entry + SHORT_TRAIL_ATR_MULT * atr if not _is_nan(atr) else np.nan
    ma_guard = ma_val * (1.0 + SHORT_MA_GUARD_PCT) if not _is_nan(ma_val) else np.nan

    cands = [c for c in (hard, atr_stop, ma_guard) if not _is_nan(c)]
    return min(cands) if cands else hard


def should_exit_short(price: float, stop: float, ma_val: float) -> bool:
    """
    Exit condition for a short:

      1) price >= stop  (hard/ATR/MA guard violated)
      2) price has reclaimed MA by ~SHORT_MA_GUARD_PCT
         (extra trend-guard).

    Used directly by the live-logic backtest; you can also reuse this
    in PROD if you want a shared definition.
    """
    if _is_nan(price):
        return False

    # 1) Stop violation
    if not _is_nan(stop) and price >= stop:
        return True

    # 2) Extra guard: reclaimed MA by ~3%
    if not _is_nan(ma_val) and price >= ma_val * (1.0 + SHORT_MA_GUARD_PCT):
        return True

    return False


# -------------------------------------------------------------------
# Stateful evaluation helper for intraday short watcher
# -------------------------------------------------------------------

def eval_short_bar(
    price: float,
    ma: float,
    pivot_low: float,
    pace_full: float,
    pace_intra: float,
    elapsed_min: Optional[int],
    closes_tail: Optional[Sequence[float]],
    state: Optional[Dict[str, Any]],
    *,
    intraday_interval: str = INTRADAY_INTERVAL,
    test_ease: bool = False,
) -> tuple[Dict[str, Any], Dict[str, bool]]:
    """
    Evaluate *one bar* of short-side logic using the same rules
    as the intraday watcher.

    Parameters
    ----------
    price : float
        Last trade for this bar.
    ma : float
        Weekly MA proxy (SMA150 on daily in PROD).
    pivot_low : float
        ~10-week pivot low from weekly report.
    pace_full : float or NaN
        Projected full-day volume / 50dma.
    pace_intra : float or NaN
        Intrabar volume pace vs avg (for 60m).
    elapsed_min : int or None
        Minutes elapsed in current bar (60m mode).
    closes_tail : sequence[float] or None
        Last N closes used for non-60m confirmation (e.g. 2 bars).
    state : dict or None
        Stateful dict with at least:
          { "short_state": str, "short_hits": list[int], "short_cooldown": int }

    Returns
    -------
    new_state : dict
        Updated state dict (same keys as input).
    flags : dict
        {
          "short_near_now": bool,
          "short_trigger_now": bool,  # one-shot event on this bar
          "ready_close_now": bool,
          "short_price_ok": bool,
          "short_vol_ok": bool,
          "short_confirm": bool,
        }
    """
    if state is None:
        state = {"short_state": "IDLE", "short_hits": [], "short_cooldown": 0}

    ma_ok = pd.notna(ma)
    pivot_ok = pd.notna(pivot_low)

    short_near_now = False
    short_price_ok = False
    short_vol_ok = True
    short_confirm = False
    ready_close_now = False

    # Easier thresholds in test mode (for sims/unit tests)
    if test_ease:
        short_near_hits_min = 1
        intrabar_confirm_min = 0
        intrabar_volpace_min = 0.0
    else:
        short_near_hits_min = SHORT_NEAR_HITS_MIN
        intrabar_confirm_min = INTRABAR_CONFIRM_MIN_ELAPSED
        intrabar_volpace_min = INTRABAR_VOLPACE_MIN

    # ----- price / confirmation logic -----
    if ma_ok and pivot_ok and price is not None and not pd.isna(price):
        short_near_now = _short_near_zone(price, ma, pivot_low)

        if intraday_interval == "60m":
            short_price_ok = _short_price_break(price, ma, pivot_low)
            short_vol_ok = (pd.isna(pace_intra) or pace_intra >= intrabar_volpace_min)
            short_confirm = bool(
                short_price_ok
                and (elapsed_min is not None and elapsed_min >= intrabar_confirm_min)
                and short_vol_ok
            )
        else:
            if closes_tail:
                short_price_ok = all(
                    _short_price_break(c, ma, pivot_low) for c in closes_tail
                )
                short_confirm = short_price_ok
                # For non-60m modes you can optionally layer in your own
                # last-bar volume confirmation in the watcher if desired.

    if ma_ok and price is not None and not pd.isna(price):
        ready_close_now = _short_ready_to_close(price, ma)

    # Gates
    pace_full_gate = pd.isna(pace_full) or pace_full >= VOL_PACE_MIN
    near_pace_gate = pd.isna(pace_full) or pace_full >= NEAR_VOL_PACE_MIN

    # ----- stateful promotion (same logic as watcher) -----
    hits = list(state.get("short_hits", []))
    hits.append(1 if short_near_now else 0)
    if len(hits) > SHORT_NEAR_HITS_WINDOW:
        hits = hits[-SHORT_NEAR_HITS_WINDOW:]
    hit_count = sum(hits)

    cooldown = int(state.get("short_cooldown", 0))
    if cooldown > 0:
        cooldown -= 1

    sstate = state.get("short_state", "IDLE")

    if sstate == "IDLE" and short_near_now:
        sstate = "NEAR"
    elif sstate in ("IDLE", "NEAR") and hit_count >= short_near_hits_min:
        sstate = "ARMED"
    elif (
        sstate == "ARMED"
        and short_confirm
        and short_vol_ok
        and pace_full_gate
    ):
        sstate = "TRIGGERED"
        cooldown = SHORT_COOLDOWN_SCANS
    elif cooldown > 0 and not short_near_now:
        sstate = "COOLDOWN"
    elif cooldown == 0 and not short_near_now and not short_confirm:
        sstate = "IDLE"

    # One-shot trigger event (mirror watcher: emit + drop to COOLDOWN)
    emit_trigger = False
    if sstate == "TRIGGERED" and pace_full_gate:
        emit_trigger = True
        sstate = "COOLDOWN"

    new_state = {
        "short_state": sstate,
        "short_hits": hits,
        "short_cooldown": cooldown,
    }

    flags = {
        "short_near_now": bool(short_near_now and near_pace_gate),
        "short_trigger_now": bool(emit_trigger),
        "ready_close_now": bool(ready_close_now),
        "short_price_ok": bool(short_price_ok),
        "short_vol_ok": bool(short_vol_ok),
        "short_confirm": bool(short_confirm),
    }
    return new_state, flags
