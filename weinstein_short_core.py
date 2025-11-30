#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared short-side core logic for:
- Weinstein Short Intraday Watcher
- Live logic backtests

This holds:
- All short-side constants
- Price/zone helpers (near / break / ready-to-close)
- Entry/stop/targets helper
- A stateful `eval_short_bar` that mimics the watcher state machine
"""

import numpy as np
import pandas as pd

# ---------------- Shared short-side constants ----------------
INTRADAY_INTERVAL        = "60m"   # override from caller if needed
PIVOT_LOOKBACK_WEEKS     = 10

# Short trigger thresholds (price/volume)
SHORT_BREAK_PCT          = 0.004   # 0.4% below pivot/MA for confirm
NEAR_ABOVE_PIVOT_PCT     = 0.02    # within +2% above pivot considered "near"
VOL_PACE_MIN             = 1.30    # full-day vol pace gate for TRIG shorts
NEAR_VOL_PACE_MIN        = 1.00    # for NEAR shorts
INTRADAY_AVG_VOL_WINDOW  = 20
INTRADAY_LASTBAR_MULT    = 1.20    # only for non-60m modes

# 60m intrabar confirmations
INTRABAR_CONFIRM_MIN_ELAPSED  = 40
INTRABAR_VOLPACE_MIN          = 1.20

# READY-TO-CLOSE threshold
READY_ABOVE_MA_PCT       = 0.005  # 0.5% above MA150 => consider short "ready to close"

# Stateful short triggers
SHORT_NEAR_HITS_WINDOW   = 6
SHORT_NEAR_HITS_MIN      = 3
SHORT_COOLDOWN_SCANS     = 24

# Short risk/profit mapping
SHORT_HARD_STOP_PCT      = 0.20   # 20% above entry
SHORT_TRAIL_ATR_MULT     = 2.0
SHORT_MA_GUARD_PCT       = 0.03   # 3% over MA150
SHORT_TARGET1_PCT        = 0.15   # 15% downside
SHORT_TARGET2_PCT        = 0.20   # 20% downside


# ---------------- Price/zone primitives (copied from watcher) ----------------
def _short_price_break(px, ma, pivot_low):
    """True if price has broken below short zone (~pivot low/MA150)."""
    conds = []
    if pd.notna(pivot_low):
        conds.append(px <= pivot_low * (1.0 - SHORT_BREAK_PCT))
    if pd.notna(ma):
        conds.append(px <= ma * (1.0 - SHORT_BREAK_PCT))
    return any(conds) if conds else False


def _short_near_zone(px, ma, pivot_low):
    """Near-breakdown zone: under MA150 but not yet breaking pivot/MA too hard."""
    if pd.isna(px) or (pd.isna(ma) and pd.isna(pivot_low)):
        return False

    # must be below MA150 (downtrend active)
    below_ma = (pd.notna(ma) and px < ma)
    if not below_ma:
        return False

    # treat "near" as above pivot low but not crazy far
    if pd.notna(pivot_low):
        if px <= pivot_low:  # already at/below pivot; let full trigger handle
            return False
        if px <= pivot_low * (1.0 + NEAR_ABOVE_PIVOT_PCT):
            return True

    # fallback: a mild cushion below MA150 but not full 0.4% break
    if pd.notna(ma):
        if (px <= ma) and (px >= ma * (1.0 - SHORT_BREAK_PCT)):
            return True

    return False


def _short_ready_to_close(px, ma):
    """Price has reclaimed MA150 by READY_ABOVE_MA_PCT (e.g. 0.5+% above)."""
    if pd.isna(px) or pd.isna(ma):
        return False
    return px >= ma * (1.0 + READY_ABOVE_MA_PCT)


def _short_entry_stop_targets(px, ma30, pivot_low, atr):
    """Compute short entry≈px, protective stop, and two downside targets."""
    if pd.isna(px):
        return np.nan, np.nan, np.nan, np.nan

    entry = float(px)

    hard = entry * (1.0 + SHORT_HARD_STOP_PCT)
    atr_stop = (entry + SHORT_TRAIL_ATR_MULT * atr) if pd.notna(atr) else np.nan
    ma_guard = (ma30 * (1.0 + SHORT_MA_GUARD_PCT)) if pd.notna(ma30) else np.nan

    cand = [c for c in (hard, atr_stop, ma_guard) if pd.notna(c)]
    stop = max(cand) if cand else hard

    t1 = entry * (1.0 - SHORT_TARGET1_PCT)
    t2 = entry * (1.0 - SHORT_TARGET2_PCT)
    return entry, stop, t1, t2


# ---------------- Stateful evaluation helper for backtests ----------------
def eval_short_bar(
    price,
    ma30,
    pivot_low,
    pace_full,
    pace_intra,
    elapsed_min,
    closes_tail,
    state,
    *,
    intraday_interval=INTRADAY_INTERVAL,
    test_ease=False,
):
    """Evaluate *one bar* of short-side logic using the same rules as the intraday watcher.

    Parameters
    ----------
    price : float
        Last trade for this bar.
    ma30 : float
        Weekly MA proxy (SMA150 on daily).
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
    state : dict
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

    ma_ok = pd.notna(ma30)
    pivot_ok = pd.notna(pivot_low)

    short_near_now = False
    short_price_ok = False
    short_vol_ok = True
    short_confirm = False
    ready_close_now = False

    # Ease thresholds for synthetic tests
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
        short_near_now = _short_near_zone(price, ma30, pivot_low)

        if intraday_interval == "60m":
            short_price_ok = _short_price_break(price, ma30, pivot_low)
            short_vol_ok = (pd.isna(pace_intra) or pace_intra >= intrabar_volpace_min)
            short_confirm = bool(
                short_price_ok
                and (elapsed_min is not None and elapsed_min >= intrabar_confirm_min)
                and short_vol_ok
            )
        else:
            if closes_tail:
                short_price_ok = all(
                    _short_price_break(c, ma30, pivot_low) for c in closes_tail
                )
                short_confirm = short_price_ok
                # For non-60m modes you can optionally layer in your own
                # last-bar volume confirmation in the backtest if desired.

    if ma_ok and price is not None and not pd.isna(price):
        ready_close_now = _short_ready_to_close(price, ma30)

    # Gates
    pace_full_gate = pd.isna(pace_full) or pace_full >= VOL_PACE_MIN
    near_pace_gate = pd.isna(pace_full) or pace_full >= NEAR_VOL_PACE_MIN

    # ----- stateful promotion (same state machine as watcher) -----
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
