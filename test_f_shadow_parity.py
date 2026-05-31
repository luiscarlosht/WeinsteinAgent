#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from weinstein_daily_sim_prod_compare import compare_signals, effective_f_signals
from weinstein_prod_account_router import read_latest_parity


def signals(*rows):
    return pd.DataFrame(rows, columns=["Ticker", "Signal", "Price", "Reason", "Source"])


def meta(profile):
    return pd.DataFrame([{"date": "2026-05-30", "meta_profile": profile}])


class EffectiveFSignalsTest(unittest.TestCase):
    def setUp(self):
        self.sim_d = signals(("D_ONLY", "BUY", 1, "", "SIM_D"))
        self.sim_e = signals(("E_ONLY", "NEAR", 2, "", "SIM_E"))
        self.sim_f_raw = signals(
            ("LONG", "BUY", 3, "", "SIM_F_RAW"),
            ("SHORT", "SHORT", 4, "", "SIM_F_RAW"),
        )

    def test_profile_d_selects_d_lane(self):
        out = effective_f_signals(self.sim_d, self.sim_e, self.sim_f_raw, meta("D"))
        self.assertEqual(set(out["Ticker"]), {"D_ONLY"})
        self.assertEqual(set(out["F_MetaProfile"]), {"D"})

    def test_profile_e_selects_strict_lane(self):
        out = effective_f_signals(self.sim_d, self.sim_e, self.sim_f_raw, meta("E"))
        self.assertEqual(set(out["Ticker"]), {"E_ONLY"})
        self.assertEqual(set(out["F_MetaProfile"]), {"E"})

    def test_profile_a_removes_short_candidates(self):
        out = effective_f_signals(self.sim_d, self.sim_e, self.sim_f_raw, meta("A"))
        self.assertEqual(set(out["Ticker"]), {"LONG"})

    def test_profile_b_uses_broad_lane(self):
        out = effective_f_signals(self.sim_d, self.sim_e, self.sim_f_raw, meta("B"))
        self.assertEqual(set(out["Ticker"]), {"LONG", "SHORT"})

    def test_comparison_reports_effective_and_raw_f_separately(self):
        prod = signals(("E_ONLY", "NEAR", 2, "", "PROD"))
        effective = effective_f_signals(self.sim_d, self.sim_e, self.sim_f_raw, meta("E"))
        out = compare_signals(prod, self.sim_d, effective, sim_f_raw=self.sim_f_raw)
        row = out[out["Ticker"].eq("E_ONLY")].iloc[0]
        self.assertEqual(row["SIM_F_EffectiveSignal"], "NEAR")
        self.assertEqual(row["SIM_F_RawSignal"], "")
        self.assertTrue(row["PROD_Latest_vs_F_Match"])


class LatestParityTest(unittest.TestCase):
    def test_router_prefers_effective_f_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            parity = Path(tmp)
            signals(("D", "BUY", 1, "", "SIM_D")).to_csv(parity / "sim_D_replay_events.csv", index=False)
            signals(("RAW", "BUY", 2, "", "SIM_F_RAW")).to_csv(parity / "sim_F_base_events.csv", index=False)
            signals(("EFFECTIVE", "NEAR", 3, "", "SIM_F_EFFECTIVE")).to_csv(parity / "sim_F_effective_events.csv", index=False)

            _, sim_f, _, info = read_latest_parity(str(parity))

            self.assertEqual(set(sim_f["Ticker"]), {"EFFECTIVE"})
            self.assertTrue(info["sim_f_path"].endswith("sim_F_effective_events.csv"))


if __name__ == "__main__":
    unittest.main()
