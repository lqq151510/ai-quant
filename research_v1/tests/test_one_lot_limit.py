from datetime import date
import unittest

import pandas as pd

from quant_research.ledger import Costs, LedgerError, opening_execution_price, simulate


FRIDAY = "2023-08-25"
MONDAY = "2023-08-28"
TUESDAY = "2023-08-29"
WEDNESDAY = "2023-08-30"


def bar(day, opening=10.0, **changes):
    return {
        "date": day,
        "symbol": "510300",
        "kind": "etf",
        "open": opening,
        "high": opening,
        "low": opening,
        "close": opening,
        "volume": 1000,
        "amount": opening * 1000,
        "tradable": True,
        "upper_limit": 11.0,
        "lower_limit": 9.0,
    } | changes


class OneLotLimitTests(unittest.TestCase):
    def test_cap_is_checked_after_rounding_and_buy_size_is_fixed(self):
        bars = pd.DataFrame([bar(FRIDAY, 4.001, lower_limit=3.6),
                             bar(MONDAY, 4.001, lower_limit=3.6),
                             bar(TUESDAY, 4.02, lower_limit=3.6)])
        result = simulate(bars, [FRIDAY, MONDAY, TUESDAY], {}, initial_cash=2000,
                          entry_limits={FRIDAY: 4.003, MONDAY: 4.004})
        skipped, buy, sell = result.orders.itertuples(index=False)

        self.assertEqual(opening_execution_price(4.001, Costs(), "buy"), 4.004)
        self.assertEqual((skipped.reason, skipped.filled_qty, skipped.cash_delta), ("price_above_limit", 0, 0))
        self.assertEqual((buy.date, buy.filled_qty, buy.price, buy.commission),
                         (date.fromisoformat(MONDAY), 100, 4.004, 5.0))
        self.assertEqual((sell.date, sell.filled_qty, sell.price), (date.fromisoformat(TUESDAY), 100, 4.017))
        self.assertEqual(result.positions["sellable_quantity"].tolist(), [0])
        self.assertEqual(result.equity["cash"].tolist(), [2000.0, 1594.60, 1991.30])

    def test_limit_down_retains_position_and_blocks_new_buy_until_exit(self):
        days = [FRIDAY, MONDAY, TUESDAY, WEDNESDAY]
        bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY, 9.0), bar(TUESDAY, 9.7), bar(WEDNESDAY, 9.8)])
        result = simulate(bars, days, {}, initial_cash=2000,
                          entry_limits={FRIDAY: 10.01, MONDAY: 9.1, TUESDAY: 9.8})
        summary = [(order.side, order.status, order.reason) for order in result.orders.itertuples(index=False)]

        self.assertEqual(summary, [
            ("buy", "filled", ""), ("sell", "rejected", "limit_down"),
            ("buy", "rejected", "position_held"), ("sell", "filled", ""),
            ("buy", "filled", ""), ("sell", "filled", ""),
        ])
        self.assertEqual(result.orders["date"].astype(str).tolist(),
                         [FRIDAY, MONDAY, MONDAY, TUESDAY, TUESDAY, WEDNESDAY])
        self.assertEqual(result.positions["quantity"].tolist(), [100, 100, 100])
        self.assertEqual(result.positions["sellable_quantity"].tolist(), [0, 100, 0])

    def test_dividend_is_received_even_after_position_exits(self):
        bars = pd.DataFrame([bar(FRIDAY), bar(MONDAY, 9.7), bar(TUESDAY, 9.8)])
        actions = pd.DataFrame([{
            "symbol": "510300", "record_date": FRIDAY, "ex_date": MONDAY, "pay_date": TUESDAY,
            "cash_per_share": 0.3, "share_multiplier": 1.0,
        }])
        result = simulate(bars, [FRIDAY, MONDAY, TUESDAY], {}, actions, initial_cash=10_000,
                          entry_limits={FRIDAY: 10.01})

        self.assertEqual(result.orders["side"].tolist(), ["buy", "sell"])
        self.assertEqual(result.cash_events["event"].tolist(), ["dividend_receivable", "dividend_paid"])
        self.assertEqual(result.equity["receivable"].tolist(), [0.0, 30.0, 0.0])
        self.assertEqual(result.equity["nav"].tolist(), [9994.50, 9989.0, 9989.0])

    def test_empty_limit_schedule_keeps_cash_and_modes_are_exclusive(self):
        bars = pd.DataFrame([bar(FRIDAY)])
        cash = simulate(bars, [FRIDAY], {}, initial_cash=1000, entry_limits={})
        self.assertTrue(cash.orders.empty)
        self.assertEqual(cash.equity["nav"].tolist(), [1000.0])
        with self.assertRaisesRegex(LedgerError, "cannot be combined"):
            simulate(bars, [FRIDAY], {FRIDAY: {"510300": 1.0}}, initial_cash=1000, entry_limits={})
        with self.assertRaisesRegex(LedgerError, "entry limit"):
            simulate(bars, [FRIDAY], {}, initial_cash=1000, entry_limits={FRIDAY: float("nan")})
