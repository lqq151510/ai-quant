"""Refit the annual forecasts from two physically bounded closes and reproduce their switching decisions."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import forecast_2022 as base
import independent
from official_forecast import load_inputs

HERE = Path(__file__).resolve().parent
SYMBOLS = base.SYMBOLS
DAYS = ("2022-01-18", "2022-03-22")
FEATURES = ("raw_pair_score_edge", "smoothed_pair_score_edge", "relative_uncertainty_edge",
            "market_exposure_cap", "proposed_l1_turnover")


def bounded_decision(root, annual, smoother, day, marked, model):
    end = pd.Timestamp(day)
    ref, inputs, evidence = load_inputs(root, end, official=True)
    assert inputs[1][-1] == end and not evidence["official_event_applied"]
    panel, mature = annual.inference_panel(ref, inputs)
    assert pd.isna(panel.label_end_dates[-1]) and len(panel.dates)-mature == 21
    # Recompute every annual fold to include the complete smoothing warm-up from January 2019.
    with patch.object(annual, "YEARS", (2019, 2020, 2021, 2022)):
        scores, folds = annual.annual_stream(ref, inputs, panel)
    fold = folds[-1]
    assert fold["seed"] == 45 and fold["last_oos"] == day
    assert fold["train"]["latest_label_end"] < fold["first_calibration_before_purge"]
    assert fold["calibration"]["latest_label_end"] == fold["hmm_last_fit_date"] == "2021-12-31"
    dates = pd.DatetimeIndex(scores["dates"])
    assert dates[0] == pd.Timestamp("2019-01-02") and dates[-1] == end
    stream = SimpleNamespace(dates=dates, oos_alpha=scores["adaptive"], oos_uncertainty=scores["uncertainty"],
                             oos_hmm_exposure=scores["hmm_cap"], vol_feats=scores["vol_feats"])
    _, smoothed = smoother(ref.engine)(stream, list(SYMBOLS), 20, 2)
    raw, uncertainty, cap = (scores[k][-1] for k in ("adaptive", "uncertainty", "hmm_cap"))
    uncertainty = uncertainty.astype(float)
    relative = uncertainty / (np.median(uncertainty) + 1e-6)
    # Use only the close's recorded book inventory, cash and accrued income; reprice it from raw bars.
    account = independent.Account(marked["shares"], marked["cash_cents"], [(None, day, marked["receivable_cents"])])
    closing = inputs[0].loc[inputs[0].date == end].set_index("symbol").close
    quotes = {(day, symbol): (None, closing[symbol]) for symbol in SYMBOLS}
    nav = account.value(day, quotes, 1)
    decision = independent.evaluate(day, account, quotes, (raw.astype(float), smoothed[-1], relative, cap), model, "donor")
    z = (np.array(decision["features"])-model["center"])/model["scale"]
    contributions = z*model["coefficients_bp"]
    np.testing.assert_allclose(model["target_mean_bp"]+sum(contributions), decision["donor_bp"], rtol=0, atol=1e-12)
    result = dict(date=day, market=evidence, forecast_seed=fold["seed"], fit_labels_through=fold["train"]["latest_label_end"],
                  calibration_labels_through=fold["calibration"]["latest_label_end"], hmm_fit_through=fold["hmm_last_fit_date"],
                  first_smoothed=str(dates[0].date()), smoothed_sessions=len(dates), forecast_sessions_2022=fold["oos_sessions"],
                  current_label_available=False, raw=raw.tolist(), smoothed=smoothed[-1].tolist(),
                  relative_uncertainty=relative.tolist(), close_prices=closing.loc[list(SYMBOLS)].tolist(), account=marked,
                  nav_cents=nav, **decision, baseline_bp=model["target_mean_bp"],
                  contributions_bp=dict(zip(FEATURES, contributions.tolist(), strict=True)))
    return result, {k:scores[k] for k in base.FIELDS}, scores["dates"]


def run():
    path = HERE/"books.json"
    book = independent.read(path)["phases"][0]["fees"]["normal"]["donor"]
    model = independent.read(independent.DONOR)["models"]["late"]["donor20"]
    assert model["last_label_exit"] < model["cutoff"] == "2020-01-02"
    owned = ("shares", "cash_cents", "receivable_cents")
    marks = {r["date"]: {k:r[k] for k in owned} for r in book["daily"] if r["date"] in DAYS}
    observations = []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory(prefix="etf-decision-cutoff-") as work, threadpool_limits(limits=1), \
            patch.object(torch.backends.mps, "is_available", return_value=False):
        root = base.replay_frozen.materialize(work)
        annual = base.annual_module(root)
        spec = spec_from_file_location("carry_prefix_annual_ledger", root/base.ANNUAL/"ledger.py")
        ledger = module_from_spec(spec)
        spec.loader.exec_module(ledger)
        for day in DAYS:
            result, scores, dates = bounded_decision(root, annual, ledger.policy_constructor, day, marks[day], model)
            # Read the corresponding saved decision and forecasts only after bounded computation has finished.
            event = next(e for e in book["events"] if e["date"] == day)
            assert result["nav_cents"] == event["nav_cents"]
            for name in ("action", "pair", "held", "target"):
                assert result[name] == event[name], (day, name)
            errors = {}
            for name in ("weights", "features", "donor_bp"):
                errors[name] = float(np.abs(np.asarray(result[name])-np.asarray(event[name])).max())
            assert errors["weights"] == 0 and errors["features"] < 1e-5, (day, errors)
            # Input differences and output basis points have different units. Bound their
            # propagation by the frozen linear coefficients, and require the sign to survive.
            delta_x = np.abs(np.asarray(result["features"])-np.asarray(event["features"]))
            slopes = np.abs(np.asarray(model["coefficients_bp"])/model["scale"])
            bp_bound = float(np.dot(delta_x, slopes)) + 1e-10
            assert errors["donor_bp"] <= bp_bound < abs(result["donor_bp"])
            result["derived_max_prediction_difference_bp"] = bp_bound
            result["archived_prediction_bp"] = event["donor_bp"]
            with np.load(HERE/"official_signals.npz", allow_pickle=False) as reference:
                assert tuple(reference["symbols"]) == SYMBOLS
                np.testing.assert_array_equal(dates, reference["dates"][:len(dates)])
                score_errors = {k:float(np.abs(scores[k].astype(float)-reference[k][:len(dates)]).max())
                                for k in base.FIELDS}
            assert max(score_errors.values()) < 1e-5
            result.update(decision_differences=errors, history_score_differences=score_errors)
            observations.append(result)
    assert [r["action"] for r in observations] == ["hold","switch"]
    sources = (Path(__file__), HERE/"forecast_2022.py", HERE/"official_forecast.py", HERE/"independent.py",
               independent.DONOR, path, HERE/"official_signals.npz")
    return dict(offset=0, fee="normal", rule="donor20", symbols=list(SYMBOLS), feature_order=list(FEATURES),
                donor_last_label_exit=model["last_label_exit"], observations=observations,
                frozen_manifest_sha256=base.replay_frozen.MANIFEST_SHA256,
                sources_sha256={str(p.relative_to(HERE.parents[1])):base.sha(p) for p in sources})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"decision_prefix.json")
    out = parser.parse_args().out
    if out.exists():
        parser.error("Choose a fresh output path")
    result = run()
    with out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
