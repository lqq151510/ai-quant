"""Extend the authenticated annual ETF scoring procedure through the last 2022 close."""
import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent
CAUSAL = ARTIFACTS/"etf-causal-inputs-20261127"
sys.path.insert(0,str(CAUSAL))
import annual_caps
import prefix_audit
import replay_frozen

END = pd.Timestamp("2022-12-30")
ANNUAL = Path("artifacts/etf-annual-expert-20261126")
FIELDS = ("adaptive","uncertainty","vol_feats","hmm_cap")
SYMBOLS = ("510050","510300","510500","510880")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annual_module(root):
    root = Path(root).resolve()
    sys.path[:0] = [str(root/ANNUAL),str(root)]
    import forecasts
    if Path(forecasts.__file__).resolve() != root/ANNUAL/"forecasts.py":
        raise RuntimeError("Use a fresh worker for each materialized model source directory")
    return forecasts


def load_inputs(root, end):
    """Return an authenticated reference loader, truncated market tuple and its provenance."""
    root, end = Path(root).resolve(), pd.Timestamp(end)
    if not pd.Timestamp("2016-07-01") <= end <= END:
        raise ValueError("Outside the registered historical market window")
    annual = annual_module(root)
    ref,_ = annual.observer.q.checked_loader()
    assert Path(ref.engine.__file__).resolve() == root/"src/application/walk_forward_engine.py"
    # Only the unversioned, study-era relationship prior comes from this warm-up query.
    with patch.object(ref,"END",pd.Timestamp("2016-07-01")):
        prior,provenance = ref.market_inputs("etf")
    assert provenance["static_relationships_are_legacy_priors"]
    inputs = prefix_audit.inputs_at(ref,root,end,prior[4])
    bars,calendar,actions = inputs[:3]
    assert pd.to_datetime(bars.date).max()==calendar[-1]==end
    assert calendar.is_monotonic_increasing and calendar.is_unique
    assert len(bars)==len(calendar)*len(SYMBOLS)
    assert not bars.duplicated(["date","symbol"]).any()
    assert tuple(sorted(bars.symbol.unique()))==SYMBOLS
    assert pd.to_datetime(actions.ex_date).max()<=end
    evidence = {k:v for k,v in provenance.items() if k!="materialized_frames"}
    evidence.update(first_market_day=str(calendar[0].date()),last_market_day=str(end.date()),
                    calendar_rows=len(calendar), bar_rows=len(bars), actions_rows=len(actions),
                    last_known_ex_date=str(pd.to_datetime(actions.ex_date).max().date()),
                    prior_read_through="2016-07-01")
    return ref,inputs,evidence


def check_prefix(root,annual,fold,scores,end):
    ref,inputs,evidence=load_inputs(root,end)
    panel,mature=annual.inference_panel(ref,inputs)
    tr,va=prefix_audit.split_ids(panel,fold["train"],fold["calibration"])
    dates=pd.DatetimeIndex(panel.dates)
    te=np.flatnonzero(dates.year==2022)
    first=pd.Timestamp(fold["first_oos"])
    assert dates[te[0]]==first and dates[te[-1]]==end
    result,_=annual.fold_scores(ref,panel,inputs[4],tr,va,te,fold["seed"])
    macro=ref.engine._build_macro_df_from_panel_returns(panel.close_returns.index,
                                                         panel.close_returns.to_numpy())
    caps,fit_dates=annual_caps.fit_caps(ref.engine,macro.set_index("trade_date",drop=False),
                                         first,dates[te],fold["seed"])
    assert str(fit_dates[-1].date())==fold["hmm_last_fit_date"]
    result.update(hmm_cap=caps,vol_feats=panel.vol_feats[te])
    idx=np.flatnonzero((scores["dates"]>=np.datetime64(first.date()))&
                       (scores["dates"]<=np.datetime64(end.date())))
    assert np.array_equal(scores["dates"][idx],dates[te].to_numpy(dtype="datetime64[D]"))
    gaps={}
    for k in FIELDS:
        a,b=scores[k][idx],result[k]
        assert a.shape==b.shape and np.isfinite(b).all()
        gaps[k]=float(np.max(np.abs(a.astype(float)-b.astype(float))))
        assert gaps[k]<1e-5,(k,gaps[k])
    assert np.array_equal(np.argsort(-result["adaptive"],axis=1,kind="stable"),
                          np.argsort(-scores["adaptive"][idx],axis=1,kind="stable"))
    return dict(cutoff=str(end.date()), sessions=len(idx),mature_panel_rows=mature,
                training_rows=len(tr),calibration_rows=len(va),max_abs_difference=gaps,
                rank_order_equal=True,market=evidence)


def build():
    with TemporaryDirectory(prefix="annual-etf-2022-") as temp, threadpool_limits(limits=1):
        root=replay_frozen.materialize(temp)
        annual=annual_module(root)
        ref,inputs,evidence=load_inputs(root,END)
        assert annual.YEARS==(2019,2020,2021) and annual.CALIBRATION_SESSIONS==230 and annual.HORIZON==20
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with patch.object(torch.backends.mps,"is_available",return_value=False):
            panel,mature=annual.inference_panel(ref,inputs)
            with patch.object(annual,"YEARS",(*annual.YEARS,2022)):
                all_scores,folds=annual.annual_stream(ref,inputs,panel)
            fold=folds[-1]
            assert fold["seed"]==45 and fold["year"]==2022
            assert fold["train"]["latest_label_end"]<fold["first_calibration_before_purge"]
            assert fold["calibration"]["latest_label_end"]<fold["first_oos"]
            assert fold["hmm_last_fit_date"]=="2021-12-31"
            archive=root/ANNUAL/"forecasts.npz"
            archive_meta=root/ANNUAL/"forecast_meta.json"
            metadata=json.loads(archive_meta.read_bytes())
            assert sha(archive)==metadata["npz_sha256"]
            assert evidence["snapshot_manifest_sha256"]==metadata["inputs"]["snapshot_manifest_sha256"]
            with np.load(archive,allow_pickle=False) as old:
                n=len(old["dates"])
                assert np.array_equal(old["symbols"],all_scores["symbols"])
                assert np.array_equal(old["dates"],all_scores["dates"][:n])
                prior_rebuild={}
                result={"symbols":old["symbols"].copy(),"dates":all_scores["dates"]}
                for k in FIELDS:
                    prior=old[k]
                    gap=np.max(np.abs(prior.astype(float)-all_scores[k][:n].astype(float)))
                    assert gap<1e-5,(k,gap)
                    prior_rebuild[k]=float(gap)
                    result[k]=np.concatenate((prior,all_scores[k][n:]))
                    assert np.array_equal(result[k][:n],prior)
            assert pd.DatetimeIndex(result["dates"])[n:].equals(inputs[1][inputs[1].year==2022])
            assert np.isfinite(np.concatenate([result[k].ravel() for k in FIELDS])).all()
            tests=[check_prefix(root,annual,fold,result,pd.Timestamp(day)) for day in (fold["first_oos"],"2022-06-30")]
            paths=[HERE/"protocol.md",Path(__file__),Path(prefix_audit.__file__),Path(annual_caps.__file__),
                   Path(replay_frozen.__file__),root/ANNUAL/"forecasts.py",archive,archive_meta]
            pins={str(p.relative_to(root) if p.is_relative_to(root) else p.relative_to(HERE.parents[1])):sha(p) for p in paths}
            meta=dict(fold=fold,prefixes=tests,market=evidence,prior_archive_sessions=n,
                      prior_2019_2021_retained_exact=True,prior_retraining_max_abs_difference=prior_rebuild,
                      fully_mature_panel_rows=mature,unlabeled_tail=len(panel.dates)-mature,
                      rows=len(result["dates"]),symbols=list(SYMBOLS),epochs=10,calibration_raw_sessions=230,
                      inputs=pins,frozen_materialization_manifest=replay_frozen.MANIFEST_SHA256,
                      versions={p:version(p) for p in metadata["versions"]},
                      array_sha256={k:ref.array_hash([v]) for k,v in result.items()})
            return result,meta


def write(directory):
    directory.mkdir(parents=True,exist_ok=True)
    names=("signals.npz","forecast.json")
    if any((directory/name).exists() for name in names):
        raise FileExistsError("Choose a directory with fresh output names")
    arrays,report=build()
    with TemporaryDirectory(dir=directory) as work:
        npz=Path(work)/names[0]
        np.savez_compressed(npz,**arrays)
        with np.load(npz,allow_pickle=False) as read:
            for k,v in arrays.items():
                np.testing.assert_array_equal(read[k],v)
        report["npz_sha256"]=sha(npz)
        manifest=Path(work)/names[1]
        manifest.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
        os.link(npz,directory/names[0])
        try:
            os.link(manifest,directory/names[1])
        except OSError:
            (directory/names[0]).unlink()
            raise


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out-dir",type=Path,default=HERE)
    write(cli.parse_args().out_dir)
