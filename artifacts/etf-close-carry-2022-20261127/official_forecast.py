"""Measure the effect of using the fund's exact August 2022 share multiplier on scoring."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import forecast_2022 as base
from split_event import corrected_actions, SOURCE_URL
import split_event

HERE=Path(__file__).resolve().parent
SNAPSHOT_INPUTS=base.load_inputs
EFFECTIVE=pd.Timestamp("2022-08-29")


def load_inputs(root,end,official=False):
    ref,inputs,evidence=SNAPSHOT_INPUTS(root,end)
    if not official:
        return ref,inputs,evidence
    actions=corrected_actions(inputs[2])
    matching=(actions.symbol=="510500") & (pd.to_datetime(actions.ex_date)==EFFECTIVE)
    assert matching.sum()==int(pd.Timestamp(end)>=EFFECTIVE)
    if matching.any():
        assert actions.loc[matching,"share_multiplier"].item()==1.14539
        assert inputs[2].loc[matching,"share_multiplier"].item()==1.1454
    else:
        pd.testing.assert_frame_equal(actions,inputs[2],check_exact=True)
    return ref,(*inputs[:2],actions,*inputs[3:]),dict(evidence,official_event_applied=bool(matching.any()))


def write_official(directory):
    directory.mkdir(parents=True,exist_ok=True)
    dest=directory/"official_signals.npz"
    manifest=directory/"official_forecast.json"
    if dest.exists() or manifest.exists():
        raise FileExistsError("Choose fresh official forecast outputs")
    snapshot=HERE/"signals.npz"
    snapshot_meta=HERE/"forecast.json"
    previous=json.loads(snapshot_meta.read_bytes())
    assert base.sha(snapshot)==previous["npz_sha256"]
    extra=[]
    original_prefix=base.check_prefix

    def verify(root,annual,fold,scores,end):
        result=original_prefix(root,annual,fold,scores,end)
        if end==pd.Timestamp("2022-06-30"):
            extra.append(original_prefix(root,annual,fold,scores,EFFECTIVE))
        return result

    with patch.object(base,"load_inputs",partial(load_inputs,official=True)), patch.object(base,"check_prefix",verify):
        arrays,report=base.build()
    assert len(extra)==1 and extra[0]["market"]["official_event_applied"]
    assert not any(x["market"]["official_event_applied"] for x in report["prefixes"])
    report["prefixes"].extend(extra)
    assert report["fold"]==previous["fold"] and report["market"]["official_event_applied"]
    prefix=arrays["dates"]<np.datetime64(EFFECTIVE.date())
    dates=pd.DatetimeIndex(arrays["dates"])
    comparison={}
    with np.load(snapshot,allow_pickle=False) as old:
        np.testing.assert_array_equal(arrays["dates"],old["dates"])
        np.testing.assert_array_equal(arrays["symbols"],old["symbols"])
        for key in base.FIELDS:
            raw,new=old[key],arrays[key]
            np.testing.assert_array_equal(raw[prefix],new[prefix])
            changes=(raw!=new).reshape(len(dates),-1).any(axis=1)
            comparison[key]=dict(changed_dates=int(changes.sum()),
                first_changed=str(dates[changes][0].date()) if changes.any() else None,
                max_abs_difference=float(np.abs(new.astype(float)-raw.astype(float)).max()))
        oldr=np.argsort(-old["adaptive"],axis=1,kind="stable")
        newr=np.argsort(-arrays["adaptive"],axis=1,kind="stable")
        reordered=(oldr!=newr).any(axis=1)
        membership=(np.sort(oldr[:,:2],axis=1)!=np.sort(newr[:,:2],axis=1)).any(axis=1)
    report.update(official_source=SOURCE_URL, share_event=dict(symbol="510500",record="2022-08-26",
                  effective="2022-08-29",snapshot_multiplier=1.1454,official_multiplier=1.14539),
                  snapshot_comparison=dict(shared_sessions_before_effective=int(prefix.sum()),
                    field_differences=comparison,changed_rank_dates=[str(d.date()) for d in dates[reordered]],
                    changed_top_two_membership_dates=[str(d.date()) for d in dates[membership]]))
    report["inputs"].update({str(p.relative_to(HERE.parents[1])):base.sha(p) for p in
                             (snapshot,snapshot_meta,Path(__file__),Path(split_event.__file__))})
    with TemporaryDirectory(dir=directory) as tmp:
        bundle=Path(tmp)/dest.name
        np.savez_compressed(bundle,**arrays)
        with np.load(bundle,allow_pickle=False) as check:
            for key,value in arrays.items():
                np.testing.assert_array_equal(check[key],value)
        report["npz_sha256"]=base.sha(bundle)
        meta=Path(tmp)/manifest.name
        meta.write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False)+"\n")
        os.link(bundle,dest)
        try:
            os.link(meta,manifest)
        except OSError:
            dest.unlink()
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=HERE)
    parser.add_argument("--official",action="store_true")
    args=parser.parse_args()
    (write_official if args.official else base.write)(args.out_dir)
