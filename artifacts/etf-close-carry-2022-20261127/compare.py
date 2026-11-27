"""Compare common-start portfolios and matched switch/hold branches in three periods."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from statistics import mean, median

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/"etf-close-asset-values-20261127"/"asset_books.json"
FORKS=HERE.parent/"etf-close-split-consensus-20261127"/"complement_books.json"
ARMS=("donor","hold","switch")
FEES=("normal","double")


def metrics(paths):
    scores={k:[] for k in ("hold","switch","both")}
    phase0={}
    by_phase=[]
    for offset,fees in enumerate(paths):
        margins,dds=[],[]
        passed={k:True for k in ("hold","switch")}
        for fee,arms in fees.items():
            assert set(arms)==set(ARMS)
            assert len({a["starting_nav"] for a in arms.values()})==1
            d=arms["donor"]
            for name in ("hold","switch"):
                c=arms[name]
                gain=100*(d["net_return"]-c["net_return"])
                reduction=100*(c["max_drawdown"]-d["max_drawdown"])
                margins.append(gain)
                dds.append(reduction)
                passed[name] &= gain>0 and reduction>=0
            if offset==0:
                phase0[fee]={k:{s:a[s] for s in ("net_return","max_drawdown","starting_nav")} for k,a in arms.items()}
        for k,v in passed.items():
            if v: scores[k].append(offset)
        if all(passed.values()): scores["both"].append(offset)
        by_phase.append(dict(offset=offset,minimum_gain_pp=min(margins),minimum_drawdown_reduction_pp=min(dds)))
    assert len(paths)==20
    return dict(phase0=phase0,passing=scores,passes=0 in scores["both"] and len(scores["both"])>=11,
                median_minimum_gain_pp=median(r["minimum_gain_pp"] for r in by_phase),offsets=by_phase)


def branch_stats(phases):
    all_rows=[dict(r,phase=i) for i,rows in enumerate(phases) for r in rows]
    changed=[r for r in all_rows if r["action"] in ("switch","hold")]
    observed=sorted((r for r in changed if r["label_bp"] is not None), key=lambda r:(r["date"],r["phase"]))
    groups={a:[r for r in observed if r["action"]==a] for a in ("switch","hold")}
    months=defaultdict(list)
    for r in observed:
        assert r["date"]<r["entry"]<r["exit"]
        assert abs(r["label_bp"]-(r["switch_exit_cents"]-r["hold_exit_cents"])*10000/r["nav_cents"])<1e-8
        assert r["actual_exit_cents"] == r[r["action"]+"_exit_cents"]
        months[r["date"][:7]].append(r)
    n=len(observed)
    selected=fsum(r["label_bp"] for r in groups["switch"])
    skipped=fsum(r["label_bp"] for r in groups["hold"])
    same_frequency=fsum(sum(r["action"]=="switch" for r in rows)*mean(r["label_bp"] for r in rows)
                         for rows in months.values())
    action_metrics={}
    for a,rows in groups.items():
        phase0=[r for r in rows if r["phase"]==0]
        action_metrics[a]=dict(count=len(rows),actual_switch_better=sum(r["label_bp"]>0 for r in rows),
                               mean_switch_minus_hold_bp=mean(r["label_bp"] for r in rows) if rows else None,
                               earliest_phase0=phase0[0] if phase0 else None)
    return dict(reviews=len(all_rows),changed=len(changed),mature=n,changed_unmatured=len(changed)-n,
                immature_all=sum(r["label_bp"] is None for r in all_rows),actions=action_metrics,
                mean_vs_hold_bp=selected/n if n else None,mean_vs_switch_bp=-skipped/n if n else None,
                same_month_selection_bp=(selected-same_frequency)/n if n else None)


def run(extension_path):
    historical=json.loads(PRIOR.read_bytes())["windows"]
    matched=json.loads(FORKS.read_bytes())["windows"]
    new=json.loads(extension_path.read_bytes())
    results={}
    for w,period in (("early","2017-2018"),("late","2020-2021")):
        paths=[]
        refs=[]
        for phase,other in zip(historical[w]["phases"],matched[w]["phases"],strict=True):
            offset=phase["offset"]
            assert offset==other["offset"]==len(paths)
            group={f:{k:phase["fees"][f][name]["periods"]["post"] for k,name in
                      zip(ARMS, ("donor20","always_hold","always_switch"),strict=True)} for f in FEES}
            paths.append(group)
            reference=other["fees"]["normal"]["combined"]
            assert reference["first_difference_from_donor"] is None
            assert reference["periods"]["post"]==group["normal"]["donor"]
            refs.append(reference["own_forks"])
        results[period]=dict(ledger=metrics(paths),matched=branch_stats(refs))
    phases=new["phases"]
    paths=[{f:{a:p["fees"][f][a]["periods"]["post"] for a in ARMS} for f in FEES} for p in phases]
    refs=[p["fees"]["normal"]["donor"]["own_forks"] for p in phases]
    results["2022"]=dict(ledger=metrics(paths),matched=branch_stats(refs))
    assert results["2022"]["ledger"]["passing"]["both"]==new["summary"]["passing_offsets"]
    paths=[PRIOR,FORKS,extension_path,HERE/"protocol.md",Path(__file__)]
    return dict(periods=results,sources={str(p.relative_to(HERE.parents[1])) if p.is_relative_to(HERE.parents[1])
                else p.name:sha256(p.read_bytes()).hexdigest() for p in paths})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--books",type=Path,default=HERE/"books.json")
    cli.add_argument("--out",type=Path,default=HERE/"comparison.json")
    args=cli.parse_args()
    if args.out.exists():
        cli.error("Choose a new output")
    result=run(args.books.resolve())
    with args.out.open("x") as dst:
        dst.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
