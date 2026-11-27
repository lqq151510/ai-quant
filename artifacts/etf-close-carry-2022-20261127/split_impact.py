"""Measure the official August distribution against the archived rounding convention."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX = "2022-08-29"
RECORD = "2022-08-26"


def run():
    names = ["books.json", "snapshot_books.json", "official_forecast.json", "510500_split_notice.pdf"]
    sources = {name: sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    official, snapshot, forecast = [json.loads((HERE / name).read_bytes()) for name in names[:3]]
    assert not official["snapshot_mode"] and snapshot["snapshot_mode"]
    assert official["dates"] == snapshot["dates"]
    count = Counter()
    forks = Counter()
    paths = []
    event_sample = None
    for po, ps in zip(official["phases"], snapshot["phases"], strict=True):
        assert po["offset"] == ps["offset"]
        for fee in ("normal", "double"):
            for arm in ("donor", "hold", "switch"):
                a, b = [p["fees"][fee][arm] for p in (po, ps)]
                assert a["initial"] == b["initial"]
                record_units = None
                max_gap = 0
                changed = []
                for da, db in zip(a["daily"], b["daily"], strict=True):
                    day = da["date"]
                    assert day == db["date"]
                    if day < EX:
                        assert da == db
                    if day == RECORD:
                        record_units = da["shares"][2]
                    if day == EX and da["shares"] != db["shares"] and event_sample is None:
                        event_sample = dict(offset=po["offset"], fee=fee, arm=arm,
                                            registered=record_units, official=da["shares"][2],
                                            snapshot=db["shares"][2], ex_day=day)
                    if da != db:
                        changed.append(day)
                    max_gap = max(max_gap, abs(da["nav_cents"] - db["nav_cents"]))
                count["paths"] += 1
                count["exposed"] += int(record_units > 0)
                count["any_daily_difference"] += bool(changed)
                count["different_year_end_nav"] += a["daily"][-1]["nav_cents"] != b["daily"][-1]["nav_cents"]
                count["different_executions"] += a["post_fills"] != b["post_fills"]
                decisions_a = [(r["date"],r["action"]) for r in a["events"]]
                decisions_b = [(r["date"],r["action"]) for r in b["events"]]
                count["different_actions"] += decisions_a != decisions_b
                for ra, rb in zip(a["own_forks"], b["own_forks"], strict=True):
                    assert ra["date"] == rb["date"]
                    if ra["exit"] is None:
                        continue
                    forks["mature_reviews"] += 1
                    forks["different_final_branch_difference_cents"] += (
                        ra["switch_exit_cents"] - ra["hold_exit_cents"] != rb["switch_exit_cents"] - rb["hold_exit_cents"])
                    forks["different_relative_winner"] += ((ra["label_bp"]>0)-(ra["label_bp"]<0) !=
                                                           (rb["label_bp"]>0)-(rb["label_bp"]<0))
                paths.append(dict(offset=po["offset"], fee=fee, arm=arm, first_difference=next(iter(changed),None),
                                  max_abs_daily_nav_cents=max_gap,
                                  year_end_delta_cents=a["daily"][-1]["nav_cents"] - b["daily"][-1]["nav_cents"]))
    return dict(sources=sources, scores=forecast["snapshot_comparison"], paths_compared=dict(count),
                forks=dict(forks), example=event_sample, paths=paths,
                maximum_abs_daily_gap_cents=max(p["max_abs_daily_nav_cents"] for p in paths),
                maximum_abs_year_end_gap_cents=max(abs(p["year_end_delta_cents"]) for p in paths),
                final_passing_offsets={k:v["summary"]["passing_offsets"] for k,v in
                                       (("official", official),("snapshot",snapshot))})


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=HERE/"split_impact.json")
    args = cli.parse_args()
    result = run()
    with args.out.open("x") as f:
        f.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
