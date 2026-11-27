"""Carry the frozen donor account into 2022 and branch only on the new year's closes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE.parent / "etf-close-asset-values-20261127"
ARCHIVE = HERE.parent / "etf-close-split-consensus-20261127/complement_books.json"
MODELS = HERE.parent / "etf-close-portfolio-states-20261127/models.json"
sys.path.insert(0, str(ASSETS))
import asset_books as old

books, fit, replay = old.books, old.previous.fit, old.replay
DECISION_START = date(2020, 1, 1)
BOUNDARY = date(2021, 12, 31)
END = pd.Timestamp("2022-12-30")
ARMS = ("donor", "hold", "switch")
STATE = {}


def load_worker(root, path, snapshot_mode):
    import forecast_2022
    books.initialize(root)
    lib = books.WORKER["lib"]
    ref, inputs, _ = forecast_2022.load_inputs(root, END)
    bars, days, actions = inputs[:3]
    assert Path(ref.engine.__file__).resolve() == root / "src/application/walk_forward_engine.py"
    days = days[days >= pd.Timestamp("2019-01-02")]
    bars = bars[bars.date.isin(days)]
    assert days[-1] == bars.date.max() == END and (actions.ex_date <= END).all()
    with np.load(path, allow_pickle=False) as z:
        quotes = {k:z[k] for k in ("adaptive", "uncertainty", "hmm_cap", "vol_feats")}
        assert pd.DatetimeIndex(z["dates"]).equals(days)
        symbols = z["symbols"].tolist()
    assert symbols == books.SYMBOLS
    from quant_research import ledger
    if not snapshot_mode:
        import split_event
        actions = split_event.configure(ledger,actions,snapshot_mode=False)
    stream = SimpleNamespace(dates=days, oos_alpha=quotes["adaptive"], oos_uncertainty=quotes["uncertainty"],
                             vol_feats=quotes["vol_feats"], oos_hmm_exposure=quotes["hmm_cap"])
    factory, smooth = lib.base.original_ledger.policy_constructor(ref.engine)(stream,symbols,20,2)
    ctx = SimpleNamespace(engine=ref.engine,bars=bars,calendar=days,actions=actions,symbols=symbols,
                          capital=100_000.,score_start_year=2020,bars_index=ledger.prepare_bars_index(bars),factory=factory)
    model = json.loads(MODELS.read_bytes())["models"]["late"]["donor20"]
    assert model["fitted"] and model["last_label_exit"] < model["cutoff"] == "2020-01-02"
    STATE.update(ctx=ctx,lib=lib,quotes=quotes,smooth=smooth,model=model,
                 archived=json.loads(ARCHIVE.read_bytes())["windows"]["late"])


def simulate(ctx, phase, fee, arm):
    lib, quotes, smoothed, model = (STATE[k] for k in ("lib","quotes","smooth","model"))
    days = ctx.calendar
    reviews = {days[i].date():i for i in range(phase,len(days),20)}
    warm = ctx.factory(decisions=reviews)
    sent, events = {}, {}
    def policy(state):
        if state.date not in reviews:
            return None
        i = reviews[state.date]
        if state.date < DECISION_START:
            target = warm(state)
        else:
            weights = np.array([state.weights.get(s,0.) for s in ctx.symbols])
            held = [s for s,w in zip(ctx.symbols,weights,strict=True) if w > 0]
            raw = quotes["adaptive"][i]
            pair = sorted(ctx.symbols[j] for j in old.data.raw_pair_indices(raw))
            x = old.data.close_features(raw, smoothed[i],quotes["uncertainty"][i],quotes["hmm_cap"][i],weights)
            rule = "donor" if state.date <= BOUNDARY else arm
            donor_bp = None
            target = None
            if x is None:
                reason, target = "fallback", warm(state)
            elif held == pair:
                reason = "same_hold"
            else:
                donor_bp = fit.predict_bp(x,model)
                do_switch = (donor_bp > 0) if rule=="donor" else (rule=="switch")
                reason = "switch" if do_switch else "hold"
                if do_switch:
                    target = dict.fromkeys(pair,float(weights.sum()/2))
            events[state.date] = dict(date=state.date.isoformat(),index=i,nav_cents=replay.cents(state.nav),
                weights=weights.tolist(),held=held,pair=pair,features=x,donor_bp=donor_bp,action=reason,
                source=rule,target=None if target is None else dict(target),
                next_open=days[i+1].date().isoformat() if i+1 < len(days) else None)
        if target is not None and i+1 < len(days):
            sent[state.date] = dict(target)
        return target

    args = lib.base.sim_args(ctx,"normal" if fee=="normal" else "double_friction")
    book = ctx.engine.simulate(ctx.bars,days,{},ctx.actions,close_policy=policy,**args)
    weights, errors = lib.reconcile(ctx, book, sent,reviews)
    if phase==0:
        static = lib.base.refresh.static_ledger(ctx.engine,sent,days,ctx.bars,ctx.actions,args)
        lib.base.refresh.same_prefix(book,static,days[-1].date())
    # There is no opening auction after the final close inside the research window.
    last = book.equity.iloc[-1]
    if last.date in reviews:
        target = policy(SimpleNamespace(date=last.date,nav=last.nav,cash=last.cash,drawdown=last.drawdown,
                                       weights=weights.loc[last.date].to_dict()))
        assert events[last.date]["target"] == target and last.date not in sent
    assert list(events)==[d for d in reviews if d>=DECISION_START]
    for d,e in events.items():
        np.testing.assert_array_equal(weights.loc[d].to_numpy(),e["weights"])
    return book,events,sent,max(errors.values())


def marked(ctx, ref, quantities, day):
    equity = ref[0].loc[day]
    shares = [int(q) for q in quantities.loc[day, ctx.symbols]]
    row = dict(date=day.isoformat(),shares=shares,nav_cents=replay.cents(equity.nav),
               cash_cents=replay.cents(equity.cash),receivable_cents=replay.cents(equity.receivable))
    # Use the frozen ledger's cash rounding for each position, including post-split odd lots.
    value = sum(old.data.events.cents(ctx.bars_index[day][s].close*q) for s,q in zip(ctx.symbols,shares,strict=True))
    assert value+row["cash_cents"]+row["receivable_cents"] == row["nav_cents"], (day,row,value)
    return row


def cash_events(book, first, last):
    return [dict(date=e.date.isoformat(), symbol=e.symbol,event=e.event,cash_delta_cents=replay.cents(e.cash_delta),
                 receivable_delta_cents=replay.cents(e.receivable_delta),quantity_delta=int(e.quantity_delta))
            for e in book.cash_events.itertuples() if first <= e.date <= last]


def historical_check(ctx,book,events,ref,qty,fills,archive,archive_dates):
    eq = book.equity
    days = [d for d in eq.date if DECISION_START <= d <= BOUNDARY]
    assert list(map(str,days))==archive_dates
    assert [replay.cents(ref[0].at[d,"nav"]) for d in days]==archive["nav_cents"]
    assert [r for r in fills if r["date"] <= str(BOUNDARY)] == archive["post_fills"]
    assert cash_events(book,DECISION_START,BOUNDARY)==archive["post_cash_events"]
    # The earlier simulation could not schedule orders from its final session.
    historical = [e for d,e in events.items() if d < BOUNDARY]
    for ours,other in zip(historical,archive["events"],strict=True):
        day = date.fromisoformat(ours["date"])
        mark = marked(ctx,ref,qty,day)
        for k in ("date","held","pair","next_open","action","weights","features","nav_cents"):
            assert ours[k]==other[k],(day,k)
        assert all(other[k]==mark[k] for k in ("shares","cash_cents","receivable_cents"))
        if ours["donor_bp"] is not None:
            assert ours["donor_bp"]==other["estimates"]["actual"]["donor_bp"]
    old_period=archive["periods"]["2019"]
    warm_eq=eq[eq.date<DECISION_START]
    warm_orders=book.orders[book.orders.date<DECISION_START]
    metrics=ctx.engine.summarize(warm_eq,warm_orders,ctx.capital)
    assert all(v==old_period[k] for k,v in metrics.items())
    return dict(days=len(days),filled_orders=len(archive["post_fills"]),review_closes=len(historical),
                cash_events=len(archive["post_cash_events"]),warmup_sessions=len(warm_eq))


def fork(ctx,book,sent,e,ref,qty):
    i=e["index"]
    result=dict(date=e["date"],entry=e["next_open"],exit=None,action=e["action"],held=e["held"],high=e["pair"],
                nav_cents=e["nav_cents"],hold_exit_cents=None,switch_exit_cents=None,actual_exit_cents=None,label_bp=None)
    if i+21>=len(ctx.calendar) or e["action"]=="fallback":
        return result
    signal,end = [ctx.calendar[i+j].date() for j in (0,21)]
    proposal = None if e["held"]==e["pair"] else dict.fromkeys(e["pair"],sum(e["weights"])/2)
    prior = {day:t for day,t in sent.items() if day<signal}
    hold = old.data.branch(ctx,book,prior,i,None,"normal")["nav"]
    switch = old.data.branch(ctx,book,prior,i,proposal,"normal")["nav"] if proposal else hold
    actual = old.before_opening_fills(ctx,ref,qty,end)
    assert actual == (switch if e["action"]=="switch" else hold)
    result.update(exit=end.isoformat(),hold_exit_cents=replay.cents(hold),switch_exit_cents=replay.cents(switch),
                  actual_exit_cents=replay.cents(actual),label_bp=10000*(replay.cents(switch)-replay.cents(hold))/e["nav_cents"])
    return result


def record(ctx,book,events,sent,error,archive,archive_dates,fee,arm,phase):
    ref,qty=old.ledger_arrays(ctx,book)
    fills=old.after_fills(ctx,book)
    history=historical_check(ctx,book,events,ref,qty,fills,archive,archive_dates)
    initial=marked(ctx,ref,qty,BOUNDARY)
    initial["pending_decision"]=events.get(BOUNDARY)
    days=[d.date() for d in ctx.calendar if d.date()>BOUNDARY]
    current=[dict(e,**{k:v for k,v in marked(ctx,ref,qty,d).items() if k not in e}) for d,e in events.items() if d>BOUNDARY]
    post_fills=[r for r in fills if r["date"]>str(BOUNDARY)]
    post_orders=book.orders[book.orders.date>BOUNDARY]
    period=ctx.engine.summarize(book.equity[book.equity.date>BOUNDARY],post_orders,initial["nav_cents"]/100)
    period.update(starting_nav=initial["nav_cents"]/100,first=str(days[0]),last=str(days[-1]))
    assert period["filled_orders"]==len(post_fills)
    order_records=json.loads(post_orders.assign(date=post_orders.date.map(str)).to_json(orient="records",double_precision=15))
    return dict(periods={"post":period}, initial=initial,daily=[marked(ctx,ref,qty,d) for d in days],
                events=current,post_fills=post_fills,post_orders=order_records,post_cash_events=cash_events(book,days[0],days[-1]),
                own_forks=[fork(ctx,book,sent,e,ref,qty) for e in current] if fee=="normal" and arm=="donor" else [],
                counts=dict(Counter(e["action"] for e in current)),historical_checks=history,
                reconciliation_max_error=error,static_replay=(phase==0))


def phase_task(phase):
    ctx,lib,group=(STATE[k] for k in ("ctx","lib","archived"))
    fees={}
    for fee in ("normal","double"):
        archive=group["phases"][phase]["fees"][fee]["combined"]
        assert archive["first_difference_from_donor"] is None
        common=None
        arms={}
        for arm in ARMS:
            book,events,sent,error=simulate(ctx,phase,fee,arm)
            first=next(d for d in events if d>BOUNDARY)
            if common is None:
                common=book
            else:
                lib.base.refresh.same_prefix(common,book,first)
            result=record(ctx,book,events,sent,error,archive,group["post_dates"],fee,arm,phase)
            if arms:
                assert result["initial"]==arms["donor"]["initial"]
            arms[arm]=result
        fees[fee]=arms
    return dict(offset=phase,fees=fees)


def run(path,workers,snapshot_mode):
    import forecast_2022
    libf=books.load_module(books.CAUSAL/"replay_frozen.py","carry_frozen_books")
    with TemporaryDirectory(prefix="etf-carry-book-") as tmp:
        root=libf.materialize(tmp)
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=load_worker,initargs=(root,path,snapshot_mode)) as pool:
            phases=list(pool.map(phase_task,range(20)))
        books.initialize(root)
        compare=books.WORKER["lib"].comparison
        comparisons={a:compare(phases,"post","donor",a) for a in ("hold","switch")}
    passed=sorted(set(comparisons["hold"]["passing_offsets"]) & set(comparisons["switch"]["passing_offsets"]))
    dates=[r["date"] for r in phases[0]["fees"]["normal"]["donor"]["daily"]]
    files=[Path(__file__),HERE/"protocol.md",path,MODELS,ARCHIVE,Path(forecast_2022.__file__),Path(old.__file__),
           Path(old.audit_data.__file__),Path(old.previous.__file__),Path(old.data.__file__),Path(fit.__file__)]
    if not snapshot_mode:
        files.extend(HERE/name for name in ("split_event.py","official_forecast.py","510500_split_notice.pdf"))
    return dict(dates=dates,phases=phases,snapshot_mode=snapshot_mode,
                summary=dict(comparisons=comparisons,passing_offsets=passed,passed=0 in passed and len(passed)>=11),
                frozen_manifest=libf.MANIFEST_SHA256,inputs={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT)
                else f.name:sha256(f.read_bytes()).hexdigest() for f in files})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot-mode",action="store_true")
    cli.add_argument("--signals",type=Path)
    cli.add_argument("--out",type=Path)
    cli.add_argument("--workers",type=int,default=4)
    args=cli.parse_args()
    path = args.signals or HERE/("signals.npz" if args.snapshot_mode else "official_signals.npz")
    out = args.out or HERE/("snapshot_books.json" if args.snapshot_mode else "books.json")
    if out.exists() or args.workers<1:
        cli.error("Choose a fresh output and a positive worker count")
    result=run(path.resolve(),args.workers,args.snapshot_mode)
    with out.open("x") as f:
        f.write(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n")
