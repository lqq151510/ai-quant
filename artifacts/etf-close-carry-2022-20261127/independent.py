"""Independently replay the cross-year ETF decisions, opening fills and distributions."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARKET = ROOT/'research_v1/artifacts/market-etf-verified-20261108'
ARCHIVE = HERE.parent/'etf-close-split-consensus-20261127/complement_books.json'
DONOR = HERE.parent/'etf-close-portfolio-states-20261127/models.json'
ANNUAL = HERE.parent/'etf-annual-expert-20261126'
SYMS = ('510050','510300','510500','510880')
START, END = '2021-12-31', '2022-12-30'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cents(value):
    return int((Decimal(str(value))*100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def near(a,b,tolerance=1e-10):
    np.testing.assert_allclose(a,b,atol=tolerance,rtol=0)


def get_market():
    with duckdb.connect() as db:
        days = [d for d, in db.execute("select strftime(date,'%Y-%m-%d') from read_parquet(?) where date<=? order by date", [str(MARKET/'calendar.parquet'),END]).fetchall()]
        bars = db.execute("select strftime(date,'%Y-%m-%d'),symbol,open,close,lower_limit,upper_limit,volume,tradable from read_parquet(?) where kind='etf' and date between '2021-11-01' and ? order by date,symbol",[str(MARKET/'bars.parquet'),END]).fetchall()
        act = db.execute("select symbol,strftime(record_date,'%Y-%m-%d'),strftime(ex_date,'%Y-%m-%d'),strftime(pay_date,'%Y-%m-%d'),cash_per_share,share_multiplier from read_parquet(?) where ex_date between '2021-11-01' and ? order by ex_date,symbol",[str(MARKET/'actions.parquet'),END]).fetchall()
    quotes={(d,s):rest for d,s,*rest in bars}
    assert days==sorted(set(days)) and days[-1]==END
    assert all([quotes[d,s] for s in SYMS] for d in days if d>='2021-11-01')
    actions=defaultdict(list)
    for s,rec,ex,pay,rate,mult in act:
        assert days[days.index(ex)-1]==rec
        actions[ex].append((s,rec,pay,rate,mult))
    assert actions['2022-08-29']==[('510500','2022-08-26',None,0.,1.1454)]
    return days,quotes,actions


def forecasts(days,metadata,npz):
    assert sha(npz)==metadata['npz_sha256']
    old_meta=read(ANNUAL/'forecast_meta.json')
    fold=metadata['fold']
    assert fold['year']==2022 and fold['seed']==45 and old_meta['folds'][-1]['seed']==44
    first,last=fold['first_oos'],fold['last_oos']
    assert (first,last)==('2022-01-04',END) and fold['hmm_last_fit_date']==START
    prior=[d for d in days if old_meta['folds'][0]['train']['first']<=d<first]
    cal=prior[-230:]
    for key,raw,limit in [('train',prior[:-230],cal[0]),('calibration',cal,first)]:
        info=fold[key]
        available=[d for d in raw if days[days.index(d)+21] < limit]
        assert info==dict(first=available[0],last=available[-1],sessions=len(available),latest_label_end=days[days.index(available[-1])+21])
        assert len(raw)-len(available)==fold[key+'_purged']==21
    assert len([d for d in days if first<=d<=last])==fold['oos_sessions']
    assert fold['first_calibration_before_purge']==cal[0]
    assert metadata['epochs']==10 and metadata['calibration_raw_sessions']==230 and metadata['unlabeled_tail']==21
    prefixes=metadata['prefixes']
    assert [r['cutoff'] for r in prefixes][:2]==[first,'2022-06-30']
    for r in prefixes:
        assert r['rank_order_equal'] and max(r['max_abs_difference'].values()) < 1e-5
        assert r['market']['last_market_day']==r['cutoff'] and r['market']['last_known_ex_date']<=r['cutoff']
        assert (r['training_rows'],r['calibration_rows'])==(fold['train']['sessions'],fold['calibration']['sessions'])
    for name,digest in metadata['inputs'].items():
        assert sha(ROOT/name)==digest, name
    streams={}
    with np.load(npz,allow_pickle=False) as z, np.load(ANNUAL/'forecasts.npz',allow_pickle=False) as old:
        assert tuple(z['symbols'])==tuple(old['symbols'])==SYMS
        sung=z['dates'].astype(str).tolist()
        assert sung==[d for d in days if d>='2019-01-01']
        for key in ('dates','adaptive','uncertainty','vol_feats','hmm_cap'):
            np.testing.assert_array_equal(z[key][:len(old['dates'])],old[key])
        moving=None
        for d,raw,u,cap in zip(sung,z['adaptive'],z['uncertainty'],z['hmm_cap'],strict=True):
            transformed=raw/(1+.20*(u/(float(np.median(u))+1e-6)))
            moving=transformed if moving is None else .955*moving+.045*transformed
            rel=u.astype(float)/(np.median(u.astype(float))+1e-6)
            assert all(np.isfinite(v).all() for v in (raw,moving,rel,cap))
            streams[d]=(raw.astype(float),moving.astype(float),rel,float(cap))
    return streams,dict(new_sessions=fold['oos_sessions'], prior_scores=len(old_meta['folds']), seed=45,train=fold['train'],calibration=fold['calibration'], first_signal=first,last_signal=last,prefixes=[r['cutoff'] for r in prefixes],max_prefix_gap=max(max(r['max_abs_difference'].values()) for r in prefixes))


@dataclass
class Account:
    units: list
    cash: int
    pending: list=field(default_factory=list)
    official: bool=False

    def opening_events(self,day,actions):
        flows=[]
        for symbol,_,pay,rate,mult in actions.get(day,[]):
            i=SYMS.index(symbol)
            if rate and self.units[i]:
                amount=cents(self.units[i]*rate)
                self.pending.append((symbol,pay,amount))
                flows.append((day,symbol,'dividend_receivable',0,amount,0))
            if mult!=1 and self.units[i]:
                assert (day,symbol)==('2022-08-29','510500')
                exact=Decimal('1.14539') if self.official else Decimal('1.1454')
                mode=ROUND_CEILING if self.official else ROUND_FLOOR
                new=int((self.units[i]*exact).to_integral_value(rounding=mode))
                flows.append((day,symbol,'share_change',0,0,new-self.units[i]))
                self.units[i]=new
        for s,pay,amount in self.pending:
            if pay<=day:
                self.cash+=amount
                flows.append((day,s,'dividend_paid',amount,-amount,0))
        self.pending=[row for row in self.pending if row[1]>day]
        return flows

    def recv(self):
        return sum(a for _,_,a in self.pending)

    def value(self,day,quotes,side):
        prices=[quotes[day,s][side] for s in SYMS]
        if side==0:  # Fork exit uses the exact published opening quote, before opening orders.
            prices=[Decimal(str(p)) for p in prices]
        return self.cash+self.recv()+sum(cents(p*q) for p,q in zip(prices,self.units,strict=True))

    def trade(self,day,quotes,target,scale):
        if target is None:
            return []
        nav=(self.cash+self.recv())/100+sum(q*quotes[day,s][0] for s,q in zip(SYMS,self.units,strict=True))
        desired=[]
        for i,s in enumerate(SYMS):
            x=target.get(s,0)
            delta=math.floor(nav*x/quotes[day,s][0]+1e-9)//100*100-self.units[i]
            if delta:
                desired.append((delta>0,-x,s,i,abs(delta)))
        fills=[]
        for buy,_,s,i,qty in sorted(desired):
            op,_,lo,hi,vol,tradable=quotes[day,s]
            price=op*(1+(.0005*scale if buy else -.0005*scale))
            if not tradable or vol<=0 or lo==hi or not lo<=op<=hi or not lo-1e-9<=price<=hi+1e-9:
                continue
            if (buy and op>=hi-1e-9) or (not buy and op<=lo+1e-9):
                continue
            if buy or qty!=self.units[i]:
                qty=qty//100*100
            while qty:
                amount=cents(qty*price)
                fee=max(500*scale,cents((amount/100)*.0003*scale))
                delta=(-amount if buy else amount)-fee
                if self.cash+delta>=0:
                    break
                qty=qty-100 if buy else 0
            if qty:
                self.units[i]+=qty if buy else -qty
                self.cash+=delta
                fills.append((day,s,'buy' if buy else 'sell',qty,amount,fee,delta))
        return fills


def evaluate(day,account,quotes,stream,model,arm):
    raw,moving,rel,cap=stream
    held=[s for s,q in zip(SYMS,account.units,strict=True) if q]
    top=sorted(SYMS[i] for i in np.argsort(-raw,kind='stable')[:2])
    assert len(held)==2, (day,held)
    nav=account.value(day,quotes,1)
    w=np.array([cents(quotes[day,s][1]*q)/nav for s,q in zip(SYMS,account.units,strict=True)])
    admitted=np.flatnonzero(w>0); high=np.argsort(-raw,kind='stable')[:2]
    target=np.zeros(4);target[high]=sum(w)/2
    x=[float(v[high].mean()-v[admitted].mean()) for v in (raw,moving,rel)]+[cap,float(sum(abs(target-w)))]
    pred=float(model['target_mean_bp']+((np.array(x)-model['center'])/model['scale'])@model['coefficients_bp'])
    choice='same_hold' if held==top else ('switch' if arm=='switch' or (arm=='donor' and pred>0) else 'hold')
    return dict(held=held,pair=top,weights=w.tolist(),features=x,donor_bp=None if held==top else pred,action=choice,target=dict.fromkeys(top,float(sum(w)/2)) if choice=='switch' else None)


def old_boundary(archive,dates,quotes,actions):
    last=archive['events'][-1];day=last['date']
    a=Account(last['shares'].copy(),last['cash_cents'])
    assert last['receivable_cents']==0
    by_day=defaultdict(list)
    for f in archive['post_fills']:
        by_day[f['date']].append(f)
    for d in dates:
        if day<d<=START:
            a.opening_events(d,actions)
            for f in by_day[d]:
                a.cash+=f['cash_delta_cents']
                a.units[SYMS.index(f['symbol'])]+=(1 if f['side']=='buy' else -1)*f['shares']
    assert a.recv()==0 and a.value(START,quotes,1)==archive['nav_cents'][-1]
    return a


def model_check():
    data=read(DONOR)
    model=data['models']['late']['donor20']
    source=DONOR.parent/'states.json'
    assert sha(source)==data['sha256'][source.name]
    cutoff=model['cutoff']
    assert cutoff=='2020-01-02'
    rows=sorted([r for r in read(source)['rows'] if r['source']=='adaptive' and r['fee']=='normal'
                 and r['date']<cutoff and r['exit'] and r['exit']<cutoff and r['eligible'] and r['changed']
                 and r['label_bp'] is not None],key=lambda r:(r['date'],r['phase'],r['window']))
    assert len(rows)==len({r['date'] for r in rows})==model['training_rows']==702
    assert max(r['exit'] for r in rows)==model['last_label_exit']
    inputs=[]
    for r in rows:
        assert r['date']<r['entry']<r['exit']<cutoff
        near(r['label_bp'],10000*(r['switch_open_nav_cents']-r['hold_open_nav_cents'])/r['nav_cents'])
        inputs.append([r['date'],r['exit'],r['phase'],r['features'],r['label_bp']])
    assert hashlib.sha256(json.dumps(inputs,separators=(',',':')).encode()).hexdigest()==model['inputs_sha256']
    x=np.array([r['features'] for r in rows])
    y=np.array([r['label_bp'] for r in rows])
    near(x.mean(axis=0),model['center']);near(x.std(axis=0),model['scale']);near(y.mean(),model['target_mean_bp'])
    z=(x-x.mean(axis=0))/x.std(axis=0)
    design=np.vstack((z/len(rows)**.5,np.eye(5)))
    estimate=np.linalg.lstsq(design,np.r_[(y-y.mean())/len(rows)**.5,np.zeros(5)],rcond=None)[0]
    near(estimate,model['coefficients_bp'])
    return model


def check_event(actual,calc,day,idx,dates,a,quotes,source):
    assert actual['date']==day and actual['index']==idx and actual['source']==source
    assert actual['next_open']==(dates[idx+1] if idx+1<len(dates) else None)
    assert actual['nav_cents']==a.value(day,quotes,1)
    for k in ('held','pair','action'):
        assert actual[k]==calc[k],(day,k)
    for k in ('features','weights','donor_bp'):
        if calc[k] is None:
            assert actual[k] is None
        else:
            near(actual[k],calc[k])
    if calc['target'] is None:
        assert actual['target'] is None
    else:
        assert actual['target'].keys()==calc['target'].keys()
        for k,v in calc['target'].items():
            near(actual['target'][k],v)


def check_state(row,day,a,quotes):
    expected=dict(shares=a.units,date=day,cash_cents=a.cash,receivable_cents=a.recv(),nav_cents=a.value(day,quotes,1))
    for k,v in expected.items():
        assert row[k]==v,(day,k,row[k],v)


def independent_fork(row,e,idx,dates,a,quotes,actions):
    day=dates[idx]
    assert row['date']==day and row['action']==e['action'] and row['high']==e['pair'] and row['held']==e['held']
    assert row['nav_cents']==a.value(day,quotes,1)
    assert row['entry']==(dates[idx+1] if idx+1<len(dates) else None)
    if idx+21>=len(dates):
        assert all(row[k] is None for k in ('exit','actual_exit_cents','label_bp','hold_exit_cents','switch_exit_cents'))
        return None
    finish=dates[idx+21]
    assert row['exit']==finish
    nav={}
    proposal=None if e['held']==e['pair'] else dict.fromkeys(e['pair'],sum(e['weights'])/2)
    for name,target in [('hold',None),('switch',proposal)]:
        branch=deepcopy(a)
        for k in range(idx+1,idx+22):
            t=dates[k]
            branch.opening_events(t,actions)
            if k==idx+1:
                branch.trade(t,quotes,target,1)
        nav[name]=branch.value(finish,quotes,0)
        assert row[name+'_exit_cents']==nav[name],(day,name,row[name+'_exit_cents'],nav[name])
    near(row['label_bp'],10000*(nav['switch']-nav['hold'])/row['nav_cents'])
    actual=nav['switch' if e['action']=='switch' else 'hold']
    assert actual==row['actual_exit_cents']
    return finish,actual


def account_check(book,phase,scale,arm,base,dates,quotes,actions,streams,model):
    a=deepcopy(base)
    first=dates.index(START)
    idx_by_day={v:k for k,v in enumerate(dates)}
    check_state(book['initial'],START,a,quotes)
    target=None
    previous=book['initial']['pending_decision']
    if first%20==phase:
        prior=evaluate(START,a,quotes,streams[START],model,'donor')
        check_event(previous,prior,START,first,dates,a,quotes,'donor')
        target=prior['target']
    else:
        assert previous is None
    close_events={r['date']:r for r in book['events']}
    assert list(close_events)==[d for i,d in enumerate(dates) if d>START and i%20==phase]
    forks={r['date']:r for r in book['own_forks']}
    assert set(forks)==(set(close_events) if scale==1 and arm=='donor' else set())
    flows=[]; fills=[]; wealth=[a.value(START,quotes,1)]; fork_exits={}
    for expected,day in zip(book['daily'],dates[first+1:],strict=True):
        i=idx_by_day[day]
        flows.extend(a.opening_events(day,actions))
        if day in fork_exits:
            assert a.value(day,quotes,0)==fork_exits[day],('selected fork vs continuous live book',day)
        fills.extend(a.trade(day,quotes,target,scale))
        target=None
        check_state(expected,day,a,quotes)
        wealth.append(a.value(day,quotes,1))
        if day in close_events:
            e=evaluate(day,a,quotes,streams[day],model,arm)
            actual=close_events[day]
            check_event(actual,e,day,i,dates,a,quotes,arm)
            check_state(actual,day,a,quotes)
            target=e['target']
            if day in forks:
                result=independent_fork(forks[day],e,i,dates,a,quotes,actions)
                if result:
                    end,cost=result
                    fork_exits[end]=cost
    assert flows==[tuple(f[k] for k in ('date','symbol','event','cash_delta_cents','receivable_delta_cents','quantity_delta')) for f in book['post_cash_events']]
    for (day,s,side,q,amount,fee,delta),record in zip(fills,book['post_fills'],strict=True):
        assert (day,s,side,q,amount,fee,delta)==tuple(record[k] for k in ('date','symbol','side','shares','notional_cents','commission_cents','cash_delta_cents'))
        direction=1 if side=='buy' else -1
        op=quotes[day,s][0]
        raw=cents(Decimal(str(op))*q)
        near(record['open'],op);near(record['price'],op*(1+direction*.0005*scale))
        assert record['stamp_cents']==record['transfer_cents']==0
        assert record['slip_cents']==(amount-raw)*direction
    nav=np.array(wealth)/100
    metrics=book['periods']['post']
    assert (metrics['sessions'],metrics['first'],metrics['last'])==(len(nav)-1,dates[first+1],END)
    checks=dict(starting_nav=nav[0],final_nav=nav[-1],net_return=nav[-1]/nav[0]-1,
               max_drawdown=float(np.max(1-nav/np.maximum.accumulate(nav))),filled_orders=len(fills),
               transaction_fees=sum(f[5] for f in fills)/100,
               traded_notional_over_average_nav=sum(f[4] for f in fills)/100/nav[1:].mean())
    for k,v in checks.items():
        near(metrics[k],v)
    return checks


def audit_scenario(official,days,quotes,actions,prior,model):
    name='official' if official else 'snapshot'
    prefix='official_' if official else ''
    path=HERE/('books.json' if official else 'snapshot_books.json')
    book=read(path)
    assert book['snapshot_mode'] is (not official)
    for key,value in book['inputs'].items():
        assert sha(ROOT/key)==value,(name,key)
    npz=HERE/(prefix+'signals.npz')
    manifest=HERE/(prefix+'forecast.json')
    metadata=read(manifest)
    assert book['inputs'][str(npz.relative_to(ROOT))]==sha(npz)
    streams,fold=forecasts(days,metadata,npz)
    if official:
        assert fold['prefixes'][-1]=='2022-08-29'
        assert metadata['share_event']['official_multiplier']==1.14539
    dates=list(streams)
    assert book['dates']==[d for d in dates if d>START]
    assert [p['offset'] for p in book['phases']]==list(range(20))
    metrics={}; counts=Counter(); splits=[]; start_close_orders=[]
    for phase,entry in enumerate(book['phases']):
        assert entry['fees'].keys()=={'normal','double'}
        common=entry['fees']
        for scale,fee in ((1,'normal'),(2,'double')):
            assert common[fee].keys()=={'donor','hold','switch'}
            archive=prior[phase]['fees'][fee]['combined']
            base=old_boundary(archive,dates,quotes,actions)
            base.official=official
            if common[fee]['donor']['initial']['pending_decision']:
                d=common[fee]['donor']['initial']['pending_decision']
                start_close_orders.append(dict(phase=phase,fee=fee,action=d['action'],target=d['target'],
                                              next_open=d['next_open'],first_open_fills=[r for r in common[fee]['donor']['post_fills'] if r['date']==book['dates'][0]]))
            for arm,b in common[fee].items():
                metrics[phase,fee,arm]=account_check(b,phase,scale,arm,base,dates,quotes,actions,streams,model)
                assert b['initial']==common[fee]['donor']['initial']
                counts.update(accounts=1,account_days=len(b['daily']),review_closes=len(b['events']),
                              filled_orders=len(b['post_fills']),cash_events=len(b['post_cash_events']))
                for e in b['post_cash_events']:
                    if e['event']=='share_change':
                        pre=next(r['shares'][2] for r in b['daily'] if r['date']=='2022-08-26')
                        splits.append(dict(offset=phase,fee=fee,arm=arm,registered=pre,increment=e['quantity_delta']))
                for r in b['own_forks']:
                    counts['fork_reviews']+=1
                    counts['mature_forks']+=int(r['exit'] is not None)
                    counts['mature_eligible_changed']+=int(r['exit'] is not None and r['high']!=r['held'])
                    counts['mature_forks_crossing_split']+=int(r['exit'] is not None and r['date']<'2022-08-29'<=r['exit'])
                    counts['immature_changed']+=int(r['exit'] is None and r['high']!=r['held'])
    assert len(start_close_orders)==2 and len({r['phase'] for r in start_close_orders})==1
    expected={}
    for control in ('hold','switch'):
        result=[]
        for p in range(20):
            if all(metrics[p,f,'donor']['net_return']>metrics[p,f,control]['net_return'] and
                   metrics[p,f,'donor']['max_drawdown']<=metrics[p,f,control]['max_drawdown'] for f in ('normal','double')):
                result.append(p)
        expected[control]=result
        assert result==book['summary']['comparisons'][control]['passing_offsets']
    both=sorted(set(expected['hold']) & set(expected['switch']))
    passed=0 in both and len(both)>=11
    assert both==book['summary']['passing_offsets'] and passed==book['summary']['passed']
    assert counts['accounts']==120 and counts['account_days']==120*len(book['dates'])
    assert counts['fork_reviews']==len(book['dates'])
    return dict(source=path.name,book_sha256=sha(path),signal_sha256=sha(npz),forecast_sha256=sha(manifest),
                fold=fold,verified=dict(counts),prior_close_decisions=start_close_orders,
                settled_share_changes=splits,passing={**expected,'both':both},acceptance=bool(passed),
                phase_zero={fee:{arm:metrics[0,fee,arm] for arm in ('donor','hold','switch')} for fee in ('normal','double')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'audit.json')
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Output already exists. Choose a fresh output path.')
    days,quotes,actions=get_market()
    prior=read(ARCHIVE)['windows']['late']['phases']
    model=model_check()
    reports={mode:audit_scenario(mode=='official',days,quotes,actions,prior,model) for mode in ('official','snapshot')}
    result=dict(through=END,initial_close=START,donor_cutoff=model['cutoff'],donor_last_training_exit=model['last_label_exit'],
                donor_training_rows=model['training_rows'],donor_sha256=sha(DONOR),archive_sha256=sha(ARCHIVE),
                archive_calendar_sha256=sha(MARKET/'calendar.parquet'),source_split_sha256=sha(HERE/'510500_split_notice.pdf'),
                verifier_sha256=sha(Path(__file__)),checks=reports,focused_tests=dict(timing_and_tradable_returns={'passed':8,'subtests_passed':17},
                ledger_and_timeline={'passed':27}))
    with args.out.open('x') as fp:
        json.dump(result,fp,indent=2,allow_nan=False)
        fp.write('\n')


if __name__=='__main__':
    main()
