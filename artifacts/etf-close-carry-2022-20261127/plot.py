"""Draw the 2022 ETF performance comparison from the published-terms replay."""
import argparse
from datetime import date, timedelta
import json
from math import isclose
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, ticker

HERE = Path(__file__).resolve().parent
ARMS = ("donor", "hold", "switch")
LABEL = {"donor":"收盘模型", "hold":"一直留仓", "switch":"前二变就换"}
COLOR = {"donor":"#202c45", "hold":"#008476", "switch":"#bc6338"}


def plot(source, output):
    books = json.loads(source.read_bytes())
    assert not books["snapshot_mode"]
    phases = books["phases"]
    assert [p["offset"] for p in phases] == list(range(20))
    accounts = phases[0]["fees"]["normal"]
    assert len({a["initial"]["nav_cents"] for a in accounts.values()}) == 1
    dates = [date(2021,12,31)] + [date.fromisoformat(s) for s in books["dates"]]
    assert len(dates)==243 and dates[-1]==date(2022,12,30)
    plt.rcParams.update({"font.family":["PingFang SC","Heiti SC","DejaVu Sans"], "font.size":11,
                         "axes.spines.top":False, "axes.spines.right":False, "axes.spines.left":False,
                         "axes.edgecolor":"#b4bdc5","xtick.color":"#475364","ytick.color":"#475364",
                         "text.color":"#263346","axes.labelcolor":"#475364", "axes.unicode_minus":False})
    fig,(top,lower) = plt.subplots(2,1,figsize=(10.2,7.8),gridspec_kw={"height_ratios":[1.22,1]})
    fig.subplots_adjust(top=.83,bottom=.105,left=.09,right=.925,hspace=.58)
    fig.text(.09,.945,"收盘模型能择时换仓吗？",fontsize=22,weight="bold")
    fig.text(.09,.903,"四只 ETF  ·  2022 年开发期延伸检验  ·  每 20 个交易日复核一次",color="#586578")
    top.set_title("原定调仓起点的累计净收益（正常费用）",loc="left",pad=12,fontsize=14)
    for ax in (top,lower):
        ax.grid(axis="y",color="#e5e9ed",lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y",length=0,pad=6)
    for arm,ls in (("hold","-"),("switch","-"),("donor","--")):
        a=accounts[arm]
        initial=a["initial"]["nav_cents"]
        curve=[0.]+[100*(r["nav_cents"]/initial-1) for r in a["daily"]]
        assert isclose(curve[-1],100*a["periods"]["post"]["net_return"],abs_tol=1e-11)
        top.plot(dates,curve,color=COLOR[arm],lw=1.8,ls=ls,label=LABEL[arm])
        top.annotate(f"{curve[-1]:.2f}%",(dates[-1],curve[-1]),xytext=(5,0),
                     textcoords="offset points",va="center",ha="left",color=COLOR[arm],fontsize=10)
    switch,=[e for e in accounts["donor"]["events"] if e["action"]=="switch"]
    trade_day=date.fromisoformat(switch["next_open"])
    assert trade_day==date(2022,3,23)
    top.axvline(trade_day,color="#718194",lw=.9,ls=":")
    top.text(trade_day+timedelta(days=4),.96,"3 月 23 日模型换仓", transform=top.get_xaxis_transform(),
             va="top",color="#546579",fontsize=9)
    top.legend(loc="lower left",frameon=True,framealpha=.94,edgecolor="none",ncol=3,fontsize=9)
    top.set_xlim(dates[0],dates[-1]+timedelta(days=6))
    months=list(range(1,13))
    top.set_xticks([date(2022,m,1) for m in months],[f"{m}月" for m in months])
    top.yaxis.set_major_locator(ticker.MultipleLocator(5))
    top.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100,decimals=0))
    lower.set_title("调仓日挪动后，模型相对两种固定做法的全年净收益差",loc="left",pad=30,fontsize=14)
    lower.text(0,1.08,"每个点取正常与双倍费用中较小的收益差，单位为百分点。",transform=lower.transAxes,color="#586578",fontsize=10)
    lower.axhline(0,color="#75808d",lw=1,zorder=1)
    lower.axvspan(-.44,.44,color="#e5eaf2",alpha=.65)
    for arm,shift,marker in (("hold",-.12,"o"),("switch",.12,"D")):
        differences=[]
        for phase in phases:
            gaps=[100*(arms["donor"]["periods"]["post"]["net_return"]-arms[arm]["periods"]["post"]["net_return"])
                  for arms in phase["fees"].values()]
            assert len(gaps)==2
            differences.append(min(gaps))
        lower.scatter([i+shift for i in range(20)],differences,label="相对"+LABEL[arm],
                      s=34,marker=marker,color=COLOR[arm],zorder=3)
    lower.set_xticks(range(20))
    lower.set_xlim(-.65,19.65)
    lower.set_ylim(-10,10.5)
    lower.yaxis.set_major_locator(ticker.MultipleLocator(5))
    lower.set_xlabel("调仓起点平移的交易日数（0 为原定起点）",labelpad=6)
    lower.legend(loc="upper left",frameon=False,ncol=2,fontsize=9)
    fig.savefig(output,dpi=155,bbox_inches="tight",pad_inches=.21,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books",type=Path,default=HERE/"books.json")
    parser.add_argument("--out",type=Path,default=HERE/"paths.png")
    args=parser.parse_args()
    plot(args.books,args.out)
