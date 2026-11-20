# 证据归档

本目录是 [`REJECTED.md`](../REJECTED.md) 引用报告的仓库内副本。原始报告位于被 Git 忽略的
`artifacts/` 与 `research_v1/artifacts/`，只在本机存在；归档后，新检出也能核对每一条候选的
关键数字。

- 副本按字节复制，`MANIFEST.json` 记录原件路径、大小、修改时间与 SHA-256。
- 已有副本不覆盖；哈希不一致会报错，而不是悄悄替换。
- `net-edge`、`entry-timing`、`rl-self-review` 三份报告原样保留了本机绝对路径链接，复制时
  未做改写，以免哈希失真；可读的仓库内路径以 `MANIFEST.json` 的 `source_path` 为准。
- 归档只覆盖登记表直接引用的摘要文件，不包含逐日账本、行情快照或源码快照。

复核：

```bash
research_v1/.venv/bin/python research_v1/evidence/build_manifest.py --check
```

## 复现 5 日对照（R-02 至 R-04）

`control-study-20261114*` 由显式四信号生成。当前脚本的默认信号是 `equal_weight_hold`（首次买入后持有），直接重跑得到的是另一个对照（500,000 元正常费用约 +9.05%），不能据此认为归档结果失真。用原信号集重跑：

```bash
research_v1/.venv/bin/python -m quant_research.capital_scale \
  --snapshot research_v1/artifacts/market-etf-verified-20261108 \
  --out /tmp/replay-control-20261114 \
  --signals momentum reverse equal_weight csi300_hold
```

2026-11-20 用当前源码重跑，500,000 元正常费用下四组分别为 −7.20%、+8.80%、+9.55%、+5.16%，最大回撤也一致。归档记录中 `capital_scale.py`、`baseline.py` 的哈希与当前源码不同，逐字节重放无法比对；同快照、同信号集下指标可复现。

新增候选时，先把结果文件加入 `build_manifest.py` 的 `SOURCES`，再运行一次
`build_manifest.py`，然后在 `REJECTED.md` 中引用 `research_v1/evidence/` 下的路径。
