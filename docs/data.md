# 业务样例数据

`fixtures/label_acquisition.json` 使用本地合成数据，不含真实个人资料。

预算按 family 采购，重复图片属于同一采购单位。候选文件没有标签；保留比较集和回执模拟文件分别存放，不属于 propose 的输入。回执 revision 是同一采购回执的更正号。

`fixtures/demo.json` 是当前单笔业务流程的样例，和批量分析资料分开保存。

## 主动采样采购闭环（`fashion_data_foundry.acquisition`）

### 三份资料的隔离关系

| 文件 | 角色 | 谁能读 |
| --- | --- | --- |
| `fixtures/label_acquisition.json` | 基线训练数据（带标签）、候选池（无标签）、撤销授权、夹带的更正片段 | propose / ingest |
| `fixtures/acquisition_receipts.json` | 模拟供货商回执（含 `unjudgeable`），`allowed_use=receipt-simulation-only` | 仅 ingest |
| `fixtures/acquisition_holdout.json` | 独立保留比较集，`allowed_use=comparison-only`，与训练/候选家族零重叠 | 仅 compare 内部解包 |

隔离是结构性的，不靠调用者自觉：

- 候选视图 `FamilyView` 由字段白名单构造，白名单不含 `label`；输入文件若给候选塞入标签，
  加载阶段直接报错（`tests/acquisition_trials/test_label_isolation.py` 固化该回归）。
- 选样策略模块不导入 `SealedLabels`/`HoldoutCase`，构造签名也没有标签或比较集参数。
- 比较集标签封在 `SealedLabels` 里，只有 `evaluation.evaluate_holdout` 能显式解包；
  compare 不回写状态、不触发再训练，因此无法把参数“试到保留集最好为止”。
- 策略权重（`BALANCED_WEIGHTS`）与停采阈值是模块级预注册常量，CLI 不暴露任何调节项。

### 三个子命令

```bash
# 1) 选样下单：输出订单、评分拆解、覆盖缺口与预算账；重复调用幂等回放
python3 -m fashion_data_foundry.acquisition propose \
  --input fixtures/label_acquisition.json \
  --state /tmp/acq/state.json --output /tmp/acq/r1-propose.json \
  --strategy balanced   # balanced | uncertainty | random

# 2) 回执入账：可多次调用，支持部分回执、乱序、更正、重复投递
python3 -m fashion_data_foundry.acquisition ingest \
  --input fixtures/label_acquisition.json \
  --state /tmp/acq/state.json --output /tmp/acq/r1-ingest.json \
  --receipts fixtures/acquisition_receipts.json

# 3) 独立比较集上的学习曲线、三策略横向对比与停采建议
python3 -m fashion_data_foundry.acquisition compare \
  --input fixtures/label_acquisition.json \
  --state /tmp/acq/state.json --output /tmp/acq/compare.json \
  --holdout fixtures/acquisition_holdout.json
```

### 状态机与计费语义

- 轮次阶段：`proposed -> ingested`；上一轮未 ingest 不能开下一轮，达到 `round_limit` 不能再 propose。
- **费用在 propose 时锁定**：采购单位是家族，重复图片折叠为同一订单只计一次 `unit_cost`；
  回执只改变证据，永不再次计费。重复 ingest 同一 `(family, revision)` 记 `duplicate_skipped`。
- 乱序：旧 revision 记 `stale_ignored`；同 `receipt_id` 的更正件靠 revision 区分，
  revision≥2 必须 `charged_cost=0`（更正不收费）。
- `unjudgeable` 回执：费用已花、不产生训练证据，订单状态保留为 `unjudgeable`。
- 撤销：`withdrawals` 中 `effective_round` 起，该家族在后续训练快照里标记 `withdrawn` 并排除，
  历史快照仍可追溯。
- 仅展示授权（无 `training` purpose）的家族永远不进训练候选；基线里的仅展示家族同样被排除，
  排除原因写入每个训练快照的 `excluded_families`。
- 状态文件用“临时文件 + `os.replace`”原子落盘，并记录输入资料指纹；
  指纹不匹配或 schema 不符会拒绝加载，中断恢复不会串单、不会重复付费。

### 策略与停采

- `balanced` = 0.4 预测熵不确定性 + 0.3 类别覆盖缺口 + 0.2 供货商多样性缺口 + 0.1 成本效率；
  `uncertainty` 只看预测熵；`random` 使用由活动种子派生的独立随机流。
- 三策略共用同一可采购池（未采购且有 training 授权）、同一 60/轮预算、同一贪心填充规则、
  同一高斯朴素贝叶斯模型与同一份比较集，差异只在评分函数。
- 停采建议基于预注册规则：轮次上限、边际准确率阈值 0.085、Wilson 95% 置信区间下界是否改善、
  每单位预算净效用阈值 0.01。训练准确率只作参考输出，不参与停采判断，
  避免把某一批保留集的偶然上涨包装成普遍有效。

### 本地两轮演示

```bash
python3 scripts/run_acquisition_demo.py
```

演示在临时目录里对三个策略各走完两轮，覆盖部分回执、乱序补录、同 ID 更正、授权撤销、
中断恢复不重复付费与预算总账核对，每步带断言；`scripts/check_project.py` 会一并执行它。
