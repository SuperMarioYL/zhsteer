<!--
  模板文件（TEMPLATE）— 不是实测数据。

  本文件是 `zhsteer demo` 输出报告的结构样例：所有 <尖括号> 均为占位符，
  不代表任何测量值。在 16GB CUDA 机器上运行

      uv run zhsteer demo --model qwen3-4b

  会把真实实测数据写回这个路径（assets/demo/ablation-report.md），覆盖本模板。
-->

# zhsteer 消融报告

- 运行日期：<YYYY-MM-DD>
- 模型：<Qwen/Qwen3-4B-Instruct-2507>
- 方向包：`packs/qwen3-4b.pack`（layer <L>，checksum `<sha256 前 12 位>`）
- 最小对：`pairs/seed_pairs.zh.yaml`（sha256 `<…>`，train 52 / held-out 12）
- 消融强度：<1.0> · 提示词数：<10> · 生成参数：greedy，max_new_tokens=<512>
- 复现：`uv run zhsteer demo --model qwen3-4b`

> 分数为实测值，不做门槛过滤。计划设定的证伪线（mvp_plan §8）：held-out 可分性 ≥ 90%，
> 平均 trope 分数降幅 ≥ 20%，且无退化输出。提示词取自最小对集中标记 held_out 的 AI 腔侧，
> 未参与方向提取。

## 方向包验证（提取时记录）

- 层选择：<训练集可分性最高的层（平手取更深）>
- train 可分性：<XX%>
- held-out 可分性：<XX%>

## 各提示词前后对比

### 1. <提示词摘要>

**Before（原始）**

```text
<模型原始输出：通常带排比三连、值得体、爆款体等腔调>
```

**After（消融，强度 <1.0>）**

```text
<投影消融后的输出：腔调应当显著下降，内容保持完整>
```

trope 分数：<XX.XX> → <XX.XX>（降幅 <XX%>）；退化检查：Before 通过 / After <通过|未通过>

### 2. <提示词摘要>

**Before（原始）**

```text
<模型原始输出>
```

**After（消融，强度 <1.0>）**

```text
<消融后输出>
```

trope 分数：<XX.XX> → <XX.XX>（降幅 <XX%>）；退化检查：Before 通过 / After <通过|未通过>

### 3. – 10. <提示词摘要>

（结构同上，共 <10> 条 held-out 提示词）

## 分数汇总

| #  | 提示词        | Before  | After   | 降幅   |
| -- | ------------- | ------- | ------- | ------ |
| 1  | <提示词摘要>  | <XX.XX> | <XX.XX> | <XX%>  |
| 2  | <提示词摘要>  | <XX.XX> | <XX.XX> | <XX%>  |
| …  | …             | …       | …       | …      |
| 10 | <提示词摘要>  | <XX.XX> | <XX.XX> | <XX%>  |

**平均 trope 分数**：<XX.XX> → <XX.XX>（平均降幅 <XX%>）

**退化输出**：<N>/<10>（After 侧含成段重复即计为未通过）
