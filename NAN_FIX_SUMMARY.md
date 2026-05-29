# NaN Filter Fix - 修复总结

## 问题描述
归因系统 `_build_chunk_training_samples` 中的 NaN 行过滤逻辑过于严格，导致动态特征（如距离、温度等）中的 NaN 值被当作整行丢弃的条件。结果：
- **旧事件数**：32,516（错误）
- **预期事件数**：数十万（正确）
- **事件丢失率**：84%

## 根本原因
`history_mask = df[required_cols].notna().all(axis=1)` 要求 **所有** 特征列都非 NaN。但动态特征（距离、温度、土地利用等）合法地存在 NaN：
- 距离 > 10km（`max_dist=10000`）→ NaN（⚠️ 此限制已在 2026-05-29 移除，现所有距离均可正常计算）
- 居民数据 NoData（127）→ NaN  
- Rolling lag（前3年无法计算）→ NaN

## 修复方案
NaN 行过滤只检查 **核心特征**（static + direct）：
- `slope`, `aspect`, `dist_to_river`
- `rainfall_prev_1m`, `rainfall_prev_2m`, `rainfall_prev_3m`, `rainfall_prev_total_3m`

动态特征中的 NaN 由下游的离散化器处理为 `"missing"` 类别，安全无害。

## 代码变更
在 `inducing_factor/core/attribution_system.py` 中修复了 **7 处** NaN 过滤：

1. **行 1001**：无背景路径 `history_mask` → 使用 `core_cols`
2. **行 1004**：无背景路径 `dropna` → 使用 `core_cols`
3. **行 1054**：主路径 `history_mask` → 使用 `core_cols`
4. **行 1063**：主路径 `dropna` → 使用 `core_cols`
5. **行 1082**：`build_training_samples` `dropna` → 使用 `core_feature_cols`
6. **行 501**：streaming run 离散化器 `dropna` → 使用核心特征过滤
7. **行 619**：streaming run 模型训练 `dropna` → 使用核心特征过滤

## 验证结果

### 实验验证（chunk_931）
| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| 事件保留数 | 2,401 | 15,037 | +526% |
| 事件保留率 | 16.0% | 100.0% | +84% |

### 全局验证（运行中）
| 指标 | 旧运行 | 当前运行（12.6% 进度） |
|------|--------|----------------------|
| 事件数 | 32,516 | 275,505+ |

## 下游安全验证
- ✅ 离散化器：`NaN` → `"missing"` 标签（不崩溃）
- ✅ 因子挖掘：`"missing"` 作为有效类别（不崩溃）
- ✅ 贝叶斯网络：`"missing"` 作为离散状态（不崩溃）

## 当前状态
- ✅ 代码修复已提交（`git commit 81aae18`）
- ✅ 实验验证通过
- 🔄 归因重跑进行中（`pty_0971865a`，预计 ~5.5 小时完成）
- ⏸ 2025 概率地图生成等待中

## 后续任务
1. 等待归因完成（自动通知）
2. 生成 2025 概率地图
3. 验证最终事件数 >> 10,489
