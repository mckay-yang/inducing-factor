# CLASS 全流域 Rust 流程交接上下文

生成时间：2026-06-05 10:41 CST

## 当前结论

当前没有需要继续运行的后台流程。最后一个全流域 PTY 已按用户要求停止：

- PTY：`pty_4654e4a1`
- 命令：`cargo run --manifest-path .temp/full_basin_driver/Cargo.toml`
- 状态：killed / exit 0 / SIGTERM
- 停止位置：`Processing data/temperature/tmp_2016.nc [1/1] : 0...10...20...30...40...50...60...70...80.`

后续如果继续跑，必须重新启动命令。重启时会重新编译，因此会使用最新代码变更。

## 用户明确约束

- 全流域必须使用同一个模型，不做 per-chunk / per-region 模型。
- CLASS 是扰动真值和目标网格，所有数据对齐到 CLASS 网格。
- CCDC 不是扰动真值，只用于给 CLASS 扰动提供发生时间。
- 训练数据只采用 2017-2024。
- 预测 2025 全流域每个像元扰动概率。
- 降雨只需要 2015-2025 范围。
- 温度也已按同样原则限制到 2015-2025。
- 缺年数据不要生成重复影像；只保存真实源年份的数据，下游读取时做 year fallback。
- 道路距离计算时，每个点需要使用完整道路数据，不按 chunk 裁剪道路几何。
- 数据缺失时，使用当年之前最新数据填充。

## 当前缓存状态

来自 2026-06-05 10:41 CST 检查：

```text
.cache/resampled              656G
.cache/resampled/rainfall     499G
.cache/resampled/temperature  151G
.cache/resampled/landuse        0B
.cache/resampled/river          0B
.cache/resampled/road           0B
.cache/resampled/resident       0B
.cache/features                 0B
```

文件计数：

```text
CCDC:        11 files
Rainfall:   120 files
Temperature: 12 files (+ test_tmp_2020.tif)
```

当前范围：

- Rainfall 已生成：`pre201501.tif` 到 `pre202412.tif`，共 120 个。源目录可能没有 2025 月文件；继续前应确认 `data/rainfall/pre2025*.tif` 是否存在。
- Temperature 已生成：`tmp_2015_01.tif` 到 `tmp_2015_12.tif`。`tmp_2016.nc` 正在处理中时被停止，可能存在中间临时文件，继续前可检查 `.cache/resampled/temperature/._tmp_resampled_2016.tif`。
- DEM 已存在：`.cache/resampled/dem.tif`。
- CCDC 已存在：`.cache/resampled/ccdc/2015.tif` 到 `2025.tif`。

## 已完成的重要验证

- CLASS 中心区域数据验证通过。
- CLASS 中心 128×128 完整流程验证通过：
  - `results/validation/class_center_probability_2025_128.tif`
  - `results/validation/class_center_training_samples.csv`
  - `results/validation/class_center_full_flow_report.md`
- BigTIFF 问题已定位并修复：旧 DEM 重采样失败是 classic GeoTIFF 4GB limit；后续大 GeoTIFF 输出均加了 `BIGTIFF=YES`。
- 最新代码在停止前已执行：

```bash
cargo check --manifest-path .temp/full_basin_driver/Cargo.toml
```

结果通过。LSP 诊断不可用，因为当前 toolchain 没有 `rust-analyzer`。

## 已修改的核心文件

### `Cargo.toml`

- GDAL 依赖更新到 `gdal = "0.19"`。

### `.temp/full_basin_driver/src/main.rs`

临时全流域 staging driver。当前逻辑：

1. 打开 CLASS 网格：`data/disturbance_class/disturbance_class_res.tif`
2. 跳过或生成 DEM。
3. 重采样 CCDC。
4. 重采样 rainfall，限制 `2015-2025`。
5. 重采样 temperature，限制 `2015-2025`。
6. 重采样 landuse、resident。
7. 计算 river distance。
8. 计算 road/railway/waterway distance。
9. 生成 training points。
10. 提取 training features。
11. 提取 2025 full-AOI prediction features。

注意：这是 `.temp/` 下的临时 driver，但目前全流域 staging 主要靠它跑。

### `crates/data-process/src/resample.rs`

主要变化：

- 新增 `resample_rainfall_to_class_for_years(...)`。
- `resample_rainfall_to_class(...)` 保留为兼容包装，默认不限制年份。
- 新增 `resample_temperature_to_class_for_years(...)`。
- `resample_temperature_to_class(...)` 保留为兼容包装，默认不限制年份。
- gdalwarp 输出大 GeoTIFF 使用：
  - `COMPRESS=DEFLATE`
  - `TILED=YES`
  - `BLOCKXSIZE=512`
  - `BLOCKYSIZE=512`
  - `BIGTIFF=YES`

### `crates/data-process/src/road_distance.rs`

最新设计变更：

- 不再在 write-time 做 gap-fill。
- 只为“源数据真实存在且属于 CLASS 年份范围”的年份生成 distance raster。
- 例如 source 有 2024，没有 2025，则只保存 `road_dist_2024.tif`，不生成重复的 `road_dist_2025.tif`。
- 缺年回填由 `feature_extraction` 读取时处理。
- 删除了 `find_prior_year(...)`。
- 更新测试：`test_gapfill` 改为 `test_no_gapfill_at_write`。

道路源数据检查到的大致范围：`data/road` 有 2014-2024。CLASS 年份是 2016-2025。因此预计只生成 2016-2024 的 road/railway/waterway distance；2025 读取时回退到 2024。

### `crates/data-process/src/feature_extraction.rs`

最新设计变更：所有缺年数据在读取时回填，不靠重复影像。

- `find_road_raster(...)` 已有回退：`year -> year-1 -> ... -> year-5`。
- 新增 `find_rainfall_raster(...)`：`year -> year-1 -> ... -> year-10`。
- 新增 `find_temperature_raster(...)`：`year -> year-1 -> ... -> year-10`。
- Training feature extraction 和 prediction feature extraction 都已改用这些 fallback 函数。

效果：如果 2025 降雨或温度不存在，预测 2025 时会读取 2024，继续往前最多 10 年。

### `crates/data-process/src/river_distance.rs`

- River distance 是单一静态 raster：`.cache/resampled/river/river_dist.tif`。
- 没有年份维度，因此无需生成重复年份影像。
- 大 GeoTIFF 输出已使用 BigTIFF 选项。

### `crates/cli/src/commands.rs`

- 新增或调整 basin-wide train/predict 路径：`run_train_basin()`、`run_predict_basin()`。
- 目标是单一全流域模型。
- 概率 GeoTIFF 输出使用 BigTIFF 创建选项。

### 其他 GDAL 0.19 修复

这些文件做过 GDAL 0.19 API 适配：

- `crates/loaders/src/ccdc.rs`
- `crates/loaders/src/climate.rs`
- `crates/loaders/src/dem.rs`
- `crates/loaders/src/disturbance_class.rs`
- `crates/loaders/src/landuse.rs`
- `crates/loaders/src/resident.rs`
- `crates/predict/src/lib.rs`

典型变更包括：

- `Buffer.data` 改为 `data()`。
- `rasterband(usize)`。
- `create_with_band_type::<T, _>(...)`。
- `band.write(..., &mut Buffer)`。

## PTY 历史

### `pty_36c93688`

- 初始全流域 run。
- 失败于 DEM resampling。
- 根因：`.cache/resampled/dem.tif` partial 正好 4.0G，classic GeoTIFF 4GB limit。
- 已通过 BigTIFF 修复。

### `pty_35787df8`

- BigTIFF 修复后的重跑。
- DEM 和 CCDC 完成。
- 误从 1901 开始跑 rainfall。
- 被停止后清理误缓存。

### `pty_b813520f`

- 修复 rainfall year filter 后重跑。
- Rainfall 2015-2024 完成，共 120 个文件。
- 后续发现 temperature 仍在从 1990 开始跑。
- 被停止后清理 1990-2014 temperature 误缓存。

### `pty_4654e4a1`

- 修复 rainfall + temperature year filter 后重跑。
- DEM/CCDC/rainfall 跳过。
- Temperature 完成 2015 全年，开始处理 2016 时按用户要求停止。
- 注意：这个 PTY 启动时还没有后来关于 road write-time 去重和 feature_extraction fallback 的最新代码。已停止，后续重启会重新编译使用最新代码。

## 已清理的误缓存

- 删除过 rainfall 1901-1926 误缓存，释放约 1.2T 级别空间。
- 删除过 temperature 1990-2014 误缓存，释放约 395G。

当前磁盘状态正常，最近检查 `/Volumes/T7` 剩余空间约 1.3Ti 级别。

## 仍需注意的问题

1. CCDC resume 逻辑会产生非致命 GDAL warning：

   ```text
   ERROR 1: Output dataset .cache/resampled/ccdc/YYYY.tif exists,
   but some command line options were provided indicating a new dataset should be created.
   ```

   目前流程会 log warn 后继续，不阻塞。但建议后续把 CCDC resample 改为真正检查 output exists 后跳过，避免噪声和浪费。

2. Temperature 当前只完成 2015。继续前建议删除可能残留的 `.cache/resampled/temperature/._tmp_resampled_2016.tif`，避免半成品影响重跑。

3. Rainfall 只到 2024-12。若 `data/rainfall` 确实没有 2025，预测 2025 会由 `feature_extraction` fallback 到 2024。

4. Temperature 源目录目前观察到 2015-2024，未看到 2025。预测 2025 会 fallback 到 2024。

5. Road distance 后续会只生成真实源年份，不会写 2025 重复栅格。读取 2025 时由 `find_road_raster` fallback 到 2024。

6. River distance 是静态单文件，后续只生成一次，不存在年份重复问题。

## 后续推荐执行顺序

如果要继续 staging：

```bash
# 可选：清理温度 2016 临时中间文件
rm -f .cache/resampled/temperature/._tmp_resampled_2016.tif

# 重新编译并继续全流域 staging
cargo run --manifest-path .temp/full_basin_driver/Cargo.toml
```

预计流程会：

1. 跳过 DEM。
2. CCDC 输出 exists warnings 后继续，或若先修 CCDC resume，则干净跳过。
3. 跳过 rainfall 2015-2024 已有文件。
4. 跳过 temperature 2015 已有文件，从 2016 继续。
5. 完成 temperature 2016-2024。
6. 生成 landuse、resident。
7. 生成 river distance 单文件。
8. 生成 road distance 2016-2024 真实源年份，不生成 2025 duplicate。
9. 生成 training points/features。
10. 生成 2025 prediction feature tiles。

然后训练和预测：

```bash
cargo run -p cli -- train-workflow --config configs/class_basin_workflow.json
cargo run -p cli -- predict-workflow --config configs/class_basin_workflow.json
```

最终需要验证：

- `results/predictions/probability_2025.tif` 存在。
- shape = CLASS grid：62166 × 35095。
- CRS = EPSG:4326。
- NoData/NaN 处理正常。
- 值域在 `[0, 1]`。
- 输出使用 BigTIFF。

## 最新代码验证状态

最新修改后已通过：

```bash
cargo check --manifest-path .temp/full_basin_driver/Cargo.toml
```

未跑完整测试和未跑最终全流域 train/predict，因为用户要求停止并生成上下文文档。
