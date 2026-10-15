# SmallDog R5 采样性能排查与修正

日期：2026-10-15，Asia/Shanghai。

## 原因

原训练每轮约 11.55 秒，其中采样 11.37 秒、网络更新 0.183 秒。现场 GPU 利用率多数采样为 1%～2%，训练进程 CPU 约 138%，不支持“GPU 算力已用满”的判断。

实际 cProfile 定位到 SmallDog 的 `_foot_ids()`：两个奖励每个控制步各调用一次，每次对四个脚名分别访问 `sensor.body_names`。Isaac 的这个属性会读取整批复制机器人的 `prim_paths`。24 步内共触发 192 次该查询。

## 最小修正

仅将静态 asset/sensor 的脚部索引按 `(asset_cfg.name, sensor_cfg.name)` 缓存在当前 env 实例内。首次读取 sensor names 一次；后续继续读取实时接触力、接触位置和刚体状态。脚的顺序沿用 `FOOT_NAMES` 和原有 `preserve_order=True`。

## 同条件短测

两个测试均为 4096 环境、24 步预热、24 步采样、相同 model_9900、seed 42、初始地形等级范围 0～2，并与原训练共享 GPU。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 24 步采样耗时 | 14.233810 s | 3.416919 s |
| prim_paths 查询次数 | 192 | 0 |
| prim_paths 累计耗时 | 10.752207 s | 0 s |
| 奖励管理器累计耗时 | 11.743725 s | 0.637858 s |

短测加速 4.166 倍。cProfile 和共享 GPU 会影响绝对时间；正式训练恢复后的实测单独记录。

## 行为验证

- 本地 6 项 tensor/unit checks 通过；脚索引测试覆盖逆序 sensor 名称、重复调用及独立 env 的不同索引顺序。
- 实际 Isaac 4096 环境同一状态下，修改前后 contacts、valid、contact-point velocity、feet slip reward 和 diagonal-contact reward 五项张量逐值完全相同（rtol=0，atol=0）。
- 受影响脚本及模块通过 Ruff E4/E7/E9/F 检查；新脚本通过 py_compile。
- 地形网格、物理步长、4096 环境、奖励权重、网络、执行器与机器人模型沿用原配置。

## 接续训练

`overlay/scripts/smalldog/resume_sampling_cache.py` 已在完整 model_10100.pt 保存点完成接续。旧运行最后完成编号 10100，未丢失已完成更新。策略及两个优化器状态已恢复，4096 环境、八类地形，追加 2697 次迭代，最终目标编号仍为 12796。旧训练状态、checkpoint 和日志保留。新训练 PID 956214，顺序驱动 PID 956008；初期实际训练迭代约 3.1 秒。

标准 checkpoint 不包含地形课程分布和速度课程状态，因此恢复后课程从 level 0 / 0.35 m/s 重建；已学策略保留。该限制已向用户说明。新运行继续为 RoughRefinedGait 八类地形任务。

证据：`validation/r5_sampling_cache_comparison.json`、`r5_sampling_profile.pstats`、`r5_sampling_profile_cached.pstats`、`r5_sampling_cache_transition_state.json`。

## 正式训练恢复后的实测

2026-10-15T13:14:05.820440+08:00，运行 `2026-10-15_13-05-49_v67_r5_sampling_cache`，已到 iteration 10242。最近 20 轮平均总耗时 3.261 秒，采样 3.0934 秒，更新 0.1663 秒，约 30173 环境步/秒。日志 ETA 02:17:57。

课程从默认状态重建后，目前平均 terrain level 0.9354，速度上限已重新解锁到 1.0 m/s。该正式训练观测与旧训练处于不同课程阶段；同条件因果对照使用前述固定 model_9900 的短测。旧进程 951039 / 951768 均已退出，新进程 956008 / 956214 运行中。证据 `validation/r5_sampling_cache_training_status.json`。
