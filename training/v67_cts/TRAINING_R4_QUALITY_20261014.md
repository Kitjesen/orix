# SmallDog V67 R4：机身俯仰与动作质量修正

日期：2026-10-14（Asia/Shanghai）。当前状态：原400轮修正及三个速度评估已经完成，最终model2797。1m/s俯仰RMS降至1.42492°，实际均速0.99750m/s。用户随后授权再追加10000轮，已启动，见TRAINING_R4_PLUS10000_20261014.md。

## 本轮目标

用户要求持续推进训练目标调整，重点减少机身前后俯仰，缩小不必要的关节摆幅，改善落脚和运动观感。R3 的速度跟踪结果保留为对照。

这次是 R3 RoughGait 策略权重向 R4 平地动作质量任务的迁移初始化，不是完全相同环境的续训。完成平地对照后再决定是否接入新的 RoughRefinedGait 多地形配置。

## 模型与算法

- V67 外壳、电池、12 关节资产仍为 `v67_sim_r3`，URDF、碰撞、质量、执行器和关节位置上下限未改动。
- 使用 `model_2398.pt` 的网络权重，重置 PPO 与 student encoder 优化器；观测维度仍为 470 / 277 / 47，动作 12 维。
- 8 专家 MoE-CTS，teacher 75%、student 25%，4096 并行环境，追加 400 次学习迭代。
- 起始全局编号 2398，预期末次 2797；保存间隔 100。
- 新任务 `RobotLab-SmallDog-V67-FlatRefinedGait-v0`；旧 R3 FlatGait / RoughGait 注册与配置保留。

## 已实现调整

| 项目 | R3 | R4 平地候选 |
|---|---:|---:|
| 步频 | 固定 1.5 Hz | ≤0.3 m/s 为1.5 Hz，1.0 m/s为2.5 Hz，中间线性插值 |
| 支撑占比 | 0.65 | 0.60 |
| 脚底净空峰值参考 | 0.04 m | 0.04 m，另对超过相位参考+0.02 m的摆动腿增加惩罚 |
| flat_orientation | -2.0 | -4.0 |
| body_pitch | 无独立项 | -4.0 × body-frame gravity X² |
| body_pitch_rate | 无独立项 | -0.15 × body-frame pitch angular velocity² |
| action_target_clip | -0.5 | -1.0 |
| action_rate / action_smoothness | -0.01 / -0.01 | -0.015 / -0.015 |
| base_height_l2 | -1.0 | -5.0 |

相位观测、接触目标、摆动净空目标及过抬脚惩罚使用同一速度相关时钟。每个回合15秒，命令完整保持15秒，回合重置时清除时钟与观测历史。平地训练速度范围0.15～1.0 m/s，横向与转向命令为0，10%站立。

新 RoughRefinedGait 配置仍保留八种地形及速度/地形课程；它不增加全局绝对 body_pitch 项，保留原orientation -2，增加pitch-rate -0.15，height -3，避免用平地绝对姿态约束否定坡面运动。本次先启动平地短程试验。

## 验证与评价标准

上线前：纯张量相位/掩码测试、Python/Ruff、实际Isaac 8环境导入与120步静态检查、相位维度与奖励源、R3 checkpoint权重加载、地面接触与净空采样。

训练后的独立学生测试固定0.3 / 0.6 / 1.0 m/s，每项至少10秒，排除前2秒启动。新增指标包括pitch/roll角度与角速度、关节实际跨度、真实地形相对脚底净空、逐关节动作裁剪、电机扭矩饱和比例、速度误差。发生重置的环境不进入姿态跨度统计，并明确报告被排除数量。

暂定平地目标是俯仰RMS接近3°以内、绝对俯仰大部分时间在5°以内，减少高抬腿和限幅，同时保持原来速度跟踪能力。这些是待验证目标，不是本轮已达到的结果。400轮后依据同命令对照判断是否继续；不把训练进程运行等同于修正成功。

## 文件

- `overlay/source/robot_lab/robot_lab/tasks/smalldog/refinement_cfg.py`
- `overlay/source/robot_lab/robot_lab/tasks/smalldog/refinement_mdp.py`
- `overlay/scripts/smalldog/train_refinement.py`
- `overlay/scripts/smalldog/check_refinement_import.py`
- `overlay/scripts/smalldog/evaluate_gait.py` 与 `record_gait_motion.py` 的 `--refined` 选项
- `overlay/start_v67_quality_r4.sh`

## 启动与实测证据

- 平地和多地形实际Isaac导入检查均通过：每项8环境、120步，0次异常终止，地面接触点采样覆盖率100%，新奖励有限，观测维度470/277/47，R3权重可加载。
- R3同条件16环境、1m/s平地基准：平均vx0.92162m/s，俯仰RMS4.50673°，俯仰角度跨度的每环境均值15.91456°，俯仰角速度RMS1.08037rad/s，0次终止。
- 新运行：`2026-10-14_14-19-28_v67_r4_pitch_refinement_flat_pilot`。
- 训练PID：915949；顺序验证/训练/评估进程PID：915285。
- `validation/r4_quality_training_run.json`：实际加载与运行参数；`validation/r4_quality_config/`：服务器导出的实际env/agent配置。
- `validation/r4_quality_training_latest.log`：最近下载的训练日志，检查到iteration2406，无Traceback、RuntimeError、AssertionError或OOM。
- `overlay/scripts/smalldog/run_quality_r4.py` 串行执行：rough接口检查 → R3基准 → 400轮平地修正 → 0.3/0.6/1.0m/s学生测试 → 1m/s真实位姿录制。结果写入服务器validation/r4_quality_*文件。
- 首轮日志已经包含并实际计算body_pitch、body_pitch_rate和excess_swing_clearance奖励项。这证明训练目标已接入，不代表策略已达标。
