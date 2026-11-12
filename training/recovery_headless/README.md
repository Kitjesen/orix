# 无头版平地恢复：配置快照

对应 2026-11-12 20:10（北京时间）启动的 `RobotLab-Isaac-Recovery-Fall-Flat-Orix-FRNet-v5` 实验，资产 `v67_sim_r5_headless`。

- `run_configuration.json`：主要参数、环境版本和后续负重要求。
- `params/env.yaml`：实际保存的场景、奖励、观测、随机化等配置。
- `params/train.yaml`：实际 FR-Net runner、PPO 和 MCP 参数。
- `params/agent.yaml`：任务 runner 参数。标准 RSL-RL 字段和 FR-Net 自定义字段并存，实际归一化由 FR-Net 的 `observation_normalization: true` 控制。
- `validation/`：资产、去头质量/惯量、短仿真和本轮生效配置检查结果。

配置中的服务器绝对资产路径已改为仓库相对路径，日志路径也改为相对路径，其余保存参数原样保留。YAML 含 Isaac Lab 的类型标记，作为实验记录使用；这不是可脱离对应 FR-Net 运行源码独立执行的训练入口。

当前运行使用 2048 环境、2500 次迭代、seed=42，从零训练。直立高斯核宽 0.25，站姿门槛 0.1，每回合 350 步，固定平地、直接随机倒地初始化，无地形或姿态难度课程。所有关节力矩上限 17 N·m，动作目标使用硬限位中间 90%；髋/大腿/小腿动作缩放为 0.2/0.5/0.5。

质量随机化在启动时对各连杆分别乘 0.9–1.1，并同步缩放惯量。用户要求下一阶段加入额外 0–5 kg 负重，当前未实现；安装位置、惯量和采样方式需在实施时明确，质量监督标签需使用叠加后的真实物理质量。

当前模型源文件位于 `../v67_cts/assets/smalldog_v67/`。完整 FR-Net 执行源码仍在远程 RobotLab 工作区；仓库已有 `../runtime/` 是历史 MoE-CTS 快照，Isaac Sim / Isaac Lab 版本也不同。此分支没有发布无头恢复策略 checkpoint。

`validation/headless_runtime.json` 是去头资产的短仿真检查，先于直立核宽改动；`validation/training_config_verification.json` 专门核对本轮实际加载的 0.25 核宽等设置。两份记录不等同于收敛验证。
