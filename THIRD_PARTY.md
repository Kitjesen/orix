# 第三方来源

机器人模型和设计工作区来源为 [Cain-A/smalldog](https://github.com/Cain-A/smalldog)，本仓库记录当前 V67 / R6 交付快照。

## RobotLab 与 MoE-CTS

- 上游：[wertyuilife2/go2_rl_robotlab](https://github.com/wertyuilife2/go2_rl_robotlab)。
- 原始 commit：`28b4516d22617b11aeaf8ead63cc00b0c0bcd1bd`。
- 当前实际运行代码包含后续 SmallDog、奖励、课程、采样优化与录制修改，保存在 `training/runtime/`。
- 原始许可证保存在 [training/runtime/LICENSE](training/runtime/LICENSE)。RSL-RL 子目录中的许可证也原样保留。

## Isaac Lab

- 上游：[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)。
- 实际源 commit：`37ddf626871758333d6ed89cf64ad702aef127d0`。
- 对应源码和 apps 保存于 `training/IsaacLab/`。
- 许可证：[LICENSE](training/IsaacLab/LICENSE)、[LICENSE-mimic](training/IsaacLab/LICENSE-mimic)。

Isaac Sim 运行库、NVIDIA 驱动、CUDA 和安装环境不在此仓库分发，需要遵循其各自的安装与使用条款。其他上游机器人的模型和预训练权重没有随本快照复制。
