# Isaac Sim 录制与训练迁移

当前正式 R6 训练保留 Isaac Sim 5.1 / Python 3.11 / Torch 2.7，以当前模型、奖励与课程继续运行。

原有多地形录像采用 Isaac 物理轨迹及实际地形网格的离线回放。用户要求更好的原生相机录像后，已新增 `record_isaac_camera.py`，使用 `render_mode='rgb_array'`、Isaac Sim 相机跟随和原生 RGB 帧。

服务器 RTX 5090 D / 驱动 595.71.05 下，相机最小测试在加载机器人前的 `librtx.scenedb.plugin.so` 初始化阶段发生 native crash。NVIDIA 上游报告了相同 Isaac Sim 5.1 / 驱动组合的问题，并有升级到 Isaac Sim 6 解决的记录：

- [官方问题 #677](https://github.com/isaac-sim/IsaacSim/issues/677)
- [同版本驱动及 RTX 初始化崩溃报告](https://github.com/isaac-sim/IsaacSim/discussions/648)
- [Isaac Sim 6 Python 安装说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html)

用户已批准另建隔离的 Sim 6 / Python 3.12 录制环境。该工作不更换现有训练环境和驱动；本仓库首次发布时，尚未宣称 Sim 6 原生录像已经通过。

整体迁移还涉及 Isaac Lab、相机和 URDF 接口、PhysX 接触、执行器、观测归一化以及 MoE-CTS runner。应先固定同一个 checkpoint，核对 12 关节顺序、限位、PD、质量和碰撞，再用短时站立及固定速度 rollout 比较两套环境。新环境通过后，再决定下一轮训练的切换时间。

如果 Sim 6 仅渲染已有真实轨迹，必须标注为“Isaac Sim 原生相机回放”；只有策略和物理也在 Sim 6 中运行，才可标注为“Sim 6 策略 rollout”。两者不能混用。
