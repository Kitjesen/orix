# V67 MoE-CTS 训练快照

本目录对应正在运行的 SmallDog V67 / R6 多地形训练，模型为 `v67_sim_r3`。正式训练资产位于 `v67_cts/assets/smalldog_v67`，当前任务的完整源码位于 `runtime/`。

## 已冻结的环境

| 组件 | 版本 / 来源 |
| --- | --- |
| Python | 3.11 |
| Isaac Sim | 5.1.0.0 |
| Torch | 2.7.0+cu128 |
| NumPy | 1.26.0 |
| Gymnasium | 1.2.1 |
| RobotLab 包 | 2.3.0，包含本项目修改 |
| RSL-RL 包 | 3.3.0，MoE-CTS fork |
| Isaac Lab 包 | 0.54.2；源码 commit `37ddf626871758333d6ed89cf64ad702aef127d0` |

完整已安装包记录见 [environment_snapshot.json](environment_snapshot.json)。这是一份来源记录，不能直接把其中所有版本当成新环境的安装清单。

`runtime/` 来自实际服务器源码快照，包含自定义 SmallDog、MoE-CTS、奖励、课程和脚本。`IsaacLab/` 包含对应的源码和应用配置。没有把服务器虚拟环境、GPU 驱动、Isaac Sim 专有运行库或所有上游机器人资产复制进 Git。

## 在已有 Isaac Sim 5.1 环境接入

使用已具备表中基础依赖的 Linux / NVIDIA CUDA 环境。安装本仓库的 Python 源码时使用 `--no-deps`，避免把已验证的 Torch / Isaac 栈自动升级：

```bash
python -m pip install --no-deps -e training/IsaacLab/source/isaaclab \
  -e training/IsaacLab/source/isaaclab_assets \
  -e training/IsaacLab/source/isaaclab_tasks \
  -e training/IsaacLab/source/isaaclab_rl
python -m pip install --no-deps -e training/runtime/source/rsl_rl \
  -e training/runtime/source/robot_lab
```

安装源依赖的硬件环境尚未在全新机器上做完整复现；本次发布验证了资产闭合、源码来源、模型可加载及文件配套关系。现有仿真记录见 `v67_cts/validation/`。

`run.sh` 使用当前激活环境的 `python`；也可通过 `SMALLDOG_PYTHON` 指定已有 Python。它只设置项目源码和资产路径，不安装依赖。

## 获取发布模型

在仓库根目录执行：

```bash
gh release download v67-r6-16100 --repo Kitjesen/orix \
  --pattern model_16100.pt --dir training/checkpoints
```

checkpoint 实际包含：

- `model_state_dict`；
- `optimizer_state_dict`：teacher 训练优化器；
- `optimizer_stu_enc_state_dict`：student encoder 优化器；
- `iter=16100`；
- `infos.smalldog_terrain`：4096 环境的地形分配、11×20 网格及速度课程状态。

完整课程恢复需要相同的地形网格和并行环境数。该发布是中间快照，不是 R6 完成模型。

## 先检查，再评估

```bash
python tools/validate_training_asset.py
bash training/run.sh scripts/smalldog/evaluate_gait.py \
  --checkpoint "$PWD/training/checkpoints/model_16100.pt" \
  --r6 --stratified-terrain-levels --max-terrain-level 10 \
  --num-envs 220 --command-vx 1.0 --duration-s 10 \
  --output "$PWD/terrain_eval.json" --headless --device cuda:0
```

每格地形为 6×6 m，10 秒前进轨迹可能跨格，第 10 行前方可能进入边界平地。因此这些数据应结合动态位置和终止情况解读，不能单靠整体均速或课程均值判定通过第 10 级。

## 从快照追加训练

下面命令会在自己的环境中创建新的训练运行，而不是连接或控制现有服务器进程：

```bash
bash training/run.sh scripts/smalldog/train_terrain_r6.py \
  --checkpoint "$PWD/training/checkpoints/model_16100.pt" \
  --num-envs 4096 --iterations 10000 --headless --device cuda:0
```

4096 环境属于当前服务器的并行配置，换机器时按实际显存调整；减少环境数后，旧 checkpoint 的逐环境课程状态不能原样恢复，应明确选择重新初始化课程。

## 修改入口

| 内容 | 文件 |
| --- | --- |
| URDF / mesh | [资产目录](v67_cts/assets/smalldog_v67/) |
| 初始姿态、限位、动作缩放 | [joint_motion_config.json](v67_cts/joint_motion_config.json) |
| RS02 执行器 / 根高度 / URDF 加载 | [smalldog.py](runtime/source/robot_lab/robot_lab/assets/smalldog.py) |
| 基础奖励与观测 | [env_cfg.py](runtime/source/robot_lab/robot_lab/tasks/smalldog/env_cfg.py) |
| 步态奖励与相位 | [gait_mdp.py](runtime/source/robot_lab/robot_lab/tasks/smalldog/gait_mdp.py) |
| 动作质量约束 | [refinement_mdp.py](runtime/source/robot_lab/robot_lab/tasks/smalldog/refinement_mdp.py) |
| 当前 R6 地形和提速 | [terrain_r6_cfg.py](runtime/source/robot_lab/robot_lab/tasks/smalldog/terrain_r6_cfg.py) |
| 课程状态 | [terrain_state.py](runtime/source/robot_lab/robot_lab/tasks/smalldog/terrain_state.py) |
| 训练与评估 | [scripts/smalldog/](runtime/scripts/smalldog/) |

`run_terrain_r6.py` 等历史服务器包装脚本保留了原服务器路径，用于追溯当时的顺序任务；跨机器使用上面的 `train_terrain_r6.py`、`evaluate_gait.py` 和 `run.sh` 入口。

## 资产重建

已发布 URDF 和 mesh 可以直接使用。`v67_cts/build_asset.py` 从冻结的机械预览输入、现有网格、V67 设计和运动参数生成仿真资产；会写入 `v67_cts/assets/smalldog_v67`，执行前应保存自己的改动。原始机械预览输入保存在 `v67_cts/source_assets/`。

重建需要 NumPy、Trimesh；静态外观图生成另需 PyVista、SciPy、Pillow。重建后先运行 `tools/validate_training_asset.py`，物理行为还需在目标仿真环境核对。
