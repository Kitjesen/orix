# SmallDog：上游 MoE-CTS 与关节限位核对（2026-10-13）

更新：下方限位表记录本日修正前的 R1 状态。模型工作空间研究和真实 Isaac 验证后，训练资产现为 `v67_sim_r2`；当前数值、默认姿态、动作软窗口和启动记录见 [JOINT_WINDOW_R2_20261013.md](JOINT_WINDOW_R2_20261013.md)。本文件保留旧窗口以说明问题来源。

## 采用的上游

按用户指定的 wty-yy/go2_rl_gym 中 go2_moe_cts 核对，同时对照它链接的 Isaac Lab 版本 wertyuilife2/go2_rl_robotlab。这里不把其他同名 CTS-MoE 论文混为本项目的上游。

- [Gym 环境配置](https://github.com/wty-yy/go2_rl_gym/blob/master/legged_gym/envs/go2/go2_config.py)
- [Gym 网络](https://github.com/wty-yy/go2_rl_gym/blob/master/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py)
- [Lab 环境配置](https://github.com/wertyuilife2/go2_rl_robotlab/blob/main/source/robot_lab/robot_lab/tasks/go2/env_cfg.py)
- [Lab 算法配置](https://github.com/wertyuilife2/go2_rl_robotlab/blob/main/source/robot_lab/robot_lab/tasks/go2/rsl_rl_cfg.py)
- [Lab Go2 资产](https://github.com/wertyuilife2/go2_rl_robotlab/blob/main/source/robot_lab/robot_lab/assets/unitree.py)

## 上游如何获得步态

当前主配置没有周期相位输入、固定步频、对角支撑时序或参考摆腿轨迹。主要通过速度跟踪、机身高度/竖向速度、角速度、近地足端水平运动、关节加速度、功耗、动作连续性和非足端碰撞等项学习步态。存在脚腾空时间函数不表示主配置启用了该项。

学生编码器是 8 专家 MoE，输出32维 latent，再与当前本体观测拼接交给共享 actor，输出12关节动作。教师读取特权观测；学生读取本体历史。75%教师环境、25%学生环境并行训练，包含latent对齐和负载均衡。它不是预先指定8种步态的控制器。

Lab动作是 default_joint_pos + .25 * action，动作项的绝对目标clip为±100rad，实际关节由物理限位和关节越软限位惩罚处理。Gym采用同类PD目标，力矩另做限幅。我们的SmallDog额外将目标严格裁剪到自身URDF范围，所以原始动作可能持续越界，但实际关节停在边界。

## 限位的修正前状态（R1 历史记录）

- 机械角度范围：这轮没有扩大。V67、portable和当前总装URDF三份范围一致。
- 旧训练软限位系数：1.0。
- 新实验 FlatGait/RoughGait：改为上游采用的0.9，并增加原始目标越界惩罚weight=-0.5。这是惩罚提前生效，不是物理范围扩大。
- 力矩/速度上限：保留hip/thigh 7N·m、42.9351rad/s，calf 14N·m、21.46755rad/s。
- 当前URDF范围是外壳试装与站姿展示的临时窗口，不是已经实测的机械止挡。来源说明见 smalldog_description/README.md。

| 关节组 | 当前URDF范围rad |
|---|---|
| hip，四腿 | [-0.524,+0.524] |
| 右thigh，前后 | [-0.86,+0.09] |
| 左thigh，前后 | [-0.09,+0.86] |
| 右calf，前后 | [0,+1.37] |
| 左calf，前后 | [-1.37,0] |

默认thigh绝对值.770796327、calf绝对值1.282614965：离最近URDF边界仅5.111°和5.007°。thigh/calf动作scale=.2rad，对raw action=1产生11.459°偏移，偏向近边界的一侧很容易被裁剪。例如右thigh原始动作低于-.446就触发下界，右calf高于+.437就触发上界。0.9软窗口会使提前惩罚的余量进一步变小；所以补软限位不等于站姿和动作范围已经校正完成。

下一步应联合核对关节坐标/零位、带壳装配运动范围、默认站姿和动作映射，之后才能给出新的机械角度上下限。不能直接复制Go2限位，不能把2:1传动再重复套在已经是关节端的角度上。

## 实验变更与证据

相位/支撑/净空辅助是SmallDog实验方案，不属于上游主配置。参数暂定1.5Hz、65%支撑期、4cm摆动峰值，低速平地命令.15～.30m/s、10%站立。独立任务注册，不覆盖旧Flat/Rough任务。

真实足端计算使用：
1. 每脚单独的地面接触传感器，接触点速度为 v_COM + omega × (p_contact - p_COM)。
2. URDF脚球体半径32mm、局部偏置X=16.58312383mm，四条地面射线测垂直脚底净空。
3. 按名字独立匹配robot和contact sensor的脚索引。

首次检查发现地面过滤器指向父Xform，力虽有效但contact points全是NaN，检查拒绝通过。修成实际CollisionPlane路径后，8环境120步平地零动作验证通过，接触点覆盖率100%，异常终止0；相位重置、历史、新观测维度470/277/47和固定支撑对整周期负分检查通过。这些仅证明配置/传感器接通，不证明学习步态改善。

新实验尚未产出训练策略。旧9795策略仍是问题基线；新增phase维度改变网络输入，不能直接原样加载旧checkpoint。
