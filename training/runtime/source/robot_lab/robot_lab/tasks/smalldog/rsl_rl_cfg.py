from isaaclab.utils import configclass
from robot_lab.tasks.go2.rsl_rl_cfg import MoECTSRunnerCfg


@configclass
class SmallDogMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = 'smalldog_v67_moe_cts'
    max_iterations = 200
    save_interval = 100
    seed = 42

    def __post_init__(self):
        self.policy.init_noise_std = 0.35
        self.algorithm.num_learning_epochs = 3
        self.algorithm.learning_rate = 3e-4
        self.algorithm.student_encoder_learning_rate = 3e-4
