import gymnasium as gym

gym.register(
    id="Tlibot-Z1-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "basic_loco_lib.tasks.mimic.agents.rsl_rl_ppo_cfg:z1_mimic_ppo_runner_cfg",
    },
)