from gymnasium.envs.registration import register

register(
     id="PtGEnv-v0",
     entry_point="ptg_gym_env.envs:PTGEnv",
)
