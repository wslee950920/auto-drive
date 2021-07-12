from gym.envs.registration import register

register(
    id='xycar-v0',
    entry_point='gym_xycar.envs:CustomEnv',
    max_episode_steps=20000,
)