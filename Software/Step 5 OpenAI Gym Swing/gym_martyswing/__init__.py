from gym.envs.registration import register
register(
    id='martyswing-v0',
    entry_point='gym_martyswing.envs:MartySwingEnv',
)
