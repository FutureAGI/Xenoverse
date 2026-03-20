from gymnasium.envs.registration import register
from xenoverse.anyhvac.anyhvac_env import HVACEnv, HVACEnvDiscreteAction, HVACEnvDiffAction
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible

register(
    id='anyhvac-v1',
    entry_point='xenoverse.anyhvac.anyhvac_env:HVACEnv',
    kwargs={"max_steps": 5040,
            "failure_upperbound": 80,
            "iter_per_step": 600,
            "set_lower_bound": 16,
            "set_upper_bound": 32,
            "verbose": False },
)

register(
    id='anyhvac-visualizer-v1',
    entry_point='xenoverse.anyhvac.anyhvac_env:HVACEnvVisible',
    kwargs={"max_steps": 5040,
            "failure_upperbound": 80,
            "iter_per_step": 600,
            "set_lower_bound": 16,
            "set_upper_bound": 32,
            "verbose": False },
)