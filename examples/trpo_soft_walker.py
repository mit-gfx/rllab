from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.soft_walker_env import SoftWalkerEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import constant_strings as cs
import os

urdf_file = os.path.join(cs.robot_rl_folder, 'test_data', 'urdf', 'soft_biped_walker.urdf')
env = normalize(SoftWalkerEnv(urdf_file))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=1000,
    discount=0.99,
    step_size=0.1,
)
algo.train()
