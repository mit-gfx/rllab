from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from subprocess import call
import numpy as np


class SoftWalkerEnv(Env):

    def __init__(self, urdf_name):
        super(SoftWalkerEnv, self).__init__()
        if not isinstance(urdf_name, str):
            raise TypeError('urdf_name shoudl be a string');
        self._urdf_name = urdf_name;

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self):
        state_file_name = 'reset_state' 
        call(['rl_api', 'reset', self._urdf_name, state_file_name])
        self._state = ReadMatrixFromFile(state_file_name)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)
