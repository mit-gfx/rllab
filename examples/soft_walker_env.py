from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from subprocess import call, Popen
import numpy as np
import constant_strings as cs
import os
import matrix_io as m
import IPython


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0

class SoftWalkerEnv(Env):

    def __init__(self, urdf_name):
        super(SoftWalkerEnv, self).__init__()
        if not isinstance(urdf_name, str):
            raise TypeError('urdf_name should be a string');
        self._urdf_name = urdf_name;
        self._rl_api = os.path.join(cs.robot_rl_build_folder, 'rl_api')
        self._count = 0
        self._file_count = 0
        self._call_schedule = CappedCubicVideoSchedule() #Currently unused

    @property
    def observation_space(self):
        state_bound_file = 'state_bound'
        call([self._rl_api, 'state-space', self._urdf_name, state_bound_file])
        bound = m.ReadMatrixFromFile(state_bound_file)
        return Box(bound[:, 0], bound[:, 1])

    @property
    def action_space(self):
        action_bound_file = 'action_bound'
        call([self._rl_api, 'action-space', self._urdf_name, action_bound_file])
        bound = m.ReadMatrixFromFile(action_bound_file)
        return Box(bound[:, 0], bound[:, 1])

    def reset(self):
        state_file_name = 'reset_state' 
        call([self._rl_api, 'reset', self._urdf_name, state_file_name])
        self._state = m.ReadMatrixFromFile(state_file_name)
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        state_file_name = 'old_state'
        action_file_name = 'action'
        new_state_file_name = 'new_state'
        dt = '0.005'
        info_file_name = 'info.txt'
        s = np.copy(self._state)
        a = np.copy(action)
        m.WriteMatrixToFile(state_file_name, s)
        m.WriteMatrixToFile(action_file_name, a)
        call([self._rl_api, 'step', self._urdf_name, state_file_name, action_file_name, \
            dt, new_state_file_name, info_file_name])
        self._state = m.ReadMatrixFromFile(new_state_file_name)
        next_observation = np.copy(self._state)
        # Now get reward and done ready from the text file.
        info = dict()
        for line in open(info_file_name):
            key, value = [s.strip() for s in line.split(':')]
            info[key.lower()] = value
        reward = float(info['reward'])
        done = True if int(info['done']) == 1 else False
        
        #Should we render?

        
        if self._call_schedule(self._count):
            self.render()
        
        self._count += 1
        
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        state_file_name = 'viewer_states/view_state'
        s = np.copy(self._state)
        write_state_file_name = ''.join([state_file_name, '_', str(self._file_count)])
        m.WriteMatrixToFile(write_state_file_name, s)
        viewer_folder = os.path.join(cs.robot_rl_build_folder, 'rl_viewer')
        
        """
        try:
            self._p.kill()
        except:
            pass
        """

        try:
            if self._p.poll() is not None:
                self._p = Popen([self._rl_api, 'initialize-viewer', self._urdf_name, state_file_name,  viewer_folder])
        except:
            self._p = Popen([self._rl_api, 'initialize-viewer', self._urdf_name, state_file_name,  viewer_folder])
        
        call([self._rl_api, 'view', self._urdf_name, state_file_name,  viewer_folder, str(self._file_count)])
        self._file_count +=1
    
        # TAO: not sure what should be done here.
        # print('current state:', self._state)
        
