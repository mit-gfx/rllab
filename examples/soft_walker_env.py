from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from subprocess import call, Popen
import numpy as np
import constant_strings as cs
import os
import matrix_io as m
import IPython
from datetime import datetime 


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        return count % 1 == 0
        #if count < 1000:
        #    return int(round(count ** (1. / 3))) ** 3 == count
        #else:
        #    return count % 1000 == 0

class SoftWalkerEnv(Env):

    def __init__(self, urdf_name, batch_size):
        super(SoftWalkerEnv, self).__init__()
        if not isinstance(urdf_name, str):
            raise TypeError('urdf_name should be a string');
        self._urdf_name = urdf_name;
        self._rl_api = os.path.join(cs.robot_rl_build_folder, 'rl_api')
        self._count = 0
        self._file_count = 0
        self._call_schedule = CappedCubicVideoSchedule() #Currently unused
        
        #lazy evaluation objects
        self._obs_space = None
        self._action_space = None
        self._reset_value = None
        self._batch_size = batch_size #Needed for rendering properly
        self._batch_num = 0
        
    def get_iter_num(self):
        return int(self._count / self._batch_size)
        
    def update_counters(self):
        if self._count % self._batch_size == 0:
            self._batch_num = 0
            self._file_count = 0

    @property
    def observation_space(self):
        startTime= datetime.now() 
        if self._obs_space is None:
            state_bound_file = 'state_bound'
            call([self._rl_api, 'state-space', self._urdf_name, state_bound_file])
            bound = m.ReadMatrixFromFile(state_bound_file)
            self._obs_space = Box(bound[:, 0], bound[:, 1])
        timeElapsed=datetime.now()-startTime 
        #print('ObservationSpace: Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
        return self._obs_space

    @property
    def action_space(self):
        if self._action_space is None:
            action_bound_file = 'action_bound'
            call([self._rl_api, 'action-space', self._urdf_name, action_bound_file])
            bound = m.ReadMatrixFromFile(action_bound_file)
            self._action_space = Box(bound[:, 0], bound[:, 1])
            return self._action_space
        return self._action_space

    def reset(self):
        #print('reset')
        if self._reset_value is None:
            state_file_name = 'reset_state' 
            call([self._rl_api, 'reset', self._urdf_name, state_file_name])
            self._state = m.ReadMatrixFromFile(state_file_name)
            observation = np.copy(self._state)
            self._reset_value = observation
        #print(self._reset_value)
        self._state = self._reset_value
        return self._reset_value

    def step(self, action):
        
        state_file_name = 'old_state'
        action_file_name = 'action'
        new_state_file_name = 'new_state'
        dt = '0.001'
        info_file_name = 'info.txt'
        #print(self._state)
        s = np.copy(self._state)
        a = np.copy(action)
        m.WriteMatrixToFile(state_file_name, s)
        m.WriteMatrixToFile(action_file_name, a)
        startTime= datetime.now() 
        call([self._rl_api, 'step', self._urdf_name, state_file_name, action_file_name, \
            dt, new_state_file_name, info_file_name])
        timeElapsed=datetime.now()-startTime 

        #print('Step: Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
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
        
        #DO we reset?
        self.update_counters()
        
        #TODO: Why are nans appearing at all?
        
        if np.isnan(reward):
            #IPython.embed()
            print('fixed a NAN')
            reward = 0.0
            
            
        #If we are done, increase the batch num
        if done:
            self._batch_num += 1
            print(self._batch_num)
            self._file_count = 0
        
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        state_file_name = 'viewer_states/view_state'
        s = np.copy(self._state)
        write_state_file_name = ''.join([state_file_name, '_', str(self.get_iter_num()), '_', str(self._batch_num), '_', str(self._file_count)])
        m.WriteMatrixToFile(write_state_file_name, s)
        viewer_folder = os.path.join(cs.robot_rl_build_folder, 'rl_viewer')
        


        try:
            if self._p.poll() is not None:
                self._p = Popen([self._rl_api, 'initialize-viewer', self._urdf_name, state_file_name,  viewer_folder])
        except:
            self._p = Popen([self._rl_api, 'initialize-viewer', self._urdf_name, state_file_name,  viewer_folder])
        
        call([self._rl_api, 'view', self._urdf_name, state_file_name,  viewer_folder, str(self.get_iter_num()), str(self._batch_num), str(self._file_count)])
        self._file_count +=1
    
        # TAO: not sure what should be done here.
        # print('current state:', self._state)
        
