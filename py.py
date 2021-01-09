import matplotlib
import numpy as np
import tensorflow as tf
import tf_agents
import math

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from selenium import webdriver

# Hyperparameters
num_iterations = 20000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
i = 16
j = 16
initialState = np.zeros(i*j)
batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000




def getList():
    driver = webdriver.Firefox(firefox_binary="C:\Program Files\Mozilla Firefox\geckodriver.exe") # Link to firefox binary
    driver.get("http://minesweeperonline.com/#intermediate") # App link
    observation = {
        "blocks": [],
        "face": ""
    }
    
    
    # Loop through i*j blocks
    for i_component in range(1, i):
        for j_component in range(1, j):
            id = i_component + "_" + j_component
            block = driver.find_element_by_id(id)
            blockClasses = block.CLASS_NAME.split(" ")
            blockClass = blockClasses[1]
            n = blockClass[4] # get number of bombs/ blank
            if math.isnan(n): # if it is blank
                elem = -1
            else: # if it is not blank
                elem = int(n)
            
            observation["blocks"].append(elem) # append that to the observation blocks array
    observation["face"] = driver.find_element_by_id(id).CLASS_NAME # append the state of game to the face element
    return observation

class SimplifiedTicTacToe(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=i*j-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,9), dtype=np.int32, minimum=0, maximum=i*j-1, name='observation')
        self._state = initialState
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._state = initialState
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def __is_spot_blank(self, position):
        return self._state[position] == 0
    
    def __all_spots_blank(self):
        return all(i == -1 for i in self._state)
    
    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        
        if self.__is_spot_blank(action):
            self._state[action] = getList["blocks"]


print(getList())
"""def sendRequest(action):
    driver = selenium.webdriver.Firefox()
    driver.get("http://minesweeperonline.com/#intermediate")
    assert "Minesweeper" in driver.title
    def getList():
        mainList = [[], ""]
        for(i_component in range(1, i)):
            for(j_component in range(1, j)):
                id = i_component + "_" + j_component
                block = driver.find_element_by_id(id)
                blockClasses = block.CLASS_NAME.split(" ")
                blockClass = blockClasses[1]
                n = blockClass[4]
                if math.isnan(n):
                    elem = -1
                elif:
                    elem = int(n)
                mainList[0].append(elem)
            mainList[1] = driver.find_element_by_id(id).CLASS_NAME
            return mainList
    return getList()"""
