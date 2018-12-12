import collections
import numpy as np

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay


class QAgent(AgentBase):
    """ Represents an intelligent agent for the Snake environment. """
    
    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def act(self, observation, reward):
        """
        Choose the next action to take.
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.
        Returns:
            The index of the action to take next.
        """
        print(observation, reward)
        return None

    def end_episode(self):
        """ Notify the agent that the episode has ended. """
        pass
    
    def train(self, env, num_episodes=1000, discount_factor=0.9, alpha=0.05, k=10):
        """
        Train the agent to perform well in the given Snake environment using Q learning.
        
        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            alpha (float): 
                moving average parameter.
            k (int): 
                exploration function parameter.
        """
        pass