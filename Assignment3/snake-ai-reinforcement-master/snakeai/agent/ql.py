import collections
import numpy as np
import operator
import itertools

from snakeai.agent import AgentBase
from snakeai.dill._dill import dump
from snakeai.dill._dill import load


class QAgent(AgentBase):
    """ Represents an intelligent agent for the Snake environment. """


    def __init__(self, env):

        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
        """
        self.env = env
        self.Q = {}


    
    def train(self, env, num_episodes=10000, discount_factor=0.9, alpha=0.5, epsilon=0.1):
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

        exploration_range=(1.0, 0.15)
        exploration_phase_size=0.6

        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        Q = collections.defaultdict(lambda: np.zeros(env.num_actions))


        for episode in range(num_episodes):

            # Reset the environment for the new episode.
            timestep = env.new_episode()

            game_over = False

            # Observe the initial state.
            state = tuple(map(tuple, timestep.observation))

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    action = np.argmax(Q[state])


                env.choose_action(action)
                timestep = env.timestep()

                next_st = tuple(map(tuple, timestep.observation))

                game_over = timestep.is_episode_end

                next_state, reward = next_st, timestep.reward
                
                best_next_action = np.argmax(Q[next_state])    

                Q[state][action] += alpha * ((reward + discount_factor * Q[next_state][best_next_action]) - Q[state][action])
                    
                state = next_state

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        print("SAVING...")
        with open('ql2.dill', 'wb') as f:
            dump(Q, f)

        return Q


    def act(self, observation, reward):
        """
        Choose the next action to take.
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.
        Returns:
            The index of the action to take next.
        """
        #ql:100.000, ql1=50.000, ql2=10.000

        action = np.random.randint(2)

        if(len(self.Q) < 1):
            with open('ql2.dill', 'rb') as f:
                self.Q = load(f)
        else:
            state = tuple(map(tuple, observation))
            action = np.argmax(self.Q[state]) 

        return action

