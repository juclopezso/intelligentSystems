import collections
import numpy as np
import operator

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay


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
        self.q_table = {}
        self.reward = 0

    
    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.reward = 0
        reward = 0


    
    def train(self, env, num_episodes=100000, discount_factor=0.9, alpha=0.05, k=10):
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
        exploration_range=(1.0, 0.1)
        exploration_phase_size=0.5
        gamma = discount_factor

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        exploration_decay = ((max_exploration_rate - min_exploration_rate) / (num_episodes * exploration_phase_size))
        exploration_rate = max_exploration_rate

        for episode in range(num_episodes):
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False

            # Observe the initial state.
            state = timestep.observation

            state = tuple(state.flatten())

            max_q = 0

            if not state in self.q_table:
                self.q_table[state] = {ac:0 for ac in range(env.num_actions)}
                #print(self.q_table)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    action = max(self.q_table[state].items(), key=operator.itemgetter(1))[0]
                    #action = np.argmax(q[0])


                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                reward = timestep.reward

                next_state = timestep.observation
                next_state = tuple(next_state.flatten())

                game_over = timestep.is_episode_end

                #check if next_state has q_values already
                if next_state not in self.q_table:
                    self.q_table[next_state] = {ac:0 for ac in range(env.num_actions)}

                # Learn policy based on state, action, reward
                old_q_value = self.q_table[state][action]

                #maximum q_value for next_state actions
                next_max = max(self.q_table[next_state].values())

                # calculate the q_value for the next_max action.
                new_q_value = (1 - alpha)*old_q_value + alpha*(reward + gamma*next_max)
                self.q_table[state][action] = new_q_value
                
                #print (state, action, reward)
                #print("REWARD: ", reward)


            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay

            summary = 'Episode {:5d}/{:5d} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes, exploration_rate,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        np.save('ql.npy', self.q_table)

        return self.q_table



    def act(self, observation, reward):
        """
        Choose the next action to take.
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.
        Returns:
            The index of the action to take next.
        """
        state = tuple(observation.flatten())

        q_table = np.load('ql.npy').item()

        #print("QQQQ-TTTAABBLLEEEE: ")
        #print(q_table[state])

        action = max(q_table[state].items(), key=operator.itemgetter(1))[0]
        #print(reward, action)
        return action



    def end_episode(self):
        """ Notify the agent that the episode has ended. """
        pass