import collections
import numpy as np
import operator
import itertools

from snakeai.agent import AgentBase
from snakeai.dill._dill import dump
from snakeai.dill._dill import load


class newQAgent(AgentBase):
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


    def make_epsilon_greedy_policy(self, Q, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.
        
        Args:
            Q: A dictionary that maps from state -> action-values.
                Each value is a numpy array of length nA (see below)
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.
        
        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
        
        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            best_action = np.argmax(Q[observation])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn


    
    def train(self, env, num_episodes=1000, discount_factor=1.0, alpha=0.5, epsilon=0.1):
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

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).

        Q = collections.defaultdict(lambda: np.zeros(env.num_actions))

        
        # The policy we're following
        policy = self.make_epsilon_greedy_policy(Q, epsilon, env.num_actions)


        for episode in range(num_episodes):

            # Reset the environment for the new episode.
            timestep = env.new_episode()

            game_over = False

            # Observe the initial state.

            state = tuple(map(tuple, timestep.observation))

            for t in itertools.count():
            
                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                env.choose_action(action)
                timestep = env.timestep()

                next_st = tuple(map(tuple, timestep.observation))

                game_over = timestep.is_episode_end

                next_state, reward, done = next_st, timestep.reward, game_over
                
                # TD Update
                best_next_action = np.argmax(Q[next_state])    
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
                    
                if done:
                    break
                    
                state = next_state


            summary = 'Episode {:5d}/{:5d} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                episode + 1, num_episodes,
                env.stats.fruits_eaten, env.stats.timesteps_survived, env.stats.sum_episode_rewards,
            ))

        print("SAVING...")
        with open('newql.dill', 'wb') as f:
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
        with open('newql.dill', 'rb') as f:
            Q = load(f)

        state = tuple(map(tuple, observation))

        print("***Q-TABLE***")
        print(observation)
        print(Q[state])

        action = np.argmax(Q[state]) 

        #action = max(q_table[state].items(), key=operator.itemgetter(1))[0]

        return action

