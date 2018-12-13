import collections
import numpy as np
import operator

from snakeai.agent import AgentBase


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
        self.q_table = {}
        self.reward = 0
        self.alpha = 0.3
        self.gamma = 0.1
        self.epsilon = 0.1

    
    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.reward = 0
        self.state = None

    def get_states(self, state):

        fruit = np.where(state == 1)
        head = np.where(state == 2)
        body = np.where(state == 3)

        fruit_pos = tuple([fruit[0][0], fruit[1][0]])
        head_pos = tuple([head[0][0], head[1][0]])

        body_pos = []
        for i in range(len(body[0])):
            body_pos.append((body[0][i], body[1][i]))
        
        tail_index = -99
        for i in range(len(body_pos)):
            dist = abs(head_pos[0] - body_pos[i][0]) + abs(head_pos[1] - body_pos[i][1])
            if(dist > tail_index):
                tail_index = dist
                head_tail = tuple([fruit_pos[1]-body_pos[i][1], fruit_pos[0]-body_pos[i][0]])


        head_fruit = tuple([fruit_pos[1]-head_pos[1], fruit_pos[0]-head_pos[0]])

        #state = (head_fruit + head_tail)
        state = head_fruit

        self.state = state

        return self.state


    
    def train(self, env, num_episodes=100000, discount_factor=0.7, alpha=0.6, k=10):
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

            max_q = 0

            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False

            # Observe the initial state.
            obs = timestep.observation

            #self.state = tuple(state.flatten())
            state = self.get_states(obs)

            if not state in self.q_table:
                self.q_table[state] = {ac:0 for ac in range(self.env.num_actions)}
                #print(self.q_table)


            while not game_over:

                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    
                    action = np.random.randint(self.env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    action = max(self.q_table[state].items(), key=operator.itemgetter(1))[0]
                    
                    #print("Q-TAAABLE")
                    #print(self.q_table[self.state])
                    #print(timestep.observation)
                    #action = np.argmax(q[0])

                #print("ACTION: ", action)

                # Act on the environment.
                self.env.choose_action(action)
                timestep = self.env.timestep()

                reward = timestep.reward


                #print("REWARD: ", reward)

                next_obs = timestep.observation
                #self.next_state = tuple(next_state.flatten())

                next_state = self.get_states(next_obs)

                game_over = timestep.is_episode_end

                #check if next_state has q_values already
                if next_state not in self.q_table:
                    self.q_table[next_state] = {ac:0 for ac in range(self.env.num_actions)}

                # Learn policy based on state, action, reward
                old_q_value = self.q_table[state][action]

                #maximum q_value for next_state actions
                next_max = max(self.q_table[next_state].values())

                # calculate the q_value for the next_max action.
                new_q_value = (1 - alpha)*old_q_value + alpha*(env.stats.sum_episode_rewards + gamma*next_max)

                self.q_table[state][action] = new_q_value

                #print("QQQQ-TTTAABBLLEEEE: ")
                #print(self.q_table[self.state])
                #print(action)
                #print(next_state)
                
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

        np.save('ql-5.npy', self.q_table)

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
        #state = tuple(observation.flatten())

        #self.state = tuple(state.flatten())
        state = self.get_states(observation)

        #ql-1: (fruit_head, fruit_tail) train: 5.000.000
        #ql-2: (fruit_head)             train: 1.000.000
        #ql-3: (fruit_head)             train: 100.000
        #ql-4: (fruit_head)             train: 1.000.000


        q_table = np.load('ql-5.npy').item()

        #print("QQQQ-TTTAABBLLEEEE: ")
        print(state)
        print(observation)
        print(q_table[state])

        action = max(q_table[state].items(), key=operator.itemgetter(1))[0]

        #print(reward, action)
        return action



    def end_episode(self):
        """ Notify the agent that the episode has ended. """
        pass