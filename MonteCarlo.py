from AgentAction import AgentAction
from Agent import Agent
import GameState as gs
import numpy as np
import random
import json
import pandas as pd
from datetime import datetime
import os


class MonteCarlo(Agent):
    """description of class"""

    def __init__(self, mission_name, env, num_episodes, gamma=1, max_simulation_time=0, alpha=.8):
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.alpha = alpha

        # self.n_values =  np.zeros((self.gs.map_width, self.gs.map_length,4))
        self.max_simulation_time = max_simulation_time
        self.episode_data = []

        self.directory = 'Alpha_' + str(self.alpha) + '_Gamma_' + str(self.gamma)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        return Agent.__init__(self, mission_name, env)

    def run_mc_for_one_episode(self):
        done = False
        observation = self.env.reset()
        self.a_counter = 0
        episode = []
        num_iterations = 0
        # TODO INITIAL STATE
        state = [0, 0]
        # game loop
        episode_list = []
        counter = 0
        self.is_first_frame = True

        t1 = datetime.now()

        while not done:
            if self.is_first_frame:
                _action = 0
                counter += 1
                if counter < 100:
                    self.is_first_frame = False
                    obs, reward, done, info = self.env.step(_action)

                    # if self.debug==1:
                    # print("The Target Block is (x,Z): ({0},{1}) , the current positions is(x,Z): ({2},{3}), taking action {4}".format(self.target_x_block_pos,self.target_z_block_pos,self.gs.current_x_block_pos, self.gs.current_z_block_pos ,self.current_action))

                    # update stats
                    self.update_stats(info, reward)

                    # update the current gamestate

                    self.gs.update_current_gamestate(self.xPos, self.yPos, self.zPos)
                    self.target_z_block_pos = int(np.floor(self.zPos))
                    self.target_x_block_pos = int(np.floor(self.xPos))
                    self.gs.current_x_block_pos = self.target_x_block_pos
                    self.gs.current_z_block_pos = self.target_z_block_pos

            else:
                _action = self.get_action()

                if self.debug == 1:
                    print(
                        "The Target Block is (x,Z): ({0},{1}) , the current positions is(x,Z): ({2},{3}), taking action {4}".format(
                            self.target_x_block_pos, self.target_z_block_pos, self.gs.current_x_block_pos,
                            self.gs.current_z_block_pos, _action))

                obs, reward, done, info = self.env.step(_action)

                # update stats
                self.update_stats(info, reward)

                # update the current gamestate
                self.rgs = self.gs.update_current_gamestate(self.xPos, self.yPos, self.zPos)

                if self.is_action_complete() == True and self.is_first_frame != True:
                    # Increase action completed count
                    self.n_actions += 1

                    # episode_list.append([ [self.previous_x_block,self.previous_z_block], self.previous_action, reward])

                    # store next state
                    self.episode_data.append(
                        [self.current_episode, self.n_actions, self.previous_action, reward, self.previous_x_block,
                         self.previous_z_block, self.gs.current_x_block_pos, self.gs.current_z_block_pos,
                         self.num_episodes, self.gamma, self.alpha, self.n_actions, 0])

                if self.visualise:
                    self.gs.visualize_gamestate(self.q_values, self.current_episode, self.n_actions,
                                                self.current_action, _action, reward, self.target_x_block_pos,
                                                self.target_z_block_pos)

                # Save full episode data to disk
                # TODO Abstract method to agent
                if done:
                    t2 = datetime.now()
                    delta = t2 - t1

                    episode_data = pd.DataFrame(data=self.episode_data,
                                                columns=['Episode_Number', 'Episode_Number_Of_Actions',
                                                         'Episode_Action', 'Episode_Reward', 'Episode_Previous_X_Block'
                                                    , 'Episode_Previous_Z_block', 'Episode_Target_X_Block',
                                                         'Episode_Target_Z_Block', 'Episode_Number', 'Episode_Gamma',
                                                         'Episode_Alpha', 'Episode_Number_Of_Actions',
                                                         'Episode_Time_Taken'])

                    episode_data['Episode_Time_Taken'] = delta.total_seconds()
                    episode_data.to_csv(self.directory + '/MC_Episode_' + str(self.current_episode) + '_Gamma_' + str(
                        self.gamma) + '_Alpha_' + str(self.alpha) + '_Epsilon_' + str(round(self.epsilon, 2)) + '.csv')

        # set previous q values
        self.previous_q_values = self.q_values

        return episode_data

    def update_q_values(self, episode):

        # prepare for discounting
        discounts = np.array([self.gamma ** i for i in range(len(episode) + 1)])

        for i, row in episode.iterrows():
            z = int(row['Episode_Previous_Z_block'])
            x = int(row['Episode_Previous_X_Block'])
            a = int(row['Episode_Action'])
            previous_q = self.q_values[x][z][a]
            self.q_values[x][z][a] = round(previous_q + row['Episode_Alpha'] * (
                        sum(episode['Episode_Reward'][i:] * discounts[:-(1 + i)]) - previous_q), 2)
            a = 1

        q_data_list = []

        # save q _values
        for x in range(len(self.q_values)):
            for z in range(len(self.q_values[0])):
                for a in range(len(self.q_values[0][0])):
                    q_data_list.append([x, z, a, self.q_values[x][z][a], self.epsilon])

        # create dataframe and save to disk
        episode_q_values = pd.DataFrame(data=q_data_list,
                                        columns=['X_Block_Position', 'Z_Block_Position', 'Action', 'Value', 'Epsilon'])
        episode_q_values.to_csv(
            self.directory + '/MC_Episode_' + str(self.current_episode) + '_Gamma_' + str(self.gamma) + '_Alpha_' + str(
                self.alpha) + '_Epsilon' + str(round(self.epsilon, 2)) + '_Q_VALUES.csv')
        return

    def mc_prediction(self, filename, iteration_number=0):

        if filename != '':
            self.load_in_q_values(filename)

        # run MC for a defined number of episodes
        for i in range(iteration_number, self.num_episodes):

            self.n_actions = 0
            self.current_episode = i

            if i % 20 == 0:
                print("Episode: " + str(i) + " ")
                sys.stdout.flush()
            # logarithmic epsilon decrease decay constant: 8
            self.epsilon = -(np.exp(4 * ((i / self.num_episodes) - 1))) + 1

            # run mc for one episode
            episode_data = self.run_mc_for_one_episode()

            # update q values
            self.update_q_values(episode_data)

        return

    def get_action_probabilities(self, n_actions):

        # defines the action policy for each valid action from that state
        # TODO check the action policy
        action_policy = np.ones(n_actions) * (self.epsilon / n_actions)
        best_action = np.argmax(self.q_values[self.gs.current_x_block_pos][self.gs.current_z_block_pos][:])
        action_policy[best_action] = 1 - self.epsilon + (self.epsilon / n_actions)
        place_holder = 1

    def get_action(self):

        if self.is_action_complete():
            self.a_counter = 0
            chance = random.random()

            if chance < self.epsilon:

                # get all valid moves for current gamestate
                self.valid_actions = self.gs.get_valid_actions()

                n_actions = len(self.valid_actions)

                # set the current action
                action_to_take = self.pick_random_valid_move()


            else:
                # greedy
                action_to_take = self.get_best_action()

            # update the target
            self.update_target_pos()

        else:
            self.a_counter += 1

            # handle server lag issues with a iteration restart
            if self.a_counter < 3:
                action_to_take = self.agentAction.getMove(self.current_action, self.yaw)
            else:
                self.reset_iteration()
        return int(action_to_take)

    pass