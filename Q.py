from AgentAction import AgentAction
from Agent import Agent
import GameState as gs
import numpy as np
import random
import json
import pandas as pd


class Q(Agent):
    """description of class"""

    def __init__(self, mission_name, env, num_episodes, gamma=1, alpha=1, max_simulation_time=0):
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes

        self.max_simulation_time = max_simulation_time
        self.episode_data = []

        self.env = env

        return Agent.__init__(self, mission_name, env)

    # Iteration over episode
    def q_prediction(self):

        # run for across defined number of episodes
        for iteration in range(self.num_episodes):
            self.current_episode = iteration
            self.n_actions = 0
            # logarithmic epsilon decrease decay constant: 3
            self.epsilon = -(np.exp(4 * ((iteration / self.num_episodes) - 1))) + 1

            # run for one episode
            self.q_learning_one_episode()

            a = 1

            # SAve to disk

        return

        # Q learning game loop

    def q_learning_one_episode(self):

        done = False
        observation = self.env.reset()
        self.is_first_frame = True
        counter = 0
        while not done:
            if self.is_first_frame:
                _action = 0
                counter += 1
                if counter < 100:
                    self.is_first_frame = False
                    obs, reward, done, info = self.env.step(_action)

                    # MIGHT NOT BE CORRECT - CHECK
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

                # update stats
                _action = self.get_action()

                if self.debug == 1:
                    print(
                        "The Target Block is (x,Z): ({0},{1}) , the current positions is(x,Z): ({2},{3}), previous position(x,Z): {4}, {5}, taking action {6}, The reward {7}".format(
                            self.target_x_block_pos, self.target_z_block_pos, self.gs.current_x_block_pos,
                            self.gs.current_z_block_pos, self.previous_x_block, self.previous_z_block,
                            self.current_action, reward))

                obs, reward, done, info = self.env.step(_action)

                self.done=done

                # update stats
                self.update_stats(info, reward)

                # update the current gamestate
                self.gs.update_current_gamestate(self.xPos, self.yPos, self.zPos)

                # TODO CHECK ENV CLOSE
                if self.is_action_complete() == True and self.is_first_frame != True:
                    self.n_actions += 1
                    # Update Q value
                    self.q_values[self.previous_x_block][self.previous_z_block][self.previous_action] = self.q_values[self.previous_x_block][self.previous_z_block][self.previous_action] + self.alpha * (reward + self.gamma * (self.get_best_next_Q()) - self.q_values[self.previous_x_block][self.previous_z_block][self.previous_action])

                    # TODO add to episode list / return episode
                    self.episode_data.append({
                        "Episode_Number": self.current_episode,
                        "Episode_Q_Values": self.q_values,
                        "Episode_Number_Of_Actions": self.n_actions,
                        "Episode_Action": self.previous_action,
                        "Episode_Reward": reward,
                        "Episode_Previous_X_Block": self.previous_x_block,
                        "Episode_Previous_Z_block": self.previous_z_block,
                        "Episode_Target_X_Block": self.gs.current_x_block_pos,
                        "Episode_Target_Z_Block": self.gs.current_z_block_pos})

                if self.visualise:
                    self.gs.visualize_gamestate(self.q_values, self.current_episode, self.n_actions,
                                                self.current_action, _action, reward, self.target_x_block_pos,
                                                self.target_z_block_pos)

                if done and 1 == 0:
                    all_data = {
                        "Episode_Number": self.num_episodes,
                        "Episode_Gamma": self.gamma,
                        "Episode_Alpha": self.alpha,
                        "Episode_Number_Of_Actions": self.n_actions,
                        "Episode_Time_Taken": 0,
                        "Episode_data": pd.Series(self.episode_data).to_json(orient='values')
                    }
                    with open("Episode_{0}.json".format(self.num_episodes), "w") as write_file:
                        json.dump(all_data, write_file)
                    a = 1

        return

    def get_action(self):

        if self.is_action_complete():

            self.chance = random.random()
            # update the current gamestate
            self.gs.update_current_gamestate(self.xPos, self.yPos, self.zPos)

            if self.chance < self.epsilon:

                # get all valid moves for current gamestate
                self.valid_actions = self.gs.get_valid_actions()

                # set the current action
                action_to_take = self.pick_random_valid_move()
                print("RANDOM ACTION")

            else:
                # greedy
                print("GREEDY ACTION")
                action_to_take = Agent.get_best_action(self)

            # update the target
            self.update_target_pos()
        else:
            print("CONTINUE ACTION")
            action_to_take = self.agentAction.getMove(self.current_action, self.yaw)

        return action_to_take


    def get_best_next_Q(self):

        next_valid_actions = self.gs.get_next_valid_actions()

        next_valid_action_q_values = np.zeros(len(next_valid_actions))

        for i, v in enumerate(next_valid_actions):

            next_valid_action_q_values[i] = self.q_values[self.target_x_block_pos][self.target_z_block_pos][v]

        next_best_Q = max(next_valid_action_q_values)

        return next_best_Q