from AgentMovement import Agent_Movement as agentMovement
from AgentAction import AgentAction  as agentAction
from bs4 import BeautifulSoup
from AgentAction import AgentAction
import numpy as np
from GameState import GameState
from random import randint
import pandas as pd


class Agent(object):
    '''
    This class serve a the agent to play the game.
    It will instancitate the player movement and actions and then return these.

    '''

    damageDealt = 0
    DamageTaken = 0
    distanceTravelled = 0
    food = 0
    isAlive = True
    life = 20
    mobsKilled = 0
    name = 'OLLIE JOHN BOT _ PLACEHODLER'
    pitch = 0
    score = 0.0
    timeAlive = 0
    totalTime = 0
    worldTime = 0
    xp = 0
    xPos = 0.0  # set to initial mission xpos
    yPos = 0.0  # set to initial mission ypos
    yaw = 0
    zPos = 0  # set to initial mission ypos

    is_in_center = False;

    def __init__(self, mission_name, env):
        self.mission_name = mission_name
        self.env = env

        mission_available_moves = env.params['comp_all_commands']
        mission_xml = BeautifulSoup(env.params['mission_xml'], features="xml")
        placement = mission_xml.find('Placement')
        self.xPos = float(placement.attrs['x'])
        self.yPos = float(placement.attrs['y'])
        self.zPos = float(placement.attrs['z'])
        discretization_mode = 1
        self.current_action = 0
        self.n_actions = 0

        self.is_first_frame = True
        self.completed_action = False
        self.visualise = True

        self.algorithm_name = 'TEST'
        self.debug = 1

        # Instantiate Player Movements
        self.agentMovement = agentMovement(discretization_mode)

        # Instantiate Game state
        # instantiate gamestate
        map_spec = mission_xml.find('specification')
        map_dimension = [int(map_spec.contents[1].text), int(map_spec.contents[2].text), int(map_spec.contents[3].text)]
        self.gs = GameState(map_dimension[0], map_dimension[1], map_dimension[2], self.xPos,
                            self.yPos, self.zPos, mission_available_moves)

        self.agentAction = AgentAction()
        self.q_values = np.zeros((self.gs.map_width, self.gs.map_length, 4))
        return

        # updates target position

    def update_target_pos(self):
        self.get_target_x()
        self.get_target_z()

    def get_target_x(self):

        # print("current x ",self.gs.current_x_block_pos," target x ",self.target_x_block_pos )
        self.previous_x_block = self.gs.current_x_block_pos
        if self.current_action == 0:
            self.target_x_block_pos = self.gs.current_x_block_pos
        elif self.current_action == 1:
            self.target_x_block_pos = self.gs.current_x_block_pos
        elif self.current_action == 2:
            self.target_x_block_pos = self.gs.current_x_block_pos + 1
        elif self.current_action == 3:
            self.target_x_block_pos = self.gs.current_x_block_pos - 1

        if (self.target_x_block_pos < 0) or (self.target_x_block_pos > 6):
            a = 1
        return self.target_x_block_pos

    def get_best_action(self):

        valid_actions = self.gs.get_valid_actions()

        best = 0
        best_index = 0
        for i, a in enumerate(valid_actions):
            if best > self.q_values[self.gs.current_x_block_pos][self.gs.current_z_block_pos][a]:
                best = self.q_values[self.gs.current_x_block_pos][self.gs.current_z_block_pos][a]
                best_index = i

        best_action = valid_actions[best_index]

        self.current_action = best_action
        return self.agentAction.getMove(best_action, self.yaw)  # best_action

    def reset_iteration(self):
        self.n_action = 0
        self.q_values = self.previous_q_values
        self.completed_action = False
        self.run_mc_for_one_episode()

    def load_in_q_values(self, filename):

        q_values_df = pd.read_csv(filename)

        for i, row in q_values_df.iterrows():
            z = row['Z_Block_Position']
            x = row['X_Block_Position']
            a = row['Action']
            self.q_values[x][z][a] = row['Value']
            self.epsilon = row['Epsilon']

        for x in range(len(self.q_values)):
            for z in range(len(self.q_values[0])):
                for a in range(len(self.q_values[0][0])):
                    self.q_values[x][z][a]

    def get_target_z(self):

        # print("current z ",self.gs.current_z_block_pos," target z ",self.target_z_block_pos )
        self.previous_z_block = self.gs.current_z_block_pos
        if self.current_action == 0:
            self.target_z_block_pos = self.gs.current_z_block_pos + 1
        elif self.current_action == 1:
            self.target_z_block_pos = self.gs.current_z_block_pos - 1
        elif self.current_action == 2:
            self.target_z_block_pos = self.gs.current_z_block_pos
        elif self.current_action == 3:
            self.target_z_block_pos = self.gs.current_z_block_pos

        if (self.target_z_block_pos < 0) or (self.target_z_block_pos > 6):
            a = 1
        return self.target_z_block_pos

    def pick_random_valid_move(self):

        action_size = len(self.valid_actions) - 1
        picked_action = self.valid_actions[randint(0, action_size)]
        self.current_action = picked_action
        return self.agentAction.getMove(picked_action, self.yaw)

    def is_action_complete(self):

        if (((self.gs.current_z_block_pos == self.target_z_block_pos) and (
                self.gs.current_x_block_pos == self.target_x_block_pos))) or self.is_first_frame == True:

            if self.debug == 1:
                print("REACHED TARGET BLOCK (x,z): ({0},{1}) , bot position (x,z): ({2},{3}), Taking action {4}".format(
                    self.target_x_block_pos, self.target_z_block_pos, self.gs.current_x_block_pos,
                    self.gs.current_z_block_pos, self.current_action))
            self.previous_action = self.current_action

            return True
        else:
            return False

    def get_action(self):

        # update the current gamestate
        self.rgs = self.gs.update_current_gamestate(self.xPos, self.yPos, self.zPos)

        # get all valid moves for current gamestate
        self.valid_actions = self.gs.get_valid_actions()

        # set the current action
        _action = self.pick_random_valid_move()

        return _action

    def update_q_values():
        # TODO update state action values

        return

    def update_stats(self, info, reward):

        # update state action values
        self.reward = reward

        self.observation = info['observation']
        self.damageDealt = self.observation['DamageDealt']
        self.DamageTaken = self.observation['DamageTaken']
        self.distanceTravelled = self.observation['DistanceTravelled']
        self.food = self.observation['Food']
        self.isAlive = self.observation['IsAlive']
        self.life = self.observation['Life']
        self.mobsKilled = self.observation['MobsKilled']

        self.pitch = self.observation['Pitch']
        self.score = self.observation['Score']
        self.timeAlive = self.observation['TimeAlive']
        self.totalTime = self.observation['TotalTime']
        self.worldTime = self.observation['WorldTime']
        self.xp = self.observation['XP']
        self.xPos = self.observation['XPos']  # set to initial mission xpos
        self.yPos = self.observation['YPos']  # set to initial mission ypos
        self.yaw = self.observation['Yaw']
        self.zPos = self.observation['ZPos']  # set to initial mission ypos

        return

    def is_action_complete(self):

        if (((self.gs.current_z_block_pos == self.target_z_block_pos) and (
                self.gs.current_x_block_pos == self.target_x_block_pos))) or (
                self.is_first_frame == True) or self.done == True:

            if self.debug == 1:
                print("REACHED TARGET BLOCK (x,z): ({0},{1}) , bot position (x,z): ({2},{3}), Taking action {4}".format(
                    self.target_x_block_pos, self.target_z_block_pos, self.gs.current_x_block_pos,
                    self.gs.current_z_block_pos, self.current_action))
            self.previous_action = self.current_action

            return True
        else:
            return False


pass




