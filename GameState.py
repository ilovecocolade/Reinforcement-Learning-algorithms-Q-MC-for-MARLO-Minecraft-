import math as m
from AgentAction import AgentAction  as agentAction
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import copy

class GameState(object):
    """description of class"""

    min_x_pos = 0
    max_x_pos = 0
    min_y_pos = 0
    max_y_pos = 0
    min_z_pos = 0
    max_z_pos = 0

    def __init__(self, map_width, map_height, map_length, playerXPos, playerYPos, playerZPos, mission_available_moves):

        self.map_width = map_width
        self.map_height = map_height
        self.map_length = map_length

        # min and max positions
        self.min_x_pos = 0  # m.floor(playerXPos);
        self.max_x_pos = self.map_width - 1;
        self.min_y_pos = 55  # m.floor(playerYPos) ;
        self.max_y_pos = 62  # m.floor(playerYPos) + self.map_height-1;
        self.min_z_pos = 0  # m.floor(playerZPos);
        self.max_z_pos = self.map_length - 1;

        # current player positions
        self.playerXPos = playerXPos
        self.playerYPos = playerYPos
        self.playerZPos = playerZPos

        # Discretization
        self.current_x_block_pos = m.floor(playerXPos);
        self.current_y_block_pos = m.floor(playerYPos);
        self.current_z_block_pos = m.floor(playerZPos);
        self.visualise = True
        if self.visualise:
            self.gamestate_fig = plt.figure(figsize=[self.map_width, self.map_length], facecolor=(0, 0, 0))
            ax = self.gamestate_fig.add_subplot(111, xticks=range(self.map_width), yticks=range(self.map_length + 1),
                                                position=[.1, .1, .8, .8])
            ax.grid(color='k', linestyle='-', linewidth=.5)
            ax.xaxis.set_tick_params(bottom='off', top='off', labelbottom='off')
            ax.yaxis.set_tick_params(left='off', right='off', labelleft='off')

            agent = mpatches.Circle((0, 0), .13, facecolor='k', edgecolor=(.8, .8, .8, 1), linewidth=1, clip_on=False,
                                    zorder=10)

            s1 = copy.copy(agent)
            s1.center = (self.current_x_block_pos + .5, self.current_z_block_pos + .5)
            ax.add_patch(s1)
            plt.ion()
            plt.show()

        return

    def get_valid_actions(self):

        for i in range(self.map_width):
            for j in range(self.map_length):
                if (self.current_x_block_pos == i) and (self.current_z_block_pos == j):
                    if (i == self.min_x_pos) and (j == self.min_z_pos):
                        actions = [0, 2]
                    elif (i == self.max_x_pos) and (j == self.min_z_pos):
                        actions = [0, 3]
                    elif (i == self.max_x_pos) and (j == self.max_z_pos):
                        actions = [1, 3]
                    elif (i == self.min_x_pos) and (j == self.max_z_pos):
                        actions = [1, 2]
                    elif i == self.min_x_pos:
                        actions = [0, 2, 1]

                    elif i == self.max_x_pos:
                        actions = [0, 3, 1]
                    elif j == self.min_z_pos:
                        actions = [0, 2, 3]
                    elif j == self.max_z_pos:
                        actions = [1, 2, 3]
                    else:
                        actions = [0, 1, 2, 3]
        return actions

    def get_next_valid_actions(self):

        for i in range(self.map_width):
            for j in range(self.map_length):
                if (self.current_x_block_pos == i) and (self.current_z_block_pos == j):
                    if (i == self.min_x_pos) and (j == self.min_z_pos):
                        actions = [0, 2]
                    elif (i == self.max_x_pos) and (j == self.min_z_pos):
                        actions = [0, 3]
                    elif (i == self.max_x_pos) and (j == self.max_z_pos):
                        actions = [1, 3]
                    elif (i == self.min_x_pos) and (j == self.max_z_pos):
                        actions = [1, 2]
                    elif i == self.min_x_pos:
                        actions = [0, 2, 1]

                    elif i == self.max_x_pos:
                        actions = [0, 3, 1]
                    elif j == self.min_z_pos:
                        actions = [0, 2, 3]
                    elif j == self.max_z_pos:
                        actions = [1, 2, 3]
                    else:
                        actions = [0, 1, 2, 3]
        return actions

    def update_current_gamestate(self, playerXPos, playerYPos, playerZPos):

        # current player positions
        self.playerXPos = playerXPos
        self.playerYPos = playerYPos
        self.playerZPos = playerZPos

        # Discretization
        self.current_x_block_pos = m.floor(playerXPos)
        self.current_y_block_pos = m.floor(playerYPos)
        self.current_z_block_pos = m.floor(playerZPos)

        return

    def visualize_gamestate(self, q_values, episode_number, n_actions, current_action, current_move, reward, targetXPos,targetZPos):

        self.gamestate_fig.clear()
        ax = self.gamestate_fig.add_subplot(111, xticks=range(self.map_width + 1), yticks=range(self.map_length + 1), position=[.1, .1, .8, .8])
        ax.grid(color='k', linestyle='-', linewidth=.5)
        # ax.xaxis.set_tick_params(bottom='off', top='off', labelbottom='off')
        # ax.yaxis.set_tick_params(left='off', right='off', labelleft='off')

        agent = mpatches.Circle((0, 0), .13, facecolor='k', edgecolor=(.8, .8, .8, 1), linewidth=1, clip_on=False, zorder=10)

        s1 = copy.copy(agent)
        s1.center = ((self.map_width - 1) - self.current_x_block_pos + .5, self.current_z_block_pos + .5)
        ax.add_patch(s1)

        if current_move == 1:
            move = 'Move Forward: 1'
        elif current_move == 2:
            move = 'Move Backwards: 2'
        elif current_move == 3:
            move = 'Move Right: 3'
        elif current_move == 4:
            move = 'Move Left: 4'
        else:
            move = 'Stand Still: 0'

        if current_action == 0:
            action_move = 'Move Forward: 0'
        elif current_action == 1:
            action_move = 'Move Backwards: 1'
        elif current_action == 2:
            action_move = 'Turn Right: 2'
        elif current_action == 3:
            action_move = 'Turn Left: 3'

        for x in range(self.map_width):
            for z in range(self.map_length):
                ax.text(x + 0.5, z + 0.95, round(q_values[(self.map_width - 1) - x][z][0], 2), fontsize=8,
                        horizontalalignment='center', verticalalignment='center')
                ax.text(x + 0.5, z + 0.05, round(q_values[(self.map_width - 1) - x][z][1], 2), fontsize=8,
                        horizontalalignment='center', verticalalignment='center')
                ax.text(x + 0.05, z + 0.5, round(q_values[(self.map_width - 1) - x][z][2], 2), fontsize=8,
                        horizontalalignment='left', verticalalignment='center')
                ax.text(x + 0.95, z + 0.5, round(q_values[(self.map_width - 1) - x][z][3], 2), fontsize=8,
                        horizontalalignment='right', verticalalignment='center')
        plt.title(
            'Episode:{0}, # Episode Actions: {1},Current Move: {2}, Current Action: {8}, Last Reward:{3}, \n Position(x,z): ({4},{5}), Target Position (x,z):({6},{7})'.format(episode_number + 1, n_actions, move, reward, self.current_x_block_pos, self.current_z_block_pos,targetXPos, targetZPos, current_action), color='white', fontsize=8)
        self.gamestate_fig.canvas.draw()
        self.gamestate_fig.canvas.flush_events()

        # plt.flush_events()
        return





