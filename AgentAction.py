import numpy as np


class AgentAction(object):

    def __init__(self):

        return

        # Moves player forwards one block

    def plus_z(self):

        # if facing forward, step forward
        if self.yaw == 0:
            return 1

        # if facing back, step backward
        elif self.yaw == 180:
            return 2

        # if facing right, turn left then move forward
        elif self.yaw == 90:
            return 4

        # if facing left, turn right then move forward
        elif self.yaw == 270:
            return 3

        return -1

    # Moves player to the backwards one block
    def minus_z(self):

        # if facing forward, step backward
        if self.yaw == 0:
            return 2

        # if facing back, step forward
        elif self.yaw == 180:
            return 1

        # if facing right, turn right then move forward
        elif self.yaw == 90:
            return 3

        # if facing left, turn left then move forward
        elif self.yaw == 270:
            return 4

        return -1

    # Moves player to the left one block
    def plus_x(self):

        # if facing left, step forward
        if self.yaw == 270:
            return 1

        # if facing right, step backward
        elif self.yaw == 90:
            return 2

        # if facing forward, turn left then move forward
        elif self.yaw == 0:
            return 4

        # if facing backward, turn right then move forward
        elif self.yaw == 180:
            return 3

        return -1

    # Moves player to the right one block
    def minus_x(self):

        # if facing left, step backward
        if self.yaw == 270:
            return 2

        # if facing right, step forward
        elif self.yaw == 90:
            return 1

        # if facing forward, turn right then move forward
        elif self.yaw == 0:
            return 3

        # if facing backward, turn left then move forward
        elif self.yaw == 180:
            return 4

        return -1

    def getMove(self, move, yaw):
        self.yaw = yaw

        if move == 0:
            return self.plus_z()
        elif move == 1:
            return self.minus_z()
        elif move == 2:
            return self.plus_x()
        elif move == 3:
            return self.minus_x()

        return 0

    pass




