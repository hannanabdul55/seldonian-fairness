import numpy as np
from numpy.core.fromnumeric import size


def create_griworld(grid=5, obstacles=None, flat=True):

    pass


class GridWorld:
    def __init__(self, size=(5,5), obstacles=None, start_state=0) -> None:
        self.m, self.n = size
        self.obs = obstacles
        self.states = np.zeros((size, size), dtype=np.uint32)
        self.start = start_state
        self.goal = (size*size)-1
        self.actions = {
            'up':0,
            'down':1,
            'right':2,
            'left':3
        }
        self.reset()
        pass

    def reset(self):
        self.state = self.start
    
    def get_states(self, flat=True, one_hot=True):
        if one_hot:
            states = np.zeros((size, size), dtype=int)
            states[self.state] = 1
            return states.flatten() if flat else states
        else:
            return self.state

    def actions(self):
        return self.actions

    def take(self, action):
        m,n = self.m, self.n
        s = self.state
        md, nd = (int(s/n), int(s%n))
        if action == 0 or action=='up':
            # up
            if self.state - n >= 0:
                self.state = self.state - n
        elif action == 'down' or action==1:
            if self.state +n <m*n:
                self.state = self.state + n
        elif action == 'right' or action==2:
            if self.state + 1 <n:
                self.state = self.state +1
        elif action=='left' or action==3:
            if self.state-1 >=0:
                self.state = self.state -1
        pass