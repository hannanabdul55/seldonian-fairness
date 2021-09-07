from numba.core.types.containers import DictType
from numba.np.ufunc import parallel
import numpy as np
from numpy.core.fromnumeric import size
from numba import int64, float32, float64, int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit

from time import time


MAX_OBS = 14

path = np.array([
    0, 1,
    6, 7,
    12, 13,
    18, 19,
    23, 24
], dtype=np.int64)


@jitclass
class GridWorld:
    m: int64
    n: int64
    states: int64[:]
    start: int64
    goal: int64
    actions: typing.Dict[str, int64]
    state: int64
    obstacle: int64[:]
    obstacles: int64[:]
    rw: float32
    rand_action_prob: float32
    len_actions: int64
    water: int64[:]
    path: int64[:]
    obs_as_cliff: bool
    len_states: int64
    t: int64
    qsa: int64[:,:]
    MAX_T: int64
    deterministic: bool

    def __init__(
        self, size=(5, 5),
        obstacle=np.array([-1], dtype=np.int64),
        start_state=0,
        rand_action_prob=np.random.rand(),
        deterministic=True,
        water=np.array([-1], dtype=np.int64),
        obs_as_cliff=True
    ) -> None:
        self.m, self.n = size
        self.MAX_T = 199
        self.obstacles = obstacle
        self.states = np.zeros(self.m*self.n, dtype=np.integer)
        self.start = start_state
        self.goal = (self.m*self.n)-1
        if not deterministic:
            self.rand_action_prob = rand_action_prob
        else:
            self.rand_action_prob = 0.0
        self.deterministic = deterministic
        self.water = water
        self.obs_as_cliff = obs_as_cliff
        self.actions = {
            'up': 0,
            'down': 1,
            'right': 2,
            'left': 3
        }

        self.len_actions = 4
        self.len_states = self.states.size
        # print(self.len_states, self.len_actions)
        self.state = self.start
        self.rw = 0.0
        self.t=0
        pass

    def reset(self):
        self.state = self.start
        self.rw = 0.0
        self.t = 0

    # def get_states(self, flat=True, one_hot=True):
    #     if one_hot:
    #         states = np.zeros((size, size), dtype=int)
    #         states[self.state] = 1
    #         return states.flatten() if flat else states
    #     else:
    #         return self.state

    def reward(self):
        if self.state in self.obstacles:
            return -20
        return -1

    def take(self, action: int):
        m, n = self.m, self.n
        s = self.state
        i = int(s/n)
        j = int(s % n)

        if (not self.deterministic) and np.random.rand() < self.rand_action_prob:
            print("took random action")
            action = np.random.choice(
                np.arange(self.len_actions, dtype=np.int32))
            # print("Random action: ", action)
        newstate = -2
        if action == 0:
            # up
            if self.state - n >= 0:
                newstate = self.state - n
        elif action == 1:
            # down
            if self.state + n < m*n:
                newstate = self.state + n
        elif action == 2:
            # right
            if self.state + 1 < m*n and j < n-1:
                newstate = self.state + 1
        elif action == 3:
            # left
            if self.state-1 >= 0 and j > 0:
                newstate = self.state - 1
        else:
            print("No action taken. THIS SHOULD NOT HAPPEN!! Action: ", action)

        rw = -1.0
        if newstate == -2:
            newstate = self.state
        if newstate in self.obstacles:
            if self.obs_as_cliff:
                # print("hit a cliff")
                newstate = self.start
            else:
                newstate = self.state
        if newstate in self.water:
            rw = -20.0

        if newstate == self.goal:
            rw = 0.0

        self.state = newstate
        # print(s, action, rw, newstate)
        self.rw+= rw
        self.t+=1
        return [s, action, rw, newstate]
        # return rw

    def is_sinf(self):
        return self.state == self.goal
    
    def is_terminated(self):
        return self.is_sinf() or self.t >=self.MAX_T

    def visualize(self):
        arr = np.zeros((self.m, self.n), dtype=np.integer)

        for w in self.water:
            if w != -1:
                i = int(w/self.n)
                j = int(w % self.n)
                arr[i, j] = 2

        for w in self.obstacles:
            if w != -1:
                i = int(w/self.n)
                j = int(w % self.n)
                arr[i, j] = 4

        i = int(self.state/self.n)
        j = int(self.state % self.n)
        print(i, j)
        arr[i, j] = 1
        print(arr)


def nparray(arr, type=np.float32):
    return np.array(arr, type)


def get_gw_from_seed(seed, path=path):
    rng = np.random.default_rng(seed)
    n_obs = rng.integers(MAX_OBS)
    pos_obs = list(filter(lambda x: x not in path, range(24)))
    obstacles = rng.choice(pos_obs, n_obs, replace=False)
    gw = GridWorld(
        obstacle=obstacles,
        rand_action_prob=rng.random()
    )
    return gw


@njit
def get_episodes_from_env(gw, eps=1000, seed=123, max_t=199):
    actions = np.array([0, 1, 2, 3], dtype=np.integer)
    epss = []
    np.random.seed(seed)
    for ep in np.arange(eps):
        t = 0
        gw.reset()
        ep = []
        while t < max_t and not gw.is_terminated():
            a = np.random.choice(actions)
            rt = gw.take(a)
            # ep.append([a, rt])
            ep.append(rt)
            t += 1
        # if t<MAX_T and gw.is_sinf():
        #     print("Reached goal state in eps: ", t)
        epss.append(ep)
    return epss


if __name__ == "__main__":
    seed = 42
    e = 1024
    gw = get_gw_from_seed(seed, path=path)
    rng = np.random.default_rng(seed)
    t = time()
    episodes = get_episodes_from_env(gw, eps=e, seed=seed)
    tot = time() - t
    print(f"Ran {e} episodes in {tot} seconds")
    t = time()
    episodes = get_episodes_from_env(gw, eps=e, seed=seed)
    tot = time() - t
    print(f"Ran {e} episodes in {tot} seconds")


if __name__ == "__main__1":
    gw = GridWorld(
        obstacle=np.array([6, 12, 13, 19, 22], dtype=np.integer),
        water=np.array([3, 4, 6], dtype=np.integer),
        rand_action_prob=0.01
    )
    print(f"Start state: {gw.state}")
    actions = gw.actions
    gw.take(actions['down'])
    print(gw.visualize())
    gw.take(actions['right'])
    print(gw.visualize())
    gw.take(actions['up'])
    print(gw.visualize())
    gw.take(actions['right'])
    print(gw.visualize())
    gw.take(actions['right'])
    print(gw.visualize())
    print(f"End state: {gw.state}")
    print(f"Final reward: {gw.rw}")
