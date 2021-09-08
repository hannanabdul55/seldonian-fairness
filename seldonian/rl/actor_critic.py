from numpy import dtype
from numpy.core.fromnumeric import size
from numba import int64, float32, float64, int32, types, typed    # import the types
from numba.experimental import jitclass
import typing
from numba import prange, njit
import numba as nb
from time import time

from rl_utils import *
from policy import *
from gridworld_obstacle import *


@jitclass
class ActorCriticGridworld:
    gamma: float64
    alpha_actor: float64
    alpha_critic: float64
    j: float64
    numactions: int64
    numstates: int64
    policy: TabularSoftmaxPolicy
    env: GridWorld
    alpha_mul: float64
    lam: float64
    eps: float64
    theta: float64[:, :]
    v: float64[:]
    e_trace_theta: float64[:, :]
    e_trace_v: float64[:]
    frozen: bool
    td: float64
    gw: GridWorld

    def __init__(
        self, env, policy,
        eps=0.0,
        gamma=1.0,
        alpha_actor=0.137731127022912,
        alpha_critic=0.31442900745165847,
        lam=0.23372572419318238,
        order=1,
        j=-8.0
    ) -> None:
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.env = env
        self.lam = lam
        self.eps = eps
        self.policy = policy
        self.j = j
        self.alpha_mul = 1.0
        self.numactions = env.len_actions
        self.numstates = env.len_states

        self.frozen = False

        self.theta = self.policy.phi
        # self.policy.set_theta(self.theta)

        self.v = np.zeros(self.numstates, dtype=np.float64)

        self.e_trace_theta = np.zeros(
            (self.numactions, self.numstates), dtype=np.float64
        )

        self.e_trace_v = np.zeros(
            self.numstates, dtype=np.float64
        )

    def set_policy(self, theta):
        self.policy.set_theta(theta)

    def td_error(self, s: int64, rw: float64, sPrime: int64):
        if self.env.is_terminated():
            # print("terminated")
            return rw - self.v[s]
        else:
            return rw + (self.gamma * self.v[sPrime]) - self.v[s]
        pass

    def newepisode(self):
        self.e_trace_theta *= 0.0
        self.e_trace_v *= 0.0
        self.env.reset()    

    def act(self):
        state = self.env.state
        action = self.policy.choose_action(state)
        s, a, r, sPrime = self.env.take(action)

        if not self.frozen:
            td = self.td_error(int(s), r, int(sPrime))
            self.update_critic(td, int(s), int(a))
            self.update_actor(td, int(s), int(a))

        return r

    def freeze(self, pol_freeze: bool = True):
        self.frozen = pol_freeze

    def update_actor(self, td: float64, s: int64,  a: int64):
        det_matrix = self.policy.derivative(s, a)
        self.e_trace_theta = self.e_trace_theta*self.gamma * self.lam
        self.e_trace_theta += det_matrix
        # print(det_matrix.shape, self.e_trace_theta.size, td)
        # self.alpha_actor * td * self.e_trace_theta
        # print("Update shape: ",(self.alpha_actor * td * self.e_trace_theta).shape)
        # print("theta shape: ",self.theta.shape)
        self.theta = self.theta + (self.alpha_actor * td * self.e_trace_theta)
        # print("Self theta shape: ", self.theta.shape)
        self.policy.set_theta(self.theta)
        # print("updated actor. TD: ", td, a)
        pass

    def update_critic(self, td: float64, s: int64, a: int64):
        self.e_trace_v *= self.gamma * self.lam
        self.e_trace_v += s_to_onehot(s, self.numstates)
        self.v += self.alpha_critic * td * self.e_trace_v
        pass


if __name__ == "__main__":
    # print(nb.typeof(list()))
    seed = 42
    gw = get_gw_from_seed(seed)
    pol = TabularSoftmaxPolicy(gw, seed=seed)
    reinforce = ActorCriticGridworld(
        gw, pol,
        alpha_actor=0.137731127022912,
        alpha_critic=0.31442900745165847,
        lam=0.23372572419318238,
        seed=seed
    )
    eps = 1024
    for i in range(eps):
        gw.reset()
        reinforce.newepisode()
        while not gw.is_terminated():
            reinforce.act()
    print(pol.phi)
    print(reinforce.v.reshape(5, 5))

    # run evaluation episode
    reinforce.freeze()
    gw.reset()
    while not gw.is_terminated():
        reinforce.act()
    print(f"Final reward: {gw.rw}")
