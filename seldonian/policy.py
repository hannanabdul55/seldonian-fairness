from sklearn.model_selection import train_test_split

from seldonian.algorithm import *
from seldonian.cmaes import *


class SeldonianCEMPolicyCMAES(SeldonianAlgorithm, CMAESModel):

    def __init__(self, data, states, actions, gamma):
        self.theta = np.random.rand(states, actions)
        self.gamma = gamma
        self.D = data
        self.D_c, self.D_s = train_test_split(data, test_size=0.2)
        # init the CMA-ES base explicitly; theta is set above, so pass it through instead
        # of letting CMAESModel re-initialize it from X's shape
        CMAESModel.__init__(self, self.D_c, None, theta=self.theta)
