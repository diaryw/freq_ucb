import numpy as np
import math
import random
from abc import ABC, abstractmethod

class RecommendationEnv:
    def __init__(self,num_message,num_maxsent,attraction_prob,reward_per_message,
                 abandonment_prob) -> None:
        # N
        self.num_message = num_message
        # M
        self.num_maxsent = num_maxsent
        # v
        self.attraction_prob = attraction_prob
        # R
        self.reward_per_message = reward_per_message
        # q(m)
        self.abandonment_prob = abandonment_prob
        self.timestep = 0

    def _setup(self) -> None:
        pass

    def _get_feedback(self) -> None:
        """get feedback for current active customer
        """
        pass

    def _customer_arrival(self) -> None:
        pass

class BaseAlgorithm(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def action(self):
        pass

    def step(self):
        # take action from env 
        return 0

if __name__ == '__main__':
    N = 35
    M = 7
    v = np.array([random.normalvariate(0.0597,0.0185) for i in range(N)])
    R = np.array([random.uniform(1, 2) for i in range(N)])
    q = np.array([1.1 * math.e ** (-0.03 * i) / (1 + math.e ** (-0.03 * i)) for i in range(M + 1)])

    env = RecommendationEnv(
        num_message=N,
        num_maxsent=M,
        attraction_prob=v,
        reward_per_message=R,
        abandonment_prob=q
    )