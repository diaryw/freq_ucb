import numpy as np
import math
from base import RecommendationEnv, BaseAlgorithm, evaluate_sequence
from frequency import optimize, alg1_basic,regret_analysis
import matplotlib.pyplot as plt

class EpsilonGreedy(BaseAlgorithm):
    """
    Epsilon-greedy for cascading bandit with delayed feedback

    Parameters
    ----------
    env : environment to interact
    epsilon : the epsilon value for non-decaying epislon-greedy
    initial_val : initial values for probability estimation. By default = None: vanilla 
        algorithm, if set a larger value: Epsilon-greedy with optimistic initialization
    decaying : whether to use decaying epsilon value, by default = False
    """
    def __init__(self,env,epsilon = 0.0,initial_val:float =None, decaying = False,c = 1e-4) -> None:
        super().__init__(env=env)
        # epsilon = 0: greedy, !=0: epsilon-greedy
        self.epsilon = epsilon
        # minimum value for probabilities
        self.inf = 1e-3
        if initial_val is None:
            # zero initialization
            self.initial_val = self.inf
        else:
            # optimistic initialization
            self.initial_val = float(initial_val)
        self.v_hat = np.full(self.num_message, self.initial_val)
        # the first element of q is not used
        self.q_hat = np.full(self.num_maxsent + 1, self.initial_val)
        self.decaying = decaying
        # rough initailization for decaying
        self.min_gap = 1e-2
        self.c = c

    def _greedy_action(self):
        optimal_m = optimize(self.v_hat,self.reward_per_message,self.q_hat,self.num_maxsent)[3]
        def get_sequence(m):
            return alg1_basic(self.v_hat,self.reward_per_message,self.q_hat,m)[2]
        return optimal_m, get_sequence

    def _random_action(self):
        v_random = np.random.rand(self.num_message)
        q_random = -np.sort(-np.random.rand(self.num_maxsent + 1))
        optimal_m = optimize(v_random,self.reward_per_message,q_random,self.num_maxsent)[3]
        def get_sequence(m):
            return alg1_basic(v_random,self.reward_per_message,q_random,m)[2]
        return optimal_m, get_sequence

    def action(self):
        """
        take actions

        Return
        ----------
        optimal_m : the optimal number of messages to send to new customer
        get_sequence: callable, input number and output optimal sequence of messages 
        """
        if np.random.rand()<self.epsilon:
            return self._random_action()
        else:
            return self._greedy_action()

    def update_param(self) -> None:
        total_fb, total_click, tilde_noclick, tilde_leave = self.env.statistic
        n_continue = tilde_noclick - tilde_leave
        self.v_hat = np.divide(total_click, total_fb, out=self.v_hat, where=(total_fb!=0))
        self.v_hat = np.maximum(self.v_hat,self.inf)
        self.q_hat = np.divide(n_continue, tilde_noclick, out=self.q_hat, where=(tilde_noclick!=0))
        self.q_hat = np.maximum(self.q_hat,self.inf)
        if self.decaying:
            self.epsilon = np.min([1,self.c*self.num_message/(self.min_gap**2 * (self.env.time+1))])

if __name__ == '__main__':
    import random
    N = 35
    M = 7
    v = np.array([random.normalvariate(0.0597,0.0185) for i in range(N)])
    R = np.array([random.uniform(1, 2) for i in range(N)])
    q = np.array([1.1 * math.e ** (-0.03 * i) / (1 + math.e ** (-0.03 * i)) for i in range(M + 1)])
    D = 200

    env = RecommendationEnv(
        num_message=N,
        num_maxsent=M,
        attraction_prob=v,
        reward_per_message=R,
        q_prob=q,
        time_window=D,
    )

    model = EpsilonGreedy(env,initial_val=1e-3)
    _rewards = model.learn(timesteps=1000)
    plt.plot(_rewards[1])
    plt.show()
    _,_, seq_theo, m_theo = optimize(v, R, q, M)
    payoff_theo = evaluate_sequence(seq_theo, v,R,q)