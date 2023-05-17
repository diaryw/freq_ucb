import numpy as np
import math
from copy import deepcopy
import random
from scipy.optimize import minimize
from base import (
    RecommendationEnv, 
    BaseAlgorithm, 
    evaluate_sequence, 
    BaseContextualAlgorithm,
    ContextualEnv,
)
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

class ContextualEpsilonGreedy(BaseContextualAlgorithm):
    """
    EpsilonGreedy based algorithm for cascading bandit with delayed feedback

    Parameters
    ----------
    epsilon : probability for randomly action
    decaying : whether to use decaying epsilon
    c : parameter for decaying factor
    """
    def __init__(self, env, epsilon:float=0.0, decaying:bool=False, c:float = 1.0,
                 regularization_alpha:float=0.1,regularization_beta:float=0.1) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        self.optimize_fun = optimize
        self.alpha_hat = np.zeros(shape=(self.num_maxsent+1,self.dim_user_feature))
        self.beta_hat = np.ones(shape=(self.dim_message_feature))
        self.alpha_default = deepcopy(self.alpha_hat)
        self.beta_default = deepcopy(self.beta_hat)
        self.regularization_alpha = regularization_alpha
        self.regularization_beta = regularization_beta
        self.decaying = decaying
        self.c = c

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
    
    def _get_prob(self,_user_feature,_message_feature):
        # for v
        v_lin_hat = np.dot(_message_feature,self.beta_hat)
        # compute probability
        v_greedy = self.sigmoid(v_lin_hat)
        # for q
        q_lin_hat = np.dot(self.alpha_hat,_user_feature)
        q_greedy = self.sigmoid(q_lin_hat)
        return v_greedy,q_greedy

    def _greedy_action(self):
        def get_optimal_m(_user_feature,_message_feature,_reward_vector) -> int:
            v_greedy,q_greedy = self._get_prob(_user_feature,_message_feature)
            return optimize(v_greedy,_reward_vector,q_greedy,self.num_maxsent)[3]

        def get_sequence(_user_feature, _message_feature, _reward_vector, _message_max) -> list:
            v_greedy,q_greedy = self._get_prob(_user_feature,_message_feature)
            return alg1_basic(v_greedy,_reward_vector,q_greedy,_message_max)[2]

        return get_optimal_m, get_sequence

    def _random_action(self):
        def get_optimal_m(_user_feature,_message_feature,_reward_vector) -> int:
            return np.random.randint(1,self.num_maxsent+1)
        
        def get_sequence(_user_feature, _message_feature, _reward_vector, _message_max) -> list:
            return np.random.choice(self.num_message,size=(_message_max),replace=False)
        
        return get_optimal_m, get_sequence
    
    def action(self):
        """
        Returns
        ----------
        get_optimal_m : a function, input two features, per reward and get optimal m
        get_sequence : a function, input two features, per reward, len of sequence,
                and output optimal sequence of messages 
        """
        if self.env.time%100 ==0:
            print('t=', self.env.time)
        if np.random.rand()<self.epsilon:
            return self._random_action()
        else:
            return self._greedy_action()

    def MLE_alpha(self,m) -> np.ndarray:
        if self.env.time==0:
            return self.alpha_default[m]
        self.m_index = self.m_record==m
        # if there is some hat_Y_rj=1
        # use MLE
        def neg_LL_alpha(_alpha):
            x_alpha = np.expand_dims((self.user_features@_alpha),axis=1)
            full_LL = self.hat_Y_rj*x_alpha - np.log(1+np.exp(x_alpha))
            neg_LL = -(full_LL*self.noclick_ind)[self.m_index].sum() + \
                np.power(_alpha[1:],2).sum()*self.regularization_alpha
            return neg_LL
        MLE_model = minimize(neg_LL_alpha, self.alpha_hat[m])
        return MLE_model.x

    def MLE_beta(self) -> np.ndarray:
        if self.env.time==0:
            return self.beta_default
        def neg_LL_beta(_beta):
            w_beta = np.dot(self.message_features,_beta)
            full_LL = self.Y_rj*w_beta - np.log(1+np.exp(w_beta))
            neg_LL = -(full_LL*self.feedback_ind).sum() + \
                np.power(_beta,2).sum()*self.regularization_beta
            return neg_LL

        MLE_model = minimize(neg_LL_beta, self.beta_hat)
        return MLE_model.x

    def update_param(self):
        m_record, noclick_ind, hat_Y_rj, feedback_ind, Y_rj = self.env.statistic
        user_features, message_features = self.env.features
        # update alpha
        self.m_record = np.array(m_record)
        self.noclick_ind = np.array(noclick_ind)
        self.hat_Y_rj = np.array(hat_Y_rj)
        self.user_features = np.array(user_features)
        # compute alpha by MLE
        self.alpha_hat = np.array([self.MLE_alpha(m) for m in range(self.num_maxsent+1)])

        # update beta
        self.feedback_ind = np.array(feedback_ind)
        self.Y_rj = np.array(Y_rj)
        self.message_features = np.array(message_features)
        # compute beta by MLE
        self.beta_hat = self.MLE_beta()
        if self.decaying:
            self.epsilon = self.c*self.num_message/(self.env.time+1)

class ContextualDecayingGreedy(ContextualEpsilonGreedy):
    def __init__(self, env, c: float = 1, 
                 regularization_alpha: float = 0.1, regularization_beta: float = 0.1) -> None:
        super().__init__(
            env = env, 
            epsilon = 0, 
            decaying = True, 
            c = c, 
            regularization_alpha = regularization_alpha, 
            regularization_beta = regularization_beta,
        )

class ContextualETC(ContextualEpsilonGreedy):
    def __init__(self, env, commit_time: int = 100,
                 regularization_alpha: float = 0.1, regularization_beta: float = 0.1) -> None:
        super().__init__(
            env=env, 
            epsilon=None, 
            decaying=False, 
            c=None, 
            regularization_alpha = regularization_alpha, 
            regularization_beta = regularization_beta,
        )
        self.commit_time = commit_time

    def action(self):
        """
        Returns
        ----------
        get_optimal_m : a function, input two features, per reward and get optimal m
        get_sequence : a function, input two features, per reward, len of sequence,
                and output optimal sequence of messages 
        """
        current_time = self.env.time
        if current_time % 100 ==0:
            print('t=', current_time)
        if current_time<self.commit_time:
            return self._random_action()
        else:
            return self._greedy_action()

if __name__ == '__main__':
    env = ContextualEnv(seed=2023)
    model = ContextualETC(env=env,commit_time=0)
    _result = model.learn(timesteps=1000)
    raise RuntimeError('Stop here')
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