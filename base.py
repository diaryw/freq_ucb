import numpy as np
import math
from abc import ABC, abstractmethod
import random
from collections import deque

class ContextGenerator:
    def __init__(self,seed:int = 2023,user_feature_range=None,message_feature_range=None,
                 reward_range = None) -> None:
        # save all input
        self.seed = seed
        self.user_feature_range = user_feature_range
        self.message_feature_range = message_feature_range
        # use default value if no input
        if self.user_feature_range is None:
            self.user_feature_range = np.array([5,5,5])
        if self.message_feature_range is None:
            self.message_feature_range = np.array([-6,1,2,-5])
        # default value for reward_range
        if reward_range is None:
            self.reward_range = [1,5]
        else:
            self.reward_range = reward_range
        # instantiate a random number generator
        self._generator = np.random.default_rng(seed=self.seed)
        # load data
        self._load()

    def _load(self) -> None:
        npzdata = np.load('data/contextual.npz')
        self._alpha = npzdata['alpha']
        self._beta = npzdata['beta']
        self._reward_per_message = npzdata['reward']
        self.num_maxsent = len(self._alpha) - 1
        # get features range
        self.user_feature_low = np.minimum(self.user_feature_range,0)
        self.user_feature_high = np.maximum(self.user_feature_range,0)
        self.message_feature_low = np.minimum(self.message_feature_range,0)
        self.message_feature_high = np.maximum(self.message_feature_range,0)
        # get reward range
        self.reward_low = np.min(self.reward_range)
        self.reward_high = np.max(self.reward_range)
        # get number of total messages from _reward_per_message
        self.num_message = len(self._reward_per_message)
        # check the dimensions
        if self._alpha.shape[1] != len(self.user_feature_range)+1:
            raise ValueError('dimension of alpha must be compatible with user_feature_range')
        if len(self._beta) != len(self.message_feature_range)+1:
            raise ValueError('dimension of beta must be compatible with message_feature_range')

    def get_user_feature(self) -> np.ndarray:
        _user_feature = self._generator.uniform(self.user_feature_low,self.user_feature_high)
        return np.concatenate([[1],_user_feature])
    
    def get_message_feature(self) -> np.ndarray:
        _intercept = np.ones(shape=(self.num_message,1))
        _message_feature = self._generator.uniform(
            low=self.message_feature_low,
            high=self.message_feature_high,
            size=(self.num_message,len(self.message_feature_range))
            )
        return np.concatenate([_intercept,_message_feature],axis=1)

    def get_reward_vector(self) -> np.ndarray:
        _reward = self._generator.uniform(
            low=self.reward_low,
            high=self.reward_high,
            size=self.num_message,
        )
        return _reward

    # add some property
    @property
    def N(self) -> int:
        return self.num_message
    
    @property
    def M(self) -> int:
        return self.num_maxsent

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha
    
    @property
    def beta(self) -> np.ndarray:
        return self._beta


class RecommendationEnv:
    def __init__(self,num_message,num_maxsent,attraction_prob,reward_per_message,
                 q_prob,time_window) -> None:
        # N
        self.num_message = num_message
        # M
        self.num_maxsent = num_maxsent
        # v
        self.attraction_prob = attraction_prob
        # R
        self.reward_per_message = reward_per_message
        # q(m)
        self.q_prob = q_prob
        # D
        self.time_window = time_window
        self._setup()

    def _setup(self) -> None:
        self.time = 0
        self.total_customer = 0
        # a list to store all active customers
        self.active_customers = []
        self.reward_customers = []

        ### for each message
        self.tot_fb = np.zeros(self.num_message, dtype = int)
        # total feedback = total click + total no click
        self.tot_click = np.zeros(self.num_message, dtype = int)

        ### for each m value
        # total no click = total abandon + total remain
        self.tilde_noclick = np.zeros(self.num_maxsent + 1, dtype = int)
        self.tilde_leave = np.zeros(self.num_maxsent + 1, dtype = int)

    @property
    def statistic(self):
        return (self.tot_fb,self.tot_click,self.tilde_noclick,self.tilde_leave)

    def _generate_feedback(self, customer: dict) -> int:
        """generate feed back for a customer"""
        m = customer['message_max']
        message_id = customer['sent'][-1]
        temp1 = random.uniform(0,1)
        if temp1 < self.attraction_prob[message_id]:
            # click
            return 1 
        else:
            temp2 = random.uniform(0,1)
            if temp2 < self.q_prob[m]:
                # not click but remain
                return 0
            else:
                # abandon
                return -1

    def _remove_inactive_customer(self) -> None:
        self.active_customers = [_customer for _customer in self.active_customers if not _customer['terminated']]

    def get_feedback(self) -> float:
        """return rewards received in current time
        """
        if len(self.active_customers)==0:
            return 0
        # total reward for current time
        total_reward = 0

        for _customer in self.active_customers:
            if _customer['next_feedback'] == self.time:
                message_id = _customer['sent'][-1]
                response = self._generate_feedback(_customer)
                self.tot_fb[message_id] += 1

                # click
                if response == 1:
                    self.tot_click[message_id] += 1
                    reward = self.reward_per_message[message_id]
                    total_reward += reward
                    self.reward_customers[_customer['id']] = reward

                # no click but remain
                if response == 0:
                    self.tilde_noclick[_customer['message_max']] += 1

                # no click and abandon
                if response == -1:
                    self.tilde_noclick[_customer['message_max']] += 1
                    self.tilde_leave[_customer['message_max']] += 1

                # check if customer is active
                if response == 0 and len(_customer['sent']) < _customer['message_max']:
                    # customer no click and remain active
                    _customer['terminated'] = False
                else:
                    # customer exit: click, abandon, or maximal messages number reached
                    _customer['terminated'] = True

        self._remove_inactive_customer()
        return total_reward

    def _customer_arrival(self, optimal_m: int) -> None:
        _customer = {
            'id': self.total_customer,
            'next_sent': self.time,
            'last_sent': None,
            'next_feedback': None,
            'message_max': optimal_m,
            'sent': [],
            'terminated': False,
        }
        self.active_customers.append(_customer)
        self.reward_customers.append(0)
        self.total_customer += 1

    def _send_active_customers(self,get_sequence: callable) -> None:
        """
        send messages to active customers whose next feedback = self.time
        """
        for _customer in self.active_customers:
            if _customer['next_sent']==self.time:
                seq = get_sequence(_customer['message_max'])
                seq_not_sent = [ele for ele in seq if ele not in _customer['sent']]
                _customer['sent'].append(seq_not_sent[0])
                _customer['last_sent'] = self.time
                _customer['next_sent'] = self.time + int(self.time_window/_customer['message_max'])
                # time for next feedback
                tau = random.randint(1,5)
                _customer['next_feedback'] = self.time + tau

    def step(self, optimal_m:int, get_sequence: callable) -> None:
        """
        update the env status, new customer arrives, send messages to customers

        Parameters
        ----------
        optimal_m : the optimal number of messages to send to new customer
        get_sequence: a function, input number and output optimal sequence of messages 
        """
        self._customer_arrival(optimal_m)
        self._send_active_customers(get_sequence)
        self.time += 1

    def expected_payoff(self, optimal_m:int, get_sequence: callable) -> float:
        seq = get_sequence(optimal_m)
        _expected_val = evaluate_sequence(
            seq = seq,
            v = self.attraction_prob,
            R = self.reward_per_message,
            q = self.q_prob)
        return _expected_val

class ContextualEnv:
    """
    environment for contextual recommendation system

    Parameters
    ----------
    seed : random seed for random generator in ContextGenerator()
    """
    def __init__(self,seed:int=2023,user_feature_range=None, message_feature_range = None,
                 time_window:int = 200) -> None:
        self.context_generator = ContextGenerator(
            seed=seed,
            user_feature_range = user_feature_range,
            message_feature_range = message_feature_range,
            )
        # N
        self.num_message = self.context_generator.N
        # M
        self.num_maxsent = self.context_generator.M
        # D
        self.time_window = time_window
        # alpha, unknown for algorithm
        self.alpha_truth = self.context_generator.alpha
        # beta, unknown for algorithm
        self.beta_truth = self.context_generator.beta
        self.max_response = 10
        # setup
        self._setup()

    def _setup(self) -> None:
        self.time = 0
        self.total_customer = 0
        # a list to store all active customers
        self.active_customers = []
        # a list to store realized reward for customers
        self.reward_customers = []

        # store customer features and content features, reward_vector
        self.user_features_record = []
        self.message_features_record = []
        self.reward_vectors_record = []
    
    def _remove_inactive_customer(self) -> None:
        self.active_customers = [_customer for _customer in self.active_customers if not _customer['terminated']]

    def _customer_arrival(self, get_optimal_m:callable) -> None:
        # generate context for this user
        _user_feature = self.context_generator.get_user_feature()
        _message_feature = self.context_generator.get_message_feature()
        _reward_vector = self.context_generator.get_reward_vector()
        optimal_m = get_optimal_m(_user_feature, _message_feature, _reward_vector)
        _customer = {
            'id': self.total_customer,  # start with 0
            'next_sent': self.time,     # time for next sent
            'last_sent': None,      # last sent time
            'next_feedback': None,     # time for next feedback
            'message_max': optimal_m,     # the optimal sequence length
            'sent': [],     # messages already sent
            'terminated': False,    # whether the customer activity is terminated
            'feature': _user_feature,      # user feature 
            'message_feature': _message_feature,      # message feature
            'reward_vector': _reward_vector,        # reward per message
        }
        # add to active customers
        self.active_customers.append(_customer)
        # set current customer's reward to 0, will change later
        self.reward_customers.append(0)
        # record generated data
        self.user_features_record.append(_user_feature)
        self.message_features_record.append(_message_feature)
        self.reward_vectors_record.append(_reward_vector)
        # index + 1
        self.total_customer += 1

    def _send_active_customers(self,get_sequence: callable) -> None:
        """
        send messages to active customers whose next feedback = self.time
        """
        for _customer in self.active_customers:
            if _customer['next_sent']==self.time:
                seq = get_sequence(_customer['feature'], _customer['message_feature'], _customer['reward_vector'])
                seq_not_sent = [ele for ele in seq if ele not in _customer['sent']]
                _customer['sent'].append(seq_not_sent[0])
                _customer['last_sent'] = self.time
                _customer['next_sent'] = self.time + int(self.time_window/_customer['message_max'])
                # time for next feedback
                tau = random.randint(1,self.max_response)
                _customer['next_feedback'] = self.time + tau

    def step(self, get_optimal_m:callable, get_sequence: callable) -> None:
        """
        update the env status, new customer arrives, send messages to customers

        Parameters
        ----------
        get_optimal_m : a function, input two features per reward and get optimal m
        get_sequence : a function, input two features per reward and output 
                optimal sequence of messages 
        """
        self._customer_arrival(get_optimal_m)
        self._send_active_customers(get_sequence)
        self.time += 1


# create aliases
ContextualRecommendationEnv = ContextualEnv

class BaseAlgorithm(ABC):
    def __init__(self, env) -> None:
        self.env = env
        # N
        self.num_message = self.env.num_message
        # M
        self.num_maxsent = self.env.num_maxsent
        # R
        self.reward_per_message = self.env.reward_per_message
        # D
        self.time_window = self.env.time_window
        self.expected_payoff_record = []

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def update_param(self):
        pass

    def step(self) -> float:
        """
        interact with env and update parameters 
        """
        current_rewards = self.env.get_feedback()
        self.update_param()
        _action = self.action()
        self.env.step(*_action)
        _expected_payoff = self.env.expected_payoff(*_action)
        self.expected_payoff_record.append(_expected_payoff)
        return current_rewards

    def learn(self,timesteps: int):
        """
        learn for timesteps and return cumulative rewards

        Parameters
        ----------
        timesteps : total time horizon

        Returns
        ----------
        rewards_record : the reward received per time
        self.expected_payoff_record : the expected payoff for action per time
        self.env.reward_customers : the rewards received from each customer
        """
        self.total_timesteps = timesteps
        # reset env
        self.env._setup()
        rewards_record = []
        for _ in range(timesteps + self.time_window + 10):
            reward = self.step()
            rewards_record.append(reward)

        return rewards_record[:timesteps], self.expected_payoff_record[:timesteps], self.env.reward_customers[:timesteps]


def evaluate_sequence(seq,v,R,q):
    m = len(seq)
    q_m = q[m]
    # prob v, reward R according to sequence
    v_kappa = v[seq]
    R_kappa = R[seq]
    # definition of w_i(m)
    w_m = np.insert(np.cumprod((1-v_kappa)*q_m),0,1)[:-1]
    return np.sum(w_m*v_kappa*R_kappa)

def get_all_sequence(elements: list, seq_len: int):
    output = []
    if seq_len==0:
        return output
    if seq_len==1:
        return [[x] for x in elements]

    # split into two parts, current and subseq
    subseq_len = seq_len - 1
    for ele_index in range(len(elements)):
        current_ele = elements[ele_index]
        other_ele = [elements[i] for i in range(len(elements)) if i!=ele_index]
        for subseq in get_all_sequence(other_ele,subseq_len):
            output.append([current_ele] + subseq)
    
    return output

def get_all_ordered_subsequence(elements: list, seq_len: int):
    output = []
    if seq_len==0:
        return output
    if seq_len==1:
        return [[x] for x in elements]
    
    # split into three parts, before current ,current and rest
    subseq_len = seq_len - 1
    for ele_index in range(len(elements)-1):
        current_ele = elements[ele_index]
        rest_ele = [elements[i] for i in range(len(elements)) if i>ele_index]
        for subseq in get_all_ordered_subsequence(rest_ele,subseq_len):
            output.append([current_ele] + subseq)

    return output

def best_sequence_given_length(v,R,q,m):
    """
    Parameters
    ----------
    v : attraction probability
    R : rewards for each message
    q : list of abandonment probability for each length
    m : the length of sequence of messages

    Returns
    -------
    payoff : the payoff of optimal sequence
    seq : the sequence of optimal sequence
    """
    message_sets = list(range(len(v)))
    seq_sets = get_all_sequence(message_sets,m)
    payoff = [evaluate_sequence(seq,v,R,q) for seq in seq_sets]
    max_id = np.argmax(payoff)
    return payoff[max_id], seq_sets[max_id]

def best_sequence_preserve_order(v,R,q,M):
    def sort_by_gamma(v, R, q): #q单值
        gamma = v * R / (1 - q * (1 - v))
        # seq is the ordered message ID
        seq = np.argsort(-gamma) #序列seq是排好序的数组中的各元素在原数组中的编号
        v_new = np.array([])
        R_new = np.array([])
        for i in range(len(seq)):
            v_new = np.append(v_new, v[seq[i]])
            R_new = np.append(R_new, R[seq[i]])
        return v_new, R_new, seq
    
    payoff_list = []
    seq_list = []
    for m in range(1,M+1):
        gamma_ordered = sort_by_gamma(v,R,q[m])[2]
        seq_sets = get_all_ordered_subsequence(gamma_ordered,m)
        payoff = [evaluate_sequence(seq,v,R,q) for seq in seq_sets]
        max_id = np.argmax(payoff)
        payoff_list.append(payoff[max_id])
        seq_list.append(seq_sets[max_id])
    max_num_id = np.argmax(payoff_list)
    return payoff_list[max_num_id], seq_list[max_num_id]

def generate_noncontextual(N,M):
    v = np.array([random.normalvariate(0.0597,0.0185) for i in range(N)])
    R = np.array([random.uniform(1, 2) for i in range(N)])
    q = np.array([1.1 * math.e ** (-0.03 * i) / (1 + math.e ** (-0.03 * i)) for i in range(M + 1)])
    np.savez('data/noncontextual',N=N,M=M,v=v,R=R,q=q)

def load_noncontextual():
    npzdata = np.load('data/noncontextual.npz')
    N = npzdata['N'].item()
    M = npzdata['M'].item()
    v = npzdata['v']
    R = npzdata['R']
    q = npzdata['q']
    return N,M,v,R,q

def generate_contextual(M=20, N = 25, alpha_range=None,beta=None):
    """
    Parameters
    ----------
    M : maximal number of sent messages
    user_feature_range : the range to define user feature
    """
    if alpha_range is None:
        alpha_range = np.array([-0.064,-0.08,-0.16])
    if beta is None:
        beta = np.array([0.05,0.2,0.1,0.3,0.4])
    alpha = np.array([np.random.uniform(low=alpha_range*m,high=0) for m in range(M+1)])
    alpha = np.concatenate([np.ones((M+1,1))*-1.04,alpha],axis = 1)
    reward = np.random.uniform(low=1,high=3,size=(N,))
    np.savez('data/contextual',alpha=alpha,beta=beta,reward=reward)


if __name__ == '__main__':
    generate_contextual(M=20)
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

    get_sequence = lambda m: [i for i in range(m)]

    for t in range(1000):
        res = env.get_feedback()
        print('t=',t,res)
        env.step(5,get_sequence)