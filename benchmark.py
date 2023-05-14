import numpy as np
import math
from base import RecommendationEnv, BaseAlgorithm
from frequency import optimize, alg1_basic

class EpsilonGreedy(BaseAlgorithm):
    def __init__(self,env,epsilon = 0.05) -> None:
        super().__init__(env=env)
        self.epsilon = epsilon

        self.inf = 1e-3 # minimum value for probabilities
        self.v_hat = np.full(self.num_message, self.inf)
        # the first element of q is not used
        self.q_hat = np.full(self.num_maxsent + 1, self.inf)

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

    model = EpsilonGreedy(env)
    cumulative_rewards = model.learn(timesteps=1000)
    print('rewards',cumulative_rewards[-1])
