from abc import ABC, abstractmethod
import numpy as np
import random
import math


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

class EpsilonGreedy(BaseAlgorithm):
    def __init__(self) -> None:
        super().__init__()

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

    model = EpsilonGreedy()
    model.action()
