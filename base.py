import numpy as np
import math
from abc import ABC, abstractmethod

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
        temp1 = np.random.uniform(0,1)
        if temp1 < self.attraction_prob[message_id]:
            # click
            return 1 
        else:
            temp2 = np.random.uniform(0,1)
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
                tau = np.random.randint(1,5)
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

    get_sequence = lambda m: [i for i in range(m)]

    for t in range(1000):
        res = env.get_feedback()
        print('t=',t,res)
        env.step(5,get_sequence)