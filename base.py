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
        current_rewards = 0

        for _customer in self.active_customers:
            if _customer['next_feedback'] == self.time:
                message_id = _customer['sent'][-1]
                response = self._generate_feedback(_customer)
                self.tot_fb[message_id] += 1

                # click
                if response == 1:
                    self.tot_click[message_id] += 1
                    current_rewards += self.reward_per_message[message_id]

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
        return current_rewards

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
        return current_rewards

    def learn(self,timesteps: int):
        """
        learn for timesteps and return cumulative rewards
        """
        self.total_timesteps = timesteps
        rewards_history = []
        for _ in range(timesteps):
            reward = self.step()
            rewards_history.append(reward)

        return np.cumsum(rewards_history)

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