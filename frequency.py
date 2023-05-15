import random
import numpy as np
import matplotlib.pyplot as plt
import math
from base import BaseAlgorithm, RecommendationEnv, evaluate_sequence

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

def alg1_basic(v, R, q, m): #q为list
    N = len(v)
    v_new, R_new, seq = sort_by_gamma(v, R, q[m])
    W = np.zeros([m, N], dtype = float)
    G = np.full([m, N], set())
    for l in range(m - 1, N):
        E_new = (v_new * R_new)[l:]
        W[m - 1, l] = np.max(E_new)
        G[m - 1, l] = set([np.argmax(E_new) + l]) #在原长度数组中的位置
    for k in range(m - 1, 0, -1):
        for l in range(k - 1, N - m + k):
            E_new = (v_new * R_new)[l: N - m + k]
            par = E_new + (1 - v_new[l: N - m + k]) * q[m] * W[k, (l+1): (N - m + k+1)]
            W[k - 1, l] = np.max(par)
            G[k - 1, l] = set.union(set([np.argmax(par) + l]), G[k, np.argmax(par) + l + 1]) #在原长度数组中的位置
    seq_r = seq[sorted(G[0,0])]
    return W[0,0], G[0,0], seq_r

def optimize(v, R, q, M): #optimize输出期望收益、排序后选择、排序前选择、list长度, q为list
    alg1_result = [alg1_basic(v, R, q, m) for m in range(1, M + 1)]
    W_res, G_res, seq_res = list(map(list, zip(*alg1_result)))
    W_m = np.max(W_res)
    G_m = G_res[np.argmax(W_res)]
    seq_m = seq_res[np.argmax(W_res)]
    return W_m, list(G_m), list(seq_m), len(G_m)

#test for alg 1
'''for test in range(20):
    v = np.array([random.normalvariate(0.0597,0.0185) for i in range(N)])
    R = np.array([random.uniform(1, 5) for i in range(N)])
    q = np.array([1.1 * math.e ** (-0.03 * i) / (1 + math.e ** (-0.03 * i)) for i in range(M + 1)]) # q[0] = 1 不用
    print(optimize(v, R, q, M))

def generate_fb(m, v, info):
    temp1 = random.uniform(0,1)
    if temp1 > v[info]:
        temp2 = random.uniform(0,1)
        if temp2 > q[m]:
            return -1
        else: return 0
    else: return 1

def alg2(N, M, T, R):
    v_hat = np.full(N, 1.0)
    q_hat = np.full(N, 0.999)
    v_ucb = np.full(N, 1.0)
    q_ucb = np.full(N, 0.999)
    tot_fb = np.zeros(N, dtype = int)
    tot_click = np.zeros(N, dtype = int)
    tot_noclick = np.zeros(N, dtype = int)
    tot_leave = np.zeros(N, dtype = int)
    next_fb = np.zeros([T, N], dtype = int)
    next_sent = np.array([x + 1 for x in range(T)])
    listm = np.zeros(T, dtype = int)
    times = np.zeros(T, dtype = int)
    selected = np.full(shape = (T, T), fill_value=-1)
    selres = np.array([], dtype = int)  #(时间，用户，消息)
    ret = np.zeros(T, dtype = float)
    for t in range(1, T + 1):
        if (t % 100 == 0):
            print('t = ', t)
        if t > 1:
            ret[t - 1] = ret[t - 2]
        for i in range(N):
            for r in range(t - 1):
                if next_fb[r, i] == t:
                    res = generate_fb(listm[t - 1], v, i)
                    tot_fb[i] += 1
                    if res != 1:
                        tot_noclick[i] += 1
                        if res == -1:
                            tot_leave[i] += 1
                            next_fb[r, i] = 0
                    else:
                        tot_click[i] += 1
                        ret[t - 1] += R[i]
                    if tot_fb[i] != 0:
                        v_hat[i] = tot_click[i] / tot_fb[i]
                        v_ucb[i] = min(v_hat[i] + np.sqrt(2 * np.log(t) / tot_fb[i]), 1)
                    if tot_noclick[i] != 0:
                        q_hat[i] = 1 - tot_leave[i]/tot_noclick[i]
                        q_ucb[i] = min(q_hat[i] + np.sqrt(2 * np.log(t) / tot_noclick[i]), 1)

        #新顾客
        sel = optimize(v_ucb, R, q_ucb, M)[2][0]
        listm[t - 1] = optimize(v_ucb, R, q_ucb, M)[3]
        next_sent[t - 1] += int(D / listm[t - 1])
        tau = random.randint(1, 5)
        selected[t - 1][0] = sel
        next_fb[t - 1, sel] = t + tau
        times[t - 1] += 1
        selres = np.append(selres, (t, t - 1, sel))
        #老顾客
        for r in range(t - 1):
            if next_sent[r] == t and times[r] <= listm[r]:
                tau = random.randint(1, 5)
                G = alg1_basic(v_ucb, R, q_ucb, listm[r])[2]
                selind = 0
                while G[selind] in selected[r]:
                    selind += 1
                    if selind == len(G):
                        next_sent[r] = 0
                        break
                if next_sent[r] != 0:
                    sel = G[selind]
                    selected[r][times[r]] = sel
                    selres = np.append(selres,(t, r, sel))
                    times[r] += 1
                    next_sent[r] += int(D / listm[r])
                    next_fb[r, sel] = t + tau
    rown = int(len(selres) / 3)
    selres = np.resize(selres, (rown,3))
    return selres, ret

#Test for alg 2
#print(alg2(N, M, 100, R))
'''

class UCB(BaseAlgorithm):
    """
    UCB based algorithm for cascading bandit with delayed feedback

    Parameters
    ----------
    c : width for confidence interval for ucb value, 
        by default = 1: UCB1-like algorithm
    """
    def __init__(self, env, confidence_level: float = 1.0, clip_ucb: bool =True, 
                 decaying: bool =True, decaying_mode: str ='log') -> None:
        super().__init__(env)
        self.confidence_level = confidence_level
        self.inf = 1e-3
        self.initial_val = 0.99
        self.v_hat = np.full(self.num_message, self.initial_val)
        self.v_ucb = np.full(self.num_message, self.initial_val)
        # the first element of q is not used
        self.q_hat = np.full(self.num_maxsent + 1, self.initial_val)
        self.q_ucb = np.full(self.num_maxsent + 1, self.initial_val)
        self.clip_ucb = clip_ucb
        self.decaying  = decaying
        self.decaying_mode = decaying_mode

    def action(self):
        """
        take actions

        Return
        ----------
        optimal_m : the optimal number of messages to send to new customer
        get_sequence: callable, input number and output optimal sequence of messages 
        """
        optimal_m = optimize(self.v_ucb,self.reward_per_message,self.q_ucb,self.num_maxsent)[3]
        def get_sequence(m):
            return alg1_basic(self.v_ucb,self.reward_per_message,self.q_ucb,m)[2]
        return optimal_m, get_sequence

    @property
    def decaying_factor(self) -> float:
        if not self.decaying:
            return 1.0
        else:
            if self.decaying_mode=='log':
                return 1/np.log(self.t + 1)
            elif self.decaying_mode=='linear':
                return 1/self.t
            else:
                raise NotImplementedError(f'{self.decaying_mode} is not defined yet')

    def update_param(self) -> None:
        t = self.env.time + 1
        self.t = t
        total_fb, total_click, tilde_noclick, tilde_leave = self.env.statistic
        n_continue = tilde_noclick - tilde_leave
        self.v_hat = np.divide(total_click, total_fb, out=self.v_hat, where=(total_fb!=0))
        self.v_ucb = self.v_hat + self.confidence_level * np.sqrt(2*np.log(t)/(total_fb + 1e-7)) * self.decaying_factor
        self.q_hat = np.divide(n_continue, tilde_noclick, out=self.q_hat, where=(tilde_noclick!=0))
        self.q_ucb = self.q_hat + self.confidence_level * np.sqrt(2*np.log(t)/(tilde_noclick + 1e-7)) * self.decaying_factor
        if self.clip_ucb:
            self.v_ucb = np.minimum(self.v_ucb, 0.99)
            self.q_ucb = np.minimum(self.q_ucb, 0.99)

def regret_analysis(v, R, q, M, ret_real):
    """
    ret_real : reward at each time
    """
    _,_, seq_theo, m_theo = optimize(v, R, q, M)
    payoff_theo = evaluate_sequence(seq_theo, v,R,q)
    instant_regret = payoff_theo - np.array(ret_real)
    regret = np.cumsum(instant_regret)
    return regret

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

    model = UCB(env,confidence_level=100,decaying_mode='linear')
    _rewards = model.learn(timesteps=16000)
    plt.plot(_rewards[1])
    plt.show()

'''
    res = np.zeros([10, T], dtype = float)
    for exptimes in range(10):
        print('The Exp ', exptimes + 1)
        r = alg2(N, M, T, R)[1]
        res[exptimes] = regret_analysis(v, R, q, M, T, r)

t = [x for x in range(T)]
avg = []
upp = []
low = []
for i in range(T):
    avg.append(np.mean(res[:,i]))
    upp.append(np.max(res[:,i]))
    low.append(np.min(res[:,i]))
plt.title("Algorithm 2  Experiment 1")
plt.xlabel('T')
plt.ylabel('Regret')
plt.plot(t, avg, label = 'average')
plt.plot(t, upp, label = 'max')
plt.plot(t, low, label = 'min')
plt.legend()
plt.show()
print(avg[-1])
'''
