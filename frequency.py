import random
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from scipy.optimize import minimize
from base import (
    BaseAlgorithm, 
    RecommendationEnv, 
    evaluate_sequence,
    BaseContextualAlgorithm,
    ContextualEnv,
    FixedContextGenerator,
)

import warnings
warnings.filterwarnings('error')

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
    confidence_level : width for confidence interval for ucb value, 
        by default = 1: UCB1-like algorithm
    """
    def __init__(self, env, confidence_level: float = 1.0, clip_ucb: bool =True, 
                 decaying: bool =False, decaying_mode: str ='log') -> None:
        super().__init__(env)
        self.confidence_level = confidence_level
        self.inf = 1e-3
        self.initial_val = self.inf
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
        self.v_hat = np.maximum(self.v_hat,self.inf)
        self.v_ucb = self.v_hat + self.confidence_level * np.sqrt(2*np.log(t)/(total_fb + 1)) * self.decaying_factor
        self.q_hat = np.divide(n_continue, tilde_noclick, out=self.q_hat, where=(tilde_noclick!=0))
        self.q_hat = np.maximum(self.q_hat,self.inf)
        self.q_ucb = self.q_hat + self.confidence_level * np.sqrt(2*np.log(t)/(tilde_noclick + 1)) * self.decaying_factor
        if self.clip_ucb:
            self.v_ucb = np.minimum(self.v_ucb, 0.999)
            self.q_ucb = np.minimum(self.q_ucb, 0.999)

# the regret analysis is only for non context case
def regret_analysis(v, R, q, M, ret_real):
    """
    ret_real : reward at each time
    """
    _,_, seq_theo, m_theo = optimize(v, R, q, M)
    payoff_theo = evaluate_sequence(seq_theo, v,R,q)
    instant_regret = payoff_theo - np.array(ret_real)
    regret = np.cumsum(instant_regret)
    return regret

class ContextualUCB(BaseContextualAlgorithm):
    """
    Linear UCB based algorithm for cascading bandit with delayed feedback

    Parameters
    ----------
    gamma1 : radius for v_ucb, i.e., confidence_level
    gamma2 : radius for q_ucb, i.e., confidence_level
    eyecoeff1 : coefficient for eye matrix added to V (for v_ucb)
    eyecoeff2 : coefficient for eye matrix added to M (for q_ucb)
    """
    def __init__(self, env, gamma1:float=0.1,gamma2:float=1e-3,
                 eyecoeff1:float=1.0,eyecoeff2:float=1.0,
                 regularization_alpha:float=0.1,regularization_beta:float=0.1) -> None:
        super().__init__(env)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eyecoeff1 = eyecoeff1
        self.eyecoeff2 = eyecoeff2
        self.optimize_fun = optimize
        self.alpha_hat = np.zeros(shape=(self.num_maxsent+1,self.dim_user_feature))
        self.beta_hat = np.ones(shape=(self.dim_message_feature))
        self.alpha_default = deepcopy(self.alpha_hat)
        self.beta_default = deepcopy(self.beta_hat)
        self.Veye = np.eye(self.dim_message_feature)*self.eyecoeff1
        self.Meye = np.array([np.eye(self.dim_user_feature)]*(self.num_maxsent+1))*self.eyecoeff2
        self.regularization_alpha = regularization_alpha
        self.regularization_beta = regularization_beta

    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def _get_ucb(self,_user_feature,_message_feature):
        # for v
        v_lin_hat = np.dot(_message_feature,self.beta_hat)
        v_mat_norm = np.sqrt(np.array([vec@self.V_inv@vec for vec in _message_feature]))
        # compute probability
        v_ucb = self.sigmoid(v_lin_hat + self.gamma1*v_mat_norm)
        # for q
        q_lin_hat = np.dot(self.alpha_hat,_user_feature)
        q_mat_norm = np.sqrt(np.array([_user_feature@_mat_M@_user_feature for _mat_M in self.M_inv]))
        q_ucb = self.sigmoid(q_lin_hat + self.gamma2*q_mat_norm)
        return v_ucb,q_ucb

    def action(self):
        def get_optimal_m(_user_feature,_message_feature,_reward_vector) -> int:
            v_ucb,q_ucb = self._get_ucb(_user_feature,_message_feature)
            return optimize(v_ucb,_reward_vector,q_ucb,self.num_maxsent)[3]

        def get_sequence(_user_feature, _message_feature, _reward_vector, _message_max) -> list:
            v_ucb,q_ucb = self._get_ucb(_user_feature,_message_feature)
            return alg1_basic(v_ucb,_reward_vector,q_ucb,_message_max)[2]

        return get_optimal_m, get_sequence

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
        mat_M, mat_V = self.env.covariance
        # update inverse matrix
        self.M_inv = np.linalg.inv(mat_M+self.Meye)
        self.V_inv = np.linalg.inv(mat_V+self.Veye)
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

class UCBFixedContext(BaseContextualAlgorithm):
    """
    UCB based algorithm for fixed context cascading bandit with delayed feedback
    The environment is contextual but context is fixed.
    No context is used in this basic UCB Algorithm

    Parameters
    ----------
    confidence_level : width for confidence interval for ucb value, 
        by default = 1: UCB1-like algorithm
    """
    def __init__(self, env, confidence_level: float = 1.0, clip_ucb: bool =True,) -> None:
        super().__init__(env)
        self.confidence_level = confidence_level
        self.inf = 1e-3
        self.initial_val = self.inf
        self.v_hat = np.full(self.num_message, self.initial_val)
        self.v_ucb = np.full(self.num_message, self.initial_val)
        # the first element of q is not used
        self.q_hat = np.full(self.num_maxsent + 1, self.initial_val)
        self.q_ucb = np.full(self.num_maxsent + 1, self.initial_val)
        self.clip_ucb = clip_ucb
        self.optimize_fun = optimize

    def action(self):
        """
        take actions

        Return
        ----------
        get_optimal_m : callable, the optimal number of messages
        get_sequence: callable, input number and output optimal sequence of messages 
        """
        def get_optimal_m(_user_feature,_message_feature,_reward_vector) -> int:
            return optimize(self.v_ucb,_reward_vector,self.q_ucb,self.num_maxsent)[3]

        def get_sequence(_user_feature, _message_feature, _reward_vector, _message_max):
            return alg1_basic(self.v_ucb,_reward_vector,self.q_ucb,_message_max)[2]

        return get_optimal_m, get_sequence

    def update_param(self) -> None:
        t = self.env.time + 1
        m_record, noclick_ind, hat_Y_rj, feedback_ind, Y_rj = self.env.statistic
        # total_fb, total_click, tilde_noclick, tilde_leave = self.env.statistic
        m_record = np.array(m_record)
        total_fb = np.array(feedback_ind).sum(axis = 0)
        total_click = np.array(Y_rj).sum(axis = 0)
        tilde_noclick = np.array([(np.array(noclick_ind)[m_record==m]).sum() for m in range(self.num_maxsent+1)])
        n_continue = np.array([(np.array(hat_Y_rj)[m_record==m]).sum() for m in range(self.num_maxsent+1)])
        self.v_hat = np.divide(total_click, total_fb, out=self.v_hat, where=(total_fb!=0))
        self.v_hat = np.maximum(self.v_hat,self.inf)
        self.v_ucb = self.v_hat + self.confidence_level * np.sqrt(2*np.log(t)/(total_fb + 1))
        self.q_hat = np.divide(n_continue, tilde_noclick, out=self.q_hat, where=(tilde_noclick!=0))
        self.q_hat = np.maximum(self.q_hat,self.inf)
        self.q_ucb = self.q_hat + self.confidence_level * np.sqrt(2*np.log(t)/(tilde_noclick + 1))
        if self.clip_ucb:
            self.v_ucb = np.minimum(self.v_ucb, 0.9)
            self.q_ucb = np.minimum(self.q_ucb, 0.9)


if __name__ == '__main__':
    T = 1000

    env = ContextualEnv(seed=2023,generator_cls=FixedContextGenerator)
    model = UCBFixedContext(env=env,confidence_level= 0.1)
    fixed_result = model.learn(timesteps=T)
    fixed_regret = np.array(fixed_result[3]-fixed_result[1]).cumsum()

    env = ContextualEnv(seed=2023,generator_cls=FixedContextGenerator)
    model = ContextualUCB(env=env)
    context_result = model.learn(timesteps=T)
    context_regret = np.array(context_result[3]-context_result[1]).cumsum()
    plt.figure()
    plt.plot(fixed_regret,label = 'noncontextual alg')
    plt.plot(context_regret,label = 'contextual alg')
    plt.legend()
    plt.show()
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

    model = UCB(env,confidence_level=0.01)
    _rewards = model.learn(timesteps=1000)
    plt.plot(_rewards[1])
    plt.show()
    _,_, seq_theo, m_theo = optimize(v, R, q, M)
    payoff_theo = evaluate_sequence(seq_theo, v,R,q)

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
