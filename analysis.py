import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy
from matplotlib.transforms import Bbox
import seaborn as sns

param_for_method = {
    'OptimisticGreedy': 'initial_val',
    'EpsilonGreedy': 'epsilon',
    'DecayingEpsilonGreedy': 'c',
    'UCB': 'confidence_level',
    'DecayingUCB': 'confidence_level',
    'LinearDecayingUCB': 'confidence_level',
}

def mean_confidence_interval(data,confidence=0.95):
    n = len(data)
    avg_regret_data = np.mean(data,axis=0)
    std_regret_data = np.std(data,axis=0)
    h = scipy.stats.t.ppf((1+confidence)/2,n-1)*std_regret_data/np.sqrt(n)
    return avg_regret_data,avg_regret_data-h,avg_regret_data+h

def read_data(method,param_val,use_expected=True):
    _data_df = pd.read_csv('experiments/noncontextual_{}.csv'.format(method))
    param_name = param_for_method[method]
    # sort by param_val and reindex
    _data_df = _data_df.sort_values(by=[param_name,'run','time']).reset_index(drop=True)
    # find T , the number of time steps
    global T
    T = _data_df['time'].max()
    data_df = _data_df[_data_df[param_name]==param_val].reset_index(drop=True)
    if use_expected:
        data_regret = data_df['expected_regret'].values
    else:
        data_regret = data_df['realized_regret'].values
    data_regret = data_regret.reshape(-1,T)
    avg_regret_data,lower_regret_data,upper_regret_data = mean_confidence_interval(data_regret)
    return avg_regret_data,lower_regret_data,upper_regret_data

def process_data_for_plot(method_param :dict, use_expected = False):
    output = []
    for method,param_val in method_param.items():
        tmp_dict = {
            'method': method,
            'param_val': param_val,
            'use_expected': use_expected,
        }
        _avg, _lb, _ub = read_data(method,param_val,use_expected)
        tmp_dict['avg'] = _avg
        tmp_dict['lb'] = _lb
        tmp_dict['ub'] = _ub
        output.append(tmp_dict)
    return output

# find the total rewards for different param_val
def get_total_rewards(method:str):
    _data_df = pd.read_csv('experiments/noncontextual_{}.csv'.format(method))
    param_name = param_for_method[method]
    # sort by param_val and reindex
    _data_df = _data_df.sort_values(by=[param_name,'run','time']).reset_index(drop=True)
    # find T , the number of time steps
    global T
    T = _data_df['time'].max()
    # find all param_val
    param_vals = np.sort(_data_df[param_name].unique())
    # find total rewards
    total_rewards = []
    for param_val in param_vals:
        temp_df = _data_df[_data_df[param_name]==param_val].reset_index(drop=True)
        _rewards = temp_df['realized_payoff'].values
        _rewards = _rewards.reshape(-1,T)
        total_rewards.append(np.sum(_rewards,axis=1).mean())
    return param_vals,total_rewards

def plot_rewards_vs_param():
    for method in param_for_method.keys():
        param_vals,total_rewards = get_total_rewards(method)
        plt.plot(param_vals,total_rewards,label=method)
    plt.legend(loc='best',fancybox=False)
    plt.xlabel('Parameter Value')
    plt.ylabel('Total Rewards')
    plt.title('Cascading Bandit Experiment')
    plt.show()

# find optimal parameters for each algorithm
plot_rewards_vs_param()

# plot regret for different methods vs time
method_param = {
    'OptimisticGreedy': 0.9,
    'EpsilonGreedy': 0.1,
    'DecayingEpsilonGreedy': 1e-4,
    'DecayingUCB': 0.1,
}
data_for_plot = process_data_for_plot(method_param,use_expected=False)
# plot
plt.figure(figsize=(8,5))
for data in data_for_plot:
    plt.plot(data['avg'],label=data['method'])
    plt.fill_between(range(T),data['lb'],data['ub'],alpha=0.1)
plt.legend(loc='best',fancybox=False)
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title('Cascading Bandit Experiment')
plt.savefig('result/regret_noncontextual.pdf',bbox_inches='tight',pad_inches=0.05)
plt.show()

