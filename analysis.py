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
    #'DecayingUCB': 'confidence_level',
    #'LinearDecayingUCB': 'confidence_level',
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
    plt.savefig('result/testparam.pdf',bbox_inches='tight',pad_inches=0.05)
    plt.show()

# find optimal parameters for each algorithm
#plot_rewards_vs_param()

# plot regret for different methods vs time
method_param = {
    'OptimisticGreedy': 0.9,
    'EpsilonGreedy': 0.01,
    'DecayingEpsilonGreedy': 3.0,
    'UCB': 0.2,
    #'DecayingUCB': 1.0,
    #'LinearDecayingUCB': 400,
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
plt.title('Noncontextual Experiments')
plt.savefig('result/regret_noncontextual.pdf',bbox_inches='tight',pad_inches=0.05)
plt.show()

### contextual
def test_contextual_ucb():
    method = 'ContextualUCB'
    _data_df = pd.read_csv('experiments/contextual_{}.csv'.format(method))
    _data_df = _data_df.sort_values(by=['gamma1','gamma2','run','time']).reset_index(drop=True)
    # find T
    T = _data_df['time'].max()
    # find all param_val
    unique_val = _data_df[['gamma1','gamma2']].copy().drop_duplicates().reset_index(drop=True)
    # how to drop some values
    unique_val = unique_val.drop([1,6,8,9])
    # iterate over the unique values
    avg_regret_data_list = []
    param_list = []
    plt.figure(figsize=(8,5))
    for i in range(unique_val.shape[0]):
        gamma1,gamma2 = unique_val.iloc[i]
        data_df = _data_df[(_data_df['gamma1']==gamma1) & (_data_df['gamma2']==gamma2)].reset_index(drop=True)
        data_regret = data_df['expected_regret'].values
        data_regret = data_regret.reshape(-1,T)
        avg_regret_data,lower_regret_data,upper_regret_data = mean_confidence_interval(data_regret)
        avg_regret_data_list.append(avg_regret_data)
        param_list.append({'gamma1':gamma1,'gamma2':gamma2})
        plt.plot(avg_regret_data,label='gamma1={},gamma2={}'.format(gamma1,gamma2))
    plt.legend(loc='best',fancybox=False)
    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.title('Contextual Experiments')
    plt.show()
    # find the best parameter
    best_param = param_list[np.argmin(np.array(avg_regret_data_list)[:,-1])]
    optimal_data_df = _data_df[(_data_df['gamma1']==best_param['gamma1']) & (_data_df['gamma2']==best_param['gamma2'])].reset_index(drop=True)
    # save the data for the best parameter
    #optimal_data_df.to_csv('experiments/contextual_{}_best.csv'.format(method),index=False)
    return best_param

best_param = test_contextual_ucb()

