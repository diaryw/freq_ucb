import os
import argparse
import numpy as np
import pandas as pd
from base import ContextualEnv
from frequency import ContextualUCB
from benchmark import ContextualEpsilonGreedy, ContextualDecayingGreedy, ContextualETC
from time import time

MPItaskID = int(os.environ['SLURM_PROCID'])

T = 3000
num_parts = 9

def run_and_save_UCB(gamma1,gamma2):
    start = time()
    env = ContextualEnv(seed=2023)
    model = ContextualUCB(env=env,gamma1=gamma1,gamma2=gamma2)
    result = model.learn(timesteps=T)
    _, expected_payoff, realized_payoff, optimal_payoff = result
    expected_regret = np.cumsum(optimal_payoff - expected_payoff)
    realized_regret = np.cumsum(optimal_payoff - realized_payoff)
    _data = {
        'run':MPItaskID,
        'time':range(1,T+1),
        'expected_payoff': expected_payoff,
        'realized_payoff': realized_payoff,
        'max_payoff': optimal_payoff,
        'expected_regret': expected_regret,
        'realized_regret': realized_regret,
        'gamma1': gamma1,
        'gamma2': gamma2,
    }
    data_df = pd.DataFrame(_data)
    # name of save file
    save_filename = 'experiments' + os.sep + 'contextual_ContextualUCB' + f'_task={MPItaskID}.csv'
    # save 
    data_df.to_csv(save_filename,index=False,mode='a',header=not os.path.exists(save_filename))
    end = time()
    print('Running time', end-start)

def run_and_save(method_cls:callable,method_name:str,param_name:str,param_val):
    start = time()
    param_dict = {param_name: param_val}

    env = ContextualEnv(seed=2023)
    model = method_cls(env = env, **param_dict)
    # result = [ , , ,]
    result = model.learn(timesteps=T)
    _, expected_payoff, realized_payoff, optimal_payoff = result
    expected_regret = np.cumsum(optimal_payoff - expected_payoff)
    realized_regret = np.cumsum(optimal_payoff - realized_payoff)
    # save the data
    _data = {
        'run':MPItaskID,
        'time':range(1,T+1),
        'expected_payoff': expected_payoff,
        'realized_payoff': realized_payoff,
        'max_payoff': optimal_payoff,
        'expected_regret': expected_regret,
        'realized_regret': realized_regret,
        param_name: param_val,
    }
    data_df = pd.DataFrame(_data)
    # name of save file
    save_filename = 'experiments' + os.sep + 'contextual_'+method_name + f'_task={MPItaskID}.csv'
    # save 
    data_df.to_csv(save_filename,index=False,mode='a',header=not os.path.exists(save_filename))
    end = time()
    print('Running time', end-start)

gamma1_list = [10**i for i in range(-5,3)]
gamma2_list = [10**i for i in range(-5,3)]
commit_time_list = [10,100,500]
epsilon_list = [0.01,0.05,0.1]
decaying_list = [1,10,100]

kwargs_list = []
"""
# UCB
for gamma1 in gamma1_list:
    for gamma2 in gamma2_list:
        temp_kwargs = {
            'gamma1': gamma1,
            'gamma2': gamma2,
        }
        kwargs_list.append(temp_kwargs)
"""

# ETC
for commit_time in commit_time_list:
    temp_kwargs = {
        'method_cls': ContextualETC,
        'method_name': 'ETC',
        'param_name': 'commit_time',
        'param_val': commit_time,
    }
    kwargs_list.append(temp_kwargs)

for epsilon in epsilon_list:
    temp_kwargs = {
        'method_cls': ContextualEpsilonGreedy,
        'method_name': 'EpsilonGreedy',
        'param_name': 'epsilon',
        'param_val': epsilon,
    }
    kwargs_list.append(temp_kwargs)

for c in decaying_list:
    temp_kwargs = {
        'method_cls': ContextualDecayingGreedy,
        'method_name': 'DecayingGreedy',
        'param_name': 'c',
        'param_val': c,
    }
    kwargs_list.append(temp_kwargs)


if __name__=='__main__':
    # 60 tasks (each MPI for 200 times) in total, each task run loop_num times
    # split them into 6 parts, 10 tasks in each part, thus 10 runs in each part, (MPI for 200 times)
    parser = argparse.ArgumentParser(description='Input the part ID.')
    parser.add_argument('--part',type=int,default=0)
    args = parser.parse_args()
    part_id = args.part
    # split the args_list into parts of num_parts
    kwargs_list_part = kwargs_list[(part_id)*len(kwargs_list)//num_parts:(part_id+1)*len(kwargs_list)//num_parts]

    for kwargs_ in kwargs_list_part:
        run_and_save(**kwargs_)

