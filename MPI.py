import os
import argparse
import numpy as np
import pandas as pd
from base import RecommendationEnv, evaluate_sequence, load_noncontextual
from frequency import UCB, regret_analysis, optimize
from benchmark import EpsilonGreedy
from time import time

MPItaskID = int(os.environ['SLURM_PROCID'])

N,M,v,R,q = load_noncontextual()
D = 200
T = 30000
num_parts = 8

def run_and_save(method_cls:callable,method_name:str,param_name:str,param_val):
    start = time()
    param_dict = {param_name: param_val}

    env = RecommendationEnv(
        num_message=N,
        num_maxsent=M,
        attraction_prob=v,
        reward_per_message=R,
        q_prob=q,
        time_window=D,
    )

    model = method_cls(env = env, **param_dict)
    # result = [ , , ,]
    result = model.learn(timesteps=T)
    _, expected_payoff, realized_payoff = result
    # compute regret
    expected_regret = regret_analysis(v,R,q,M,expected_payoff)
    realized_regret = regret_analysis(v,R,q,M,realized_payoff)
    # maximal payoff
    _,_, seq_theo, _ = optimize(v, R, q, M)
    payoff_theo = evaluate_sequence(seq_theo, v,R,q)
    # save the data
    _data = {
        'run':MPItaskID,
        'time':range(1,T+1),
        'expected_payoff': expected_payoff,
        'realized_payoff': realized_payoff,
        'max_payoff': payoff_theo,
        'expected_regret': expected_regret,
        'realized_regret': realized_regret,
        param_name: param_val,
    }
    data_df = pd.DataFrame(_data)
    # name of save file
    save_filename = 'experiments' + os.sep + 'noncontextual_'+method_name + f'_task={MPItaskID}.csv'
    # save 
    data_df.to_csv(save_filename,index=False,mode='a',header=not os.path.exists(save_filename))
    end = time()
    print('Running time', end-start)

confidence_level_list = [0] + [x*10**i for i in range(-5,2) for x in range(1,6) ]
epsilon_list = np.arange(0,21)*0.01
c_list = [x*10**i for i in range(-2,2) for x in range(1,4)]
initial_list = np.arange(5,11)*0.1

kwargs_list = []
# UCB
for confidence_level in confidence_level_list:
    temp_kwargs = {
        'method_cls': UCB,
        'method_name': 'UCB',
        'param_name': 'confidence_level',
        'param_val': confidence_level,
    }
    kwargs_list.append(temp_kwargs)

# decaying epsilon greedy
for c in c_list:
    temp_kwargs = {
        'method_cls': EpsilonGreedy,
        'method_name': 'DecayingEpsilonGreedy',
        'param_name': 'c',
        'param_val': c,
    }
    kwargs_list.append(temp_kwargs)

# epsilon greedy
for epsilon in epsilon_list:
    temp_kwargs = {
        'method_cls': EpsilonGreedy,
        'method_name': 'EpsilonGreedy',
        'param_name': 'epsilon',
        'param_val': epsilon,
    }
    kwargs_list.append(temp_kwargs)


# greedy with optimistic initialzation
for initial_val in initial_list:
    temp_kwargs = {
        'method_cls': EpsilonGreedy,
        'method_name': 'OptimisticGreedy',
        'param_name': 'initial_val',
        'param_val': initial_val,
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

