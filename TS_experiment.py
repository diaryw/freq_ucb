import os
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from base import RecommendationEnv, evaluate_sequence, load_noncontextual
from frequency import UCB, regret_analysis, optimize
from benchmark import EpsilonGreedy, TS_Cascade
from time import time

N,M,v,R,q = load_noncontextual()
D = 200
T = 30000

def run_and_save(method_cls:callable,method_name:str,param_name:str,param_val,run_id:int):
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
        'run' : run_id,
        'time' : range(1,T+1),
        'expected_payoff': expected_payoff,
        'realized_payoff': realized_payoff,
        'max_payoff': payoff_theo,
        'expected_regret': expected_regret,
        'realized_regret': realized_regret,
        param_name: param_val,
    }
    data_df = pd.DataFrame(_data)
    # name of save file
    save_filename = 'experiments' + os.sep + 'noncontextual_'+method_name + f'_task={run_id}.csv'
    # save 
    data_df.to_csv(save_filename,index=False,mode='a',header=not os.path.exists(save_filename))
    end = time()
    print('Running time', end-start)

kwargs_list = []
for c in [0.01,0.1,0.2,1.0,2.0]:
    for run_id in range(100):
        temp_kwargs = {
            'method_cls': TS_Cascade,
            'method_name' : 'TS-Cascade',
            'param_name' : 'std_c',
            'param_val' : c,
            'run_id' : run_id,
        }
        temp_kwargs = [TS_Cascade, 'TS-Cascade', 'std_c', c, run_id]
        kwargs_list.append(temp_kwargs)

if __name__=='__main__':
    number_processes = 124
    with multiprocessing.Pool(number_processes) as pool:
        pool.starmap(run_and_save, kwargs_list)


