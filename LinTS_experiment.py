import os
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from base import ContextualEnv, FixedContextGenerator
from benchmark import LinTS_Cascade
from time import time

T = 3000

def run_and_save(method_cls:callable,method_name:str,param_name:str,param_val,run_id:int):
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
        'run':run_id,
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
    save_filename = 'experiments' + os.sep + 'contextual_'+method_name + f'_task={run_id}.csv'
    # save 
    data_df.to_csv(save_filename,index=False,mode='a',header=not os.path.exists(save_filename))
    end = time()
    print('Running time', end-start)

kwargs_list = []
for c in [1e-3,1e-2,1e-1]:
    for run_id in range(50):
        temp_kwargs = [LinTS_Cascade, 'LinTS-Cascade', 'std_c', c, run_id]
        kwargs_list.append(temp_kwargs)

if __name__=='__main__':
    number_processes = 50
    with multiprocessing.Pool(number_processes) as pool:
        pool.starmap(run_and_save, kwargs_list)

