import pandas as pd
import glob
import os

def merge_files(filenamepattern):
    # setting the path for joining multiple files
    filepath = os.path.join('./experiments/',filenamepattern+'_task=*.csv')

    # list of merged files returned
    files = glob.glob(filepath)

    if len(files)>0:
        print(filenamepattern+', {} files are found'.format(len(files)))

        # joining files with concat and read_csv
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df2 = df.sort_values(['run','time'], ascending=[True, True],ignore_index=True)

        # save to file
        df2.to_csv('./experiments/{}.csv'.format(filenamepattern),index=False)
        print('Merge completed.')
    else:
        print(filenamepattern, 'No files found.')

method_list = ['UCB','DecayingEpsilonGreedy','EpsilonGreedy','OptimisticGreedy']
for method in method_list:
    merge_files(f'noncontextual_{method}')

method_list = ['ContextualUCB','ETC','EpsilonGreedy','DecayingGreedy']
for method in method_list:
    merge_files(f'contextual_{method}')

