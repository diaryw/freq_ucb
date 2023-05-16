#!/bin/bash
#SBATCH --account=def-weiqi
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mail-user=zhangyanking00@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err\n

module load python/3.10
module load scipy-stack
python3 mergeresult.py

find ./experiments/ ! -name '*task*' -type f -exec tar -cvf experiments.tar {} +
