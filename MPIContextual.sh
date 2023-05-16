#!/bin/bash
#SBATCH --account=def-weiqi
#SBATCH --ntasks=50
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --mail-user=zhangyanking00@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-9
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err\n

module load python/3.10
module load scipy-stack
srun python3 MPIContextual.py --part=$SLURM_ARRAY_TASK_ID
