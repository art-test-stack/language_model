#!/bin/sh
#SBATCH -J llm
#SBATCH --partition=GPUQ
#SBATCH --time=0-60:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --output=logs/output2.out
### SBATCH --mail-user=$EMAIL
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
module avail Python
module load Python/3.11.3-GCCcore-12.3.0
source /cluster/home/ajtestar/lmenv/bin/activate
torchrun --nproc_per_node=${2:-4} prl_train.py