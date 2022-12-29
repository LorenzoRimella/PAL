#$ -S /bin/bash
#$ -q long
#$ -l ngpus=1
#$ -l ncpus=1
#$ -l h_vmem=32G
#$ -l h_rt=72:00:00

#$ -N PAL-M

source /etc/profile

module add cuda
module add anaconda3

source activate tf-gpu

python Measles/experiment.py
