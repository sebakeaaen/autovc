#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J AutoVC
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- For requesting a GPU with 32GB of memory, then please add a
#BSUB -R "select[gpu32gb]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=32GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 128GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
#BSUB -gpu "num=1"
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# here follow the commands you want to execute 
source ~/miniconda3/bin/activate
conda activate py39

nvidia-smi
# Load the cuda module
module load cuda/10.2

module load gcc/9.2.0

python main.py --run_name ditte_stft_test