#!/bin/bash 
#SBATCH --job-name=DP800_New
#SBATCH --output=DP800_New.txt 
#SBATCH --time=10:00:00 
#SBATCH --nodes=1 
##SBATCH --account=rwth0744
#SBATCH --mem-per-cpu=2G 
#SBATCH --cpus-per-task=32

#cd /rwthfs/rz/cluster/home/nf838244/IEHK/CPCal/V11_new

chmod +x damask.sh

apptainer exec docker://eisenforschung/damask-grid:3.0.0-alpha7 ./damask.sh

sbatch batch_2.sh
