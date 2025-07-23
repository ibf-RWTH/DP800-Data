#!/bin/bash 
#SBATCH --job-name=DP800_New
#SBATCH --output=DP800_postproc.txt 
#SBATCH --time=10:00:00 
#SBATCH --nodes=1 
##SBATCH --account=rwth0744
#SBATCH --mem-per-cpu=64G 
#SBATCH --cpus-per-task=1

#cd /rwthfs/rz/cluster/home/nf838244/IEHK/CPCal/V11_new

#chmod +x damask.sh

#apptainer exec /rwthfs/rz/SW/UTIL.common/singularity/damask-grid-alpha7 ./damask.sh

module load Python/3.10.4
python3 postproc_dist.py 
