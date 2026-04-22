#!/bin/bash --login
#SBATCH -p multicore_small # (or --partition=) Parallel job using cores on a single node
#SBATCH -n 12                # (or --ntasks=) Number of cores (2--40)
#SBATCH -N 1                # (or --nodes=) Number of nodes
#SBATCH -t 7-0
#SBATCH --job-name opi_wf
###########################
job=${SLURM_JOB_NAME}

job=$(echo ${job%%.*})
##########################

# Load the modulefile in a clean environment
module purge
module load apps/binapps/orca/6.1.0-avx2
module load apps/binapps/anaconda3/2024.10
######### Activate conda environment THIS NEEDS TO BE CHANGED TO YOUR ENV NAME ########
source activate opi

export RSH_COMMAND=ssh
##########################
export scratchlocation=/scratch

export localscratch=$HOME/localscratch

##########################

# Copy only the necessary stuff in submit directory to scratch directory

#########################
# Determine scratch strategy based on hostname
if [[ $HOSTNAME == *"csf3"* ]]; then
    echo "CSF3 detected - using TMPDIR approach"

    # Create calculation directory in TMPDIR (scratch on node)
    calcdir=$(mktemp -d $HOME/localscratch/orcacalc_$SLURM_JOB_ID-XXXX)
    echo "Calculation directory: $calcdir"


    # Start ORCA job
    echo "RUNNING OPI SCRIPT"
    ########### THIS NEEDS TO BE CHANGED TO YOUR ENV NAME ########
    $HOME/.conda/envs/opi/bin/python opi_wf.py -c $SLURM_NTASKS -wrkd $calcdir -subd $SLURM_SUBMIT_DIR 
    echo "COPYING RESULTS DIR"
    cp -r $calcdir $SLURM_SUBMIT_DIR

    # Clean up calculation directory
    echo "Cleaning up calculation directory: $calcdir"
    rm -rf $calcdir
fi
echo "DONE"
# Copy important output files back to submit directory

