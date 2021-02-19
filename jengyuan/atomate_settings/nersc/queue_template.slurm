#!/bin/bash -l

#SBATCH --nodes=$${nodes}
#SBATCH --ntasks=$${ntasks}
#SBATCH --ntasks-per-node=$${ntasks_per_node}
#SBATCH --cpus-per-task=$${cpus_per_task}
#SBATCH --gres=$${gres}
#SBATCH --qos=$${qos}
#SBATCH --time=$${walltime}
#SBATCH --partition=$${queue}
#SBATCH --account=$${account}
#SBATCH --job-name=$${job_name}
#SBATCH --license=$${license}
#SBATCH --output=$${job_name}-%j.out
#SBATCH --error=$${job_name}-%j.error
#SBATCH --constraint=$${constraint}

ssh -N -f -L localhost:12345:155.247.53.219:27017 tug03990@shell.cst.temple.edu
ssh -N -f -L localhost:12346:155.247.53.219:22 tug03990@shell.cst.temple.edu
# change "tug03990@shell.cst.temple.edu" to your account on Owlsnesttwo, and
# Make sure you are able to SSH from Cori to Owlsnesttwo without passwords
conda activate wf
export OMP_NUM_THREADS=8
cd $${launch_dir}
$${pre_rocket}
$${rocket_launch}
$${post_rocket}

# CommonAdapter (SLURM) completed writing Template