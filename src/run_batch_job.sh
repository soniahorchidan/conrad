#!/bin/bash
#SBATCH -A naiss2026-4-106      # Project ID
#SBATCH -J conrad_experiment    # Name of the job
#SBATCH -N 2                    # Request 2 nodes
#SBATCH --gpus-per-node=1       # Request 1 GPU per node (2 total)
#SBATCH --mem=64G               # 64GB RAM per node
#SBATCH -t 01:00:00             # Time limit (HH:MM:SS)
#SBATCH -o logs/job_%j.out      # Standard output log (%j = Job ID)
#SBATCH -e logs/job_%j.err      # Error log
#SBATCH --mail-type=END,FAIL    # Send mail when job ends or fails
#SBATCH --mail-user=sfhor@kth.se

# --- Setup Environment ---
module purge
module load Python/3.10.4-bare-hpc1-gcc-2022a-eb
module load Java/17.0.6-hpc1-bdist
source /home/x_sonho/conrad/orb/venv/bin/activate

# --- Run the Experiment ---
# Use srun to ensure the code runs on BOTH nodes
srun python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 1000  --model_to_infer ThreeHopPipeline --disable-raps --calib_batch_size 12 --dataset fb15k-237