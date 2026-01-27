#!/bin/bash
#SBATCH -A naiss2026-4-106      # Project ID
#SBATCH -J conrad_experiment    # Name of the job
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --mem=32G               # 32GB RAM per node
#SBATCH -t 06:00:00             # Time limit (HH:MM:SS)
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
# Start Neo4j, run your script, then ensure Neo4j stops
neo4j start
sleep 10 # Give Neo4j a moment to initialize

srun ./run.sh

neo4j stop