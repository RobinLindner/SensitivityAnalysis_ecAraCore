#!/bin/bash
#SBATCH --job-name=Flux_sampling           # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G            # Memory per node (specify how much memory you need per node)
#SBATCH --time=120:00:00             # Walltime (time limit)
#SBATCH --output=Debug/run_%A_%a.out      # Standard output log file
#SBATCH --error=Debug/run_%A_%a.err       # Standard error log file
#SBATCH --mail-user=lindner5@uni-potsdam.de
#SBATCH --mail-type=all

temperatures=({10..40})
alpha=0.95
samplesizes=(100 1000 10000) # vary this between 100 and 10000
temp=${temperatures[$SLURM_ARRAY_TASK_ID]}

for n_samples in "${samplesizes[@]}"; do
    output_file="results/tables/FluxSampling/n${n_samples}/fluxes_nsamples.${n_samples}_vbio.${alpha}_${temp}.csv"
    if [ -f "$output_file" ]
        then
        echo "Output exists. Skipping iteration"
        continue
    fi

    echo "Processing Temperature: $temp"
    python3 scripts/dataGen/FluxSamplingAtTemp.py ${n_samples} 4 ${alpha} ${temp}

done