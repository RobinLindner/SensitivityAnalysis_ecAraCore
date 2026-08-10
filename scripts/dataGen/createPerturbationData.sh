#!/bin/bash
#SBATCH --job-name=vary_kcats           # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --nodelist=n-hpc-ca1        # specific node with gurobi licence
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --mem-per-cpu=4G            # Memory per node (specify how much memory you need per node)
#SBATCH --time=120:00:00             # Walltime (time limit)
#SBATCH --output=vary_kcats.out      # Standard output log file
#SBATCH --error=vary_kcats.err       # Standard error log file
#SBATCH --mail-user=lindner5@uni-potsdam.de
#SBATCH --mail-type=all


temperatures=({10..40})
n_samples=1000
max_change=0.025
temp=${temperatures[$SLURM_ARRAY_TASK_ID]}

output_file="results/tables/PerturbationAnalysis/temperatureSpecific/PerturbedESC_${temp}.tsv"
if [ -f "$output_file" ]
then
echo "Output exists. Skipping iteration"
exit 0
fi

echo "Processing Temperature: $temp"
python3 scripts/dataGen/createPerturbationData.py ${n_samples} ${max_change} ${temp}