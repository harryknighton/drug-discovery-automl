for name in datasets/*; do
  sbatch src/scripts/slurm.sh baseline "$name" "$1";
done