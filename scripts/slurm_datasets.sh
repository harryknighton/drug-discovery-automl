for name in $(ls datasets); do
  sbatch src/scripts/slurm.sh "$1" "$name" "$2";
done