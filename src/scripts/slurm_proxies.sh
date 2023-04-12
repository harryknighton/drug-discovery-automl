for proxy in num_params synflow gradnorm jacobcov snip grasp fisher zico ensemble; do
   sbatch src/scripts/slurm.sh hyperopt_proxy_"$proxy" "$1" "$2";
done