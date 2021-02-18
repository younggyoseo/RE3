for seed in 12345 23451 34512 45123 51234
do
    python train.py env=walker_run_sparse batch_size=512 action_repeat=2 logdir=runs_rad_re3 beta_decay=0.000025 seed=$seed
done