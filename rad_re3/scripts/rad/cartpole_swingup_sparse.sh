for seed in 12345 23451 34512 45123 51234
do
    python train.py env=cartpole_swingup_sparse batch_size=512 action_repeat=2 logdir=runs_rad use_state_entropy=false seed=$seed
done