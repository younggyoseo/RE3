for seed in 12345 23451 34512 45123 51234
do
    python train.py env=quadruped_run batch_size=512 action_repeat=4 num_seed_steps=10000 logdir=runs_rad use_state_entropy=false seed=$seed
done