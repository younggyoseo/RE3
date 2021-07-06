for seed in 12345 23451 34512 45123 51234
do
    python train.py env=pendulum_swingup batch_size=512 action_repeat=2 logdir=runs_rad_re3 beta_init=0.05 seed=$seed
done
