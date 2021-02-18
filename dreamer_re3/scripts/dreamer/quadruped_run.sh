for seed in 12345 23451 34512 45123 51234
do
    python dreamer.py --logdir ./logdir/dmc_quadruped_run/dreamer/$seed --task dmc_quadruped_run --action_repeat 4 --precision 32 --beta 0.0 --seed $seed
done