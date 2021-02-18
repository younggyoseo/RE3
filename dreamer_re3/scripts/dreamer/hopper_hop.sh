for seed in 12345 23451 34512 45123 51234
do
    python dreamer.py --logdir ./logdir/dmc_hopper_hop/dreamer/$seed --task dmc_hopper_hop --precision 32 --beta 0.0 --seed $seed
done