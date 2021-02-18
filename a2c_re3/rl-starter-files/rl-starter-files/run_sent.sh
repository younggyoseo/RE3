# A2C + RE3
for task in 'DoorKey-6x6'
do
    for seed in 1 2 3 4 5
    do
        CUDA_VISIBLE_DEVICES=4 python3 -m scripts.train --algo a2c --env MiniGrid-$task-v0 --model $task/MiniGrid-$task-v0-sent-0.005-$seed \
        --save-interval 100 --frames 400000 --use_entropy_reward --seed $seed --beta 0.005
    done
done