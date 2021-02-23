# State Entropy Maximization with Random Encoders for Efficient Exploration (RE3)

Code for [State Entropy Maximization with Random Encoders for Efficient Exploration](https://arxiv.org/abs/2102.09430)

In this repository, we provide code for RE3 algorithm described in the paper linked above. We provide code in three sub-directories: `rad_re3` containing code for the combination of RE3 and RAD, `dreamer_re3` containing code for the combination of RE3 and Dreamer, and `a2c_re3` containing code for the combination of RE3 and A2C.

We also provide raw data(.csv) and code for visualization in the `data` directory.

If you find this repository useful for your research, please cite:
```
@article{seo2021state,
  title={State Entropy Maximization with Random Encoders for Efficient Exploration},
  author={Seo, Younggyo and Chen, Lili and Shin, Jinwoo and Lee, Honglak and Abbeel, Pieter and Lee, Kimin},
  journal={arXiv preprint arXiv:2102.09430},
  year={2021}
}
```

# RAD + RE3
Our code is built on top of the [DrQ](https://github.com/denisyarats/drq) repository. 

## Installation
You could install all dependencies by following command:

```
conda env install -f conda_env.yml
```

You should also install custom version of `dm_control` to run experiments on `Walker Run Sparse` and `Cheetah Run Sparse`. You could do this by following command:

```
cd ../envs/dm_control
pip install .
```

## Instructions
### RAD
```
python train.py env=hopper_hop batch_size=512 action_repeat=2 logdir=runs_rad_re3 use_state_entropy=false
```

### RAD + RE3
```
python train.py env=hopper_hop batch_size=512 action_repeat=2 logdir=runs_rad_re3
```

We provide all scripts to reproduce Figure 4 (RAD, RAD + RE3) in `scripts` directory.


# Dreamer + RE3
Our code is built on top of the [Dreamer](https://github.com/danijar/dreamer) repository.

## Installation

You could install all dependencies by following command:

```
pip3 install --user tensorflow-gpu==2.2.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib

# Install custom dm_control environments for walker_run_sparse / cheetah_run_sparse
cd ../envs/dm_control
pip3 install .
```

## Instructions
### Dreamer
```
python dreamer.py --logdir ./logdir/dmc_pendulum_swingup/dreamer/12345 --task dmc_pendulum_swingup --precision 32 --beta 0.0 --seed 12345
```

### Dreamer + RE3
```
python dreamer.py --logdir ./logdir/dmc_pendulum_swingup/dreamer_re3/12345 --task dmc_pendulum_swingup --precision 32 --k 53 --beta 0.1 --seed 12345
```

We provide all scripts to reproduce Figure 4 (Dreamer, Dreamer + RE3) in `scripts` directory.

# A2C + RE3
Training code can be found in `rl-starter-files` directory, which is forked from [rl-starter-files](https://github.com/lcswillems/rl-starter-files), which uses a modified A2C implementation from [torch-ac](https://github.com/lcswillems/torch-ac). Note that currently there is only support for A2C.

## Installation 

All of the dependencies are in the `requirements.txt` file in `rl-starter-files`. They can be installed manually or with the following command:

```
pip3 install -r requirements.txt
```

You will also need to install our cloned version of `torch-ac` with these commands:

```
cd torch-ac
pip3 install -e .
```

## Instructions
See instructions in `rl-starter-files` directory. Example scripts can be found in `rl-starter-files/rl-starter-files/run_sent.sh`.
