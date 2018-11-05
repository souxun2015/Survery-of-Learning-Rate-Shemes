#!/usr/bin/env bash

python lr_with_momentumoptimizer.py --lr_scheme exponential_decay
python lr_with_momentumoptimizer.py --lr_scheme piecewise_constant
python lr_with_momentumoptimizer.py --lr_scheme polynomial_decay
python lr_with_momentumoptimizer.py --lr_scheme natural_exp_decay
python lr_with_momentumoptimizer.py --lr_scheme inverse_time_decay
python lr_with_momentumoptimizer.py --lr_scheme cosine_decay
python lr_with_momentumoptimizer.py --lr_scheme cosine_decay_restarts
python lr_with_momentumoptimizer.py --lr_scheme linear_cosine_decay
python lr_with_momentumoptimizer.py --lr_scheme noisy_linear_cosine_decay
