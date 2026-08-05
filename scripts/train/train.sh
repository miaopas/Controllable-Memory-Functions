#!/bin/bash


poetry run python src/train.py trainer=gpu trainer.devices=1 experiment=Train/LSTM data=lf data.rho_name=impulse data.rho_param='{alpha:1}' +skip_confirm=true


python src/train.py trainer=gpu trainer.devices=1 experiment=Train/ContinuousTransformer data=lf data.rho_name=impulse data.rho_param='{alpha:1}' +skip_confirm=true


python src/train.py trainer=gpu trainer.devices=1 experiment=Train/Transformer data=lf data.rho_name=impulse data.rho_param='{alpha:1}' +skip_confirm=true




python src/train.py trainer=gpu trainer.devices=1 experiment=Train/SSM