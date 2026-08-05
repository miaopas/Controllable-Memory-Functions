python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/exp/coarse data.T=512 logger.wandb.project=LSTM_exp_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/exp/coarse data.T=2048 logger.wandb.project=LSTM_exp_T2048 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/polynomial/coarse data.T=512 logger.wandb.project=LSTM_poly_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/polynomial/coarse data.T=2048 logger.wandb.project=LSTM_poly_T2048 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/impulse/coarse data.T=512 logger.wandb.project=LSTM_impulse_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/impulse/coarse data.T=2048 logger.wandb.project=LSTM_impulse_T2048 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/airy/coarse data.T=512 logger.wandb.project=LSTM_airy_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/airy/coarse data.T=2048 logger.wandb.project=LSTM_airy_T2048 +skip_confirm=true
