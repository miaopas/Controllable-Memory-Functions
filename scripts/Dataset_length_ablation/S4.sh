python src/train_multiple.py trainer=gpu experiment=Dataset/S4/exp/coarse data.T=512 logger.wandb.project=S4_exp_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/exp/coarse data.T=2048 logger.wandb.project=S4_exp_T2048 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/polynomial/coarse data.T=512 logger.wandb.project=S4_poly_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/polynomial/coarse data.T=2048 logger.wandb.project=S4_poly_T2048 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/impulse/coarse data.T=512 logger.wandb.project=S4_impulse_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/impulse/coarse data.T=2048 logger.wandb.project=S4_impulse_T2048 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/airy/coarse data.T=512 logger.wandb.project=S4_airy_T512 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/S4/airy/coarse data.T=2048 logger.wandb.project=S4_airy_T2048 +skip_confirm=true
