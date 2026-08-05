python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/exp/coarse data.T=64 module.layers.Layer1.src_length=64 logger.wandb.project=Transformer_exp_T64 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/exp/coarse data.T=256 module.layers.Layer1.src_length=256 logger.wandb.project=Transformer_exp_T256 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/polynomial/coarse data.T=64 module.layers.Layer1.src_length=64 logger.wandb.project=Transformer_poly_T64 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/polynomial/coarse data.T=256 module.layers.Layer1.src_length=256 logger.wandb.project=Transformer_poly_T256 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/impulse/coarse data.T=64 module.layers.Layer1.src_length=64 logger.wandb.project=Transformer_impulse_T64 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/impulse/coarse data.T=256 module.layers.Layer1.src_length=256 logger.wandb.project=Transformer_impulse_T256 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/airy/coarse data.T=64 module.layers.Layer1.src_length=64 logger.wandb.project=Transformer_airy_T64 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/airy/coarse data.T=256 module.layers.Layer1.src_length=256 logger.wandb.project=Transformer_airy_T256 +skip_confirm=true
