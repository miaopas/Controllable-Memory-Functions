python src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/exp/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/polynomial/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/impulse/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/airy/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
