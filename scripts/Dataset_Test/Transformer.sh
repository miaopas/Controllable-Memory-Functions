python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/exp/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/polynomial/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/impulse/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/Transformer/airy/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
