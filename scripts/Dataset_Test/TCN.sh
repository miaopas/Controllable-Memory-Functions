python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/exp/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/polynomial/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/impulse/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/airy/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
