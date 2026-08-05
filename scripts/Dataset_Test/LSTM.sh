python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/exp/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/polynomial/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/impulse/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/airy/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true