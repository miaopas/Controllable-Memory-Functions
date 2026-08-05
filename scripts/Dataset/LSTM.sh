python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/exp/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/polynomial/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/impulse/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/airy/coarse +skip_confirm=true