python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/exp/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/polynomial/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/impulse/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/TCN/airy/coarse +skip_confirm=true