python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/exp/multi_layer_large_epoch +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/polynomial/multi_layer_large_epoch +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/impulse/multi_layer_large_epoch +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/LSTM/airy/multi_layer_large_epoch +skip_confirm=true
