python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/exp/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/polynomial/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/impulse/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/airy/coarse +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/impulse/PE_sweep +skip_confirm=true
