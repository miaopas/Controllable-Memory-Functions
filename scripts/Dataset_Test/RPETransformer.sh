python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/exp/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/polynomial/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/impulse/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/airy/coarse trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
python src/train_multiple.py trainer=gpu experiment=Dataset/RPETransformer/impulse/PE_sweep trainer.max_epochs=1 trainer.min_epochs=1 +skip_confirm=true
