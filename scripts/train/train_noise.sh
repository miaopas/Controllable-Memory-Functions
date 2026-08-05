python src/train.py trainer=gpu experiment=LF/train/s4 data=lf data.rho_name=exponential data.rho_param='{alpha:1}' +data.dt=0.05 +data.processing='noise'


python src/train.py trainer=gpu experiment=LF/train/rnn data=lf data.rho_name=exponential data.rho_param='{alpha:1}' +data.dt=0.05 +data.processing='noise'

python src/train.py trainer=gpu experiment=LF/train/s4 data=lf data.rho_name=exponential data.rho_param='{alpha:1}' +data.dt=0.05


python src/train.py trainer=gpu experiment=LF/train/gatedRNN data=lf data.rho_name=exponential data.rho_param='{alpha:1}' +data.dt=0.05
