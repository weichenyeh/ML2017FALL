#! /bin/sh
wget -O ke4_model.02-0.8162.h5 'https://www.dropbox.com/s/zq8ax847wksac01/ke4_model.02-0.8162.h5?dl=1'
python3 ke4_predict_deli.py $@

