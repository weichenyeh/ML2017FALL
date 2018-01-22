#!/bin/sh
python3 test.py submit/Model.01-0.7805-0.7537-0.5007.hdf5 $1 
python3 test.py submit/Model.01-0.7807-0.7543-0.5002.hdf5 $1
python3 test.py submit/Model.01-0.7810-0.7540-0.5013.hdf5 $1
python3 test.py submit/Model.01-0.7812-0.7561-0.4977.hdf5 $1
python3 test.py submit/Model.01-0.7805-0.7548-0.4992.hdf5 $1
python3 test.py submit/Model.01-0.7807-0.7544-0.4998.hdf5 $1
python3 test.py submit/Model.01-0.7811-0.7525-0.5023.hdf5 $1
python3 test.py submit/Model.01-0.7814-0.7560-0.4995.hdf5 $1
python3 test.py submit/Model.01-0.7805-0.7555-0.4996.hdf5 $1
python3 test.py submit/Model.01-0.7812-0.7531-0.5048.hdf5 $1

python3 ensemble.py results/Model.01-0.7805-0.7537-0.5007.csv results/Model.01-0.7807-0.7543-0.5002.csv results/Model.01-0.7810-0.7540-0.5013.csv results/Model.01-0.7812-0.7561-0.4977.csv results/Model.01-0.7805-0.7548-0.4992.csv results/Model.01-0.7807-0.7544-0.4998.csv results/Model.01-0.7811-0.7525-0.5023.csv results/Model.01-0.7814-0.7560-0.4995.csv results/Model.01-0.7805-0.7555-0.4996.csv results/Model.01-0.7812-0.7531-0.5048.csv final.csv
