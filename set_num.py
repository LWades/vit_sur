import h5py
from args import args
import sys
from os.path import abspath, dirname

sys.path.append(abspath(dirname(__file__)).strip('decoding'))
from module import log

with h5py.File(args.file, 'r') as f:
    log(f[args.key][()].shape[0])
# python3 set_num.py --file '/root/Surface_code_and_Toric_code/sur_pe/sur_d13_p0.200_trnsz10000000_imgsdr_seed0.hdf5' --key image_syndromes
