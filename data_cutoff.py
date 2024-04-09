from args import args
# from utils import log
import h5py

path_data = '/root/Surface_code_and_Toric_code/{}_pe/'.format(args.c_type)
filename_read_data = '{}_d{}_p{}_trnsz{}_seed0.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.poolsz)
filename_write_data = '{}_d{}_p{}_trnsz{}_seed0.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.trnsz)

# log("Cut off...")
with h5py.File(path_data + filename_read_data, 'r') as f:
    syndromes = f['syndromes'][()]
    logical_errors = f['logical_errors'][()]
    num = syndromes.shape[0]
with h5py.File(path_data + filename_write_data, 'w') as f:
    syndromes_cutoff = f.create_dataset('syndromes', data=syndromes[:num//2],
                                     chunks=True, compression="gzip")
    logical_errors_cutoff = f.create_dataset('logical_errors', data=logical_errors[:num//2], chunks=True, compression="gzip")
# log("Cut off... Done.")
# python3 data_cutoff.py --c_type torc --d 5 --p 0.04 --poolsz 10000000 --trnsz 5000000
# python3 data_cutoff.py --c_type torc --d 3 --p 0.01 --poolsz 10000000 --trnsz 5000000
# python3 data_cutoff.py --c_type torc --d 5 --p 0.01 --poolsz 10000000 --trnsz 5000000

