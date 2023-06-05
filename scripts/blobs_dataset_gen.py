# Usage:
# python blobs_dataset_generator.py --datasets_path=/datasets/<username>/blobs_dataset --n_index_rows=16000 --n_parts=16


import sys
import os
import argparse
import numpy as np
#import cupy as cp
from sklearn.datasets import make_blobs


def gen_blob_dataset(datasets_path, n_rows, n_cols, n_centers, n_parts, dtype=np.float32):
    gen_dir = '{}/{}_{}_{}_{}'.format(datasets_path, n_rows, n_cols, n_centers, n_parts)
    os.mkdir(gen_dir)

    print('Generating dataset at :', gen_dir)

    X, _ = make_blobs(n_samples=n_rows,
                      n_features=n_cols,
                      centers=n_centers,
                      shuffle=True)

    blobs = X[:n_rows].astype(dtype)

    print("Outputting dataset parts")

    chunks = np.array_split(X, n_parts)
    for i, chunk in enumerate(chunks):
        with open('{}/part_{}.npy'.format(gen_dir, i), 'wb') as f:
            np.save(f, chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate partitioned isotropic blobs dataset')
    parser.add_argument('-datasets_path','--datasets_path', help='Path to precomputed datasets', required=True, type=str)
    parser.add_argument('-n_rows','--n_rows', help='Number of rows', required=True, type=int)
    parser.add_argument('-n_cols', '--n_cols', help='Number of columns', required=True, type=int)
    parser.add_argument('-n_centers', '--n_centers', help='Number of cluster centers', required=True, type=int)
    parser.add_argument('-n_parts','--n_parts', help='Number of workers or index chunks', required=True, type=int)
    args = vars(parser.parse_args())

    gen_blob_dataset(args['datasets_path'], args['n_rows'], args['n_cols'], args['n_centers'],  args['n_parts'])

