# Usage:
# python blobs_dataset_generator.py --datasets_path=/datasets/<username>/blobs_dataset --n_index_rows=16000 --n_parts=16


import sys
import os
import argparse
import numpy as np
from legate.raft.datasets import gen_blob_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate partitioned isotropic blobs dataset')
    parser.add_argument('-datasets_path','--datasets_path', help='Path to precomputed datasets', required=True, type=str)
    parser.add_argument('-n_rows','--n_rows', help='Number of rows', required=True, type=int)
    parser.add_argument('-n_cols', '--n_cols', help='Number of columns', required=True, type=int)
    parser.add_argument('-n_centers', '--n_centers', help='Number of cluster centers', required=True, type=int)
    parser.add_argument('-n_parts','--n_parts', help='Number of workers or index chunks', required=True, type=int)
    args = vars(parser.parse_args())

    gen_blob_dataset(args['datasets_path'], args['n_rows'], args['n_cols'], args['n_centers'],  args['n_parts'])

