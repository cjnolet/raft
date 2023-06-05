# legate.raft

Currently a sharing ground for building legate-enabled versions of naive bayes, knn, kmeans so that we can minimize duplication.


## Build

1. Build and install `legate.core` using [these instructions](https://github.com/nv-legate/legate.core#how-do-i-install-legate).

    Most recently tested against _branch-23.05_ (`59b59235`).


2. Install RAFT into your `legate` conda environment:

    ```bash
    mamba install -c conda-forge -c rapidsai-nightly -c nvidia libraft==23.04* raft-dask
    ```

3. Use `build.sh` to build this repository.

## Development Dependencies

```bash
mamba install -c conda-forge pytest hypothesis scikit-learn
```

## Notes

### Naive Bayes

_Implementation state:_

|         | cpu | gpu | mcpu | mgpu | mncpu | mngpu
|---------|-----|-----|------|------|-------|------
| fit     | ✅  |  ✅ |  ✅  | ✅   | ❔    |❔
| predict | ✅  |  ✅ |  🛑  | 🛑   | ❔    |❔

🛑: Blocked by missing features to express sparse constraints in legate.core.

❔: Not yet tested.

To run tests for Naive Bayes, execute:

```bash
legate naive_bayes/tests.py
```
or with `pytest`:
```bash
legate --module pytest naive_bayes/tests.py
```

_The tests require **scikit-learn**._

### K-NN

- Can use the nearest neighbors primitives in `raft::neighbors`

### K-Means

K-means uses the multi-gpu implementation which is based entirely on building blocks and comms API from RAFT (relies on NCCL for collectives).

Run the kmeans pytest on multiple GPUs. See below section for generating data at scale.
```bash
legate --gpus <ngpu> legate/raft/test/test_kmeans.py <dataset_path> <k>
```

Example of running `legate.raft` kmeans on 8 gpus:
```bash
legate --gpus 8 legate/raft/test/test_kmeans.py 8 50000 500 k
```

## Generating Data for Testing/Benchmarking

To generate data for testing/benchmarking at scale:
```bash
mkdir -p data/blobs1B
python scripts/blobs_dataset_gen.py --n_rows 1000000000 --n_cols 16 --n_centers 10 --n_parts 32 --datasets_path data/blobs1B
```


## Multi-node setup
To build and run on multiple nodes there are many choices to be made and the setup can be a bit fiddly. Here one way that we made work:


Create your conda environment in a location that is accessible from all nodes you want to use. For ASELAB machines `/datasets` is such a location.


Build legate from source (branch-23.05) more or less as usual. We used `./scripts/generate-conda-envs.py --python 3.10 --ctk 11.6 --os linux --compilers --openmpi` on `dgx12,13,14,15`, explicitly installed `gcc=11.2` after updating the environment based on the generated conda env file, and then used `./install.py --cuda --openmp --network mpi` for the `./install.py` step.



Note: With CUDA 11.6 we had to explicitly install gcc=11.2 to make things work. This is after using the


Build cunumeric from source (branch-23.05) as usual, but use ./install.py --cuda --openmp --network mpi for the ./install.py step.


Note: on dgx12,13,14,15 make sure to update your PATH: export PATH=/usr/local/cuda/bin:$PATH


Make sure you can use plain mpirun to execute code on all nodes. The following will test that you can run Python from your desired conda environment on all nodes with hostname dgx14 and dgx15:


```
mpirun -v --prefix $CONDA_PREFIX -H dgx14,dgx15 -n 2 python -c 'import sys; print(sys.path)'
```

It should print the module search path on each node. The path should contain your conda environment.
Next check that multiple instances of a MPI program can talk to each other. Use the ping pong example from the MPI tutorial (Makefile template).
If you can no make plain mpirun work, you need to fix this first before trying multi-node legate. Unfortuantely there are many ways in which this can fail, the best advice is to ask someone who has made it work in the past.
