# legate.raft

Currently a sharing ground for building legate-enabled versions of naive bayes, knn, kmeans so that we can minimize duplication.


## Build

1. Build and install `legate.core` using [these instructions](https://github.com/nv-legate/legate.core#how-do-i-install-legate).

    Most recently tested against _branch-23.05_ (`59b59235`).


2. Install RAFT into your `legate` conda environment:

    ```bash
    mamba install -c conda-forge -c rapidsai-nightly -c nvidia libraft==23.06*
    ```

2. Use `build.sh` to build this repository.

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

Run the kmeans pytest on multiple GPUs
```bash
legate --gpus <ngpu> legate/raft/test/test_kmeans.py <ngpu> <nrow> <ncol> <k> 
```

Example of running `legate.raft` kmeans on 8 gpus:
```bash
legate --gpus 8 legate/raft/test/test_kmeans.py 8 50000 500 k
```
