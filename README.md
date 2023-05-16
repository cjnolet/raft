# legate.raft

Currently a sharing ground for building legate-enabled versions of naive bayes, knn, kmeans so that we can minimize duplication.


## Build

1. Build and install `legate.core` using [these instructions](https://github.com/nv-legate/legate.core#how-do-i-install-legate).

    Most recently tested against _branch-23.05_ (`d756ff9`).


2. Install RAFT into your `legate` conda environment:

    ```bash
    mamba install -c conda-forge -c rapidsai-nightly -c nvidia libraft==23.06*
    ```

2. Use `build.sh` to build this repository.

## Notes

### Naive Bayes

To run tests for Naive Bayes, execute:

```bash
legate naive_bayes/tests.py
```
or with `pytest`:
```bash
legate --module pytest naive_bayes/tests.py
```

_The tests require **scikit-learn** and **cunumeric**._

- Can potentially copy in `bincount` from legate.core or the `raft::stats::histogram` or other RAFT primitives.
- maybe we could try to use pylibraft as a legate task if they make progress on exposing legate tasks through Python UDFs?
- Interested in an end-to-end prototype that uses the underlying indices array from the legate.sparse csr w/ bincount/histogram primitive.

### K-NN

- Can use the nearest neighbors primitives in `raft::neighbors`

### K-Means

- Can use the kmeans primitives in `raft::cluster::kmeans` (which are already used to compose the mnmg kmeans in cuml)
