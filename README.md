# legate.raft

Currently a sharing ground for building legate-enabled versions of naive bayes, knn, kmeans so that we can minimize duplication.


## Build

After you install `libraft` into your `legate` conda environment, use `build.sh` to build this repository.

## Notes

### Naive Bayes

- Can potentially use `bincount` from legate.core or the `raft::stats::histogram` primitives.
- maybe we could try to use pylibraft as a legate task if they make progress on exposing legate tasks through Python UDFs?

### K-NN

- Can use the nearest neighbors primitives in `raft::neighbors`

### K-Means

- Can use the kmeans primitives in `raft::cluster::kmeans` (which are already used to compose the mnmg kmeans in cuml)
