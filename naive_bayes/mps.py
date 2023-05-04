# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from legate.raft import add, argmax, bincount, fill, log, subtract, sum_over_axis
from legate.raft.core import as_array, as_store
from legate.raft.prims import count_features, invert_labels, make_monotonic
from legate.raft.sparse import COOStore, CSRStore


def T(s):
    return s.transpose((1, 0))


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = COOStore.from_sparse_array(X)
        y = as_store(y)

        Y, self.classes_ = make_monotonic(y)

        self.n_classes_ = self.classes_.shape[0]
        self.n_features_ = X.shape[1]

        self.feature_count_ = count_features(X, y, self.n_classes_)
        smoothed_fc = add(self.feature_count_, self.alpha)
        smoothed_cc = sum_over_axis(smoothed_fc, axis=1)
        self.feature_log_prob_ = T(subtract(T(log(smoothed_fc)), log(smoothed_cc)))

        self.class_count_ = bincount(Y, num_bins=self.n_classes_)
        self.class_log_prior_ = fill(
            self.n_classes_, -log(self.n_classes_), dtype=self.feature_log_prob_.type
        )

    def predict(self, X):
        X = CSRStore.from_sparse_array(X).to_type(self.feature_log_prob_.type)
        jll = add(X @ T(self.feature_log_prob_), self.class_log_prior_)
        indices = argmax(jll, axis=1)  # Signal 11!!!
        return as_array(invert_labels(indices, self.classes_))


if __name__ == "__main__":
    from conftest import _nlp_20news
    from numpy.testing import assert_allclose
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB as skNB

    X, y = _nlp_20news()

    n_rows = 500
    n_cols = 10000

    X = X[:n_rows, :n_cols]
    y = y[:n_rows]

    print("START")

    estimator = MultinomialNB()
    sk_estimator = skNB()

    estimator.fit(X, y)
    sk_estimator.fit(X, y)

    print("feature count sum (sklearn)", sk_estimator.feature_count_.sum())
    print("feature count sum (legate) ", as_array(estimator.feature_count_).sum())

    assert_allclose(as_array(estimator.feature_count_), sk_estimator.feature_count_)
    assert_allclose(as_array(estimator.class_count_), sk_estimator.class_count_)

    y_hat = estimator.predict(X)
    sk_y_hat = sk_estimator.predict(X)

    print(accuracy_score(y, sk_y_hat))
    print(accuracy_score(y, y_hat))
    print("END")
