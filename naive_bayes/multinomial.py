# Part of this code was adapted from sklearn/naive_bayes.py which
# is distributed under the following copyright:
#     Copyright (c) 2007–2020 The scikit-learn developers.
# and licensed under the New BSD License (see LICENSE.scikit_learn).

import numpy as np
import pyarrow as pa
from legate.core import Store

from legate.raft import (
    add,
    argmax,
    as_array,
    as_store,
    convert,
    exp,
    fill,
    log,
    subtract,
    sum_over_axis,
)
from legate.raft.prims import (
    count_classes,
    count_features,
    invert_labels,
    make_monotonic,
)
from legate.raft.sparse import COOStore, CSRStore, SparseArray, SparseStore
from legate.raft.special import logsumexp


class MultinomialNB:
    def __init__(self, alpha: float = 1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = False
        self.class_prior = class_prior
        self._feature_log_prob_ = None
        self._class_log_prior_ = None

    def fit(self, X, y):
        X = COOStore.from_sparse_array(X)
        y = as_store(y)

        Y, self._classes_ = make_monotonic(y)

        self.n_classes_ = self._classes_.shape[0]
        self.n_features = X.shape[1]

        self._feature_count_ = count_features(X, Y, self.n_classes_)
        # Consider to call bincount directly since that's all that's happening...
        self.class_count_ = count_classes(Y, self.n_classes_)

        self._update_feature_log_proba(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def _update_feature_log_proba(self, alpha, dtype=pa.float64()):
        fc_ = convert(self._feature_count_, dtype)
        smoothed_fc = add(fc_, alpha)
        smoothed_cc = sum_over_axis(smoothed_fc, axis=1)
        x1 = log(smoothed_fc)
        x2 = log(smoothed_cc)
        self._feature_log_prob_ = subtract(x1.transpose((1, 0)), x2).transpose((1, 0))

    @property
    def feature_count_(self):
        if self._feature_count_ is not None:
            return as_array(self._feature_count_)

    @property
    def classes_(self):
        if self._classes_ is not None:
            return as_array(self._classes_)

    @property
    def feature_log_prob_(self):
        if self._feature_log_prob_ is not None:
            return as_array(self._feature_log_prob_)

    @property
    def class_log_prior_(self):
        if self._class_log_prior_ is not None:
            return as_array(self._class_log_prior_)

    def _update_class_log_prior(self, class_prior=None):
        n_classes = self.n_classes_
        if class_prior is not None:
            assert len(class_prior) == n_classes
            self._class_log_prior_ = log(class_prior)
        elif self.fit_prior:
            raise NotImplementedError
        else:
            self._class_log_prior_ = fill(n_classes, -np.log(n_classes))

    def _joint_log_likelihood(self, X):
        X_converted = X.to_type(self._feature_log_prob_.type)
        x1 = X_converted @ self._feature_log_prob_.transpose((1, 0))
        x2 = self._class_log_prior_
        return add(x1, x2)

    def _predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        x1 = jll
        x2 = log_prob_x
        return subtract(x1.transpose((1, 0)), x2).transpose((1, 0))

    def predict_log_proba(self, X: SparseArray):
        X = CSRStore.from_sparse_array(X)
        ret = self._predict_log_proba(X)
        return as_array(ret)

    def predict_proba(self, X: Store | SparseStore):
        X = CSRStore.from_sparse_array(X)
        ret = exp(self._predict_log_proba(X))
        return as_array(ret)

    def predict(self, X: SparseArray):
        X = CSRStore.from_sparse_array(X)
        jll = self._joint_log_likelihood(X)
        indices = argmax(jll, axis=1)
        y_hat = invert_labels(indices, self._classes_)

        return as_array(y_hat)
