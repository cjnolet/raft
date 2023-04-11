# Part of this code was adapted from sklearn/naive_bayes.py which
# is distributed under the following copyright:
#     Copyright (c) 2007–2020 The scikit-learn developers.
# and licensed under the New BSD License (see LICENSE.scikit_learn).

import numpy as np
import pyarrow as pa

# import cunumeric as np
from sklearn.preprocessing import LabelBinarizer

from legate.raft import (
    add,
    array_to_store,
    convert,
    exp,
    fill,
    log,
    matmul,
    store_to_array,
    subtract,
    sum_over_axis,
)
from legate.raft.special import logsumexp


class MultinomialNB:
    def __init__(self, alpha: float = 1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = False
        self.class_prior = class_prior
        self._feature_log_prob_ = None
        self._class_log_prior_ = None

    def fit(self, X, y):
        # Convert to a legate stores
        X = array_to_store(X)
        _, n_features = X.shape
        self.n_features_in_ = n_features

        labelbin = LabelBinarizer()
        Y = array_to_store(labelbin.fit_transform(y))

        self.classes_ = labelbin.classes_
        assert Y.shape[1] != 1

        Y_transposed = Y.transpose((1, 0))
        self.feature_count_ = matmul(Y_transposed, X)
        self.class_count_ = sum_over_axis(Y, axis=0)
        self._update_feature_log_proba(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def _update_feature_log_proba(self, alpha, dtype=pa.float64()):
        # Adapted from sklearn/naive_bayes.py

        fc_ = convert(self.feature_count_, dtype)
        smoothed_fc = add(fc_, alpha)
        smoothed_cc = sum_over_axis(smoothed_fc, axis=1)

        # self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
        #     smoothed_cc.reshape(-1, 1)
        # )
        x1 = log(smoothed_fc)
        x2 = log(smoothed_cc)
        self._feature_log_prob_ = subtract(x1.transpose((1, 0)), x2).transpose((1, 0))

    @property
    def feature_log_prob_(self):
        if self._feature_log_prob_ is not None:
            return store_to_array(self._feature_log_prob_)

    @property
    def class_log_prior_(self):
        if self._class_log_prior_ is not None:
            return store_to_array(self._class_log_prior_)

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            assert len(class_prior) == n_classes
            self._class_log_prior_ = log(class_prior)
        elif self.fit_prior:
            raise NotImplementedError
        else:
            self._class_log_prior_ = fill(n_classes, -np.log(n_classes))

    def _joint_log_likelihood(self, X):
        X_converted = convert(X, pa.float64())

        x1 = matmul(X_converted, self._feature_log_prob_.transpose((1, 0)))
        x2 = self._class_log_prior_
        return add(x1, x2)

    def _predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)

        log_prob_x = logsumexp(jll, axis=1)

        x1 = jll
        x2 = log_prob_x
        return subtract(x1.transpose((1, 0)), x2).transpose((1, 0))

    def predict_log_proba(self, X):
        X = array_to_store(X)
        ret = self._predict_log_proba(X)
        return store_to_array(ret)

    def predict_proba(self, X):
        X = array_to_store(X)
        ret = exp(self._predict_log_proba(X))
        return store_to_array(ret)

    def predict(self, X):
        X = array_to_store(X)
        jll = store_to_array(self._joint_log_likelihood(X))
        ret = self.classes_[np.argmax(jll, axis=1)]
        return ret
