# Part of this code was adapted from sklearn/naive_bayes.py which
# is distributed under the following copyright:
#     Copyright (c) 2007–2020 The scikit-learn developers.
# and licensed under the New BSD License (see LICENSE.scikit_learn).

# import numpy as np
import cunumeric as np
from scipy.special import logsumexp
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot


def sparse_dot(a, b):
    # TODO: Actually implement handling of sparse matrices.
    # cunumeric does not support integer types for this operation.
    # TODO: Handle a.dtype != b.dtype
    return np.asarray(
        safe_sparse_dot(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ),
        dtype=a.dtype,
    )


def sum_(a, axis=None):
    return np.sum(a, axis=axis)


class MultinomialNB:
    def __init__(self, alpha: float = 1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = False
        self.class_prior = class_prior

    def fit(self, X, y):
        # convert to cunumeric arrays
        X = np.asarray(X)
        y = np.asarray(y)
        _, n_features = X.shape
        self.n_features_in_ = n_features

        labelbin = LabelBinarizer()
        Y = np.asarray(labelbin.fit_transform(y))  # need to convert output again

        self.classes_ = labelbin.classes_
        assert Y.shape[1] != 1

        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        self._update_feature_log_proba(self.alpha)
        self._update_class_log_prior(class_prior=self.class_prior)

        return self

    def _count(self, X, Y):
        self.feature_count_ += sparse_dot(Y.T, X)
        self.class_count_ += sum_(Y, axis=0)

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def _update_feature_log_proba(self, alpha):
        # Adapted from sklearn/naive_bayes.py
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = sum_(smoothed_fc, axis=1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            assert len(class_prior) == n_classes
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            raise NotImplementedError
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _joint_log_likelihood(self, X):
        return sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict_log_proba(self, X):
        X = np.asarray(X)
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        X = np.asarray(X)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
