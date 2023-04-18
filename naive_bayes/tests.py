# import numpy as np
import cunumeric as np
from numpy.testing import assert_allclose
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB as skNB

from naive_bayes import MultinomialNB
from naive_bayes.cn.multinomial import MultinomialNB as CNMultinomialNB


def test_multinomial(nlp_20news):
    X_sparse, y = nlp_20news
    n_rows = 500
    n_cols = 10000

    X = X_sparse[:n_rows, :n_cols]
    # TODO: Implement support for sparse arrays.
    X = np.ascontiguousarray(X.todense())
    y = y[:n_rows]

    legate_model = MultinomialNB()
    cn_legate_model = CNMultinomialNB()
    sk_model = skNB()

    sk_model.fit(X, y)
    cn_legate_model.fit(X, y)
    legate_model.fit(X, y)

    sk_log_proba = sk_model.predict_log_proba(X)
    legate_log_proba = legate_model.predict_log_proba(X)
    cn_legate_log_proba = cn_legate_model.predict_log_proba(X)
    sk_proba = sk_model.predict_proba(X)
    legate_proba = legate_model.predict_proba(X)
    cn_legate_proba = cn_legate_model.predict_proba(X)
    # sk_score = sk_model.score(X, y)
    # legate_score = legate_model.score(X, y)

    y_sk = sk_model.predict(X)
    y_legate = legate_model.predict(X)
    y_cn_legate = cn_legate_model.predict(X)

    assert_allclose(cn_legate_log_proba, sk_log_proba, atol=5e-1, rtol=5e-1)
    assert_allclose(cn_legate_proba, sk_proba, atol=2e-1, rtol=2.5)
    assert_allclose(legate_log_proba, sk_log_proba, atol=5e-1, rtol=5e-1)
    assert_allclose(legate_proba, sk_proba, atol=2e-1, rtol=2.5)
    assert accuracy_score(y, y_sk) >= 0.45
    assert accuracy_score(y, y_cn_legate) >= 0.45
    assert accuracy_score(y, y_legate) >= 0.45
    print("PASS")


if __name__ == "__main__":
    from conftest import _nlp_20news

    test_multinomial(_nlp_20news())
