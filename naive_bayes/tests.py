try:
    import cunumeric as cn
except ImportError:
    raise ImportError("The Naive Bayes tests require cunumeric.")
from numpy.testing import assert_allclose

try:
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB as skNB
except ImportError:
    raise ImportError("The Naive Bayes tests require scikit-learn.")

from naive_bayes import MultinomialNB
from naive_bayes.cn.multinomial import MultinomialNB as CNMultinomialNB


def test_multinomial(nlp_20news):
    X_sparse, y = nlp_20news
    n_rows = 500
    n_cols = 10000

    X = X_sparse[:n_rows, :n_cols]
    y = y[:n_rows]

    legate_model = MultinomialNB()
    cn_legate_model = CNMultinomialNB()
    sk_model = skNB()

    sk_model.fit(X, y)
    # Cunumeric does not natively support sparse arrays.
    X_cn_dense = cn.ascontiguousarray(X.todense())
    cn_legate_model.fit(X_cn_dense, y)
    legate_model.fit(X, y)
    assert_allclose(sk_model.classes_, cn_legate_model.classes_)
    assert_allclose(sk_model.classes_, legate_model.classes_)
    assert_allclose(sk_model.feature_count_, cn_legate_model.feature_count_)
    assert_allclose(sk_model.feature_count_, legate_model.feature_count_)

    sk_log_proba = sk_model.predict_log_proba(X)
    legate_log_proba = legate_model.predict_log_proba(X)
    cn_legate_log_proba = cn_legate_model.predict_log_proba(X_cn_dense)
    sk_proba = sk_model.predict_proba(X)
    legate_proba = legate_model.predict_proba(X)
    cn_legate_proba = cn_legate_model.predict_proba(X_cn_dense)
    # sk_score = sk_model.score(X, y)
    # legate_score = legate_model.score(X, y)

    y_sk = sk_model.predict(X)
    y_legate = legate_model.predict(X)
    y_cn_legate = cn_legate_model.predict(X_cn_dense)

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
