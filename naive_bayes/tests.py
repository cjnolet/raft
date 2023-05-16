from time import perf_counter

from numpy.testing import assert_equal

try:
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB as skNB
except ImportError:
    raise ImportError("The Naive Bayes tests require scikit-learn.")

from naive_bayes import MultinomialNB


def test_multinomial(nlp_20news):
    X, y = nlp_20news

    n_rows = -1
    n_cols = -1

    X = X[:n_rows, :n_cols]
    y = y[:n_rows]

    tic = perf_counter()
    sk_estimator = skNB()
    sk_estimator.fit(X, y)
    sk_y_hat = sk_estimator.predict(X)
    toc = perf_counter()
    print("sklearn", toc - tic)

    tic = perf_counter()
    estimator = MultinomialNB()
    estimator.fit(X, y)
    y_hat = estimator.predict(X)
    toc = perf_counter()
    print("legate", toc - tic)

    assert_equal(estimator.feature_count_, sk_estimator.feature_count_)
    assert_equal(estimator.class_count_, sk_estimator.class_count_)
    print(accuracy_score(y, sk_y_hat))
    print(accuracy_score(y, y_hat))


if __name__ == "__main__":
    from conftest import _nlp_20news

    test_multinomial(_nlp_20news())
