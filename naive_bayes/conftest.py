import numpy as np
import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def _nlp_20news():
    try:
        twenty_train = fetch_20newsgroups(subset="train", shuffle=True, random_state=42)
    except:  # noqa E722
        pytest.xfail(reason="Error fetching 20 newsgroup dataset")

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(twenty_train.data)
    Y = np.array(twenty_train.target)

    return X, Y


@pytest.fixture(scope="module")
def nlp_20news():
    return _nlp_20news()
