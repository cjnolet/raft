from legate.raft.kmeans import KMeans
import numpy as np

def test_kmeans():
    X = np.random.random((100, 50))
    
    model = KMeans()
    model.fit(X, 5)


if __name__ == "__main__":
    test_kmeans()
