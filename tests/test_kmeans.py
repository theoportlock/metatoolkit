import pandas as pd
from sklearn.cluster import KMeans

def test_kmeans():
    df = pd.DataFrame({
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6]
    })
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(df)
    assert len(kmeans.labels_) == 3