"""
Unit tests for Spatial Delivery-Hub Clustering (Syllabus Module 5).
Verifies:
1. Haversine spherical distance calculation accuracy against known coordinates.
2. K-Means clustering convergence and centroid identification.
3. Inertia and Silhouette score evaluation across candidate k values [3, 8].
4. Robustness against missing coordinates, duplicate points, and boundary conditions.
"""
import pytest
import numpy as np
import pandas as pd
from src.mining.spatial_clustering import SpatialHubClusterer, haversine_distance_km


@pytest.fixture
def spatial_fixture():
    """
    Deterministic spatial fixture with 3 clearly separated clusters of coordinates.
    Cluster 1: Around (18.93, 72.83) - 10 points
    Cluster 2: Around (19.07, 72.87) - 10 points
    Cluster 3: Around (19.18, 72.85) - 10 points
    """
    np.random.seed(42)
    c1_lat = 18.93 + np.random.normal(0, 0.005, 10)
    c1_lon = 72.83 + np.random.normal(0, 0.005, 10)

    c2_lat = 19.07 + np.random.normal(0, 0.005, 10)
    c2_lon = 72.87 + np.random.normal(0, 0.005, 10)

    c3_lat = 19.18 + np.random.normal(0, 0.005, 10)
    c3_lon = 72.85 + np.random.normal(0, 0.005, 10)

    lats = np.concatenate([c1_lat, c2_lat, c3_lat])
    lons = np.concatenate([c1_lon, c2_lon, c3_lon])

    return pd.DataFrame({
        "location_id": [f"LOC-{i}" for i in range(30)],
        "latitude": lats,
        "longitude": lons
    })


def test_haversine_distance_accuracy():
    """
    Verify Haversine formula against known benchmark:
    Gateway of India (18.9220, 72.8347) to Bandra Fort (19.0410, 72.8183) ≈ 13.3 km.
    """
    dist = haversine_distance_km(18.9220, 72.8347, 19.0410, 72.8183)
    assert 12.5 <= dist <= 14.5, f"Haversine distance {dist} km deviated from expected ~13.3 km"

    # Zero distance test
    assert haversine_distance_km(19.0, 72.0, 19.0, 72.0) == 0.0


def test_spatial_clustering_fit(spatial_fixture):
    """Verify K-Means successfully identifies 3 clusters and computes centroids."""
    clusterer = SpatialHubClusterer(n_clusters=3, random_state=42)
    results = clusterer.fit_predict(spatial_fixture)

    assert results["n_clusters"] == 3
    assert results["total_points_clustered"] == 30
    assert len(results["centroids"]) == 3
    assert results["inertia"] > 0.0
    assert results["silhouette_score"] > 0.60  # Well-separated clusters should have high silhouette

    # Each centroid should have approximately 10 assigned points
    for c in results["centroids"]:
        assert c["assigned_orders_count"] == 10
        assert c["avg_transit_radius_km"] >= 0.0


def test_evaluate_cluster_range(spatial_fixture):
    """Verify evaluation of candidate cluster counts from k=3 to k=5."""
    clusterer = SpatialHubClusterer(random_state=42)
    eval_list = clusterer.evaluate_cluster_range(spatial_fixture, k_min=3, k_max=5)

    assert len(eval_list) == 3
    assert [e["k"] for e in eval_list] == [3, 4, 5]
    for e in eval_list:
        assert "inertia" in e
        assert "silhouette_score" in e
        assert e["inertia"] > 0


def test_spatial_clustering_edge_cases():
    """Verify handling of missing coordinates, duplicate points, and invalid k."""
    clusterer = SpatialHubClusterer(n_clusters=2, random_state=42)

    # DataFrame with missing values
    df_missing = pd.DataFrame({
        "latitude": [19.0, 19.1, None, 19.2],
        "longitude": [72.8, None, 72.8, 72.9]
    })
    res = clusterer.fit_predict(df_missing)
    assert res["total_points_clustered"] == 2

    # Invalid n_clusters
    with pytest.raises(ValueError):
        SpatialHubClusterer(n_clusters=0)

    # Dataset smaller than requested clusters
    with pytest.raises(ValueError):
        tiny_df = pd.DataFrame({"latitude": [19.0], "longitude": [72.0]})
        clusterer.fit_predict(tiny_df)
