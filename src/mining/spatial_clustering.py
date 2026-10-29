"""
Spatial Delivery-Hub Clustering (Syllabus Module 5: Clustering & Big Data Mining).
Identifies optimal courier staging hubs and delivery cluster centers using K-Means clustering.

THEORETICAL & DOMAIN CONSIDERATIONS:
1. Proposed Hubs vs. Real-World Road Networks:
   Mathematical cluster centroids represent proposed central aggregation points based on spatial
   density. In real-world urban deployments, road networks, traffic corridors, flyovers, and zoning
   laws further constrain actual staging facility locations.
2. Coordinate Distance Metric:
   Latitude and longitude are spherical coordinates on Earth's ellipsoid, not Cartesian plane coordinates.
   While K-Means optimizes squared Euclidean distance in degree-space over localized metropolitan
   regions (~15-20 km), we compute true transit radii using the Haversine Great-Circle formula:
     d = 2 * R * arcsin(sqrt(sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)))
     where R = 6,371.0 km (mean radius of Earth).
3. Cluster Validation:
   Evaluates candidate k in [3, 8] using Inertia (within-cluster sum of squares) and
   the Silhouette Score (measuring cluster cohesion vs. separation).
"""
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two geographic coordinates in kilometers.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(EARTH_RADIUS_KM * c, 3)


class SpatialHubClusterer:
    """
    K-Means spatial clustering engine for identifying courier staging hubs.
    """
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
        self.n_clusters: int = n_clusters
        self.random_state: int = random_state
        self.model: Optional[KMeans] = None
        self.centroids_: Optional[np.ndarray] = None

    def evaluate_cluster_range(
        self,
        df: pd.DataFrame,
        k_min: int = 3,
        k_max: int = 8,
        lat_col: str = "latitude",
        lon_col: str = "longitude"
    ) -> List[Dict[str, Any]]:
        """
        Evaluates cluster counts from k_min to k_max computing Inertia and Silhouette scores.
        """
        clean_coords = df[[lat_col, lon_col]].dropna().values
        n_samples = len(clean_coords)

        if n_samples < k_min:
            return []

        max_k = min(k_max, n_samples - 1)
        results = []

        for k in range(k_min, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = km.fit_predict(clean_coords)
            inertia = float(km.inertia_)

            # Silhouette score is mathematically defined only when 2 <= k <= n_samples - 1
            # and when distinct clusters exist
            if len(set(cluster_labels)) > 1:
                sil = float(silhouette_score(clean_coords, cluster_labels))
            else:
                sil = 0.0

            results.append({
                "k": k,
                "inertia": round(inertia, 5),
                "silhouette_score": round(sil, 4)
            })

        return results

    def fit_predict(
        self,
        df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude"
    ) -> Dict[str, Any]:
        """
        Fits K-Means clustering on the spatial coordinates and assigns each record to a hub.
        """
        valid_df = df.dropna(subset=[lat_col, lon_col]).copy()
        n_samples = len(valid_df)

        if n_samples == 0:
            raise ValueError("Input dataset contains zero valid coordinate records.")
        if n_samples < self.n_clusters:
            raise ValueError(
                f"Sample size ({n_samples}) is smaller than requested clusters ({self.n_clusters})."
            )

        coords = valid_df[[lat_col, lon_col]].values

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = self.model.fit_predict(coords)
        self.centroids_ = self.model.cluster_centers_

        valid_df["assigned_hub_id"] = labels
        valid_df["hub_name"] = [f"Hub-{chr(65 + l)}" for l in labels]

        # Calculate Haversine transit distance to assigned centroid for each point
        transit_distances_km = []
        for i, row in valid_df.iterrows():
            c_lat, c_lon = self.centroids_[row["assigned_hub_id"]]
            d_km = haversine_distance_km(row[lat_col], row[lon_col], c_lat, c_lon)
            transit_distances_km.append(d_km)

        valid_df["distance_to_hub_km"] = transit_distances_km

        # Build centroids summary
        centroids_summary = []
        for c_id in range(self.n_clusters):
            c_lat, c_lon = self.centroids_[c_id]
            points_in_cluster = valid_df[valid_df["assigned_hub_id"] == c_id]
            avg_radius = points_in_cluster["distance_to_hub_km"].mean() if not points_in_cluster.empty else 0.0

            centroids_summary.append({
                "hub_id": c_id,
                "hub_name": f"Hub-{chr(65 + c_id)}",
                "centroid_latitude": round(float(c_lat), 6),
                "centroid_longitude": round(float(c_lon), 6),
                "assigned_orders_count": len(points_in_cluster),
                "avg_transit_radius_km": round(float(avg_radius), 2)
            })

        sil_score = 0.0
        if len(set(labels)) > 1 and n_samples > self.n_clusters:
            sil_score = float(silhouette_score(coords, labels))

        return {
            "n_clusters": self.n_clusters,
            "total_points_clustered": n_samples,
            "inertia": round(float(self.model.inertia_), 5),
            "silhouette_score": round(sil_score, 4),
            "centroids": centroids_summary,
            "clustered_df": valid_df
        }
