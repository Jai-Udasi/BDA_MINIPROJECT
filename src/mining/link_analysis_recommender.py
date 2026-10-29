"""
Link Analysis, Social Network Graph Centrality, and Dual Recommendation Engines.
Module: Syllabus Module 6 (Link Analysis, Social Network Graphs, and Recommendation Systems).

Implements:
1. Bipartite Customer-Restaurant Graph Analysis:
   - PageRank Algorithm (Power iteration with damping factor d=0.85).
   - HITS Algorithm (Hyperlink-Induced Topic Search: Hub and Authority scores).
2. Dual Recommendation Engines:
   - Content-Based Filtering: Cosine similarity on restaurant cuisine & operational profile vectors.
   - Collaborative Filtering: User-Item interaction matrix similarity predicting unvisited preferences.
"""
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


def ensure_customer_id(df: pd.DataFrame, num_customers: int = 150) -> pd.DataFrame:
    """
    Ensures DataFrame has a 'customer_id' column.
    If absent, assigns deterministic customer identifiers based on order_id hash,
    modeling a realistic customer pool of 150 repeating patrons across restaurants.
    """
    if "customer_id" in df.columns:
        return df

    df_out = df.copy()
    if "order_id" in df_out.columns:
        def _hash_to_cust(oid: Any) -> str:
            h_val = int(hashlib.md5(str(oid).encode()).hexdigest(), 16)
            return f"CUST-{(h_val % num_customers) + 100:03d}"
        df_out["customer_id"] = df_out["order_id"].apply(_hash_to_cust)
    else:
        df_out["customer_id"] = [f"CUST-{(i % num_customers) + 100:03d}" for i in range(len(df_out))]
    return df_out


class BipartiteGraphAnalyzer:
    """
    Constructs a customer-restaurant interaction graph from historical orders
    and executes PageRank and HITS link analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes graph from order DataFrame containing 'customer_id' and 'restaurant_id'.
        """
        if "restaurant_id" not in df.columns:
            raise ValueError("DataFrame must contain 'restaurant_id'")

        df_ready = ensure_customer_id(df)
        self.df = df_ready[["customer_id", "restaurant_id"]].dropna()
        self.customers = sorted(self.df["customer_id"].unique())
        self.restaurants = sorted(self.df["restaurant_id"].unique())

        self.num_customers = len(self.customers)
        self.num_restaurants = len(self.restaurants)
        self.nodes = self.customers + self.restaurants
        self.N = len(self.nodes)

        self.cust_to_idx = {c: i for i, c in enumerate(self.customers)}
        self.rest_to_idx = {r: i for i, r in enumerate(self.restaurants)}
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # Build Adjacency Matrix A where rows = customers, cols = restaurants
        self.adj_matrix = np.zeros((self.num_customers, self.num_restaurants), dtype=np.float64)
        for _, row in self.df.iterrows():
            c_idx = self.cust_to_idx[row["customer_id"]]
            r_idx = self.rest_to_idx[row["restaurant_id"]]
            self.adj_matrix[c_idx, r_idx] += 1.0

    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Computes PageRank centrality on the full undirected/bipartite network.
        PR_{t+1}(p) = (1-d)/N + d * sum_{q in In(p)} (PR_t(q) / Out(q))
        """
        if self.N == 0:
            return {"customer_ranks": {}, "restaurant_ranks": {}, "iterations": 0}

        # Build N x N transition probability matrix M
        # Customers link to restaurants, restaurants link back to customers
        full_adj = np.zeros((self.N, self.N), dtype=np.float64)
        for c_idx in range(self.num_customers):
            for r_idx in range(self.num_restaurants):
                w = self.adj_matrix[c_idx, r_idx]
                if w > 0:
                    node_c = c_idx
                    node_r = self.num_customers + r_idx
                    full_adj[node_c, node_r] = w
                    full_adj[node_r, node_c] = w

        # Normalize rows to form stochastic transition matrix
        row_sums = full_adj.sum(axis=1)
        # Avoid division by zero for dangling nodes
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.where(row_sums[:, None] > 0, full_adj / row_sums[:, None], 1.0 / self.N)

        # Power iteration
        pr = np.full(self.N, 1.0 / self.N, dtype=np.float64)
        iterations = 0

        for it in range(max_iter):
            iterations = it + 1
            # Random walk + teleportation
            pr_next = (1.0 - damping) / self.N + damping * (M.T @ pr)
            # Re-normalize to ensure sum is strictly 1
            pr_next = pr_next / np.sum(pr_next)
            if np.linalg.norm(pr_next - pr, 1) < tol:
                pr = pr_next
                break
            pr = pr_next

        # Split into customers and restaurants
        cust_pr = {self.customers[i]: float(pr[i]) for i in range(self.num_customers)}
        rest_pr = {self.restaurants[j]: float(pr[self.num_customers + j]) for j in range(self.num_restaurants)}

        # Sort descending
        sorted_cust = sorted(cust_pr.items(), key=lambda x: x[1], reverse=True)
        sorted_rest = sorted(rest_pr.items(), key=lambda x: x[1], reverse=True)

        return {
            "top_restaurants": sorted_rest[:10],
            "top_customers": sorted_cust[:10],
            "all_restaurant_pagerank": rest_pr,
            "all_customer_pagerank": cust_pr,
            "damping_factor": damping,
            "iterations": iterations,
            "converged": iterations < max_iter
        }

    def compute_hits(
        self,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Computes Kleinberg's HITS algorithm:
        Hubs h: Customers who order from diverse prominent restaurants.
        Authorities a: Restaurants that attract orders from influential hubs.
        a = A^T * h,  h = A * a  (with L2 normalization at each step)
        """
        if self.num_customers == 0 or self.num_restaurants == 0:
            return {"hub_scores": {}, "authority_scores": {}, "iterations": 0}

        A = self.adj_matrix.copy()

        # Initialize hub vector uniformly
        h = np.ones(self.num_customers, dtype=np.float64) / np.sqrt(self.num_customers)
        a = np.ones(self.num_restaurants, dtype=np.float64) / np.sqrt(self.num_restaurants)

        iterations = 0
        for it in range(max_iter):
            iterations = it + 1
            # Authority update: sum of hubs pointing to it
            a_next = A.T @ h
            norm_a = np.linalg.norm(a_next)
            if norm_a > 0:
                a_next /= norm_a

            # Hub update: sum of authorities it points to
            h_next = A @ a_next
            norm_h = np.linalg.norm(h_next)
            if norm_h > 0:
                h_next /= norm_h

            diff_a = np.linalg.norm(a_next - a)
            diff_h = np.linalg.norm(h_next - h)

            a = a_next
            h = h_next

            if diff_a < tol and diff_h < tol:
                break

        hub_scores = {self.customers[i]: float(h[i]) for i in range(self.num_customers)}
        authority_scores = {self.restaurants[j]: float(a[j]) for j in range(self.num_restaurants)}

        sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_authorities = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "top_authorities": sorted_authorities[:10],
            "top_hubs": sorted_hubs[:10],
            "authority_scores": authority_scores,
            "hub_scores": hub_scores,
            "iterations": iterations
        }


class ContentBasedRecommender:
    """
    Recommends restaurants based on cosine similarity of feature profiles
    (cuisine tags, zone, price tier, delivery speed, and prep time).
    """

    def __init__(self, orders_df: pd.DataFrame):
        self.orders_df = ensure_customer_id(orders_df)
        self.restaurant_profiles = self._build_restaurant_profiles()

    def _build_restaurant_profiles(self) -> pd.DataFrame:
        """Aggregates restaurant-level behavioral and categorical attributes."""
        grouped = self.orders_df.groupby("restaurant_id").agg({
            "order_amount": "mean",
            "preparation_time_min": "mean",
            "delivery_duration_min": "mean",
            "item_count": "mean",
            "order_id": "count"
        }).rename(columns={"order_id": "total_orders"})

        # One-hot encode restaurant_type and delivery_zone if present
        metadata = self.orders_df.drop_duplicates(subset=["restaurant_id"])[
            ["restaurant_id", "restaurant_type", "delivery_zone"]
        ].set_index("restaurant_id")

        merged = grouped.join(metadata, how="left")
        encoded = pd.get_dummies(merged, columns=["restaurant_type", "delivery_zone"], drop_first=False)
        
        # Min-max scale numeric columns
        numeric_cols = ["order_amount", "preparation_time_min", "delivery_duration_min", "item_count", "total_orders"]
        for col in numeric_cols:
            if col in encoded.columns:
                min_v = encoded[col].min()
                max_v = encoded[col].max()
                if max_v > min_v:
                    encoded[col] = (encoded[col] - min_v) / (max_v - min_v)
                else:
                    encoded[col] = 1.0

        return encoded.fillna(0.0)

    def recommend_for_restaurant(self, restaurant_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds most similar restaurants using Cosine Similarity."""
        if restaurant_id not in self.restaurant_profiles.index:
            return []

        target_vec = self.restaurant_profiles.loc[restaurant_id].values
        norm_target = np.linalg.norm(target_vec)
        if norm_target == 0:
            return []

        recommendations = []
        for rest_id, row in self.restaurant_profiles.iterrows():
            if rest_id == restaurant_id:
                continue
            cand_vec = row.values
            norm_cand = np.linalg.norm(cand_vec)
            if norm_cand > 0:
                sim = float(np.dot(target_vec, cand_vec) / (norm_target * norm_cand))
                recommendations.append({
                    "restaurant_id": str(rest_id),
                    "similarity_score": round(sim, 4)
                })

        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations[:top_k]

    def recommend_for_customer(self, customer_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Builds customer preference vector from past orders and scores unvisited restaurants."""
        cust_orders = self.orders_df[self.orders_df["customer_id"] == customer_id]
        if cust_orders.empty:
            # Cold-start fallback: recommend highest volume restaurants
            top_rests = self.orders_df["restaurant_id"].value_counts().head(top_k).index.tolist()
            return [{"restaurant_id": r, "similarity_score": 0.5, "note": "Cold-start fallback"} for r in top_rests]

        visited_restaurants = set(cust_orders["restaurant_id"].unique())
        # Customer vector is average of visited restaurant vectors
        visited_vectors = self.restaurant_profiles.loc[list(visited_restaurants)]
        cust_vec = visited_vectors.mean(axis=0).values
        norm_cust = np.linalg.norm(cust_vec)

        if norm_cust == 0:
            return []

        candidates = []
        for rest_id, row in self.restaurant_profiles.iterrows():
            if rest_id in visited_restaurants:
                continue
            cand_vec = row.values
            norm_cand = np.linalg.norm(cand_vec)
            if norm_cand > 0:
                sim = float(np.dot(cust_vec, cand_vec) / (norm_cust * norm_cand))
                candidates.append({
                    "restaurant_id": str(rest_id),
                    "similarity_score": round(sim, 4),
                    "affinity_basis": "Cuisine & Operational Profile Match"
                })

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]


class CollaborativeFilteringRecommender:
    """
    User-Item Collaborative Filtering Recommender using Cosine similarity
    on order interaction frequencies.
    """

    def __init__(self, orders_df: pd.DataFrame):
        self.orders_df = ensure_customer_id(orders_df)
        # Pivot table: rows = customer_id, cols = restaurant_id, values = order_count
        self.interaction_matrix = self.orders_df.pivot_table(
            index="customer_id",
            columns="restaurant_id",
            aggfunc="size",
            fill_value=0
        )
        self.user_similarity = self._compute_user_similarity()

    def _compute_user_similarity(self) -> pd.DataFrame:
        """Computes pairwise cosine similarity between customers."""
        M = self.interaction_matrix.values
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_M = np.where(norms > 0, M / norms, 0.0)

        sim_matrix = normalized_M @ normalized_M.T
        return pd.DataFrame(
            sim_matrix,
            index=self.interaction_matrix.index,
            columns=self.interaction_matrix.index
        )

    def recommend(self, customer_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recommends unvisited restaurants based on peer order habits."""
        if customer_id not in self.interaction_matrix.index:
            # Fallback for unseen customer
            top_rests = self.orders_df["restaurant_id"].value_counts().head(top_k).index.tolist()
            return [{"restaurant_id": r, "predicted_score": 0.5, "note": "Cold-start fallback"} for r in top_rests]

        user_sims = self.user_similarity.loc[customer_id]
        # Exclude self
        user_sims = user_sims.drop(customer_id)

        # Get restaurants customer has already ordered from
        customer_history = self.interaction_matrix.loc[customer_id]
        unvisited = customer_history[customer_history == 0].index

        if len(unvisited) == 0:
            return []

        # Predict affinity: sum(sim(u, v) * orders(v, r)) / sum(|sim(u, v)|)
        peer_interactions = self.interaction_matrix.loc[user_sims.index, unvisited]
        sim_weights = user_sims.values[:, None]

        weighted_orders = peer_interactions.values * sim_weights
        predicted_scores = weighted_orders.sum(axis=0) / (np.abs(sim_weights).sum() + 1e-9)

        recommendations = []
        for rest_id, score in zip(unvisited, predicted_scores):
            if score > 0:
                recommendations.append({
                    "restaurant_id": str(rest_id),
                    "predicted_score": round(float(score), 4),
                    "rationale": "High peer affinity among customers with similar order histories"
                })

        recommendations.sort(key=lambda x: x["predicted_score"], reverse=True)
        return recommendations[:top_k]
