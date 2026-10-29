"""
Unit tests for Link Analysis (PageRank & HITS) and Recommendation Engines.
Module 6 Syllabus Verification.
"""
import pytest
import pandas as pd
import numpy as np
from src.mining.link_analysis_recommender import (
    BipartiteGraphAnalyzer,
    ContentBasedRecommender,
    CollaborativeFilteringRecommender
)


@pytest.fixture
def sample_orders_df():
    """Provides a deterministic mini-order dataframe with known bipartite links."""
    data = {
        "order_id": [f"ORD-{i:03d}" for i in range(1, 13)],
        "customer_id": [
            "CUST-A", "CUST-A", "CUST-A",
            "CUST-B", "CUST-B",
            "CUST-C", "CUST-C", "CUST-C",
            "CUST-D", "CUST-D", "CUST-D", "CUST-D"
        ],
        "restaurant_id": [
            "REST-1", "REST-2", "REST-3",
            "REST-1", "REST-2",
            "REST-3", "REST-4", "REST-5",
            "REST-1", "REST-2", "REST-4", "REST-5"
        ],
        "restaurant_type": [
            "Fast Food", "Fast Food", "Cafe",
            "Fast Food", "Fast Food",
            "Cafe", "Fine Dining", "Fine Dining",
            "Fast Food", "Fast Food", "Fine Dining", "Fine Dining"
        ],
        "delivery_zone": [
            "Downtown", "Downtown", "North Suburbs",
            "Downtown", "Downtown",
            "North Suburbs", "South Suburbs", "South Suburbs",
            "Downtown", "Downtown", "South Suburbs", "South Suburbs"
        ],
        "order_amount": [250.0, 300.0, 150.0, 200.0, 350.0, 180.0, 600.0, 750.0, 280.0, 320.0, 580.0, 700.0],
        "preparation_time_min": [15.0, 20.0, 10.0, 12.0, 18.0, 14.0, 30.0, 35.0, 16.0, 22.0, 28.0, 32.0],
        "delivery_duration_min": [25.0, 30.0, 20.0, 22.0, 28.0, 26.0, 45.0, 50.0, 27.0, 33.0, 42.0, 48.0],
        "item_count": [2, 3, 1, 2, 3, 1, 4, 5, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)


def test_pagerank_convergence_and_conservation(sample_orders_df):
    """Verifies PageRank converges and conserves total network probability of 1.0."""
    analyzer = BipartiteGraphAnalyzer(sample_orders_df)
    results = analyzer.compute_pagerank(damping=0.85, max_iter=100)

    assert results["converged"] is True
    assert len(results["top_restaurants"]) > 0
    assert len(results["top_customers"]) > 0

    # Total PageRank probability must sum to 1.0
    total_pr = sum(results["all_customer_pagerank"].values()) + sum(results["all_restaurant_pagerank"].values())
    assert pytest.approx(total_pr, rel=1e-4) == 1.0

    # REST-1 and REST-2 have the most incoming links (3 each)
    top_rest = results["top_restaurants"][0][0]
    assert top_rest in ["REST-1", "REST-2"]


def test_hits_hub_and_authority_mechanics(sample_orders_df):
    """Verifies HITS computes non-negative normalized hub and authority scores."""
    analyzer = BipartiteGraphAnalyzer(sample_orders_df)
    hits_res = analyzer.compute_hits(max_iter=100)

    authorities = hits_res["authority_scores"]
    hubs = hits_res["hub_scores"]

    assert len(authorities) == 5
    assert len(hubs) == 4

    # All scores must be non-negative
    for a in authorities.values():
        assert a >= 0.0
    for h in hubs.values():
        assert h >= 0.0

    # CUST-D ordered from 4 diverse restaurants, should have highest or near-highest hub score
    top_hub = hits_res["top_hubs"][0][0]
    assert top_hub in ["CUST-D", "CUST-C", "CUST-A"]


def test_content_based_recommender(sample_orders_df):
    """Tests feature vector cosine similarity recommendations."""
    cb = ContentBasedRecommender(sample_orders_df)
    
    # Recommend similar restaurants for REST-1 (Fast Food, Downtown)
    sim_rests = cb.recommend_for_restaurant("REST-1", top_k=3)
    assert len(sim_rests) > 0
    # REST-2 is also Fast Food in Downtown, so it should have highest similarity
    assert sim_rests[0]["restaurant_id"] == "REST-2"
    assert 0.0 <= sim_rests[0]["similarity_score"] <= 1.0

    # Recommend for CUST-A (loves Fast Food, visited REST-1, REST-2, REST-3)
    cust_recs = cb.recommend_for_customer("CUST-A", top_k=2)
    assert len(cust_recs) > 0
    # Must only recommend unvisited restaurants
    unvisited_ids = [r["restaurant_id"] for r in cust_recs]
    assert "REST-1" not in unvisited_ids
    assert "REST-2" not in unvisited_ids


def test_collaborative_filtering_recommendation(sample_orders_df):
    """Tests collaborative user-item affinity recommendations."""
    cf = CollaborativeFilteringRecommender(sample_orders_df)

    # CUST-B has only ordered from REST-1 and REST-2
    recs = cf.recommend("CUST-B", top_k=3)
    assert len(recs) > 0
    recommended_ids = [r["restaurant_id"] for r in recs]
    # Cannot recommend what CUST-B already ordered
    assert "REST-1" not in recommended_ids
    assert "REST-2" not in recommended_ids
    assert all(r["predicted_score"] >= 0 for r in recs)


def test_recommenders_handle_cold_start(sample_orders_df):
    """Verifies that unknown customer gracefully receives popular fallback recommendations."""
    cb = ContentBasedRecommender(sample_orders_df)
    cf = CollaborativeFilteringRecommender(sample_orders_df)

    cb_recs = cb.recommend_for_customer("CUST-UNKNOWN-99", top_k=3)
    cf_recs = cf.recommend("CUST-UNKNOWN-99", top_k=3)

    assert len(cb_recs) == 3
    assert len(cf_recs) == 3
    assert "Cold-start fallback" in cb_recs[0].get("note", "")
    assert "Cold-start fallback" in cf_recs[0].get("note", "")


def test_invalid_dataframe_raises():
    """Verifies ValueError when required columns are absent."""
    bad_df = pd.DataFrame({"col_a": [1, 2]})
    with pytest.raises(ValueError):
        BipartiteGraphAnalyzer(bad_df)
