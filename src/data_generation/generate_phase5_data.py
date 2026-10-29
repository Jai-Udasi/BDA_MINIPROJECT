"""
Synthetic Data Generator for Phase 5 (Big Data Mining & Machine Learning).
Generates:
1. Market basket transactions for PCY frequent itemset mining (data/synthetic/synthetic_market_baskets.json).
2. Spatial delivery coordinates for K-Means hub clustering (data/synthetic/synthetic_delivery_coordinates.csv).

IMPORTANT ARCHITECTURAL NOTE:
These datasets are explicitly labeled as synthetic demonstrations. They are generated separately
to satisfy Syllabus Module 5 without altering or corrupting the main 14-column orders dataset.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

# Distinct food menu items across categories
MENU_ITEMS = [
    # Biryani & Rice
    "Chicken Biryani", "Mutton Biryani", "Veg Biryani", "Jeera Rice",
    # Burgers & Fast Food
    "Veg Burger", "Chicken Burger", "Peri-Peri Fries", "Crispy Chicken Wings",
    # Pizzas & Pastas
    "Margherita Pizza", "Farmhouse Pizza", "Garlic Bread", "White Sauce Pasta",
    # North Indian Curries & Breads
    "Paneer Butter Masala", "Dal Makhani", "Butter Chicken", "Butter Naan", "Garlic Naan", "Tandoori Roti",
    # Desserts & Beverages
    "Gulab Jamun", "Rasgulla", "Chocolate Brownie", "Mango Lassi", "Cold Coffee", "Masala Chai"
]

# Targeted frequent co-ordered pairings with high correlation
FREQUENT_COMBOS = [
    ("Chicken Biryani", "Gulab Jamun"),
    ("Veg Burger", "Peri-Peri Fries"),
    ("Paneer Butter Masala", "Butter Naan"),
    ("Margherita Pizza", "Garlic Bread"),
    ("Dal Makhani", "Tandoori Roti"),
    ("Butter Chicken", "Garlic Naan")
]

# Zone centroid coordinates for spatial simulation (Mumbai / VESIT Urban Area)
ZONE_CENTROIDS = {
    "Downtown": (18.9388, 72.8354),
    "East Tech Park": (19.0688, 72.8704),
    "North Suburbs": (19.1750, 72.8450),
    "South Suburbs": (19.0150, 72.8450),
    "West University Campus": (19.0720, 72.8550),
    "Central Waterfront": (18.9650, 72.8180)
}


def generate_market_baskets(num_baskets: int = 1000, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates a deterministic dataset of food order transactions (baskets) with embedded frequent pairs.
    """
    rng = random.Random(seed)
    baskets = []

    for i in range(num_baskets):
        order_id = f"BSK-{10000 + i}"
        basket_items = set()

        # Decide if this basket injects one of the known frequent combos (~40% probability)
        if rng.random() < 0.45:
            combo = rng.choice(FREQUENT_COMBOS)
            basket_items.add(combo[0])
            basket_items.add(combo[1])

        # Add 1 to 4 additional random menu items
        extra_count = rng.randint(1, 3)
        for _ in range(extra_count):
            basket_items.add(rng.choice(MENU_ITEMS))

        baskets.append({
            "order_id": order_id,
            "items": sorted(list(basket_items)),
            "item_count": len(basket_items)
        })

    return baskets


def generate_delivery_coordinates(num_points: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic delivery and pickup coordinates clustered around realistic urban delivery zones.
    """
    np.random.seed(seed)
    zones = list(ZONE_CENTROIDS.keys())
    zone_probs = [0.28, 0.20, 0.18, 0.16, 0.12, 0.06]

    chosen_zones = np.random.choice(zones, size=num_points, p=zone_probs)
    records = []

    for i, zone in enumerate(chosen_zones):
        lat_center, lon_center = ZONE_CENTROIDS[zone]
        # Gaussian dispersion around zone centroid (~0.015 degrees ≈ 1.6 km)
        lat = lat_center + np.random.normal(0, 0.012)
        lon = lon_center + np.random.normal(0, 0.012)
        loc_type = "restaurant_pickup" if (i % 5 == 0) else "customer_dropoff"

        records.append({
            "location_id": f"LOC-{20000 + i}",
            "delivery_zone": zone,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "location_type": loc_type
        })

    return pd.DataFrame(records)


def save_phase5_synthetic_data():
    """Generates and writes both synthetic datasets to data/synthetic/."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Market Baskets JSON
    baskets = generate_market_baskets(num_baskets=1000, seed=42)
    baskets_file = SYNTHETIC_DIR / "synthetic_market_baskets.json"
    with open(baskets_file, "w", encoding="utf-8") as f:
        json.dump(baskets, f, indent=2)

    # 2. Save Spatial Coordinates CSV
    coords_df = generate_delivery_coordinates(num_points=500, seed=42)
    coords_file = SYNTHETIC_DIR / "synthetic_delivery_coordinates.csv"
    coords_df.to_csv(coords_file, index=False)

    return baskets_file, coords_file


if __name__ == "__main__":
    b_file, c_file = save_phase5_synthetic_data()
    print(f"Generated market baskets at: {b_file}")
    print(f"Generated delivery coordinates at: {c_file}")
