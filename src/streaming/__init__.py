"""
Streaming Big Data Mining Package (Syllabus Module 4).
Provides real-time event simulation, Bloom Filter stream deduplication,
Flajolet-Martin count-distinct estimation, and DGIM sliding-window delay counting.
"""
from src.streaming.order_stream_simulator import OrderStreamSimulator
from src.streaming.bloom_filter import BloomFilter
from src.streaming.flajolet_martin import FlajoletMartin
from src.streaming.dgim_window import DGIMSlidingWindow

__all__ = [
    "OrderStreamSimulator",
    "BloomFilter",
    "FlajoletMartin",
    "DGIMSlidingWindow"
]
