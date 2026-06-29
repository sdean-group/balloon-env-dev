"""windeval — conditioned wind-field evaluation harness.

Decouples generation from evaluation through a frozen artifact format:
generators -> WindArtifact -> metric suite -> calibration/leaderboard.
"""
__all__ = ["artifact", "anchors", "ingest_era5", "metrics", "benchmark"]
