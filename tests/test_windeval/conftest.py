"""pytest configuration for the windeval suite.

Two suites here are SCRIPT-STYLE (they run their checks at import time and exit with a
status code), so pytest must not import them during collection; run them directly:

    PYTHONPATH=. python tests/test_windeval/test_coarse_conditioning.py   # 22 checks
    PYTHONPATH=. python tests/test_windeval/test_structure_metrics.py     # 13 checks

Everything else is ordinary pytest.
"""
collect_ignore = ["test_coarse_conditioning.py", "test_structure_metrics.py"]
