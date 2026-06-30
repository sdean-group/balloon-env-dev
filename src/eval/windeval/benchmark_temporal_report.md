# Temporal Leaderboard — does the wind field evolve like weather? (Phase 4d)

Peer-matched realism (real evolution lives in a band — too-frozen AND too-chaotic are both wrong): `temporal realism` = mean(persistence-match, tendency-match). `drift` = spatial COMPOSITE vs lead time (a generator should stay ~flat). Higher = better; **N/A** = needs >1 frame / a reference.

Peer `era5_real.zarr` · dt 1.0h · 12 frames · crop 32 · device `mps`. Peer refs: persistence 0.924, tendency 1.818 m/s/step.

## Headline

| Metric | era5 (peer) | kinematic-toy (M1) | shuffled (anchor) |
|---|---|---|---|
| score: temporal realism | **1.000** | **0.531** | **0.335** |
|   └ persistence match | 1.000 | 0.931 | 0.387 |
|   └ tendency match | 1.000 | 0.131 | 0.283 |
| drift slope/step (→0 = flat) | +0.0001 | +0.0000 | -0.0004 |
| drift mean (spatial COMPOSITE) | 0.914 | 0.500 | 0.914 |

## Diagnostics (not realism scores)

| Metric | era5 (peer) | kinematic-toy (M1) | shuffled (anchor) |
|---|---|---|---|
| temporal persistence (consec. corr) | 0.924 | 0.993 | 0.312 |
| tendency (m/s/step) | 1.818 | 0.239 | 6.432 |
| structure advection (diag) | 0.769 | 0.996 | 0.401 |

## Verdict (automated checks)
- ✅ peer realism (1.00) > shuffled (0.33) — the metric ranks coherent over incoherent
- ✅ peer is temporally coherent (persistence 0.92)
- ✅ shuffled anchor is incoherent (persistence 0.31)
- ✅ kinematic toy flagged TOO FROZEN by tendency-match (0.13) — the naive floor, as designed