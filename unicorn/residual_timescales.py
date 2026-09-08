"""How fast does the fine-scale residual change in time, and how wrong is linear time
interpolation of the 6-hourly coarse field? Runs on the node that holds the fine store."""
import numpy as np, zarr, time
root = zarr.open_group("/scratch/sps252/era5_hpx", mode="r")
print("groups:", list(root), {k: list(root[k]) for k in root if hasattr(root[k], "keys")})
fine, coarse = root["fine/uv"], root["coarse/uv"]
hours = np.asarray(root["time"][:], dtype=np.int64); n = len(hours)
rng = np.random.default_rng(0)
def pool(x):                      # NEST: 64 consecutive fine pixels = one nside-32 pixel
    return x.reshape(x.shape[0], -1, 64).mean(-1)
def lift(c):
    return np.repeat(c, 64, axis=-1)
# ---- residual scale and Eulerian temporal autocorrelation (3 starts x 13 consecutive hours)
lags = [1, 2, 3, 4, 6, 9, 12]
acc = {k: [] for k in lags}; rms_r, rms_x, rms_c = [], [], []
t0 = time.time()
for start in rng.integers(0, n - 13, size=3):
    while np.isnan(coarse[start, 0, :5]).any():
        start = rng.integers(0, n - 13)
    xs = [np.asarray(fine[start + k]).astype(np.float32) for k in range(13)]     # (36, 786432)
    rs = []
    for x in xs:
        c = pool(x); r = x - lift(c); rs.append(r)
        rms_r.append(np.sqrt((r ** 2).mean())); rms_x.append(np.sqrt(((x - x.mean(-1, keepdims=True)) ** 2).mean())); rms_c.append(np.sqrt(((c - c.mean(-1, keepdims=True)) ** 2).mean()))
    for k in lags:
        for i in range(0, 13 - k):
            a, b = rs[i].ravel(), rs[i + k].ravel()
            acc[k].append(float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean())))
    print(f"  start row {start} done ({time.time() - t0:.0f}s)", flush=True)
print(f"fine-field RMS (about its mean) {np.mean(rms_x):.2f} m/s; coarse-field RMS {np.mean(rms_c):.2f}; residual RMS {np.mean(rms_r):.2f} m/s "
      f"= {np.mean(rms_r) / np.mean(rms_x):.3f} of the fine RMS ({(np.mean(rms_r) / np.mean(rms_x)) ** 2 * 100:.1f}% of variance)")
print("residual Eulerian autocorrelation by lag (h):", {k: round(float(np.mean(v)), 3) for k, v in acc.items()})
# ---- linear time interpolation error of the 6-hourly coarse field (40 random t)
errs, steps, mids = [], [], []
for t in rng.integers(0, n - 7, size=40):
    c0, c3, c6 = (np.asarray(coarse[t + k]).astype(np.float32) for k in (0, 3, 6))
    if np.isnan(c0).any() or np.isnan(c3).any() or np.isnan(c6).any():
        continue
    errs.append(np.sqrt(((c3 - 0.5 * (c0 + c6)) ** 2).mean())); steps.append(np.sqrt(((c6 - c0) ** 2).mean()))
    mids.append(np.sqrt(((c3 - c0) ** 2).mean()))
print(f"coarse field: RMS change over 6 h {np.mean(steps):.2f} m/s, over 3 h {np.mean(mids):.2f}; "
      f"linear-interpolation error at the 3 h midpoint {np.mean(errs):.2f} m/s ({len(errs)} samples)")
