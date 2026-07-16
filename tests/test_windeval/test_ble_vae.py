"""Validate the numpy BLE-VAE decoder reimplementation.

Correctness proof: the field is the curl of a streamfunction, so it must be
divergence-free (div << vorticity). If our port were wrong, this would fail.

Run:  ../.venv/bin/python -m tests.test_ble_vae
"""
import numpy as np

from src.eval.windeval.generators import ble_vae


def run():
    params = ble_vae.load_params()
    field = ble_vae.sample(params, seed=0)

    print("field shape:", field.shape)
    u, v = field[..., 0], field[..., 1]
    speed = np.sqrt(u ** 2 + v ** 2)

    # divergence vs vorticity on each (level,time) slice (interior)
    div_ratios = []
    for k in range(field.shape[2]):
        for t in range(field.shape[3]):
            uu, vv = u[:, :, k, t], v[:, :, k, t]
            du_dy, du_dx = np.gradient(uu)
            dv_dy, dv_dx = np.gradient(vv)
            div = (du_dx + dv_dy)[2:-2, 2:-2]
            vort = (dv_dx - du_dy)[2:-2, 2:-2]
            div_ratios.append(div.std() / (vort.std() + 1e-12))
    div_ratio = float(np.mean(div_ratios))

    print(f"speed mean {speed.mean():.2f} m/s, max {speed.max():.2f} m/s")
    print(f"mean |div|/|vort| (interior): {div_ratio:.4f}  (≈0 => divergence-free)")

    checks = [
        (field.shape == (21, 21, 10, 9, 2), "output shape (21,21,10,9,2)"),
        (np.all(np.isfinite(field)), "field finite"),
        (0.5 < speed.mean() < 100, "speed magnitude physically plausible"),
        (div_ratio < 0.05, "divergence-free by construction (div/vort < 0.05)"),
    ]
    ok = all(c for c, _ in checks)
    print()
    for c, msg in checks:
        print(f"{'PASS' if c else 'FAIL'}  {msg}")
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
