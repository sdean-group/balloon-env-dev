import sys, numpy as np, healpy as hp, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src/eval/windeval/hpx")
from patches import padded_index_map
nside, pad = 256, 32
pidx = padded_index_map(nside, pad, "/home/sps252/data/hpx_layout"); S = nside + 2 * pad
vec = np.stack(hp.pix2vec(nside, np.arange(12 * nside * nside), nest=True), -1)   # (npix, 3)
def dist(a, b):  # great-circle distance in pixel units (0.229 deg)
    return np.degrees(np.arccos(np.clip((vec[a] * vec[b]).sum(-1), -1, 1))) / 0.2289
edges = {"top": (pidx[:, pad, pad:-pad], pidx[:, pad - 1, pad:-pad], pidx[:, pad + 1, pad:-pad]),
         "bottom": (pidx[:, S - pad - 1, pad:-pad], pidx[:, S - pad, pad:-pad], pidx[:, S - pad - 2, pad:-pad]),
         "left": (pidx[:, pad:-pad, pad], pidx[:, pad:-pad, pad - 1], pidx[:, pad:-pad, pad + 1]),
         "right": (pidx[:, pad:-pad, S - pad - 1], pidx[:, pad:-pad, S - pad], pidx[:, pad:-pad, S - pad - 2])}
print("geometry: great-circle distance (pixel units) interior-pixel -> halo-pixel vs interior -> next interior, per face type")
for name, (edge, halo, inner) in edges.items():
    d_h, d_i = dist(edge, halo), dist(edge, inner)
    print(f"  {name:6s} polar faces 0-3: halo {d_h[:4].mean():.2f} (min {d_h[:4].min():.2f} max {d_h[:4].max():.2f}) interior {d_i[:4].mean():.2f} | equatorial 4-7: halo {d_h[4:8].mean():.2f} interior {d_i[4:8].mean():.2f} | south 8-11: halo {d_h[8:].mean():.2f} interior {d_i[8:].mean():.2f}")
# duplicated halo? fraction of halo pixels identical to the edge pixel or to any pixel of the edge row
dup = np.mean([np.mean(np.isin(edges[k][1], edges[k][0])) for k in edges]); print("fraction of halo pixels that duplicate an edge pixel:", round(float(dup), 4))
z = np.load("/scratch/sps252/runs/stage2_hpx256/diag_jul10_f6.npz")
era, gen = z["era"][0].astype(np.float32), z["smin0.002"][0].astype(np.float32)
print("jump RMS (ch0, frame 6) across each edge type, gen vs ERA5, and interior; then along-edge profile in 32-px bins (top edge, all faces):")
for name, (edge, halo, inner) in edges.items():
    for lab, x in (("gen", gen), ("era", era)):
        je, ji = x[edge] - x[halo], x[edge] - x[inner]
        print(f"  {name:6s} {lab}: edge {np.sqrt((je**2).mean()):.3f} (polar {np.sqrt((je[:4]**2).mean()):.3f} eq {np.sqrt((je[4:8]**2).mean()):.3f} south {np.sqrt((je[8:]**2).mean()):.3f})  interior {np.sqrt((ji**2).mean()):.3f}")
edge, halo, inner = edges["top"]
for lab, x in (("gen", gen), ("era", era)):
    je = (x[edge] - x[halo]) ** 2
    print(f"  top-edge along-edge RMS by 32px bin {lab}:", [round(float(np.sqrt(je[:, a:a + 32].mean())), 3) for a in range(0, 256, 32)])
# residual-only seams (remove block means)
def resid(x): return x - np.repeat(x.reshape(12288, 64).mean(-1), 64)
rg, re = resid(gen), resid(era)
for lab, x in (("gen-resid", rg), ("era-resid", re)):
    je, ji = x[edge] - x[halo], x[edge] - x[inner]
    print(f"  residual only, top edge {lab}: edge {np.sqrt((je**2).mean()):.3f} interior {np.sqrt((ji**2).mean()):.3f}")
