"""Are the extra jumps at face edges or at every coarse-cell (8 px) boundary? ch 0, frame 6."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src/eval/windeval/hpx")
from patches import padded_index_map
nside, pad = 256, 32
pidx = padded_index_map(nside, pad, "/home/sps252/data/hpx_layout"); S = nside + 2 * pad
z = np.load("/scratch/sps252/runs/stage2_hpx256/diag_jul10_f6.npz")
era, gen = z["era"][0].astype(np.float32), z["smin0.002"][0].astype(np.float32)
interior = pidx[:, pad:-pad, pad:-pad]                                    # (12, 256, 256) NEST indices, face images
def rms(a): return float(np.sqrt((a ** 2).mean()))
for lab, x in (("gen", gen), ("era", era)):
    f = x[interior]                                                       # (12, 256, 256)
    dy = f[:, 1:, :] - f[:, :-1, :]                                       # jumps between row r and r+1
    rows = np.arange(255)
    at_block = (rows % 8) == 7                                            # boundary between block rows
    print(f"{lab}: row-jump RMS at coarse-cell boundaries {rms(dy[:, at_block]):.3f} vs within blocks {rms(dy[:, ~at_block]):.3f}; "
          f"by offset within block 0..7: {[round(rms(dy[:, (rows % 8) == k]), 3) for k in range(8)]}")
for lab, x in (("gen", gen), ("era", era)):
    je = x[pidx[:, pad, pad:-pad]] - x[pidx[:, pad - 1, pad:-pad]]
    print(f"{lab}: face-edge jump RMS {rms(je):.3f}")
