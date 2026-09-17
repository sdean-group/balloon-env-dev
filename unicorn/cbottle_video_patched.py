"""Run NVIDIA's scripts/inference_coarse_video.py with one bug in its clip chaining fixed.

cBottle-video makes 12 frames at a time. To continue a rollout, the last frame of one clip is handed
to the next clip as its known first frame. In cbottle's FrameMasker.__call__, the handed-over frames
(`condition_to_insert`) are only used to read a shape; the tensor actually packed for the network is
`batch["target"] * mask`, i.e. the dataset's own frame. With real ERA5 loaded that is the true weather
at that time (so the bug hides); in the data-free AMIP mode it is blank, and every chained clip
restarts from a near-calm atmosphere (measured 2026-09-17: wind std 18 -> 4 m/s at the first boundary,
correlation with the previous frame 0.98 -> 0.14).

Fix: when frames are handed over, pack those. Model weights and sampler are untouched.

Second, smaller change: NVIDIA's script gives every clip the same seed, so every clip starts from the
same noise (a hint of 11-frame repetition showed up in a short test). Here clip n uses seed + n, which
stays reproducible. Use base seeds far apart (e.g. 100, 200) so two rollouts never share a clip seed.

    python cbottle_video_patched.py <the same arguments as inference_coarse_video.py>
"""
import os, runpy, sys
from cbottle.training.video import frame_masker

_original = frame_masker.FrameMasker.__call__

def _call(self, batch, condition_to_insert=None):
    if condition_to_insert is not None:
        handed = condition_to_insert.to(batch["target"].dtype)
        if handed.ndim == batch["target"].ndim + 1:            # drop a leading batch axis the dataset frame lacks
            handed = handed.squeeze(0)
        batch = {**batch, "target": handed}
    return _original(self, batch, condition_to_insert)

frame_masker.FrameMasker.__call__ = _call

from cbottle.inference import _video_autoregression as _va
_generate_original = _va.VideoAutoregression._generate_frames

def _generate(self, *args, **kwargs):
    out = _generate_original(self, *args, **kwargs)
    if self.sample_kwargs.get("seed") is not None:
        self.sample_kwargs["seed"] += 1                      # next clip, next seed
    return out

_va.VideoAutoregression._generate_frames = _generate
script = os.path.expanduser("~/cBottle/scripts/inference_coarse_video.py")
sys.path.insert(0, os.path.dirname(script))                 # the script imports helpers from its own folder
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
