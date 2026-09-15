"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-10
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Builds the fused DinoV3 checkpoint (NOT trained) for key_encoder.network2,
starting from the published DINOv2 checkpoint.

Fuses:
  - Base weights: weights/DINOv2FeatureV6_LocalAtten_s2_154000.pth,
    with the key_encoder.network2.* keys removed (they were specific to the
    DINOv2 Segmentor - internal architecture incompatible with
    Segmentor_DINOv3)
  - key_encoder.network2.backbone.*: DinoV3 ViT-B/16 loaded from
    weights/dinov3-vitb16/ (local project directory, config.json +
    model.safetensors - NEVER from the canonical HuggingFace name / the
    user's global cache), generic - NOT the partial checkpoints of the old
    train_proj.py (discarded)
  - key_encoder.network2.proj.*: random initialization (Conv1x1 3072->1536 + BN)

NOTE: the runtime loading (ColorMNet.load_weights, network.py) now filters
the 'key_encoder.network2.*' keys BY SHAPE COMPATIBILITY (no longer an
unconditional blanket drop) - the 217 keys produced by this script have
shapes identical to those of a fresh Segmentor_DINOv3 (by construction),
so they are ALL actually reloaded at runtime (verified: 0 discarded). The
file produced here is therefore an artifact actually used, not inert as in
the previous version of this script / of the load logic.

Output: weights/DINOv3FeatureV6_LocalAtten_untrained.pth (in .gitignore,
NOT to be committed - verified at the end of the script).

Usage:
    python tools/build_dinov3_checkpoint.py
"""
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import torch

from colormnet.model.resnet import Segmentor_DINOv3

BASE_CKPT = script_dir / "weights" / "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
OUT_CKPT = script_dir / "weights" / "DINOv3FeatureV6_LocalAtten_untrained.pth"
DINOV3_WEIGHTS_DIR = script_dir / "weights" / "dinov3-vitb16"
NETWORK2_PREFIX = "key_encoder.network2."


def main():
    print(f"Loading base checkpoint: {BASE_CKPT}")
    base_dict = torch.load(BASE_CKPT, map_location="cpu")

    n_before = len(base_dict)
    base_dict = {k: v for k, v in base_dict.items() if not k.startswith(NETWORK2_PREFIX)}
    n_removed = n_before - len(base_dict)
    print(f"Removed {n_removed} keys '{NETWORK2_PREFIX}*' (DINOv2 Segmentor)")

    if not DINOV3_WEIGHTS_DIR.exists():
        raise FileNotFoundError(
            f"Missing backbone directory: {DINOV3_WEIGHTS_DIR} "
            "(config.json + model.safetensors required)")
    print(f"Instantiating Segmentor_DINOv3 (weights_dir={DINOV3_WEIGHTS_DIR}, local_files_only=True)")
    seg = Segmentor_DINOv3(str(DINOV3_WEIGHTS_DIR))
    seg_dict = seg.state_dict()
    print(f"  {len(seg_dict)} keys from Segmentor_DINOv3 (backbone + random-init proj)")

    merged = dict(base_dict)
    for k, v in seg_dict.items():
        merged[NETWORK2_PREFIX + k] = v

    print(f"Fused checkpoint: {len(merged)} total keys")

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, OUT_CKPT)
    print(f"Saved: {OUT_CKPT} ({OUT_CKPT.stat().st_size / 1e6:.1f} MB)")

    # .gitignore check: the file must NOT be addable to git
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-v", str(OUT_CKPT)],
                       cwd=script_dir, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"OK: the file is in .gitignore -> {r.stdout.strip()}")
    else:
        print("WARNING: the file is NOT in .gitignore! Check before every commit.")


if __name__ == "__main__":
    main()
