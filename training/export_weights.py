"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-11 (updated 2026-09-15: passthrough for files already in weights format)
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Exports ONLY the weights (the model state_dict) from a training checkpoint
(checkpoint_io.py format: dict with keys effective_step/model/optimizer/
scaler/scheduler) into a .pth file directly usable by
colormnet_render.py - of the same nature as
weights/DINOv3FeatureV6_LocalAtten_untrained.pth.

Format detection based on the same principle already verified in
training/verify/benchmark_full_validation.py (_detect_and_extract_state_dict/
TRAINING_CHECKPOINT_KEYS): the mere presence of the first-level 'model' key
unambiguously distinguishes a full training checkpoint from a weights-only
file (no ColorMNet parameter is named 'model' - they are all dotted paths
like 'key_encoder.conv1.weight').

- Full training checkpoint -> extracts ONLY 'model' (pure passthrough,
  no modification of the weights) - about half of the original size (which
  also includes the AdamW optimizer state, exp_avg/exp_avg_sq for every
  trainable parameter, plus scaler/scheduler).
- File already in weights format (none of the 5 known keys) -> copied as-is
  with a warning, not an error: whoever runs the script on a file already
  ready for inference must not get a failure or an empty file.

Usage:
    python training/export_weights.py --input training_runs/dinov3_full/step_1000.pth --output weights/DINOv3FeatureV6_LocalAtten_step1000.pth
"""
import argparse
import os
import shutil

import torch

# Schema from training/checkpoint_io.py:25-36 (save_checkpoint) - the 5 first-level
# keys known for a full training checkpoint. Same constant as
# training/verify/benchmark_full_validation.py.
TRAINING_CHECKPOINT_KEYS = {"effective_step", "model", "optimizer", "scaler", "scheduler"}


def export_weights(input_path, output_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    if os.path.exists(output_path):
        raise FileExistsError(
            f"'{output_path}' already exists - specify a different --output to avoid "
            f"accidentally overwriting an existing file."
        )

    checkpoint = torch.load(input_path, map_location="cpu")
    input_size_mb = os.path.getsize(input_path) / (1024 ** 2)

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        # No first-level 'model' key: it is already a weights-only file -
        # direct copy, no conversion needed (not an error, not an empty file).
        n_keys = len(checkpoint) if isinstance(checkpoint, dict) else None
        print(f"'{input_path}' is already in inference format (no first-level 'model' "
              f"key, {n_keys} keys) - no conversion needed, copying the file as-is.")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        shutil.copyfile(input_path, output_path)
        output_size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"Copied: {output_path} ({output_size_mb:.1f} MB, unchanged with respect to "
              f"'{input_path}', {input_size_mb:.1f} MB).")
        return

    found_keys = set(checkpoint.keys())
    missing_known_keys = TRAINING_CHECKPOINT_KEYS - found_keys
    if missing_known_keys:
        print(f"WARNING: detected a training checkpoint (first-level 'model' key "
              f"present) but the keys {sorted(missing_known_keys)} are missing with respect to the "
              f"known checkpoint_io.py schema - proceeding anyway by extracting 'model' "
              f"(the only key actually needed here).")

    state_dict = checkpoint["model"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(state_dict, output_path)

    output_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Weights exported: {output_path} ({output_size_mb:.1f} MB, {len(state_dict)} keys) "
          f"from '{input_path}' ({input_size_mb:.1f} MB, effective_step={checkpoint.get('effective_step', '?')}).")


def parse_args():
    p = argparse.ArgumentParser(
        description="Exports only the weights (checkpoint['model']) from a "
                    "training checkpoint, into a .pth file directly usable by "
                    "colormnet_render.py, without the optimizer state."
    )
    p.add_argument("--input", required=True,
                    help="Path to the training checkpoint (e.g. training_runs/dinov3_full/step_1000.pth).")
    p.add_argument("--output", required=True,
                    help="Path of the output .pth file (weights only). No default - "
                         "must be specified explicitly to avoid overwriting "
                         "something by mistake.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_weights(args.input, args.output)
