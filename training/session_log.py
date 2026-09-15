"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-10
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Local append-only log (CSV) for the DinoV3 training - no wandb,
no remote service. One line for every effective step in which logging
happens (--checkpoint_every cadence, coincides with save+validation,
"paired as in the original").
"""
import csv
import os
import time

# val_deltae00_mean/val_deltae00_p90 added at the END -
# the original 5 columns keep the same position so as not to break
# scripts/spreadsheets that read the CSV by position.
FIELDNAMES = ["timestamp", "effective_step", "train_loss_smoothed", "val_psnr", "session_elapsed_seconds",
              "val_deltae00_mean", "val_deltae00_p90"]


def append_row(csv_path, effective_step, train_loss_smoothed=None, val_psnr=None, session_elapsed_seconds=None,
                val_deltae00_mean=None, val_deltae00_p90=None):
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "effective_step": effective_step,
            "train_loss_smoothed": "" if train_loss_smoothed is None else f"{train_loss_smoothed:.6f}",
            "val_psnr": "" if val_psnr is None else f"{val_psnr:.4f}",
            "session_elapsed_seconds": "" if session_elapsed_seconds is None else f"{session_elapsed_seconds:.1f}",
            "val_deltae00_mean": "" if val_deltae00_mean is None else f"{val_deltae00_mean:.4f}",
            "val_deltae00_p90": "" if val_deltae00_p90 is None else f"{val_deltae00_p90:.4f}",
        })
