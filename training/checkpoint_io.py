"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-10
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Atomic save/load of the training checkpoint.

save_checkpoint writes to a temporary file and renames it with os.replace(),
so a partially written file with the final name is never visible (if the
process is interrupted mid-write, the incomplete temporary file stays
separate from 'latest.pth', which continues to point to the last successful
save).

Does not serialize RNG/dataloader state: DAVISVidevoDataset sampling is
already random at every __getitem__, there is no notion of an "epoch" to
resume from halfway - restoring it would gain nothing useful.
"""
import os
import torch


def save_checkpoint(path, model, optimizer, scaler, scheduler, effective_step):
    tmp_path = path + ".tmp"
    checkpoint = {
        "effective_step": effective_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, scaler, scheduler, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["effective_step"]
