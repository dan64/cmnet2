"""
-------------------------------------------------------------------------------
Author: dan64
Date: 2026-09-10
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Entry point CLI for the full fine-tuning of the DinoV3 backbone (unlocked
backbone) on cmnet2_dev, branch feature/dinov3. Time-boxed sessions,
resumable (full checkpoint/resume via checkpoint_io.py), local logging
(session_log.py, no wandb), gradient accumulation (physical batch 1 ->
effective accum_steps).

Reuses ColorMNetTrainer.do_pass/do_val EXACTLY AS THEY ARE (no modification
beyond what was already applied in trainer.py: accum_steps, the
_SingleProcessWrapper fix for world_size=1, the device= fix on 'hidden').
This script only provides the surrounding orchestration: config, real
datasets (DAVISVidevoDataset /
DAVISTestDataset_221128_TransColorization_batch), local logger/wandb shim,
effective step loop, timeout, signals.

Usage:
    python training/train_dinov3.py --target_iterations 15000 \
        --checkpoint_every 1000 --max_session_hours 8
"""
import argparse
import gc
import glob
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()  # cmnet2_dev/
TRAINING_DIR = Path(__file__).parent.resolve()
for p in (str(PROJECT_DIR), str(TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from colormnet.dataset.vos_dataset import DAVISVidevoDataset
from colormnet.inference.data.test_datasets import DAVISTestDataset_221128_TransColorization_batch
from colormnet.model.trainer import ColorMNetTrainer

import checkpoint_io
import heartbeat
import session_log

# No line printed by this script (including those from LocalTrainLogger
# below, which calls print() and therefore automatically inherits this
# same behavior) may be left without a timestamp. Centralized in
# colormnet/util/console_log.py (shared with trainer.py) - init_log_file()
# is called further down in main(), once checkpoint_dir is known, so this
# script too writes its log directly to file (session_console.log inside
# checkpoint_dir) WITHOUT going through an external tee (PowerShell/
# run_session.bat) - that pipe was verified to prevent reliable delivery
# of a real Ctrl+C to the process.
from colormnet.util.console_log import init_log_file, log as print


# Same sample of 20/131 videos used to empirically verify the .png/.jpg
# loading (same fixed seed, same selection - not recomputed/re-invented
# here). Generated with:
#   videos = sorted(os.listdir(val_root)); random.seed(42)
#   sample = random.sample(videos, 20)
# Hardcoded (not recomputed on every start) so the subset stays stable even
# if the content of val_root changes in the future (the full val_root,
# --val_full, remains available for a rarer final evaluation - it is not
# this subset that limits it).
VAL_SUBSET_20 = [
    "bmx-trees", "breakdance", "camel", "dog", "dogs-jump", "drift-straight",
    "duello_al_sole_1946_1946", "ferdinando_i_re_di_napoli_1959_1959",
    "ieri_oggi_domani_1963_1963", "il_fantasma_dell_opera_1943_1943",
    "il_figlio_di_ali_baba_1952_1952",
    "il_giro_del_mondo_in_80_giorni_around_the_world_in_80_days_1956_1956",
    "incantesimo_the_eddy_duchin_story_1956_1956", "intrigo_a_stoccolma_1963_1963",
    "l_occhio_che_uccide_peeping_tom_1960_1960",
    "l_orribile_segreto_del_dr_hichcock_1962_1962",
    "questo_pazzo_pazzo_mondo_1963_1963", "riccardo_iii_1955_1955",
    "sinbad_il_marinario_1958_1947", "vera_cruz_1954_1954",
]


class LocalTrainLogger:
    """Replaces TensorboardLogger (colormnet/util/logger.py - requires the
    'tensorboard' package, not installed in this environment). Implements
    only the interface actually used by trainer.py: log_metrics
    (called from Integrator.finalize, untouched), log_scalar/log_string/
    log_cv2 (called from do_pass/do_val when logger is not None). Console
    only, no file - the "real" log is session_log.csv."""
    def log_metrics(self, l1_tag, l2_tag, val, step, f=None):
        print(f"[{l1_tag}/{l2_tag}] step={step} {val:.6f}")

    def log_scalar(self, tag, x, step):
        pass

    def log_string(self, tag, x):
        print(f"[{tag}] {x}")

    def log_cv2(self, tag, x, step):
        pass


class _NoOpWandbImage:
    def __init__(self, *a, **kw):
        pass


class LocalWandbShim:
    """Intercepts the wandb.log/wandb.Image calls from trainer.py (do_pass/
    do_val, untouched - they call them unconditionally) without sending
    anything to a remote service. Captures only the last logged dict: lets
    the orchestrator read the loss just computed by do_pass without having
    to modify the function to make it return it explicitly."""
    def __init__(self):
        self.last = {}

    def log(self, data, step=None):
        self.last.update(data)

    def Image(self, *a, **kw):
        return _NoOpWandbImage(*a, **kw)


_stop_requested = False


def _signal_handler(signum, frame):
    global _stop_requested
    print(f"\n[train_dinov3] Signal {signum} received - finishing the current "
          f"effective step and saving before exiting...", flush=True)
    _stop_requested = True


# --shutdown: delay before actually shutting down, to leave a window for
# cancellation (shutdown /a) if someone is at the terminal when it fires.
_SHUTDOWN_DELAY_SECONDS = 60


def _trigger_shutdown():
    """Starts the PC shutdown at the end of the session (--shutdown). Windows
    only (the project's platform) - on other platforms it prints a warning
    instead of failing silently or running a command that does not exist."""
    print(f"\n[train_dinov3] --shutdown active: the PC will shut down in "
          f"{_SHUTDOWN_DELAY_SECONDS}s. To cancel: 'shutdown /a' in another "
          f"terminal within this time.", flush=True)
    if sys.platform != "win32":
        print(f"[train_dinov3] WARNING: --shutdown only supports Windows "
              f"(sys.platform={sys.platform!r}) - no shutdown started.",
              flush=True)
        return
    try:
        subprocess.run(["shutdown", "/s", "/t", str(_SHUTDOWN_DELAY_SECONDS)], check=True)
    except Exception as e:
        print(f"[train_dinov3] WARNING: shutdown command failed: {e}", flush=True)


def build_config(args, dinov3_weights_dir, val_root):
    return {
        "single_object": False,
        "key_dim": 64, "value_dim": 512, "hidden_dim": 64,
        "deep_update_prob": 0.2,
        "backbone": args.backbone,
        "dinov3_weights_dir": dinov3_weights_dir,
        "unlock_backbone": args.unlock_backbone,
        "num_frames": args.num_frames, "num_ref_frames": args.num_ref_frames,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "steps": args.lr_milestones, "gamma": 0.1,
        "start_warm": 20000, "end_warm": 70000,
        "amp": True,
        "log_text_interval": 100,
        "log_image_interval": 10 ** 9,      # never - no image grid needed for this script
        "save_network_interval": 10 ** 9,   # never automatic - we control the validation
        "save_checkpoint_interval": 10 ** 9,  # never - checkpoint_io controls the saving
        "debug": False,
        "validation_root": val_root,
    }


def infinite_train_loader(dataset, batch_size):
    """DAVISVidevoDataset has no meaningful notion of 'epoch' (every
    __getitem__ re-samples randomly) - a DataLoader that is recreated when
    exhausted is enough to provide an indefinite stream of batches."""
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, drop_last=True)
        for batch in loader:
            yield batch


def resolve_path(p, project_dir):
    p = Path(p)
    return str(p if p.is_absolute() else project_dir / p)


_BEST_CHECKPOINT_RE = re.compile(r"^latest_best_psnr_([0-9]+\.[0-9]+)\.pth$")


def find_best_checkpoint(checkpoint_dir):
    """Finds latest_best_psnr_*.pth in checkpoint_dir, parsing the NUMERIC
    value from the name - never an alphabetical string sort ('51.9800'
    precedes '6.0500' alphabetically despite being numerically greater).
    Returns (path, val_psnr) of the file with the highest psnr found, or
    (None, None) if none. By construction there should be at most one alive
    at a time; if a crash halfway between writing the new one and removing
    the old one (see below) leaves more than one, here we simply pick the
    highest value, without cleaning up the others (no unrequested
    deletion)."""
    best_path, best_val = None, None
    for path in glob.glob(os.path.join(checkpoint_dir, "latest_best_psnr_*.pth")):
        m = _BEST_CHECKPOINT_RE.match(os.path.basename(path))
        if not m:
            continue
        val = float(m.group(1))
        if best_val is None or val > best_val:
            best_path, best_val = path, val
    return best_path, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="datasets/train_root")
    ap.add_argument("--val_root", default="datasets/val_root")
    ap.add_argument("--checkpoint_dir", default="training_runs/dinov3_full/")
    ap.add_argument("--log_filename", default="session_console.log",
                     help="Name of the textual log file (timestamp + same content as the "
                          "console), written INSIDE --checkpoint_dir - no longer an external "
                          "tee (PowerShell), the Python process itself writes it "
                          "(colormnet/util/console_log.py). Default: session_console.log "
                          "(same name/location already used before).")
    ap.add_argument("--init_checkpoint", default="weights/DINOv3FeatureV6_LocalAtten_untrained.pth")
    ap.add_argument("--backbone", default="dinov3")
    ap.add_argument("--unlock_backbone", action="store_true",
                     help="Trainable DinoV3 backbone (requires_grad + "
                          "HuggingFace gradient checkpointing) instead of frozen. Default "
                          "disabled (unchanged behavior) - must be passed explicitly, "
                          "never the implicit default. See colormnet/model/resnet.py "
                          "(Segmentor_DINOv3) for details on what changes.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--accum_steps", type=int, default=2)
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--num_ref_frames", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lr_milestones", type=int, nargs="*", default=[120000])
    ap.add_argument("--target_iterations", type=int, required=True,
                     help="Total effective steps for this phase (required, no default)")
    ap.add_argument("--max_session_hours", type=float, default=8)
    ap.add_argument("--shutdown", action="store_true",
                     help="Shuts down the PC (Windows: 'shutdown /s /t 60', 60s margin to "
                          "cancel with 'shutdown /a') when the session ends by itself "
                          "without needing further intervention: target reached or timeout "
                          "of --max_session_hours. Does NOT fire on manual interruption "
                          "(SIGINT/SIGTERM/SIGBREAK - the user is present at the terminal) "
                          "nor on a crash (unhandled exception - requires attention, not a "
                          "shutdown). Default disabled.")
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    ap.add_argument("--log_level", choices=["quiet", "normal", "verbose"], default="normal",
                     help="quiet: only the summary at each checkpoint. normal (default): "
                          "quiet + watchdog heartbeat.log. verbose: normal + one line per "
                          "video processed in validation.")
    ap.add_argument("--heartbeat_interval", type=float, default=30.0,
                     help="Seconds between one heartbeat.log line and the next.")
    ap.add_argument("--val_full", action="store_true",
                     help="Use the entire val_root (131 videos) instead of the fixed sample "
                          "of 20 - for a rarer final evaluation, not for the periodic "
                          "cadence of every checkpoint.")
    ap.add_argument("--val_max_side", type=int, default=1280,
                     help="Caps the longest side of the validation frames (aspect ratio "
                          "preserved, same computation as compute_process_size() in "
                          "test_video_full.py) - limits the VRAM peak on frames with very "
                          "high native resolution (up to 1920px in the library). The "
                          "prediction is still re-projected to the original resolution before "
                          "the PSNR computation (no impact on metric fidelity beyond the "
                          "normal downscale/upscale). <=0 = no cap (original behavior).")
    ap.add_argument("--save_best", action="store_true",
                     help="In addition to latest.pth/step_N.pth at every "
                          "checkpoint_every, also saves a full checkpoint (same 5 keys as "
                          "checkpoint_io.py) of the best val_psnr seen so far, as "
                          "latest_best_psnr_<value>.pth inside checkpoint_dir (one live file "
                          "at a time). Default disabled: no new file, no extra check if not "
                          "passed.")
    args = ap.parse_args()

    # datasets/train_root and datasets/val_root live one level above cmnet2_dev
    # (sibling of cmnet2_dev/cmnet2_dinov3/...), not inside it -
    # checkpoint_dir/init_checkpoint/weights instead live inside cmnet2_dev.
    train_root = resolve_path(args.train_root, PROJECT_DIR.parent)
    val_root = resolve_path(args.val_root, PROJECT_DIR.parent)
    checkpoint_dir = resolve_path(args.checkpoint_dir, PROJECT_DIR)
    init_checkpoint = resolve_path(args.init_checkpoint, PROJECT_DIR)
    # Same logic as ColorMNetRender: local project path, never the canonical
    # HuggingFace name / the user's global cache.
    dinov3_weights_dir = os.path.join(str(PROJECT_DIR), "weights", "dinov3-vitb16")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Log to file directly from the process (see comment at the top of the
    # file) - configurable name (--log_filename, default session_console.log,
    # same name/location already used by the old tee of run_session.bat),
    # always inside checkpoint_dir. Append (no truncation) so a resume
    # continues the same file instead of overwriting it, same behavior as
    # the old 'New-Object System.IO.StreamWriter($logPath, $true, ...)'.
    log_path = os.path.join(checkpoint_dir, args.log_filename)
    log_path = os.path.join(checkpoint_dir, args.log_filename)
    init_log_file(log_path)
    print(f"[train_dinov3] Text log: {log_path}")

    cudnn.benchmark = False  # matches the original

    if args.amp_dtype != "float16":
        print(f"[train_dinov3] WARNING: --amp_dtype={args.amp_dtype} requested, but "
              f"trainer.py (untouched in this aspect) always uses the "
              f"torch.cuda.amp.autocast()/GradScaler() legacy, default dtype float16. "
              f"The parameter is accepted for interface consistency but has no effect "
              f"on the actual dtype used internally.")

    config = build_config(args, dinov3_weights_dir, val_root)

    logger = LocalTrainLogger()
    wandb_shim = LocalWandbShim()

    print(f"[train_dinov3] Building ColorMNetTrainer (backbone={args.backbone}, "
          f"accum_steps={args.accum_steps}, unlock_backbone={args.unlock_backbone})...")
    trainer = ColorMNetTrainer(config, logger=logger, save_path=None,
                                local_rank=0, world_size=1, wandb=wandb_shim,
                                accum_steps=args.accum_steps)

    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.exists(latest_path):
        print(f"[train_dinov3] Found {latest_path} - resuming from there.")
        effective_step = checkpoint_io.load_checkpoint(
            latest_path, trainer.model.module, trainer.optimizer, trainer.scaler,
            trainer.scheduler, map_location="cuda:0")
        print(f"[train_dinov3] effective_step resumed: {effective_step}")
    else:
        print(f"[train_dinov3] No checkpoint in {checkpoint_dir}, loading "
              f"init_checkpoint: {init_checkpoint}")
        src_dict = torch.load(init_checkpoint, map_location="cpu")
        trainer.load_network_in_memory(src_dict)
        del src_dict
        effective_step = 0

    train_dataset = DAVISVidevoDataset(im_root=train_root, gt_root=train_root, max_jump=5,
                                        is_bl=False, subset=None, num_frames=args.num_frames,
                                        max_num_obj=2, finetune=False)
    train_iter = infinite_train_loader(train_dataset, args.batch_size)
    val_subset = None if args.val_full else VAL_SUBSET_20
    val_dataset = DAVISTestDataset_221128_TransColorization_batch(
        data_root=val_root, imset=val_root, size=-1, subset=val_subset, max_side=args.val_max_side)
    print(f"[train_dinov3] Periodic validation on "
          f"{'full val_root' if val_subset is None else f'a fixed sample of {len(val_subset)}/{len(os.listdir(val_root))} videos'}"
          f" ({len(val_dataset)} videos), val_max_side={args.val_max_side}.")

    # best_psnr is not among the keys saved by checkpoint_io.py (only
    # effective_step/model/optimizer/scaler/scheduler) - it must be recovered
    # explicitly from the filename latest_best_psnr_*.pth, or it is lost at
    # every process restart. Same default (0) as trainer.py:116 if
    # --save_best was never used before (no file found).
    best_psnr = 0
    if args.save_best:
        found_path, found_val = find_best_checkpoint(checkpoint_dir)
        if found_val is not None:
            best_psnr = found_val
            print(f"[train_dinov3] --save_best: resumed best_psnr={best_psnr:.4f} from {found_path}")
        else:
            print(f"[train_dinov3] --save_best: no latest_best_psnr_*.pth found in "
                  f"{checkpoint_dir}, starting from best_psnr={best_psnr} (default, trainer.py:116).")

    session_log_path = os.path.join(checkpoint_dir, "session_log.csv")
    heartbeat_path = os.path.join(checkpoint_dir, "heartbeat.log")
    progress_state = heartbeat.ProgressState()
    watchdog = None
    if args.log_level != "quiet":
        watchdog = heartbeat.Watchdog(progress_state, heartbeat_path,
                                       interval_seconds=args.heartbeat_interval)
        watchdog.start()
        print(f"[train_dinov3] Watchdog started: {heartbeat_path}, every "
              f"{args.heartbeat_interval:.0f}s.")

    def val_progress_cb(video_name, video_idx, video_total, elapsed):
        progress_state.update(f"validating video {video_idx + 1}/{video_total} ({video_name})")
        if args.log_level == "verbose":
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[val] video {video_idx + 1}/{video_total} '{video_name}' - "
                  f"{elapsed:.2f}s - vram_allocated={allocated_gb:.2f}GB "
                  f"vram_reserved={reserved_gb:.2f}GB")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGBREAK"):
        # Windows: os.kill/taskkill do not deliver a manageable SIGTERM
        # (TerminateProcess forces the exit without invoking handlers). The
        # only signal deliverable in a targeted way to a SINGLE child process
        # (without killing the calling process too) is CTRL_BREAK_EVENT,
        # which Python maps to SIGBREAK - the exact same clean-exit path.
        signal.signal(signal.SIGBREAK, _signal_handler)

    session_start = time.time()
    loss_ema = None
    loss_ema_alpha = 0.1
    # --shutdown: set to True only on the exit paths "the session no longer
    # needs me" (target reached, timeout) - NEVER on manual interruption
    # (the user is there) nor implicitly on a crash (an exception skips the
    # lines that set it, it stays False).
    should_shutdown = False

    print(f"[train_dinov3] Start: effective_step={effective_step} -> "
          f"target_iterations={args.target_iterations}, checkpoint_every={args.checkpoint_every}, "
          f"max_session_hours={args.max_session_hours}")

    try:
        while True:
            if effective_step >= args.target_iterations:
                print(f"[train_dinov3] TARGET ALREADY REACHED: effective_step={effective_step} "
                      f">= target_iterations={args.target_iterations}. Exiting without running steps.")
                should_shutdown = args.shutdown
                return 0

            elapsed_h = (time.time() - session_start) / 3600.0
            if elapsed_h >= args.max_session_hours:
                checkpoint_io.save_checkpoint(latest_path, trainer.model.module, trainer.optimizer,
                                               trainer.scaler, trainer.scheduler, effective_step)
                print(f"[train_dinov3] SESSION ENDED BY TIMEOUT ({elapsed_h:.2f}h >= "
                      f"{args.max_session_hours}h). Re-run the same command to resume "
                      f"from effective_step={effective_step}.")
                should_shutdown = args.shutdown
                return 0

            # A full accumulation cycle (accum_steps micro-steps) = 1 effective
            # step. The 'it' passed to do_pass is ALWAYS >=1 (never 0): do_pass
            # has an internal automatic validation trigger 'it %
            # save_network_interval == 0' which at it=0 fires ALWAYS (0 mod N
            # == 0 for any N) regardless of the configured interval - not
            # avoidable without touching do_pass. That internal do_val is not
            # protected by torch.no_grad() (see below), so it is avoided by
            # keeping 'it' always >=1: effective_step (0-indexed, completed
            # steps) and 'it' (1-indexed, step in progress) stay coherent.
            it_for_this_step = effective_step + 1
            step_losses = []
            for _ in range(args.accum_steps):
                data = next(train_iter)
                trainer.do_pass(data, it=it_for_this_step, val_dataset=val_dataset)
                loss_val = wandb_shim.last.get("train/loss")
                if loss_val is not None:
                    step_losses.append(loss_val)

            effective_step += 1
            progress_state.update(f"training step {effective_step}/{args.target_iterations}")
            if step_losses:
                step_loss = sum(step_losses) / len(step_losses)
                loss_ema = step_loss if loss_ema is None else (
                    loss_ema_alpha * step_loss + (1 - loss_ema_alpha) * loss_ema)

            # 'do_checkpoint' originally included '_stop_requested' in the
            # same OR as checkpoint_every/target - a manual interruption
            # (Ctrl+C/SIGTERM) therefore ALWAYS ended up triggering the entire
            # do_val() on VAL_SUBSET_20 (20 videos, ~30-45s/video, several
            # minutes) before being able to actually save and exit - even
            # when checkpoint_every had been set deliberately very high to
            # avoid it (verified with a manual test: Ctrl+C on a session with
            # checkpoint_every=100000 still started the full do_val()).
            # Separated: only a REAL checkpoint_every/target boundary runs
            # validation+snapshot+save_best+session_log (unchanged, 'if'
            # branch below); an interruption that does not land on that
            # boundary does a QUICK save (only weights/optimizer/scaler/
            # scheduler to latest_path, no do_val()) in the 'elif' branch -
            # the user must be able to exit immediately, not wait for a
            # full validation pass.
            is_periodic_or_final_checkpoint = (
                (effective_step % args.checkpoint_every == 0)
                or (effective_step >= args.target_iterations)
            )

            if is_periodic_or_final_checkpoint:
                # do_val (untouched in the computation logic, only extended
                # with per-video progress_cb/reset) never disables grad
                # tracking internally (the 'with torch.no_grad():' line is
                # commented out in the original) - it inherits
                # grad_enabled=True from the last do_pass, accumulating an
                # autograd graph that is never freed -> VRAM growing without
                # limit (empirically observed: from ~3GB to >15GB in a few
                # frames). Fix on the caller side, without touching do_val:
                # we explicitly disable grad here, and free the cache
                # before/after.
                gc.collect()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    val_psnr = trainer.do_val(it=it_for_this_step, val_dataset=val_dataset,
                                               progress_cb=val_progress_cb)
                gc.collect()
                torch.cuda.empty_cache()

                checkpoint_io.save_checkpoint(latest_path, trainer.model.module, trainer.optimizer,
                                               trainer.scaler, trainer.scheduler, effective_step)
                snapshot_path = os.path.join(checkpoint_dir, f"step_{effective_step}.pth")
                checkpoint_io.save_checkpoint(snapshot_path, trainer.model.module, trainer.optimizer,
                                               trainer.scaler, trainer.scheduler, effective_step)

                # --save_best. Strict comparison (>), not >= as in
                # trainer.py:473 (save_best_network(), never reactivated - it
                # stays unreachable behind save_network_interval=10**9): a
                # tie (val_psnr==best_psnr, observed not rarely on several
                # consecutive checkpoints in the early phase) would still use
                # the same filename (latest_best_psnr_{val_psnr:.4f}
                # unchanged) - with >= it would rewrite/delete the same file
                # for nothing; with > the block does nothing, correctly.
                # Reuses the val_psnr already computed above, no extra
                # validation. Writes the NEW file FIRST (already atomic:
                # checkpoint_io.save_checkpoint writes to a temporary and
                # renames with os.replace()), THEN removes the old one - never
                # the other way around: deleting first and writing after would
                # leave a window in which a mid-way interruption would lose
                # BOTH files (no valid best left). In the worst case
                # (interrupted between the two lines) at most two files
                # remain temporarily, never zero.
                if args.save_best and val_psnr > best_psnr:
                    old_best_path, _ = find_best_checkpoint(checkpoint_dir)
                    new_best_path = os.path.join(checkpoint_dir, f"latest_best_psnr_{val_psnr:.4f}.pth")
                    checkpoint_io.save_checkpoint(new_best_path, trainer.model.module, trainer.optimizer,
                                                   trainer.scaler, trainer.scheduler, effective_step)
                    if old_best_path is not None and old_best_path != new_best_path and os.path.exists(old_best_path):
                        os.remove(old_best_path)
                    print(f"[train_dinov3] New best PSNR: {val_psnr:.4f} (previous: "
                          f"{best_psnr:.4f}) -> saved {new_best_path}")
                    best_psnr = val_psnr

                # do_val() (above) populates these two attributes on each
                # call - no extra validation.
                session_log.append_row(session_log_path, effective_step,
                                        train_loss_smoothed=loss_ema, val_psnr=val_psnr,
                                        session_elapsed_seconds=time.time() - session_start,
                                        val_deltae00_mean=trainer.last_val_deltae_mean,
                                        val_deltae00_p90=trainer.last_val_deltae_p90)
                print(f"[train_dinov3] step={effective_step} loss_smoothed={loss_ema} "
                      f"val_psnr={val_psnr} -> salvato {latest_path} + {snapshot_path}")
            elif _stop_requested:
                # Interruption that does NOT land on a checkpoint_every/
                # target boundary - quick save (only weights/optimizer/scaler/
                # scheduler to latest_path), NO validation: the confirmation
                # message is already printed by the 'if _stop_requested:'
                # block right below, unchanged.
                checkpoint_io.save_checkpoint(latest_path, trainer.model.module, trainer.optimizer,
                                               trainer.scaler, trainer.scheduler, effective_step)

            if _stop_requested:
                print(f"[train_dinov3] Interrupted (SIGINT/SIGTERM) - state saved at "
                      f"effective_step={effective_step}. Re-run the same command to resume.")
                # No shutdown here, regardless of --shutdown: a manual
                # signal means the user is at the terminal right now -
                # shutting down the PC immediately after would be an
                # undesired surprise, not the "I don't need to check on it
                # anymore" behavior for which --shutdown exists.
                return 0

            if effective_step >= args.target_iterations:
                print(f"[train_dinov3] TARGET REACHED: effective_step={effective_step} >= "
                      f"target_iterations={args.target_iterations}. Exiting.")
                should_shutdown = args.shutdown
                return 0
    finally:
        if watchdog is not None:
            watchdog.stop()
        if should_shutdown:
            _trigger_shutdown()


if __name__ == "__main__":
    sys.exit(main())
