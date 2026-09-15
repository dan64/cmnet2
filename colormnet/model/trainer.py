"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import gc
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from colormnet.model.network import ColorMNet

from colormnet.model.losses import LossComputer
from colormnet.util.log_integrator import Integrator
from colormnet.util.image_saver import pool_pairs_221128_TransColorization

# val
from torch.utils.data import DataLoader
from colormnet.inference.data.mask_mapper import MaskMapper
from colormnet.inference.inference_core import InferenceCore
import torch.nn.functional as F
from PIL import Image
from colormnet.util.transforms import lab2rgb_transform_PIL, calculate_psnr
from colormnet.dataset.range_transform import RGB2Lab, ToTensor, im_rgb2lab_normalization
# Second validation metric (delta CIEDE2000, perceptual color difference)
# computed on the same uint8 RGB out_img/gt_img already used for
# calculate_psnr() above - see do_val(). Purely additive,
# calculate_psnr() is not touched.
from skimage.color import rgb2lab, deltaE_ciede2000

# Every line printed by this module (both the scattered print() calls in
# do_val/do_pass/save_*, and the "[train/...]" lines produced by
# LocalTrainLogger.log_metrics/log_string when invoked from here) must
# carry a timestamp and keep being tracked to file even in long sessions.
# Centralized in colormnet/util/console_log.py (shared with train_dinov3.py,
# which calls init_log_file() once at session start) - it writes to file
# DIRECTLY from the Python process, without depending on an external tee
# (PowerShell/run_session.bat), because that pipe was verified to prevent
# reliable delivery of a real Ctrl+C to the process. If nobody ever called
# init_log_file() (e.g. the diagnostic scripts in training/verify/, which
# import this module but do not go through train_dinov3.py), log() behaves
# like a plain print() with a timestamp only, no file - unchanged behavior
# for those cases.
from colormnet.util.console_log import log as print


class _SingleProcessWrapper(nn.Module):
    """Minimal stand-in for DistributedDataParallel for world_size=1 (no real
    communication needed, a single process). Exposes .module the way DDP does,
    so all the self.model.module.* calls in do_pass/do_val (untouched) and
    self.model('mode', ...) keep working unchanged."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ColorMNetTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1, wandb = None,
                 accum_steps: int = 1):
        self.wandb= wandb
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank
        # Gradient accumulation: accum_steps=1 -> identical behavior to the
        # original. accum_steps>1 -> zero_grad() only on the first micro-step
        # of the cycle, optimizer.step()/scaler.update()/scheduler.step() only
        # on the last one. self._accum_counter counts the micro-steps called
        # (do_pass), not the effective steps - the effective step (for
        # scheduler/cadences) stays the one passed explicitly as 'it' by the
        # caller.
        self.accum_steps = accum_steps
        self._accum_counter = 0

        # world_size==1 (single process/GPU): no real DDP or distributed reduce
        # - those would require an initialized torch.distributed process group,
        # which on Windows/gloo in this environment does not work with any
        # available backend (verified: gloo always fails in creating the
        # transport device, NCCL/MPI absent on Windows). _SingleProcessWrapper
        # exposes .module the way DDP does, so do_pass/do_val (untouched) keep
        # working unchanged.
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                ColorMNet(config).cuda(),
                device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
        else:
            self.model = _SingleProcessWrapper(ColorMNet(config).cuda())

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size', str(sum([param.nelement() for param in self.model.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=(world_size > 1), local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

        self.best_psnr = 0
        self.best_it = 0
        # Populated by do_val() on each call (mean/p90 of deltaE00 over the
        # last validation run), None until do_val() has ever been invoked.
        self.last_val_deltae_mean = None
        self.last_val_deltae_p90 = None

    def do_val(self, it=0, val_dataset=None, progress_cb=None):
        self.model.module.eval()
        self.val()

        print('starting validation at it: %s'%(it))
        val_loader = val_dataset.get_datasets()
        video_total = len(val_dataset)

        config = {}
        config['enable_long_term'] = True
        config['max_mid_term_frames'] = 10
        config['min_mid_term_frames'] = 5
        config['max_long_term_elements'] = 10000
        config['num_prototypes'] = 128
        config['benchmark'] = False
        config['flip'] = False
        config['top_k'] = 30
        config['mem_every'] = 5
        config['deep_update_every'] = -1
        config['hidden_dim'] = 64

        avg_psnr = []
        wb_frames = []
        # Session-level lists, one entry per video (mean/p90 already aggregated
        # for that video) - avg_deltae_p90 below is therefore the mean of the
        # per-video p90s, not the global p90 over all pixels of all videos
        # (that would require accumulating the entire validation set in memory
        # - not done here, see the note at the end of the function).
        all_video_deltae_means = []
        all_video_deltae_p90 = []

        # with torch.no_grad():
        for video_idx, vid_reader in enumerate(val_loader):
            t_video_start = time.time()
            clip_psnr = []
            # An HxW (ravel) array of deltaE00 for every validated frame of
            # this video, concatenated at the end of the video.
            clip_deltae_frames = []
            loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
            vid_name = vid_reader.vid_name
            vid_length = len(loader)
            # no need to count usage for LT if the video is not that long anyway
            config['enable_long_term_count_usage'] = (
                config['enable_long_term'] and
                (vid_length
                    / (config['max_mid_term_frames']-config['min_mid_term_frames'])
                    * config['num_prototypes'])
                >= config['max_long_term_elements']
            )

            mapper = MaskMapper()
            processor = InferenceCore(self.model.module, config=config)
            first_mask_loaded = False

            t0 = 0
            for ti, data in enumerate(loader):
                t0 += 1 
                if (t0-1) % 10 != 0: # Do not validate every frame for speed up training
                    continue

                with torch.cuda.amp.autocast(enabled=not config['benchmark']):
                    rgb = data['rgb'].cuda()[0]
                    msk = data.get('mask')
                    msk = msk[:,1:3,:,:] if msk is not None else None

                    info = data['info']
                    frame = info['frame'][0]
                    shape = info['shape']
                    need_resize = info['need_resize'][0]

                    """
                    For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
                    Seems to be very similar in testing as my previous timing method 
                    with two cuda sync + time.time() in STCN though 
                    """

                    if not first_mask_loaded:
                        if msk is not None:
                            first_mask_loaded = True
                        else:
                            # no point to do anything without a mask
                            continue

                    if config['flip']:
                        rgb = torch.flip(rgb, dims=[-1])
                        msk = torch.flip(msk, dims=[-1]) if msk is not None else None

                    # Map possibly non-continuous labels to continuous ones
                    if msk is not None:
                        msk = torch.Tensor(msk[0]).cuda()
                        if need_resize:
                            msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]

                        processor.set_all_labels(list(range(1,3)))
                        labels = range(1,3)
                    else:
                        labels = None
            
                    # Run the model on this frame
                    # print('******************* START %s *************'%ti)
                    prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))
                    # print('******************* END %s *************'%ti)

                    # Upsample to original size if needed (only the predicted
                    # chrominance - 'rgb' is no longer resized here: see below
                    # for the why).
                    if need_resize:
                        prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

                    if config['flip']:
                        prob = torch.flip(prob, dims=[-1])

                    # Save the mask
                    if info['save'][0]:
                        gt_folder = self.config['validation_root']
                        gt_path = os.path.join(gt_folder, info['vid_name'][0], info['frame'][0])
                        gt_img = np.array(Image.open(gt_path))

                        if need_resize:
                            # When --val_max_side has enabled a resize, 'rgb'
                            # (built by VideoReader) is ALREADY at the processing
                            # resolution (resized upstream, BEFORE arriving here)
                            # - "simply not resizing it" would leave it small,
                            # shape-incompatible with 'prob' (upsampled to
                            # 'shape'). A genuinely native L is needed: it is
                            # derived from gt_img, the SAME source file the
                            # grayscale frame comes from (same path as 'im_path'
                            # in VideoReader, never resized), with the exact
                            # same normalization pipeline (RGB2Lab+ToTensor+
                            # im_rgb2lab_normalization) used for the network
                            # inputs, so the numeric scale stays coherent with
                            # 'prob'. Aligned with the real production path
                            # (colormnet_render.py/chroma_transfer): final
                            # luminance always from the original frame, never
                            # passed through a resize - only the predicted
                            # chrominance is.
                            native_lab = im_rgb2lab_normalization(ToTensor()(RGB2Lab()(gt_img))).cuda()
                            l_channel = native_lab[:1, :, :]
                        else:
                            # No resize involved (library clips, never subject
                            # to --val_max_side): unchanged behavior, bit-for-bit
                            # identical to before.
                            l_channel = rgb[:1, :, :]

                        out_mask_final = lab2rgb_transform_PIL(torch.cat([l_channel, prob], dim=0))
                        out_mask_final = out_mask_final * 255
                        out_mask_final = out_mask_final.astype(np.uint8)

                        out_img = np.array(Image.fromarray(out_mask_final))
                        psnr = calculate_psnr(gt_img, out_img)
                        # avg_psnr.append(psnr)
                        clip_psnr.append(psnr)

                        # delta CIEDE2000 on the same uint8 RGB out_img/gt_img
                        # just used for the PSNR above (no alteration of the
                        # PSNR calculation).
                        out_lab = rgb2lab(out_img.astype(np.float64) / 255.0)
                        gt_lab = rgb2lab(gt_img.astype(np.float64) / 255.0)
                        deltae_map = deltaE_ciede2000(gt_lab, out_lab)
                        clip_deltae_frames.append(deltae_map.ravel())

                        save_shape = (384, 384) # resize to save wandb space
                        out_img = np.array(Image.fromarray(out_img).resize(save_shape, resample=0))
                        gt_img = np.array(Image.fromarray(gt_img).resize(save_shape, resample=0))

                        wb_frames.append(self.wandb.Image(out_img, caption="Pred_%s_it%s"%(t0, it)))
                        wb_frames.append(self.wandb.Image(gt_img, caption="GT%s_it%s"%(t0, it)))
            self.wandb.log({"val/pairs": wb_frames},step=it)


            # deltaE00 aggregation for this video (mean + p90 - the p90
            # catches concentrated local errors, e.g. edge bleeding, which a
            # plain mean over the whole frame masks).
            video_deltae_all = np.concatenate(clip_deltae_frames)
            video_deltae_mean = float(np.mean(video_deltae_all))
            video_deltae_p90 = float(np.percentile(video_deltae_all, 90))
            all_video_deltae_means.append(video_deltae_mean)
            all_video_deltae_p90.append(video_deltae_p90)

            print('current item: %s clip_psnr is: %s deltae00_mean is: %.4f deltae00_p90 is: %.4f'%(
                info['vid_name'][0], np.mean(clip_psnr), video_deltae_mean, video_deltae_p90))
            avg_psnr += clip_psnr

            video_elapsed = time.time() - t_video_start
            if progress_cb is not None:
                progress_cb(video_name=vid_name, video_idx=video_idx,
                            video_total=video_total, elapsed=video_elapsed)

            # 'processor'/'mapper' are recreated from scratch for every video
            # (no InferenceCore/MemoryManager instance survives to the next
            # video), so application state does NOT persist. But no
            # torch.cuda.empty_cache() was ever called between one video and
            # the next: with 131 videos of highly variable native resolution
            # (123 distinct resolutions over 500 clips, versus the uniform
            # dimensions of DAVIS), the blocks freed by PyTorch's cache
            # allocator for one video are almost never reusable by the next
            # video (different shape) - the "reserved" memory can grow
            # monotonically through the validation pass until it spills into
            # Windows shared memory (WDDM driver), with an order-of-magnitude
            # slowdown that shows up as a hang. Fix: same pattern already used
            # in train_dinov3.py around the entire do_val() call
            # (gc.collect()+torch.cuda.empty_cache()), applied here between one
            # video and the next instead of once for the entire validation.
            del processor, mapper
            gc.collect()
            torch.cuda.empty_cache()

        self.wandb.log({'val/psnr': np.mean(avg_psnr)}, step=it)
        self.logger.log_scalar('val/psnr', np.mean(avg_psnr), it)

        # Mean of the per-video means and mean of the per-video p90s (NOT the
        # global p90 over all pixels of all videos in the sample - that would
        # require accumulating the entire validation set in memory, not done
        # here; if a true global p90 is ever needed it must be recomputed
        # differently). Exposed as instance attributes (not in the return,
        # to not alter the signature of do_val() and the existing callers
        # that expect a single float/avg_psnr).
        avg_deltae_mean = float(np.mean(all_video_deltae_means))
        avg_deltae_p90 = float(np.mean(all_video_deltae_p90))
        self.last_val_deltae_mean = avg_deltae_mean
        self.last_val_deltae_p90 = avg_deltae_p90

        print('finish validation at it: %s avg_psnr: %s best_psnr: %s best_it: %s '
              'avg_deltae00_mean: %.4f avg_deltae00_p90(mean of the per-video p90s): %.4f'%(
                  it, np.mean(avg_psnr), self.best_psnr, self.best_it, avg_deltae_mean, avg_deltae_p90))

        return np.mean(avg_psnr)

    def _freeze_calibrated_batchnorms(self):
        """Selective override ON TOP of the "everything in train()" fix
        (the fix around trainer.py:602-620, NOT touched here - it stays the
        baseline: without this override the whole model would stay in real
        .train(), including the Dropout(p=0.1) of short_term_attn.dw_conv
        (discovered in a previous session) which MUST stay active - for this
        reason the solution is not a return to self.model.eval(), but putting
        back into .eval() only the BatchNorms listed below, AFTER do_pass()
        has already put everything else in .train() (called right after
        self.train() in do_pass()).

        Standard "freeze BN" practice in fine-tuning when the physical batch
        is too small (here: 1) to produce reliable batch statistics - the
        variance computed on a single sample is noise, not a useful signal.
        It applies to:
          - key_encoder's resnet50 (conv1/bn1/res2/layer2/layer3 - the same
            5 submodules, network2/fuse1-3 deliberately EXCLUDED, iterated
            individually below instead of doing key_encoder.modules() to
            avoid touching them by mistake): inherited from the mature
            154k-iteration checkpoint, statistics already valid.
          - value_encoder (ResNet18): same reason.

          - value_encoder (ResNet18): same reason.
          - key_encoder.network2.proj (the DinoV3 adapter):
            added here after the one-time recalibration
            (training/verify/recalibrate_proj_bn.py, momentum=None/cumulative
            moving average, 400 forward passes - verified stable, two
            independent runs converge within 1.5% relative). Before that
            recalibration its BN intentionally stayed in .train() so as not
            to freeze it at the arbitrary default values
            (running_mean=0/running_var=1) which never represented the real
            statistics - now that it has been recalibrated on a mature
            checkpoint, it behaves like resnet50/value_encoder.
          - 'dinov2' guard: the backbone='dinov2' branch of
            key_encoder.network2 (Segmentor class, not Segmentor_DINOv3)
            has no .proj attribute - hasattr() excludes it safely, no crash
            on that backbone.

        decoder/key_proj/fuse1-3/short_term_attn have no BatchNorm at all
        (GroupNorm/LayerNorm) - they do not appear here because there is
        nothing to freeze. proj's gamma/beta (the trainable parameters of
        its BN) keep receiving gradients normally - .eval() does not touch
        requires_grad, only the normalization behavior (running stats vs
        batch stats).
        """
        key_encoder = self.model.module.key_encoder
        resnet50_children = (key_encoder.conv1, key_encoder.bn1, key_encoder.res2,
                              key_encoder.layer2, key_encoder.layer3)
        for child in resnet50_children:
            for m in child.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

        for m in self.model.module.value_encoder.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        if hasattr(key_encoder.network2, "proj"):
            key_encoder.network2.proj[1].eval()

    def do_pass(self, data, it=0, val_dataset=None):
        self.model.module.train()
        self.train()
        self._freeze_calibrated_batchnorms()

        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        wb_frames = []
        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            # image features never change, compute once
            key, shrinkage, selection, f16, f8, f4 = self.model('encode_key', frames)

            filler_one = torch.zeros(1, dtype=torch.int64)
            # device=frames.device: necessary fix, same bug already
            # documented for the VRAM probe - the original omits 'device',
            # causing a device-mismatch crash outside the exact DDP
            # multi-GPU context in which this code ran.
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]), device=frames.device)

            v16, hidden = self.model('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
            
            values = v16.unsqueeze(3) # add the time dimension

            for ti in range(1, self.num_frames):
                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                else:
                    # pick num_ref_frames random frames
                    # this is not very efficient but I think we would
                    # need broadcasting in gather which we don't have
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None

                # Segment frame ti
                memory_readout = self.model('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                
                # short term memory
                memory_readout_short = self.model('read_memory_short', key[:,:,ti], key[:,:,ti-1], values[:, :, :, ti-1])
                memory_readout += memory_readout_short

                hidden, logits, masks = self.model('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
                        hidden, selector, h_out=(ti < (self.num_frames-1)))

                # No need to encode the last frame
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob
                    v16, hidden = self.model('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3)

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits

                if self._is_train:
                    if it % self.log_image_interval == 0 and it != 0:
                        if self.logger is not None:

                            out_mask_final = lab2rgb_transform_PIL(torch.cat([frames[0,ti,:1,:,:], masks[0]], dim=0))
                            out_mask_final = out_mask_final * 255
                            out_mask_final = out_mask_final.astype(np.uint8)
                            out_img = np.array(Image.fromarray(out_mask_final))
                            
                            gt_image_final = lab2rgb_transform_PIL(torch.cat([frames[0,ti,:1,:,:], data['cls_gt'][0,ti]], dim=0))
                            gt_image_final = gt_image_final * 255
                            gt_image_final = gt_image_final.astype(np.uint8)
                            gt_img = np.array(Image.fromarray(gt_image_final))

                            wb_frames.append(self.wandb.Image(out_img, caption="Pred_%s"%ti))
                            wb_frames.append(self.wandb.Image(gt_img, caption="GT_%s"%ti))


            if self._do_log or self._is_train:
                losses = self.loss_computer.compute_l1loss({**data, **out}, num_filled_objects, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)

                    self.wandb.log({'train/loss': losses['total_loss'].item()}, step=it)
                    self.wandb.log({'train/dice_loss_7': losses['dice_loss_7'].item()}, step=it)
                    self.wandb.log({'train/lr': self.scheduler.get_last_lr()[0]}, step=it)

                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384) # resize to save wandb space
                                self.logger.log_cv2('train/pairs', pool_pairs_221128_TransColorization(images, size, num_filled_objects), it)

                                self.wandb.log({"train/pairs": wb_frames},step=it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0 and it >= 129999:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # Backward pass (optional gradient accumulation, accum_steps=1 by
        # default -> bit-for-bit identical to the original, see comment in __init__)
        is_first_microstep = (self._accum_counter % self.accum_steps == 0)
        is_last_microstep = ((self._accum_counter + 1) % self.accum_steps == 0)

        if is_first_microstep:
            self.optimizer.zero_grad(set_to_none=True)

        loss = losses['total_loss']
        if self.accum_steps > 1:
            loss = loss / self.accum_steps

        if self.config['amp']:
            self.scaler.scale(loss).backward()
            if is_last_microstep:
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss.backward()
            if is_last_microstep:
                self.optimizer.step()

        self._accum_counter += 1

        if is_last_microstep:
            self.scheduler.step()

        # validation
        if it % self.save_network_interval == 0: # log_text_interval
            current_psnr = self.do_val(it, val_dataset=val_dataset)

            if current_psnr >= self.best_psnr:
                self.best_psnr = current_psnr
                self.best_it = it
                self.save_best_network(it)


    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.model.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_best_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_best.pth'
        torch.save(self.model.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.model.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.model.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Fix (was 'self.model.eval()', a bug inherited verbatim from the
        # upstream colormnet/XMem, never fixed in any git history available):
        # it put the ENTIRE model in eval() at every do_pass() (trainer.py:
        # 364-365 calls self.model.module.train() and IMMEDIATELY AFTER
        # self.train(), which overwrote that train() a moment later) - no
        # BatchNorm of the model (resnet50/value_encoder/proj, 59 out of 59
        # verified) ever updated its running stats throughout the dinov3_full
        # run (num_batches_tracked=0 after 36000 steps). The DinoV3 backbone
        # (frozen by default) has no BatchNorm (verified: 25 LayerNorm, 0
        # BatchNorm) - no risk that this fix "recalibrates" it: it stays
        # frozen via requires_grad_(False)+torch.no_grad() in resnet.py,
        # independent of this train()/eval().
        self.model.train()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.model.eval()
        return self
