import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as Ff
from PIL import Image
import numpy as np

from colormnet.dataset.range_transform import im_normalization, im_rgb2lab_normalization, ToTensor, RGB2Lab


def compute_process_size(orig_w, orig_h, max_side):
    """Same as compute_process_size() in test_video_full.py (same pattern
    already used for inference/rendering, reused here for periodic
    validation) - caps the longest side to max_side preserving the aspect
    ratio, dimensions rounded to the nearest even number. Returns the
    original dimensions if max_side<=0 or if the frame is already smaller
    than max_side."""
    if max_side <= 0:
        return orig_w, orig_h
    longest = max(orig_w, orig_h)
    if longest <= max_side:
        return orig_w, orig_h
    scale = max_side / longest
    new_w = int(round(orig_w * scale / 2)) * 2
    new_h = int(round(orig_h * scale / 2)) * 2
    return new_w, new_h


class VideoReader_221128_TransColorization(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None,
                 max_side=-1):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        max_side - caps the longest side of the frame (and of the mask,
            resized identically) before processing it, to limit the VRAM
            peak on frames with very high native resolution (library clips
            up to 1920px, against the uniform dimensions of DAVIS) - without
            touching 'size'/'need_resize' already used for the rendering
            resize. No effect on PSNR fidelity: the prediction is still
            re-projected to the original resolution ('shape', captured BEFORE
            this resize) inside do_val() through the same upsample mechanism
            already in place for need_resize. No effect if <=0 (default).
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        self.max_side = max_side
        # print('use_all_mask', use_all_mask);assert 1==0
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        self.frames = [img for img in sorted(os.listdir(self.image_dir)) if (img.endswith('.jpg') or img.endswith('.png')) and not img.startswith('.')]
        self.palette = Image.open(path.join(mask_dir, sorted([msk for msk in os.listdir(mask_dir) if not msk.startswith('.')])[0])).getpalette()
        self.first_gt_path = path.join(self.mask_dir, sorted([msk for msk in os.listdir(self.mask_dir) if not msk.startswith('.')])[0])
        self.suffix = self.first_gt_path.split('.')[-1]

        if size < 0:
            self.im_transform = transforms.Compose([
                RGB2Lab(),
                ToTensor(),
                im_rgb2lab_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = size


    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['vid_name'] = self.vid_name
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        # The 'shape' above was already captured from the ORIGINAL image,
        # untouched by this resize - it stays the correct target for the
        # subsequent upsample of the prediction in do_val(). new_w/new_h
        # remain in scope for the mask resize below only if max_side_resize
        # is True (the exact same resizing, otherwise rgb and ab would not
        # have the same spatial dimensions when concatenated).
        max_side_resize = False
        if self.max_side > 0:
            orig_w, orig_h = img.size
            new_w, new_h = compute_process_size(orig_w, orig_h, self.max_side)
            if (new_w, new_h) != (orig_w, orig_h):
                img = img.resize((new_w, new_h), Image.BILINEAR)
                max_side_resize = True

        gt_path = path.join(self.mask_dir, sorted(os.listdir(self.mask_dir))[idx]) if idx < len(os.listdir(self.mask_dir)) else None

        img = self.im_transform(img)
        img_l = img[:1,:,:]
        img_lll = img_l.repeat(3,1,1)

        load_mask = (self.use_all_mask or (gt_path == self.first_gt_path)) and gt_path is not None
        if load_mask and path.exists(gt_path):
            mask = Image.open(gt_path).convert('RGB')
            if max_side_resize:
                mask = mask.resize((new_w, new_h), Image.BILINEAR)
            mask = self.im_transform(mask)

            # keep L channel of reference image in case First frame is not exemplar
            # mask_ab = mask[1:3,:,:]
            # data['mask'] = mask_ab
            data['mask'] = mask

        info['shape'] = shape
        info['need_resize'] = (not (self.size < 0)) or max_side_resize
        data['rgb'] = img_lll
        data['info'] = info

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        if self.size < 0:
            # With size<0, 'need_resize' can be True only because of the
            # max_side cap (never because of this self.size, which would be
            # negative) - in that case rgb AND mask have already been
            # resized identically inside __getitem__, BEFORE im_transform.
            # No further resize here: using self.size (<0) in the formula
            # below would produce a negative target dimension.
            return mask
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return Ff.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)),
                    mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
