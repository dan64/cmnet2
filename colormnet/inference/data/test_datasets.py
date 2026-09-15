import os
from os import path
import json

from colormnet.inference.data.video_reader import VideoReader_221128_TransColorization


class DAVISTestDataset_221128_TransColorization_batch:
    def __init__(self, data_root, imset='2017/val.txt', size=-1, subset=None, max_side=-1):
        """
        subset - if provided (iterable of video names), limits
            self.vid_list to only these videos (in the sorted order of
            data_root, not in the order of 'subset'), instead of using all
            folders present in data_root. None (default) = original behavior,
            no subset. Allows reusing the same class both for the reduced
            periodic validation (fixed sample) and for a final evaluation on
            the entire val_root.
        max_side - passed through unchanged to every
            VideoReader_221128_TransColorization (see there for details).
            <=0 (default) = no cap, original behavior.
        """
        self.image_dir = data_root
        self.mask_dir = imset
        self.size_dir = data_root
        self.size = size
        self.max_side = max_side

        self.vid_list = [clip_name for clip_name in sorted(os.listdir(data_root)) if
                         clip_name != '.DS_Store' and not clip_name.startswith('.')]

        if subset is not None:
            subset = set(subset)
            missing = subset - set(self.vid_list)
            if missing:
                raise ValueError(f"subset contains videos absent from {data_root}: {sorted(missing)}")
            self.vid_list = [v for v in self.vid_list if v in subset]

        # print(lst, len(lst), self.vid_list, self.vid_list_DAVIS2016, path.join(data_root, 'ImageSets', imset));assert 1==0

    def get_datasets(self):
        for video in self.vid_list:
            # print(self.image_dir, video, path.join(self.image_dir, video));assert 1==0
            # use_all_mask=False (was True) - with True, data['mask'] was
            # populated with the ground truth on EVERY frame (not just the
            # first exemplar), triggering the bypass 'if mask is not None:
            # pred_prob_with_bg = mask' on every processed frame in
            # InferenceCore.step() (colormnet/inference/inference_core.py:
            # 118-123) - the network (network.segment(), the decoder) was
            # NEVER called during do_val(), which therefore only measured the
            # fidelity of the pad/resize round-trip of the ground truth, with
            # zero dependence on the model's weights (isolated and confirmed
            # with an instrumented test). False restores the intended use of
            # step(): mask only on the first frame (initial exemplar,
            # gt_path == self.first_gt_path in video_reader.py:128), true
            # propagation of the model on the subsequent frames.
            yield VideoReader_221128_TransColorization(vid_name=video,
                                                       image_dir=path.join(self.image_dir, video),
                                                       mask_dir=path.join(self.mask_dir, video),
                                                       size=self.size,
                                                       to_save=None,
                                                       use_all_mask=False,
                                                       size_dir=path.join(self.size_dir, video),
                                                       max_side=self.max_side,
                                                       )

    def __len__(self):
        return len(self.vid_list)
