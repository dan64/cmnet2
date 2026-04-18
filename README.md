# CMNET2 — Reference-Based Video Colorization

**CMNET2** is a deep-learning system for colorizing grayscale images and videos using colored reference frames. It is built on top of [ColorMNet](https://github.com/yyang181/colormnet) and extends it with an improved three-tier memory architecture inspired by [XMem++](https://github.com/mbzuai-metaverse/XMem2), enabling robust colorization of long videos with hundreds of reference frames.

---

## Key Features

- **Reference-based colorization** — propagates color from one or more colored reference frames to a grayscale video, operating in the LAB color space for perceptual accuracy.
- **Permanent memory (XMem++ style)** — reference frames are stored in a dedicated `perm_mem` store that is never compressed or evicted, ensuring color fidelity across the entire video.
- **Preloading API** — reference frames can be bulk-loaded into memory before colorization begins, decoupling the reference ingestion phase from the inference phase.
- **Sliding window memory management** — for long videos with thousands of reference frames, a configurable sliding window evicts the oldest references and loads new ones as the video progresses, keeping VRAM usage bounded.
- **Adaptive VRAM management** — gradual memory pressure response: slides 70% of permanent memory when VRAM drops below 500 MB, full reset only as a last resort below 100 MB.
- **DINOv2 + ResNet50 fusion backbone** — multi-scale key features are extracted by fusing DINOv2 ViT-S/14 semantic features with ResNet50 spatial features at 1/4, 1/8, and 1/16 scales.

---

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- CUDA-capable GPU (16 GB VRAM recommended for long videos)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python pillow scikit-image tqdm numpy
```

---

## Directory Structure

```
cmnet2/
├── weights/
│   └── DINOv2FeatureV6_LocalAtten_s2_154000.pth   # ColorMNet pre-trained weights
│
├── models/
│   ├── checkpoints/
│   │   ├── dinov2_vits14_pretrain.pth              # DINOv2 ViT-S/14 backbone weights
│   │   ├── resnet18-5c106cde.pth                   # ResNet18 pre-trained weights
│   │   └── resnet50-19c8e357.pth                   # ResNet50 pre-trained weights
│   │
│   └── facebookresearch_dinov2_main/               # DINOv2 source code (required by torch.hub)
│
├── assets/                                         # sample inputs/outputs for testing
├── colormnet/                                      # model source code
├── test_imge.py                                    # single image colorization script
├── test_video.py                                   # video colorization script
└── test_video_slide.py                             # long video with sliding window
```

> **Note:** The `weights/` and `models/` directories are not included in the repository.
> Download all required files from the [Releases page](https://github.com/dan64/cmnet2/releases/tag/v1.0.0) as described below.

---

## Download Model Weights

Download the following files from the [v1.0.0 Release](https://github.com/dan64/cmnet2/releases/tag/v1.0.0) and place them in the correct directories:

| File                                       | Destination           | Download                                                                                                      |
| ------------------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `DINOv2FeatureV6_LocalAtten_s2_154000.pth` | `weights/`            | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/DINOv2FeatureV6_LocalAtten_s2_154000.pth) |
| `dinov2_vits14_pretrain.pth`               | `models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/dinov2_vits14_pretrain.pth)               |
| `resnet18-5c106cde.pth`                    | `models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet18-5c106cde.pth)                    |
| `resnet50-19c8e357.pth`                    | `models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet50-19c8e357.pth)                    |
| `facebookresearch_dinov2_main.zip`         | extract to `models/`  | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/facebookresearch_dinov2_main.zip)         |

> **Note:** `facebookresearch_dinov2_main/` contains the DINOv2 source code required by
> `torch.hub` to instantiate the model. Extract the zip so that the folder is located at
> `models/facebookresearch_dinov2_main/`.

---

## Usage

### Colorize a single image

```bash
python test_imge.py \
  --input  assets/image/image_bw.jpg \
  --ref    assets/image/image_color_ref.jpg \
  --output assets/image/output.jpg
```

### Colorize a video (all references preloaded)

Reference images must be named with the target frame number embedded in the filename
(e.g. `ref_000040.jpg` → applies to frame 40).

```bash
python test_video.py \
  --input    assets/video/sample_bw.mp4 \
  --ref_path assets/video/ref/ \
  --output   assets/video/output.mp4
```

All reference frames are preloaded into `perm_mem` before colorization begins.
The first reference frame is also passed normally at frame 0 to initialize the working memory.

### Colorize a long video with sliding window

For videos with more reference frames than the configured window size, use the sliding
window script. References are loaded progressively as colorization advances through the video.

```bash
python test_video_slide.py \
  --input    assets/video_slide/sample_bw.mp4 \
  --ref_path assets/video_slide/ref/ \
  --output   assets/video_slide/output.mp4
```

Configure the window parameters at the top of `test_video_slide.py`:

```python
WINDOW_SIZE = 50   # max reference frames kept in perm_mem at any time
SLIDE_STEP  = 5    # frames removed/added per slide step
```

The slide trigger is frame-id aware: the window advances when the current video frame
surpasses the frame ID of the oldest reference that is no longer needed.

---

## Architecture

```
Grayscale input frame (L channel in LAB)
    ↓
KeyEncoder  ←  ResNet50 (1/4, 1/8, 1/16) + DINOv2 ViT-S/14 (fused via Fuse blocks)
    ↓
Key / Shrinkage / Selection tensors
    ↓
MemoryManager — 3-tier memory
    ├── perm_mem   — reference frames, never evicted         ← XMem++ extension
    ├── work_mem   — recent colorized frames (LRU tracking)
    └── long_mem   — compressed prototypes (128 per consolidation)
    ↓
Memory readout (scaled L2 affinity + softmax, top-k=30)
    ↓
ValueEncoder  ←  ResNet18-based, fuses image features + memory readout
    ↓
Decoder (GRU hidden state + upsampling blocks)
    ↓
AB color channels → LAB → RGB → colorized frame
```

### Core classes

| Class                  | File                                    | Description                                                                                |
| ---------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------ |
| `ColorMNetRender`      | `colormnet/colormnet_render.py`         | Public API. Singleton. Handles GPU memory, reference management, sliding window.           |
| `InferenceCore`        | `colormnet/inference/inference_core.py` | Frame-by-frame inference loop. Exposes `step()`, `step_AnyExemplar()`, `load_reference()`. |
| `MemoryManager`        | `colormnet/inference/memory_manager.py` | Manages `perm_mem`, `work_mem`, `long_mem`. Handles consolidation and sliding.             |
| `ColorMNet`            | `colormnet/model/network.py`            | Top-level `nn.Module`.                                                                     |
| `KeyEncoder_DINOv2_v6` | `colormnet/model/modules.py`            | DINOv2 + ResNet50 fusion backbone.                                                         |

---

## Public API

```python
from colormnet.colormnet_render import ColorMNetRender
from PIL import Image

colorizer = ColorMNetRender(
    image_size=-1,          # -1 = original resolution
    vid_length=1000,        # total number of frames to colorize
    max_memory_frames=5000, # long-term memory capacity
    encode_mode=1,          # 0=remote, 1=async, 2=sync
    project_dir="."
)

# Option A — preload all references before colorization
for ref_img in reference_images:
    colorizer.preload_reference(ref_img)          # loads into perm_mem

colorizer.set_ref_frame(reference_images[0])      # initialize work_mem
frame_colored = colorizer.colorize_frame(ti=0, frame_i=grayscale_frame)

# Option B — pass reference alongside each frame
colorizer.set_ref_frame(ref_img)
frame_colored = colorizer.colorize_frame(ti=i, frame_i=grayscale_frame)

# Sliding window control
count = colorizer.get_perm_mem_frame_count()      # current perm_mem size
colorizer.slide_permanent_memory(n_frames=50)     # evict oldest 50 refs
```

---

## Differences from the original ColorMNet

| Feature                | Original ColorMNet         | CMNET2                                   |
| ---------------------- | -------------------------- | ---------------------------------------- |
| Memory stores          | working + long-term        | **permanent** + working + long-term      |
| Reference handling     | passed with each frame     | **preloadable in bulk** before inference |
| Long video support     | resets memory periodically | **sliding window** over permanent memory |
| VRAM pressure response | full reset                 | **graduated**: slide 70% → full reset    |
| `reset_on_ref_update`  | active                     | deprecated (permanent memory handles it) |

---

## Credits

CMNET2 is based on:

- **ColorMNet** — [yyang181/colormnet](https://github.com/yyang181/colormnet)
- **XMem** — [hkchengrex/XMem](https://github.com/hkchengrex/XMem)
- **XMem++** — [mbzuai-metaverse/XMem2](https://github.com/mbzuai-metaverse/XMem2)
- **DINOv2** — [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

---

## License

This project inherits the license terms of the original ColorMNet repository.
Please refer to the [original repository](https://github.com/yyang181/colormnet) for details.
