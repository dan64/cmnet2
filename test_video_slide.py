import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from skimage import color
from argparse import ArgumentParser
from pathlib import Path
import sys

# Assicura che i moduli locali siano trovati
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

# configuring torch
import torch

from colormnet.colormnet_render import ColorMNetRender

def cv2_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# parametri sliding window
WINDOW_SIZE = 6      # max reference frame in perm_mem
SLIDE_STEP = 3        # quanti rimuovere/aggiungere per volta

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', default='./assets/video_slide/sample_bw.mp4', help='video target')
    parser.add_argument('--ref_path', default='./assets/video_slide/ref', help='immagini di riferimento a colori')
    parser.add_argument('--output', default='./assets/video_slide/sample_bw_slide_cmnet2.mp4', help='Risultato colorizzato')
    args = parser.parse_args()

    torch.hub.set_dir(model_dir)

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {args.input}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MP4 Direct
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    refs = sorted([f for f in os.listdir(args.ref_path) if f.lower().endswith(('.png', '.jpg'))])
    refs_id = sorted(int(''.join(filter(str.isdigit, f))) for f in refs if any(c.isdigit() for c in f))

    print("--- Caricamento modello CMNET2 ---")
    colorizer = ColorMNetRender(image_size=-1, vid_length=total_frames, enable_resize=False,
                               encode_mode=1, max_memory_frames=1000, reset_on_ref_update=False,
                               project_dir=package_dir)

    # fase 1: precarica i primi WINDOW_SIZE reference
    print("Precaricamento reference...")
    refs_loaded = 0
    for f in refs[:WINDOW_SIZE]:
        ref = Image.open(os.path.join(args.ref_path, f)).convert('RGB')
        colorizer.preload_reference(ref)
        refs_loaded += 1
    refs_queue_idx = refs_loaded  # indice del prossimo ref da caricare

    print(f"perm_mem frames: {colorizer.get_perm_mem_frame_count()}")

    # fase 2: colora i frame con sliding window
    first_ref = Image.open(os.path.join(args.ref_path, refs[0])).convert('RGB')

    print(f"--- [VIDEO] Salvataggio di {total_frames} frames a colori ---")
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break

        # sliding: quando il frame corrente supera il frame_id
        # del (refs_queue_idx - WINDOW_SIZE + SLIDE_STEP)-esimo reference
        # significa che i primi SLIDE_STEP reference sono "nel passato"
        if refs_queue_idx < len(refs):
            oldest_ref_still_needed = refs_id[refs_queue_idx - WINDOW_SIZE + SLIDE_STEP]
            if i > oldest_ref_still_needed:
                colorizer.slide_permanent_memory(SLIDE_STEP)
                for f in refs[refs_queue_idx:refs_queue_idx + SLIDE_STEP]:
                    ref = Image.open(os.path.join(args.ref_path, f)).convert('RGB')
                    colorizer.preload_reference(ref)
                refs_queue_idx += SLIDE_STEP
        # primo frame: passa il ref normalmente per inizializzare work_mem
        colorizer.set_ref_frame(first_ref if i == 0 else None)

        # Frame
        rgb_frame = cv2_to_pil(frame)
        img_color = colorizer.colorize_frame(ti=i, frame_i=rgb_frame)
        final_bgr = pil_to_cv2(img_color)
        out_video.write(final_bgr)

    cap.release()
    out_video.release()
    print(f"\n--- [FINE] Video salvato in : {args.output} ---")

if __name__ == '__main__':
    main()
