import os
import torch
import numpy as np
import cv2
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

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', default='./assets/image/image_bw.jpg', help='Immagine target (grigio o colore, verrà estratto L)')
    parser.add_argument('--ref', default='./assets/image/image_color_ref.jpg', help='Immagine di riferimento a colori')
    parser.add_argument('--output', default='./assets/image/image_bw_cmnet2.jpg', help='Risultato colorizzato')
    args = parser.parse_args()

    torch.hub.set_dir(model_dir)

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    # 1. Caricamento immagini
    ref_raw = Image.open(args.ref).convert('RGB')
    target_raw = Image.open(args.input).convert('RGB')
    if ref_raw is None or target_raw is None:
        print("❌ Errore nel caricamento delle immagini. Verifica i percorsi.")
        return

    print("--- Caricamento modello CMNET2 ---")
    colorizer = ColorMNetRender(image_size=-1, vid_length=1, enable_resize=False,
                               encode_mode=1, max_memory_frames=1000, reset_on_ref_update=False,
                               project_dir=package_dir)

    colorizer.set_ref_frame(ref_raw, False)
    img_color = colorizer.colorize_frame(ti=0, frame_i=target_raw)

    img_color.save(args.output)
    img_color.show()

if __name__ == '__main__':
    main()
