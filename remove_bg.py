import os
import cv2
import numpy as np
import argparse
from rembg import remove, new_session
from PIL import Image


# 🔥 créer UNE seule session (important pour perf)
session = new_session("u2net_human_seg")


def remove_background_white(input_path, output_path):
    try:
        img = Image.open(input_path).convert("RGBA")

        # ✅ utilisation correcte
        output = remove(img, session=session)

        data = np.array(output).astype(np.float32)

        rgb = data[:, :, :3]
        alpha = data[:, :, 3] / 255.0

        # =========================
        # AMÉLIORATION MASQUE
        # =========================
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        alpha = np.clip(alpha * 1.2, 0, 1)

        # =========================
        # FOND BLANC
        # =========================
        white_bg = np.ones_like(rgb) * 255

        result = rgb * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])
        result = np.clip(result, 0, 255).astype(np.uint8)

        Image.fromarray(result).save(output_path, quality=95)

    except Exception as e:
        raise RuntimeError(f"Erreur traitement image : {e}")


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        try:
            remove_background_white(input_path, output_path)
            print(f"✅ {file} traité proprement")
        except Exception as e:
            print(f"❌ {file} : {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background and set white background")

    parser.add_argument("input", help="Dossier d'entrée")
    parser.add_argument("output", help="Dossier de sortie")

    args = parser.parse_args()

    process_folder(args.input, args.output)