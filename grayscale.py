import cv2
import os
import numpy as np
import re


# ==========================================
# TRI NUMÉRIQUE
# ==========================================
def natural_sort_key(filename):

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', filename)
    ]


# ==========================================
# LECTURE IMAGE SAFE WINDOWS
# ==========================================
def imread_unicode(path):

    try:
        stream = np.fromfile(path, np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img

    except Exception:
        return None


# ==========================================
# SAVE IMAGE SAFE WINDOWS
# ==========================================
def imwrite_unicode(path, img):

    ext = os.path.splitext(path)[1]

    result, encoded = cv2.imencode(ext, img)

    if result:
        encoded.tofile(path)


# ==========================================
# SHARPEN
# ==========================================
def sharpen_image(img, strength=0):

    if strength <= 0:
        return img

    blurred = cv2.GaussianBlur(img, (0, 0), 3)

    sharpened = cv2.addWeighted(
        img,
        1 + (strength / 10.0),
        blurred,
        -(strength / 10.0),
        0
    )

    return sharpened


# ==========================================
# MAIN
# ==========================================
def convert_folder_to_grayscale(
    input_folder,
    output_folder="./sortie_bw/",
    brightness=0,
    contrast=0,
    sharpness=0,
    start_index=0,
    end_index=None
):

    os.makedirs(output_folder, exist_ok=True)

    # ======================================
    # FILES
    # ======================================
    files = sorted(
        [
            f for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ],
        key=natural_sort_key
    )

    # ======================================
    # RANGE
    # ======================================
    files = files[start_index:end_index]

    # ======================================
    # CONTRASTE
    # ======================================
    alpha = 1.0 + (contrast / 100.0)

    for i, file in enumerate(files, start=start_index):

        path = os.path.join(input_folder, file)

        # ==================================
        # READ SAFE
        # ==================================
        img = imread_unicode(path)

        if img is None:
            print(f"❌ Impossible de lire : {file}")
            continue

        # ==================================
        # GRAYSCALE
        # ==================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ==================================
        # BRIGHTNESS + CONTRAST
        # ==================================
        adjusted = cv2.convertScaleAbs(
            gray,
            alpha=alpha,
            beta=brightness
        )

        # ==================================
        # SHARPNESS
        # ==================================
        adjusted = sharpen_image(
            adjusted,
            sharpness
        )

        output_path = os.path.join(output_folder, file)

        # ==================================
        # SAVE SAFE
        # ==================================
        imwrite_unicode(output_path, adjusted)

        print(f"🖤 [{output_path}] -> {file}")


# ==========================================
# RUN
# ==========================================
convert_folder_to_grayscale(
    "./PB/AID/[traité]BGR_AID_10MARS_27AVRIL",
    "./PB/AID/[br-60_cr-10_sh-15]Grayscale_AID_10MARS_27AVRIL",
    brightness=60,
    contrast=10,
    sharpness=15
)