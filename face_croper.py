import cv2
import os
import shutil
import numpy as np


# =========================
# LOG
# =========================
def log(msg):
    print(f"[INFO] {msg}")


# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(img):
    h, w = img.shape[:2]

    # resize si image trop grande
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # améliore contraste (meilleure détection visage)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return img, gray


# =========================
# DETECTION VISAGE
# =========================
def detect_faces(gray, cascade):
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50)
    )

    # filtre bruit
    return [f for f in faces if f[2] > 70 and f[3] > 70]


# =========================
# SHARPEN IMAGE
# =========================
def enhance_image(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)

def denoise(img):
    return cv2.bilateralFilter(img, 7, 50, 50)


# =========================
# RESIZE HAUTE QUALITÉ
# =========================
def high_quality_resize(img, size=224):
    return cv2.resize(
        img,
        (size, size),
        interpolation=cv2.INTER_LANCZOS4
    )


# =========================
# NORMALISATION VISAGE
# =========================
def normalize_face(img, x, y, w, h, output_size=224, face_ratio=0.70):

    cx = x + w // 2
    cy = int(y + h * 0.45)

    size = int(max(w, h) / face_ratio)

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(img.shape[1], cx + size // 2)
    y2 = min(img.shape[0], cy + size // 2)

    crop = img[y1:y2, x1:x2]

    # 🔥 réduction du grain (IMPORTANT)
    #crop = denoise(crop)

    # 🔥 amélioration qualité
    #crop = enhance_image(crop)

    h_crop, w_crop = crop.shape[:2]
    size_final = max(h_crop, w_crop)

    square = cv2.copyMakeBorder(
        crop,
        (size_final - h_crop) // 2,
        (size_final - h_crop + 1) // 2,
        (size_final - w_crop) // 2,
        (size_final - w_crop + 1) // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return high_quality_resize(square, output_size)


# =========================
# ANOMALIES
# =========================
def save_anomaly(path, folder, file, reason):
    name, ext = os.path.splitext(file)
    new_name = f"{name}__{reason}{ext}"
    shutil.copy(path, os.path.join(folder, new_name))
    log(f"❌ ANOMALIE: {file} -> {reason}")


# =========================
# SAVE IMAGE PNG + JPG
# =========================
def save_image(output_path, img):
    base, _ = os.path.splitext(output_path)

    # PNG (sans perte)
    png_path = base + ".png"
    cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    

    log(f"💾 Sauvegardé PNG  → {base}")


# =========================
# PIPELINE PRINCIPAL
# =========================
def process_dataset(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    anomaly_folder = os.path.join(output_folder, "anomalies")
    os.makedirs(anomaly_folder, exist_ok=True)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    success, failed = 0, 0

    for file in files:
        path = os.path.join(input_folder, file)

        log("\n==============================")
        log(f"Traitement: {file}")

        img = cv2.imread(path)

        if img is None:
            save_anomaly(path, anomaly_folder, file, "image_corrompue")
            failed += 1
            continue

        img_original = img.copy()  # 🔥 on garde l'image propre

        img_detect, gray = preprocess_image(img)  # pour détection uniquement
        faces = detect_faces(gray, cascade)

        log(f"Visages détectés: {len(faces)}")

        if len(faces) == 0:
            save_anomaly(path, anomaly_folder, file, "aucun_visage")
            failed += 1
            continue

        if len(faces) > 1:
            save_anomaly(path, anomaly_folder, file, "plusieurs_visages")
            failed += 1
            continue

        x, y, w, h = faces[0]

        log(f"Face retenue: x={x}, y={y}, w={w}, h={h}")

        try:
            face = normalize_face(img_original, x, y, w, h)

            output_path = os.path.join(output_folder, file)

            # 🔥 sauvegarde haute qualité PNG + JPG
            save_image(output_path, face)

            success += 1

        except Exception as e:
            save_anomaly(path, anomaly_folder, file, f"erreur_{e}")
            failed += 1

    log("\n==============================")
    log(f"FINISHED → OK: {success} | FAIL: {failed}")


# =========================
# RUN
# =========================
process_dataset("./storage/images/", "./storage/cropped/")