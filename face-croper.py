import cv2
import os
import shutil


# =========================
# HAAR DETECTION
# =========================
def detect_faces_haar(gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)
    )
    return [(x, y, w, h) for (x, y, w, h) in faces]


# =========================
# NORMALISATION VISAGE
# =========================
def normalize_face(img, x, y, w, h, output_size=224, scale=2.0):
    cx = x + w // 2
    cy = y + h // 2

    size = int(max(w, h) * scale)

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(img.shape[1], cx + size // 2)
    y2 = min(img.shape[0], cy + size // 2)

    crop = img[y1:y2, x1:x2]

    h_crop, w_crop = crop.shape[:2]
    size_final = max(h_crop, w_crop)

    # Padding pour carré
    square = cv2.copyMakeBorder(
        crop,
        top=(size_final - h_crop) // 2,
        bottom=(size_final - h_crop + 1) // 2,
        left=(size_final - w_crop) // 2,
        right=(size_final - w_crop + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # 🔥 Choix intelligent interpolation
    if size_final > output_size:
        interpolation = cv2.INTER_AREA       # réduction
    else:
        interpolation = cv2.INTER_CUBIC      # agrandissement

    resized = cv2.resize(square, (output_size, output_size), interpolation=interpolation)

    return resized

# =========================
# PIPELINE
# =========================
def process_dataset(input_folder, output_folder="./sortie/"):

    os.makedirs(output_folder, exist_ok=True)
    anomaly_folder = os.path.join(output_folder, "anomalie")
    os.makedirs(anomaly_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    counter = 1

    for file in files:
        path = os.path.join(input_folder, file)
        img = cv2.imread(path)

        if img is None:
            shutil.copy(path, os.path.join(anomaly_folder, file))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_haar(gray, face_cascade)

        output_path = os.path.join(output_folder, f"{counter}.jpg")

        if len(faces) == 0:
            shutil.copy(path, os.path.join(anomaly_folder, file))
            counter += 1
            continue

        x, y, w, h = faces[0]

        # 🔥 NORMALISATION
        face_normalized = normalize_face(img, x, y, w, h, output_size=224, scale=2.2)

        cv2.imwrite(output_path, face_normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"✅ Normalisé -> {output_path}")

        counter += 1


# =========================
# TEST
# =========================
process_dataset("./entree/", "./sortie/")