import cv2
import os

def convert_folder_to_grayscale(input_folder, output_folder="./sortie_bw/"):

    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file in files:
        path = os.path.join(input_folder, file)
        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        output_path = os.path.join(output_folder, file)

        cv2.imwrite(output_path, gray)

        print(f"🖤 Noir & blanc -> {output_path}")
        

convert_folder_to_grayscale("./sortie/", "./sortie_bw/")