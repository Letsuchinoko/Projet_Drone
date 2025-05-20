import cv2
import numpy as np
import os

# === CONFIGURATION ===
input_dir = "images"
output_detection = "detection"
output_masks = "masks"

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Créer les dossiers si besoin
os.makedirs(output_detection, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

def contour_in_face(cnt, faces):
    x, y, w, h = cv2.boundingRect(cnt)
    for (fx, fy, fw, fh) in faces:
        if fx <= x <= fx + fw and fy <= y <= fy + fh:
            return True
    return False

def process_image(filename):
    path = os.path.join(input_dir, filename)
    image = cv2.imread(path)
    if image is None:
        print(f"⚠️ Impossible de lire {filename}")
        return

    image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Rouge ciblé (évite les objets jaunes)
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Nettoyage
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_mask = np.zeros_like(mask)
    image_with_contour = image.copy()
    img_h, img_w = image.shape[:2]
    best_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if area < 1000 or area > (img_w * img_h * 0.4) or y < img_h * 0.1:
            continue
        if aspect_ratio < 0.3 or aspect_ratio > 2.5:
            continue
        if contour_in_face(cnt, faces):
            continue

        if area > max_area:
            max_area = area
            best_cnt = cnt

    if best_cnt is not None:
        hull = cv2.convexHull(best_cnt)
        cv2.drawContours(image_with_contour, [hull], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(hull)
        cv2.putText(image_with_contour, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.drawContours(best_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Générer image détourée
    detailed = cv2.bitwise_and(image, image, mask=best_mask)

    # Sauvegarder
    cv2.imwrite(os.path.join(output_detection, filename), image_with_contour)
    cv2.imwrite(os.path.join(output_masks, filename), detailed)
    print(f"✅ {filename} traité")

# === Lancement du traitement ===
for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(file)

print("\n🎉 Traitement terminé !")
