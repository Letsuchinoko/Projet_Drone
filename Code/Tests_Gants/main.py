import cv2
import numpy as np
import os

# === CONFIGURATION ===
input_dir = "images"
output_detection = "detection"
output_masks = "masks"

os.makedirs(output_detection, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

def process_image(filename):
    path = os.path.join(input_dir, filename)
    image = cv2.imread(path)
    if image is None:
        print(f"⚠️ Impossible de lire {filename}")
        return

    image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = image.shape[:2]

    # === Étape 1 : Masque des zones rouges
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Nettoyage morphologique
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # === Étape 2 : Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_mask = np.zeros_like(mask)
    image_with_contour = image.copy()
    max_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0 or perimeter == 0:
            continue

        aspect_ratio = w / float(h)
        solidity = float(area) / hull_area
        complexity = area / perimeter

        # Exclure les zones trop hautes/centrées (visages)
        center_x = x + w // 2
        center_y = y + h // 2
        if center_y < img_h * 0.25 and img_w * 0.3 < center_x < img_w * 0.7:
            continue

        # Filtres souples pour accepter différentes formes de main
        if area < 800 or area > img_w * img_h * 0.45:
            continue
        if aspect_ratio < 0.3 or aspect_ratio > 1.8:
            continue
        if solidity > 0.96:
            continue
        if complexity > 25:
            continue

        # Score basé sur surface + irrégularité
        score = area * (1 - solidity) * complexity

        if score > max_score:
            max_score = score
            best_cnt = cnt

    # === Étape 3 : Affichage & export
    if best_cnt is not None:
        hull = cv2.convexHull(best_cnt)
        cv2.drawContours(image_with_contour, [hull], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(hull)
        cv2.putText(image_with_contour, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.drawContours(best_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Gant détouré sur fond noir
    detailed = cv2.bitwise_and(image, image, mask=best_mask)

    # Sauvegardes
    cv2.imwrite(os.path.join(output_detection, filename), image_with_contour)
    cv2.imwrite(os.path.join(output_masks, filename), detailed)
    print(f"✅ {filename} traité")

# === TRAITEMENT DU DOSSIER ===
for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(file)

print("\n🎉 Tous les fichiers ont été traités avec succès.")
