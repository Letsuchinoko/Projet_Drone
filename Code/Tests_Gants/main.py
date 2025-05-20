import cv2
import numpy as np
import os

# === DOSSIERS ===
input_dir = "images"
output_detection = "detection"
output_masks = "masks"
output_red = "redzones"

os.makedirs(output_detection, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)
os.makedirs(output_red, exist_ok=True)

def process_image(filename):
    path = os.path.join(input_dir, filename)
    image = cv2.imread(path)
    if image is None:
        print(f"⚠️ Impossible de lire {filename}")
        return

    # Resize image
    image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = image.shape[:2]

    # ✅ Plage rouge étendue pour ton gant (même s'il est foncé)
    lower_red = np.array([0, 80, 30])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # DEBUG : masque rouge visible
    red_visible = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(os.path.join(output_red, filename), red_visible)

    # Nettoyage morphologique
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Trouver contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_mask = np.zeros_like(mask)
    image_with_contour = image.copy()
    max_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        aspect_ratio = w / float(h)
        solidity = float(area) / hull_area
        complexity = area / perimeter
        center_x = x + w // 2
        center_y = y + h // 2

        # 🔒 Anti-visage : trop centré en haut
        if center_y < img_h * 0.25 and img_w * 0.3 < center_x < img_w * 0.7:
            continue

        # Filtres géométriques souples
        if area < 800 or area > img_w * img_h * 0.5:
            continue
        if aspect_ratio < 0.3 or aspect_ratio > 2.0:
            continue
        if solidity > 0.98:
            continue
        if complexity > 28:
            continue

        score = area * (1 - solidity) * complexity
        # Debug print (optionnel)
        print(f"[{filename}] Area={area:.0f} Solidity={solidity:.2f} "
              f"Complexity={complexity:.2f} Score={score:.1f}")

        if score > max_score:
            max_score = score
            best_cnt = cnt

    if best_cnt is not None:
        # === Création masque précis à partir du contour
        mask_gant = np.zeros_like(mask)
        cv2.drawContours(mask_gant, [best_cnt], -1, 255, thickness=cv2.FILLED)

        # Flouter puis binariser pour un détourage propre
        blurred = cv2.GaussianBlur(mask_gant, (7, 7), 0)
        _, mask_final = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Annoter image originale
        cv2.drawContours(image_with_contour, [best_cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.putText(image_with_contour, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Appliquer masque détouré
        detailed = cv2.bitwise_and(image, image, mask=mask_final)

        # Sauvegarder
        cv2.imwrite(os.path.join(output_detection, filename), image_with_contour)
        cv2.imwrite(os.path.join(output_masks, filename), detailed)
    else:
        print(f"❌ Aucun gant détecté dans {filename}")
        cv2.imwrite(os.path.join(output_detection, filename), image)

    print(f"✅ {filename} terminé.")

# === TRAITEMENT DU DOSSIER ENTIER
for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(file)

print("\n🎯 Traitement terminé — Résultats dans 'detection/', 'masks/', et 'redzones/'")
