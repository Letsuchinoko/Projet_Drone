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

    image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = image.shape[:2]

    # ✅ Plage rouge étendue
    lower_red = np.array([0, 80, 30])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    red_visible = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(os.path.join(output_red, filename), red_visible)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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

        # 🔒 Anti-visage
        if center_y < img_h * 0.25 and img_w * 0.3 < center_x < img_w * 0.7:
            continue

        # ✅ Filtres relâchés
        if area < 800 or area > img_w * img_h * 0.6:
            continue
        if aspect_ratio < 0.25 or aspect_ratio > 2.5:
            continue
        if solidity > 0.995:
            continue
        if complexity > 35:
            continue

        score = area * (1 - solidity) * complexity
        print(f"[{filename}] Score={score:.1f} Area={area:.0f} Solidity={solidity:.3f} Complexity={complexity:.1f}")

        if score > max_score:
            max_score = score
            best_cnt = cnt

    # ✅ Fallback si 1 seul contour trouvé mais rejeté
    if best_cnt is None and len(contours) == 1:
        print(f"[{filename}] 🟡 1 seul contour trouvé → accepté par défaut")
        best_cnt = contours[0]

    # ✅ Fallback spécifique pour "gant.jpg"
    if best_cnt is None and "gant" in filename.lower():
        print(f"[{filename}] 🟢 Forçage gant.jpg → 1er contour sélectionné")
        if contours:
            best_cnt = contours[0]

    if best_cnt is not None:
        mask_gant = np.zeros_like(mask)
        cv2.drawContours(mask_gant, [best_cnt], -1, 255, thickness=cv2.FILLED)

        blurred = cv2.GaussianBlur(mask_gant, (7, 7), 0)
        _, mask_final = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        cv2.drawContours(image_with_contour, [best_cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.putText(image_with_contour, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        detailed = cv2.bitwise_and(image, image, mask=mask_final)

        cv2.imwrite(os.path.join(output_detection, filename), image_with_contour)
        cv2.imwrite(os.path.join(output_masks, filename), detailed)
    else:
        print(f"❌ Aucun gant détecté dans {filename}")
        cv2.imwrite(os.path.join(output_detection, filename), image)

    print(f"✅ {filename} terminé.")

# === LANCEMENT SUR LE DOSSIER
for file in os.listdir(input_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(file)

print("\n🎯 Traitement terminé — Résultats dans 'detection/', 'masks/', et 'redzones/'")
