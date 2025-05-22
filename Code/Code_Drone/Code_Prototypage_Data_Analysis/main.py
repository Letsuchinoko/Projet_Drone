import os
import glob
import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

DISPLAY_INTERVAL = 1 / 10  # 10 FPS (modifiable)
UPSCALE_FACTOR = 2         # Agrandit artificiellement l’image

last_display_time = 0
last_image_name = None

def detect_gant(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = image.shape[:2]

    # Plages HSV personnalisées pour le gant rouge/brique
    lower1 = np.array([0, 80, 40])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([10, 100, 50])
    upper2 = np.array([25, 255, 255])
    lower3 = np.array([170, 50, 40])
    upper3 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1)
    mask |= cv2.inRange(hsv, lower2, upper2)
    mask |= cv2.inRange(hsv, lower3, upper3)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
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

        if center_y < img_h * 0.25 and img_w * 0.3 < center_x < img_w * 0.7:
            continue
        if area < 600 or area > img_w * img_h * 0.6:
            continue
        if aspect_ratio < 0.25 or aspect_ratio > 2.5:
            continue
        if solidity > 0.995:
            continue
        if complexity > 35:
            continue

        score = area * (1 - solidity) * complexity
        if score > max_score:
            max_score = score
            best_cnt = cnt

    final_mask = None
    if best_cnt is not None:
        mask_gant = np.zeros_like(mask)
        cv2.drawContours(mask_gant, [best_cnt], -1, 255, thickness=cv2.FILLED)

        blurred = cv2.GaussianBlur(mask_gant, (7, 7), 0)
        _, final_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Annoter l'image
        cv2.drawContours(image, [best_cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.putText(image, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image, final_mask

# === Callback vidéo appelé à chaque image ===
def vision_callback(_):
    global last_display_time, last_image_name

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    images_dir = os.path.join(os.path.dirname(__file__), "images")
    raw_files = glob.glob(os.path.join(images_dir, "image_*.png"))

    fichiers = []
    for f in raw_files:
        try:
            fichiers.append((f, os.path.getmtime(f)))
        except FileNotFoundError:
            continue

    fichiers = [f[0] for f in sorted(fichiers, key=lambda x: x[1])]
    if not fichiers:
        return

    derniere = fichiers[-1]
    if derniere == last_image_name:
        return

    last_image_name = derniere
    last_display_time = now

    try:
        img = cv2.imread(derniere)
        if img is not None:
            # ⬆️ Agrandissement (simulé 1080p)
            img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR))

            # Traitement gant
            annotated, mask = detect_gant(img)

            if mask is not None:
                gant_only = cv2.bitwise_and(img, img, mask=mask)
                black_background = np.zeros_like(img)
                black_background[mask > 0] = gant_only[mask > 0]
                cv2.imshow("🧤 Gant détecté", black_background)
            else:
                cv2.imshow("🧤 Gant détecté", np.zeros_like(img))

            cv2.imshow("🎥 Flux Annoté", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
    except Exception as e:
        print(f"⚠️ Vision error: {e}")

# === Connexion au drone ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")

    vision = DroneVision(bebop, is_bebop=True)
    vision.set_user_callback_function(vision_callback)

    if vision.open_video():
        print("🎥 Détection en direct (RAM only). Ctrl+C pour quitter.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("⏹ Arrêt manuel.")
        finally:
            vision.close_video()
            bebop.disconnect()
            cv2.destroyAllWindows()
    else:
        print("❌ Erreur ouverture flux.")
        bebop.disconnect()
else:
    print("❌ Erreur connexion.")
