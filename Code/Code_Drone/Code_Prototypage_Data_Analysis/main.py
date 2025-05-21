import os
import glob
import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(LOCAL_DIR, "images")
MASKS_DIR = os.path.join(LOCAL_DIR, "masks")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

DISPLAY_INTERVAL = 1 / 10  # 10 FPS
KEEP_LAST = 10
last_display_time = 0
last_image_name = None

# === Détection + sauvegarde fond noir ===
def detect_gant_and_save_mask(image, filename):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = image.shape[:2]

    # Plages HSV personnalisées (gant rouge/brique)
    lower1 = np.array([0, 80, 40])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([10, 100, 50])
    upper2 = np.array([25, 255, 255])
    lower3 = np.array([170, 50, 40])
    upper3 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.bitwise_or(cv2.inRange(hsv, lower2, upper2), cv2.inRange(hsv, lower3, upper3))
    )

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

    if best_cnt is not None:
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
        result = cv2.bitwise_and(image, image, mask=final_mask)
        cv2.imwrite(os.path.join(MASKS_DIR, filename), result)

        cv2.drawContours(image, [best_cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.putText(image, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image

# === Lecture protégée ===
def lire_image_sans_crash(path, essais=5, delai=0.05):
    for _ in range(essais):
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
        time.sleep(delai)
    return None

# === Nettoyage images ===
def nettoyer_images():
    while True:
        fichiers = sorted(
            glob.glob(os.path.join(IMAGES_DIR, "image_*.png")),
            key=os.path.getmtime
        )
        for f in fichiers[:-KEEP_LAST]:
            try:
                os.remove(f)
            except:
                continue
        time.sleep(10)

# === Callback vidéo ===
def vision_callback(_):
    global last_display_time, last_image_name

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    fichiers = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "image_*.png")),
        key=os.path.getmtime
    )
    if not fichiers:
        return

    derniere = fichiers[-1]
    if derniere == last_image_name:
        return
    last_image_name = derniere
    last_display_time = now

    try:
        image = lire_image_sans_crash(derniere)
        if image is None:
            return

        image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))
        image = detect_gant_and_save_mask(image, os.path.basename(derniere))
        cv2.imshow("🧤 Gant Detecté", image)
        cv2.waitKey(1)
    except Exception as e:
        print(f"⚠️ Callback crash évité : {e}")

# === Connexion drone ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")

    # Nettoyage en fond
    threading.Thread(target=nettoyer_images, daemon=True).start()

    # Dossier d’enregistrement dans PyParrot modifié
    vision = DroneVision(bebop, is_bebop=True, image_path=IMAGES_DIR)
    vision.set_user_callback_function(vision_callback)

    if vision.open_video():
        print("🎥 Détection en direct. Ctrl+C pour quitter.")
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
