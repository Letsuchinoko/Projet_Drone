import os
import glob
import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

DISPLAY_INTERVAL = 1 / 10  # 10 FPS
KEEP_LAST = 10

last_display_time = 0
last_image_name = None

# === Détection Gant (simplifié pour test) ===
def detect_gant(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (10, 100, 50), (25, 255, 255))
    mask3 = cv2.inRange(hsv, (170, 50, 40), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image

# === Nettoyage régulier ===
def nettoyer_images():
    while True:
        raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
        fichiers = []
        for f in raw_files:
            try:
                fichiers.append((f, os.path.getmtime(f)))
            except FileNotFoundError:
                continue
        fichiers = [f[0] for f in sorted(fichiers, key=lambda x: x[1])]
        if len(fichiers) > KEEP_LAST:
            for f in fichiers[:-KEEP_LAST]:
                try:
                    os.remove(f)
                except:
                    pass
        time.sleep(10)

# === Callback appelé à chaque image ===
def vision_callback(_):
    global last_display_time, last_image_name

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
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
        if img is None:
            return
        img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
        img = detect_gant(img)
        cv2.imshow("🧤 Gant Detection", img)
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

    threading.Thread(target=nettoyer_images, daemon=True).start()

    # Patch : on modifie le path dans le script temporairement
    import pyparrot.DroneVision
    pyparrot.DroneVision.IMAGE_DIR = IMAGES_DIR

    vision = DroneVision(bebop, is_bebop=True)
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
