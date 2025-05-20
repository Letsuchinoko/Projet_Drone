import os
import glob
import cv2
import time
from collections import deque
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
DISPLAY_INTERVAL = 1 /10  # 5 FPS
BUFFER_IMAGES_TO_KEEP = 5

last_display_time = 0
last_image_name = None

# === CALLBACK appelé à chaque frame ===
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

    last_display_time = now
    derniere = fichiers[-1]

    # Ne pas retraiter la même image
    if derniere == last_image_name:
        return

    last_image_name = derniere

    try:
        img = cv2.imread(derniere)
        if img is not None:
            cv2.imshow("🖼️ Flux Bebop2 (final)", img)
            cv2.waitKey(1)
    except:
        pass

    # Suppression décalée des anciennes images (hors des 5 dernières)
    try:
        for f in fichiers[:-BUFFER_IMAGES_TO_KEEP]:
            if f != derniere:
                os.remove(f)
    except:
        pass

# === Connexion ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")
    vision = DroneVision(bebop, is_bebop=True)
    vision.set_user_callback_function(vision_callback)

    if vision.open_video():
        print("🎥 Flux vidéo actif. Ctrl+C pour arrêter.")
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
        print("❌ Impossible d’ouvrir le flux.")
        bebop.disconnect()
else:
    print("❌ Connexion échouée.")
