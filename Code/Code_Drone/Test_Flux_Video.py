import os
import glob
import cv2
import time
import threading
from collections import deque
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
FRAME_BUFFER_SIZE = 2
KEEP_LAST = 10  # On garde au moins 10 images
CLEANUP_INTERVAL = 15  # Nettoyage toutes les 15 sec
DISPLAY_INTERVAL = 1 / 10  # 10 FPS

last_display_time = 0
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

# === THREAD de nettoyage ===
def nettoyer_images():
    while True:
        fichiers = sorted(
            glob.glob(os.path.join(IMAGES_DIR, "image_*.png")),
            key=os.path.getmtime
        )
        if len(fichiers) > KEEP_LAST:
            for f in fichiers[:-KEEP_LAST]:
                try:
                    os.remove(f)
                except:
                    pass
        time.sleep(CLEANUP_INTERVAL)

# === CALLBACK vidéo ===
def vision_callback(_):
    global last_display_time, frame_buffer

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    last_display_time = now

    try:
        fichiers = sorted(
            glob.glob(os.path.join(IMAGES_DIR, "image_*.png")),
            key=os.path.getmtime
        )

        if not fichiers:
            return

        derniere = fichiers[-1]
        img = cv2.imread(derniere)
        if img is not None:
            frame_buffer.append(img)
            cv2.imshow("🖼️ Flux Bebop2 Live (RAM buffer)", frame_buffer[-1])
            cv2.waitKey(1)

    except Exception as e:
        print(f"⚠️ Erreur d'affichage : {e}")

# === Connexion ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")

    # Lancer le thread de nettoyage
    cleaner_thread = threading.Thread(target=nettoyer_images, daemon=True)
    cleaner_thread.start()

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
