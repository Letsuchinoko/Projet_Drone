import os
import glob
import cv2
import time
from collections import deque
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
FRAME_BUFFER_SIZE = 2
DISPLAY_INTERVAL = 1 / 5  # 5 FPS

last_display_time = 0
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
dernier_fichier_affiche = None

def vision_callback(_):
    global last_display_time, frame_buffer, dernier_fichier_affiche

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    fichiers = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "image_*.png")),
        key=os.path.getmtime
    )

    if not fichiers:
        return

    # Ne traiter que la nouvelle image
    derniere = fichiers[-1]
    if derniere == dernier_fichier_affiche:
        return

    dernier_fichier_affiche = derniere
    last_display_time = now

    try:
        img = cv2.imread(derniere)
        if img is not None:
            frame_buffer.append(img)
            cv2.imshow("🖼️ Flux Bebop2 (stable)", img)
            cv2.waitKey(1)
    except:
        pass

    # Nettoyage des anciennes images
    try:
        for f in fichiers[:-FRAME_BUFFER_SIZE]:
            if f != dernier_fichier_affiche:
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
        print("❌ Erreur ouverture flux.")
        bebop.disconnect()
else:
    print("❌ Erreur connexion drone.")
