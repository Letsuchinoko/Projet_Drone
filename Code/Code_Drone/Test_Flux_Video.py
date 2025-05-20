import os
import glob
import cv2
import time
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_IMAGES = 15
DISPLAY_INTERVAL = 1 / 5  # 5 FPS
last_display_time = 0

def afficher_image(_):
    global last_display_time

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    last_display_time = now

    try:
        fichiers = sorted(
            glob.glob(os.path.join(IMAGES_DIR, "image_*.png")),
            key=os.path.getmtime
        )

        # Supprimer seulement les plus anciens (hors des 5 derniers)
        if len(fichiers) > MAX_IMAGES:
            for f in fichiers[:-5]:  # ← garde un petit historique
                try:
                    os.remove(f)
                except:
                    pass  # on ignore proprement

        if fichiers:
            derniere = fichiers[-1]
            frame = cv2.imread(derniere)

            if frame is not None:
                cv2.imshow("🖼️ Flux Bebop2 (5 FPS)", frame)
                cv2.waitKey(1)

    except:
        pass  # on ignore toute erreur d’affichage sans log

# === CONNEXION DRONE & VISION ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté au drone.")

    vision = DroneVision(bebop, is_bebop=True)
    vision.set_user_callback_function(afficher_image)

    if vision.open_video():
        print("🎥 Flux vidéo actif. Ctrl+C pour quitter.")
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
        print("❌ Impossible d'ouvrir le flux vidéo.")
        bebop.disconnect()
else:
    print("❌ Connexion échouée.")
