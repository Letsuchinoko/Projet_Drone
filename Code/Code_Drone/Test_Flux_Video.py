from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import cv2
import time
import os
import glob
import re

# === CONFIGURATION ===
MAX_IMAGES = 15
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"

# === CALLBACK pour le flux vidéo ===
def show_and_manage_images(image):
    if image is None:
        return

    # Supprimer les fichiers les plus anciens (garder seulement les 15 derniers)
    image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "image_*.png")), key=os.path.getmtime)

    if len(image_files) > MAX_IMAGES:
        for f in image_files[:-MAX_IMAGES]:
            try:
                os.remove(f)
            except Exception as e:
                print(f"⚠️ Erreur suppression {f} : {e}")

    # Afficher la dernière image du dossier
    if image_files:
        last_img_path = image_files[-1]
        try:
            frame = cv2.imread(last_img_path)
            if frame is not None:
                cv2.imshow("Flux vidéo (image la plus récente)", frame)
                cv2.waitKey(1)
        except Exception as e:
            print(f"⚠️ Erreur lecture image : {e}")

# === INITIALISATION DU DRONE ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"
print("Connexion au drone...")

if bebop.connect(10):
    print("✅ Connecté au drone !")

    bebopVision = DroneVision(bebop, is_bebop=True)
    bebopVision.set_user_callback_function(show_and_manage_images)

    if bebopVision.open_video():
        print("🎥 Flux vidéo actif. Ctrl+C pour quitter.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("⏹ Arrêt manuel.")
        finally:
            bebopVision.close_video()
            bebop.disconnect()
            cv2.destroyAllWindows()
    else:
        print("❌ Impossible d'ouvrir le flux vidéo.")
        bebop.disconnect()
else:
    print("❌ Connexion échouée.")
