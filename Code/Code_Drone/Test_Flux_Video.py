import os
import glob
import cv2
import time
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_IMAGES = 15
DISPLAY_INTERVAL = 1 / 20  # 20 FPS = 0.05 sec

last_display_time = 0  # Pour contrôler le rythme d'affichage

# === CALLBACK pour afficher le flux vidéo ===
def afficher_image(image):
    global last_display_time

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return  # Attendre pour respecter les 20 FPS

    last_display_time = now

    try:
        # Liste des fichiers image triés par date
        fichiers = sorted(glob.glob(os.path.join(IMAGES_DIR, "image_*.png")), key=os.path.getmtime)

        # Supprimer les plus anciennes
        if len(fichiers) > MAX_IMAGES:
            for f in fichiers[:-MAX_IMAGES]:
                try:
                    os.remove(f)
                except:
                    pass

        # Afficher la dernière image
        if fichiers:
            derniere = fichiers[-1]
            img = cv2.imread(derniere)
            if img is not None:
                cv2.imshow("🖼️ Flux Bebop2 (dernière image)", img)
                cv2.waitKey(1)
                print(f"Affichage : {derniere}")
    except Exception as e:
        print(f"⚠️ Erreur d'affichage : {e}")

# === CONNEXION AU DRONE ===
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
