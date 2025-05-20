import os
import glob
import cv2
import time
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_IMAGES = 15

def afficher_image(image):
    # On ignore le paramètre image (pas utilisable directement)
    try:
        # Lister et trier les images par date
        fichiers = sorted(glob.glob(os.path.join(IMAGES_DIR, "image_*.png")), key=os.path.getmtime)

        # Supprimer les plus anciennes
        if len(fichiers) > MAX_IMAGES:
            for f in fichiers[:-MAX_IMAGES]:
                try:
                    os.remove(f)
                except:
                    pass

        # Afficher la dernière image disponible
        if fichiers:
            derniere = fichiers[-1]
            img = cv2.imread(derniere)
            if img is not None:
                cv2.imshow("🖼️ Flux Bebop2 (image la plus récente)", img)
                cv2.waitKey(1)

    except Exception as e:
        print(f"⚠️ Erreur affichage : {e}")

# === CONNEXION AU DRONE ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté au drone.")

    vision = DroneVision(bebop, is_bebop=True)
    vision.set_user_callback_function(afficher_image)

    if vision.open_video():
        print("🎥 Flux vidéo actif. Appuyez sur Ctrl+C pour arrêter.")
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
        print("❌ Échec d’ouverture du flux vidéo.")
        bebop.disconnect()
else:
    print("❌ Échec de connexion au drone.")
