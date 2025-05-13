from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import cv2
import time
import os
import glob

# === CALLBACK : Affiche l'image + nettoie les images PNG ===
def show_frame(image):
    if image is not None:
        cv2.imshow("Flux vidéo du Bebop 2", image)
        cv2.waitKey(1)

        # Supprime les images PNG pour éviter l'accumulation
        for f in glob.glob("C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images/image_*.png"):
            try:
                os.remove(f)
            except:
                pass

# === INITIALISATION DU DRONE ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"
print("Connexion au drone...")

if bebop.connect(10):
    print("✅ Connecté au drone !")

    bebopVision = DroneVision(bebop, is_bebop=True)
    bebopVision.set_user_callback_function(show_frame)

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
