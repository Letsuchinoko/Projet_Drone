import os
import time
import cv2
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_CAPTURE = 2
captured_files = []
capture_done = False

# === CALLBACK de capture ===
def capture_images(_):
    global captured_files, capture_done

    fichiers = sorted(
        [f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")],
        key=lambda x: os.path.getmtime(os.path.join(IMAGES_DIR, x))
    )

    if len(fichiers) > 0:
        last_image = fichiers[-1]
        full_path = os.path.join(IMAGES_DIR, last_image)
        if full_path not in captured_files:
            captured_files.append(full_path)
            print(f"📸 Capturée : {last_image}")

    if len(captured_files) >= MAX_CAPTURE and not capture_done:
        capture_done = True
        print("✅ 2 images capturées. Fermeture du flux.")
        bebopVision.close_video()
        bebop.disconnect()

# === Connexion au drone ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")
    bebopVision = DroneVision(bebop, is_bebop=True)
    bebopVision.set_user_callback_function(capture_images)

    if bebopVision.open_video():
        print("🎥 Capture en cours (2 images)...")
        while not capture_done:
            time.sleep(1)
    else:
        print("❌ Erreur ouverture flux vidéo.")
        bebop.disconnect()
        exit()
else:
    print("❌ Erreur connexion drone.")
    exit()

# === AFFICHAGE DES 2 IMAGES CAPTURÉES ===
print("🖼️ Chargement des images...")
frames = []
for f in captured_files:
    img = cv2.imread(f)
    if img is not None:
        frames.append(img)

if len(frames) < 2:
    print("❌ Moins de 2 images lisibles. Abandon.")
    exit()

print("▶️ Affichage alterné. Appuyez sur 'q' pour quitter.")

i = 0
while True:
    cv2.imshow("Flux Alterné (Test)", frames[i % 2])
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    i += 1

cv2.destroyAllWindows()
