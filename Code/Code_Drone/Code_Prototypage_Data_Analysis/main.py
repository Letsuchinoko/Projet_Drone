import os
import glob
import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

# === CONFIGURATION ===
LOCAL_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")
PYPARROT_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images")
GANT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gant_only")

DISPLAY_INTERVAL = 1 / 10  # 10 FPS
KEEP_LAST = 10

os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
os.makedirs(GANT_OUTPUT_DIR, exist_ok=True)

last_display_time = 0
last_image_name = None

# === Détection du gant rouge ===a
def detect_gant(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Plages élargies
    mask = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([25, 255, 255]))

    # Nettoyage
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Affichage du masque
    cv2.imshow("MASK DEBUG", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Aucun contour détecté")
        return image

    print(f"{len(contours)} contour(s) trouvés")
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    print(f"→ Meilleur contour : {area:.0f} px")

    if area < 500:
        print("Contour trop petit, ignoré.")
        return image

    # Dessin et sauvegarde
    mask_final = np.zeros_like(mask)
    cv2.drawContours(mask_final, [best], -1, 255, cv2.FILLED)

    gant_only = cv2.bitwise_and(image, image, mask=mask_final)
    fond_noir = np.zeros_like(image)
    fond_noir[mask_final == 255] = gant_only[mask_final == 255]

    out_path = os.path.join("gant_only", f"debug_{int(time.time()*1000)%100000}.png")
    os.makedirs("gant_only", exist_ok=True)
    cv2.imwrite(out_path, fond_noir)
    print(f"✅ Image sauvegardée : {out_path}")

    cv2.drawContours(image, [best], -1, (0, 255, 0), 2)
    return image


# === Nettoyage images obsolètes ===
def nettoyer_images():
    pyparrot_dir = os.path.join(os.path.dirname(__file__), "../../../../../anaconda3/Lib/site-packages/pyparrot/images")
    while True:
        fichiers = glob.glob(os.path.join(pyparrot_dir, "image_*.png"))
        fichiers = sorted(fichiers, key=os.path.getmtime)
        if len(fichiers) > KEEP_LAST:
            for f in fichiers[:-KEEP_LAST]:
                try:
                    os.remove(f)
                except:
                    pass
        time.sleep(5)

# === Callback vidéo appelé à chaque image ===
def vision_callback(_):
    global last_display_time, last_image_name

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    fichiers = []
    for f in glob.glob(os.path.join(PYPARROT_IMAGE_DIR, "image_*.png")):
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
        # 🔁 Lecture robuste de l'image (attente si en cours d'écriture)
        for _ in range(3):
            img = cv2.imread(derniere)
            if img is not None:
                break
            time.sleep(0.05)  # 50ms

        if img is None:
            print("⚠️ Image corrompue ou non disponible")
            return

        img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
        img = detect_gant(img)

        # Affiche l’image
        cv2.imshow("🧤 Gant Détecté (Live)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        # Sauvegarde locale
        local_name = os.path.basename(derniere)
        local_path = os.path.join(LOCAL_IMAGE_DIR, local_name)
        cv2.imwrite(local_path, img)

    except Exception as e:
        print(f"⚠️ Erreur vision_callback : {e}")

# === Connexion au drone ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")

    threading.Thread(target=nettoyer_images, daemon=True).start()

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
