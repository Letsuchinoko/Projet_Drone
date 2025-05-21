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

# === Détection du gant rouge ===
def detect_gant(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_h, img_w = image.shape[:2]

    # Plages HSV élargies (rouge profond → orange clair)
    ranges = [
        (np.array([0, 50, 50]),   np.array([10, 255, 255])),  # rouge foncé
        (np.array([11, 100, 60]), np.array([25, 255, 255])),  # orange
        (np.array([160, 50, 50]), np.array([180, 255, 255]))  # rouge clair
    ]

    mask = sum([cv2.inRange(hsv, low, high) for (low, high) in ranges])

    # Nettoyage
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    max_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        aspect_ratio = w / float(h)
        solidity = float(area) / hull_area
        complexity = area / perimeter
        center_y = y + h // 2

        # Anti-visage
        if center_y < img_h * 0.25:
            continue

        # Filtres plus permissifs
        if area < 500 or area > img_w * img_h * 0.7:
            continue
        if solidity > 0.995 or complexity > 45:
            continue

        score = area * (1 - solidity) * complexity
        if score > max_score:
            max_score = score
            best_cnt = cnt

    if best_cnt is not None:
        # Affichage sur image
        cv2.drawContours(image, [best_cnt], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(best_cnt)
        cv2.putText(image, "Gant detecte", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Masque gant pour fond noir
        mask_gant = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_gant, [best_cnt], -1, 255, cv2.FILLED)

        gant_seul = cv2.bitwise_and(image, image, mask=mask_gant)
        fond_noir = np.zeros_like(image)
        fond_noir[mask_gant == 255] = gant_seul[mask_gant == 255]

        # Enregistrement pour IA
        filename = f"gant_{int(time.time()*1000)%100000}.png"
        cv2.imwrite(os.path.join(GANT_OUTPUT_DIR, filename), fond_noir)

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
