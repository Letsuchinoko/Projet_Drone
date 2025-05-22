import os
import cv2
import time
import numpy as np
import subprocess
from pyparrot.Bebop import Bebop

# === CONFIGURATION ===
WIDTH, HEIGHT = 856, 480  # Résolution du Bebop2
PIX_FMT = "bgr24"
DISPLAY_INTERVAL = 1 / 10  # 10 FPS
FFMPEG_BIN = "ffmpeg"
SDP_NAME = "bebop.sdp"

# === Récupérer chemin absolu vers le .sdp généré par pyparrot ===
from pyparrot.DroneVision import IMAGE_DIR  # Ce répertoire contient bebop.sdp
SDP_PATH = os.path.join(IMAGE_DIR, SDP_NAME)

# === Connexion PyParrot ===
bebop = Bebop()
print("Connexion au drone...")
if not bebop.connect(10):
    print("❌ Erreur de connexion")
    exit(1)
print("✅ Connecté")

# === Lancer le flux vidéo ffmpeg en RAM (stdout) ===
print("🎥 Démarrage du flux vidéo direct...")
cmd = [
    FFMPEG_BIN,
    "-protocol_whitelist", "file,udp,rtp",
    "-fflags", "nobuffer",
    "-i", SDP_PATH,
    "-f", "rawvideo",
    "-pix_fmt", PIX_FMT,
    "-"
]

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL  # cacher logs ffmpeg
    )
except Exception as e:
    print(f"❌ Impossible de lancer ffmpeg : {e}")
    bebop.disconnect()
    exit(1)

frame_size = WIDTH * HEIGHT * 3
last_display = 0

try:
    while True:
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            print("⚠️ Trame incomplète")
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
        now = time.time()

        if now - last_display >= DISPLAY_INTERVAL:
            cv2.imshow("🎥 Bebop2 Live (RAM only)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_display = now

except KeyboardInterrupt:
    print("⏹ Arrêt manuel.")

finally:
    print("🧹 Fermeture...")
    process.kill()
    bebop.disconnect()
    cv2.destroyAllWindows()
