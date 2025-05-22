import os
import cv2
import time
import numpy as np
import subprocess
from pyparrot.Bebop import Bebop

# === CONFIGURATION ===
WIDTH, HEIGHT = 856, 480  # Résolution native du Bebop2
PIX_FMT = "bgr24"
SDP_FILENAME = "bebop.sdp"
FFMPEG_BIN = "ffmpeg"
DISPLAY_INTERVAL = 1 / 10  # 10 FPS

# === Chemin absolu du fichier .sdp ===
script_dir = os.path.dirname(os.path.abspath(__file__))
sdp_path = os.path.join(script_dir, SDP_FILENAME)

# === Connexion au drone ===
bebop = Bebop()
print("Connexion au drone...")
if not bebop.connect(10):
    print("❌ Connexion échouée.")
    exit(1)

print("✅ Connecté.")

# === Vérifie que le fichier SDP est prêt ===
sdp_template = """v=0
o=- 0 0 IN IP4 127.0.0.1
s=Parrot Bebop2 Video
c=IN IP4 224.1.1.1
t=0 0
m=video 5004 RTP/AVP 96
a=rtpmap:96 H264/90000
"""

try:
    with open(sdp_path, 'w') as f:
        f.write(sdp_template)
except Exception as e:
    print(f"❌ Erreur lors de l'écriture du fichier SDP : {e}")
    bebop.disconnect()
    exit(1)

# === Lancer ffmpeg pour lire directement le flux vidéo ===
cmd = [
    FFMPEG_BIN,
    "-protocol_whitelist", "file,udp,rtp",
    "-fflags", "nobuffer",
    "-i", sdp_path,
    "-f", "rawvideo",
    "-pix_fmt", PIX_FMT,
    "-"
]

print("🎥 Lancement du flux vidéo...")
try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL  # ignore les logs ffmpeg
    )
except Exception as e:
    print(f"❌ Erreur lancement ffmpeg : {e}")
    bebop.disconnect()
    exit(1)

frame_size = WIDTH * HEIGHT * 3  # chaque pixel = 3 bytes (bgr)
last_display = 0

try:
    while True:
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            print("⚠️ Trame incomplète")
            continue

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        now = time.time()

        if now - last_display >= DISPLAY_INTERVAL:
            cv2.imshow("🎥 Bebop2 - Flux Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_display = now

except KeyboardInterrupt:
    print("\n⏹ Arrêt manuel.")

finally:
    print("🔌 Déconnexion et fermeture...")
    process.kill()
    bebop.disconnect()
    cv2.destroyAllWindows()
