import os
import cv2
import time
import numpy as np
import subprocess
import threading

# === CONFIGURATION ===
WIDTH, HEIGHT = 1280, 720  # Résolution souhaitée
PIX_FMT = "bgr24"
SDP_FILENAME = "bebop.sdp"
FFMPEG_BIN = "ffmpeg"
DISPLAY_INTERVAL = 1 / 10  # 10 FPS

# === Préparation chemins ===
script_dir = os.path.dirname(os.path.abspath(__file__))
sdp_path = os.path.join(script_dir, SDP_FILENAME)

# === Vérifie que le fichier SDP existe ===
if not os.path.exists(sdp_path):
    print(f"❌ Fichier SDP introuvable : {sdp_path}")
    exit(1)

# === Démarrage du flux ffmpeg ===
print("🎥 Démarrage du flux vidéo direct...")
cmd = [
    FFMPEG_BIN,
    "-protocol_whitelist", "file,udp,rtp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", sdp_path,
    "-f", "rawvideo",
    "-pix_fmt", PIX_FMT,
    "-"
]

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL  # ignorer les erreurs flood
    )
    time.sleep(2)  # Laisse ffmpeg initier le flux correctement
except Exception as e:
    print(f"❌ Impossible de lancer ffmpeg : {e}")
    exit(1)

frame_size = WIDTH * HEIGHT * 3
last_display = 0

try:
    # Attente de la première frame complète
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) == frame_size:
            break
        time.sleep(0.01)

    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            time.sleep(0.01)
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))

        now = time.time()
        if now - last_display >= DISPLAY_INTERVAL:
            cv2.imshow("🎥 Bebop2 Live Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_display = now

except KeyboardInterrupt:
    print("\n⏹ Arrêt manuel.")

finally:
    print("Fermeture du flux...")
    process.kill()
    cv2.destroyAllWindows()
