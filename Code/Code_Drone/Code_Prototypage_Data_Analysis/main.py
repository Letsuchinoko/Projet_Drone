import os
import cv2
import time
import shutil
import subprocess
import numpy as np
import pyparrot

# === CONSTANTES ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDP_SOURCE = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
SDP_LOCAL = os.path.join(SCRIPT_DIR, "bebop.sdp")

# === COPIE SDP LOCAL ===
def copier_sdp_local():
    if os.path.exists(SDP_SOURCE):
        shutil.copy2(SDP_SOURCE, SDP_LOCAL)
        print(f"✅ bebop.sdp copié dans le dossier local.")
    else:
        print("❌ Fichier bebop.sdp introuvable dans pyparrot.")

# === DÉCODAGE FLUX EN DIRECT ===
def lire_flux_video_direct(path_sdp):
    if not os.path.exists(path_sdp):
        print(f"❌ Fichier SDP introuvable : {path_sdp}")
        return

    print("🎥 Démarrage du flux vidéo direct...")

    ffmpeg_cmd = [
        "ffmpeg",
        "-protocol_whitelist", "file,rtp,udp",
        "-i", path_sdp,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]

    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # désactive les logs bruyants
            bufsize=10**8
        )

        width, height = 856, 480  # dimensions par défaut
        frame_size = width * height * 3

        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                print("⚠️ Trame incomplète")
                break

            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            cv2.imshow("📡 Flux Drone Direct", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"❌ Erreur durant le décodage : {e}")

    finally:
        cv2.destroyAllWindows()
        process.kill()

# === POINT D’ENTRÉE ===
if __name__ == "__main__":
    copier_sdp_local()
    lire_flux_video_direct(SDP_LOCAL)
