import os
import cv2
import time
import shutil
import subprocess
import numpy as np
import pyparrot
import json

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

# === DÉTECTION AUTOMATIQUE DE LA TAILLE VIA Ffprobe ===
def get_resolution(sdp_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        sdp_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        if video_stream:
            width = int(video_stream["width"])
            height = int(video_stream["height"])
            return width, height
    except Exception as e:
        print(f"❌ Erreur détection résolution : {e}")
    return 856, 480  # fallback

# === LECTURE FLUX VIDEO EN TEMPS RÉEL ===
def lire_flux_video_direct(path_sdp):
    if not os.path.exists(path_sdp):
        print(f"❌ Fichier SDP introuvable : {path_sdp}")
        return

    print("🎥 Démarrage du flux vidéo direct...")

    width, height = get_resolution(path_sdp)
    print(f"🖼️ Résolution détectée : {width}x{height}")
    frame_size = width * height * 3

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
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

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
