import subprocess
import numpy as np
import cv2
import os

# Chemin vers bebop.sdp dans le même dossier que le script
current_dir = os.path.dirname(os.path.abspath(__file__))
sdp_path = os.path.join(current_dir, "bebop.sdp")

if not os.path.exists(sdp_path):
    print("❌ Le fichier bebop.sdp est introuvable.")
    exit()

print("🎥 Démarrage du flux via FFmpeg en lecture mémoire...")

# Commande FFmpeg avec protocol_whitelist
cmd = [
    "ffmpeg",
    "-protocol_whitelist", "file,rtp,udp",
    "-i", sdp_path,
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

# Taille des images attendues (856x480), modifiable si besoin
width, height = 856, 480
frame_size = width * height * 3  # bgr24 = 3 bytes par pixel

# Lancement de FFmpeg
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

try:
    while True:
        raw_frame = proc.stdout.read(frame_size)
        if not raw_frame:
            print("⚠️ Flux interrompu")
            break

        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        cv2.imshow("🎥 Flux Bebop direct", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹ Arrêt manuel.")
finally:
    proc.terminate()
    cv2.destroyAllWindows()
