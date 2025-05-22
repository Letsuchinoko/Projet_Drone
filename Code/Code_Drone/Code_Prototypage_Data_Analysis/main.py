import os
import cv2
import subprocess
import numpy as np

# === Configuration ===
FFMPEG_BIN = "ffmpeg"
SDP_FILENAME = "bebop.sdp"
WIDTH, HEIGHT = 1920, 1080  # Résolution fixée
PIX_FMT = "bgr24"
FPS = 10

# === Génère le fichier .sdp requis pour le flux Bebop2 ===
def create_sdp_file(path):
    sdp_content = """v=0
o=- 0 0 IN IP4 127.0.0.1
s=Parrot Bebop2
c=IN IP4 224.1.1.1
t=0 0
a=recvonly
m=video 5004 RTP/AVP 96
a=rtpmap:96 H264/90000
"""
    with open(path, "w") as f:
        f.write(sdp_content)

# === Initialise et lance ffmpeg pour lire le flux vidéo ===
def open_ffmpeg_stream(sdp_path):
    cmd = [
        FFMPEG_BIN,
        "-protocol_whitelist", "file,udp,rtp",
        "-fflags", "nobuffer",
        "-i", sdp_path,
        "-f", "rawvideo",
        "-pix_fmt", PIX_FMT,
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# === Affichage boucle principale ===
def stream_and_display():
    sdp_path = os.path.join(os.path.dirname(__file__), SDP_FILENAME)
    create_sdp_file(sdp_path)

    print("🎥 Démarrage du flux vidéo direct...")
    process = open_ffmpeg_stream(sdp_path)
    frame_size = WIDTH * HEIGHT * 3  # bgr24 = 3 bytes par pixel

    try:
        while True:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                print("⚠️ Trame incomplète")
                continue

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            frame = cv2.resize(frame, (960, 540))  # Affichage réduit
            cv2.imshow("🎥 Flux Bebop2 direct", frame)

            if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹ Arrêt manuel.")
    finally:
        process.kill()
        cv2.destroyAllWindows()

# === Lancement ===
if __name__ == "__main__":
    stream_and_display()
