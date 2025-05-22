import ffmpeg
import numpy as np
import cv2
import subprocess

# === Flux vidéo du Bebop 2 (via UDP multicast par défaut)
SDP_PATH = "bebop.sdp"

# ⚠️ Chemin vers un fichier bebop.sdp contenant :
# v=0
# m=video 55004 RTP/AVP 96
# a=rtpmap:96 H264/90000
# c=IN IP4 0.0.0.0

FFMPEG_CMD = [
    'ffmpeg',
    '-protocol_whitelist', 'file,udp,rtp',
    '-i', SDP_PATH,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-loglevel', 'error',
    '-r', '10',  # 10 FPS
    '-'
]

try:
    print("🎥 Démarrage du flux vidéo direct...")

    # Lance ffmpeg pour streamer le flux vers stdout
    process = subprocess.Popen(
        FFMPEG_CMD, stdout=subprocess.PIPE, bufsize=10**8
    )

    width, height = 856, 480  # résolution par défaut du Bebop 2

    while True:
        # Lit une image brute (bgr24)
        raw_frame = process.stdout.read(width * height * 3)
        if len(raw_frame) != width * height * 3:
            print("⚠️ Trame incomplète")
            break

        # Transforme en image OpenCV (numpy)
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        cv2.imshow("🎥 Flux Bebop2 Direct (ffmpeg + numpy)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("⏹ Interruption manuelle.")

finally:
    process.kill()
    cv2.destroyAllWindows()
