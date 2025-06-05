import cv2
import numpy as np
import time
import subprocess
import sys
import logging
import os
import pyparrot
from pyparrot.Bebop import Bebop

BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Connexion au drone...")
    bebop = Bebop()
    if not bebop.connect(10):
        logger.error("Echec connexion drone")
        return

    logger.info("Drone connecté !")

    # Démarre seulement le flux vidéo côté drone, PAS DroneVision/open_video
    bebop.start_video_stream()
    logger.info("Flux vidéo demandé au drone (start_video_stream).")
    time.sleep(2)

    sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
    if not os.path.exists(sdp_path):
        logger.error(f"Fichier SDP introuvable: {sdp_path}")
        return

    ffmpeg_cmd = [
        'ffmpeg',
        '-protocol_whitelist', 'file,rtp,udp',
        '-i', sdp_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    logger.info(f"Lancement de ffmpeg avec SDP : {' '.join(ffmpeg_cmd)}")
    try:
        pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    except FileNotFoundError:
        logger.error("ffmpeg non trouvé ! Ajoute ffmpeg à ton PATH système.")
        return

    window_name = "Bebop 2 - Flux vidéo direct (RAM, ffmpeg SDP)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
            if len(raw_frame) != WIDTH * HEIGHT * 3:
                logger.error("Problème lecture frame video, arrêt.")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Arrêt demandé par l'utilisateur.")
                break
    except KeyboardInterrupt:
        logger.info("Arrêt clavier demandé.")
    finally:
        logger.info("Nettoyage et arrêt du flux video...")
        pipe.terminate()
        cv2.destroyAllWindows()
        bebop.disconnect()
        logger.info("Drone déconnecté et script terminé.")

if __name__ == "__main__":
    main()
