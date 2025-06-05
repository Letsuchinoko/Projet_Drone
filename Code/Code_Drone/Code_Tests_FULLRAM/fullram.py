import cv2
import numpy as np
import time
import subprocess
import threading
import sys
import logging
from collections import deque

from pyparrot.Bebop import Bebop

# === PARAMÈTRES BEBOP 2 VIDEO ===
BEBOP_IP = "192.168.42.1"
BEBOP_PORT = 5600
WIDTH, HEIGHT = 856, 480   # Résolution standard Bebop 2 (tu peux ajuster)

# === LOGGING CONFIG ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_detection_ffmpeg.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedGloveDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=25)
        self.min_area = 400
        self.max_area = 70000
        self.min_contour_points = 10
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 7), np.uint8)
        self.stable_detections = deque(maxlen=5)
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        self.fps_start_time = time.time()
        self.last_fps = 0

    def detect_glove(self, frame):
        if frame is None:
            return frame, False
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        try:
            scale_factor = 1.0
            if w > 640:
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)

            # --- Masque couleur peau (à exclure) ---
            skin_lower = np.array([0, 30, 80])
            skin_upper = np.array([25, 130, 255])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)

            # --- Masque orange du gant ---
            orange_lower = np.array([10, 120, 120])
            orange_upper = np.array([23, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)

            # --- Masque rouge du gant (deux plages à cause du Hue circulaire) ---
            red_lower1 = np.array([0, 140, 120])
            red_upper1 = np.array([8, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_lower2 = np.array([170, 140, 120])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # --- Fusionne orange + rouge, puis enlève la peau ---
            mask_gant = cv2.bitwise_or(mask_orange, mask_red)
            mask = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin))

            # --- Nettoyage morpho ---
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours, work_frame.shape)
            detected = best_contour is not None
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= 2
            self.detection_history.append(stable_detection)

            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_detection(original_frame, best_contour)
                self.detection_count += 1
            result_frame = self._add_overlay(original_frame, stable_detection, mask)
            return result_frame, stable_detection

        except Exception as e:
            logger.error(f"Detection error: {e}")
            self.error_count += 1
            return original_frame, False

    def _select_best_contour(self, contours, frame_shape):
        if not contours:
            return None
        h, w = frame_shape[:2]
        best_contour = None
        best_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            if len(contour) < self.min_contour_points:
                continue
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = w_rect / float(h_rect)
            if not (0.2 <= aspect_ratio <= 4.0):
                continue
            if x < 5 or y < 5 or (x + w_rect) > (w - 5) or (y + h_rect) > (h - 5):
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.35:
                    continue
            position_score = 1.0 if y > h * 0.1 else 0.5
            area_score = min(area / 4000.0, 1.0)
            score = area_score * position_score
            if score > best_score:
                best_score = score
                best_contour = contour
        return best_contour

    def _draw_detection(self, frame, contour):
        try:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT DETECTE", (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Aire: {int(area)}", (x, max(y - 35, 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            logger.debug(f"Drawing error: {e}")

    def _add_overlay(self, frame, detected, mask=None):
        try:
            h, w = frame.shape[:2]
            status = "GANT DETECTE" if detected else "RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Detections: {self.detection_count} ({detection_rate:.1f}%)"
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            history_text = "Historique: " + "".join(["●" if x else "○" for x in list(self.detection_history)[-25:]])
            cv2.putText(frame, history_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if mask is not None and mask.size > 0:
                mask_small = cv2.resize(mask, (160, 120))
                mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                frame[10:130, w-170:w-10] = mask_colored
                cv2.rectangle(frame, (w-170, 10), (w-10, 130), (255, 255, 255), 1)
                cv2.putText(frame, "Masque", (w-160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return frame
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def drone_control_thread(bebop):
    logger.info("Contrôle du drone (pyparrot) démarré.")
    print("\n[Commandes clavier]\n"
          "  t = décoller\n"
          "  l = atterrir\n"
          "  e = quitter\n"
          "  f = avancer\n"
          "  b = reculer\n"
          "  g = gauche\n"
          "  d = droite\n"
          "  h = haut\n"
          "  m = bas\n"
          "  a = rotation gauche\n"
          "  c = rotation droite\n")
    while True:
        try:
            key = input("> ").strip().lower()
        except EOFError:
            print("Arrêt du thread contrôle drone (entrée clavier coupée).")
            break
        key = input("> ").strip().lower()
        if key == 't':
            bebop.safe_takeoff(10)
            print("Décollage")
        elif key == 'l':
            bebop.safe_land(10)
            print("Atterrissage")
        elif key == 'e':
            bebop.safe_land(10)
            bebop.disconnect()
            print("Fin du vol, arrêt du script.")
            break
        elif key == 'f':
            bebop.fly_direct(roll=0, pitch=40, yaw=0, vertical_movement=0, duration=1)
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-40, yaw=0, vertical_movement=0, duration=1)
        elif key == 'g':
            bebop.fly_direct(roll=-40, pitch=0, yaw=0, vertical_movement=0, duration=1)
        elif key == 'd':
            bebop.fly_direct(roll=40, pitch=0, yaw=0, vertical_movement=0, duration=1)
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=1)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-30, duration=1)
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-50, vertical_movement=0, duration=1)
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=50, vertical_movement=0, duration=1)
        else:
            print("Commande inconnue.")

def main():
    logger.info("Démarrage détection gant Bebop 2 en mode flux RAM (ffmpeg + OpenCV)")

    # -- Démarre le contrôle drone en thread séparé --
    bebop = Bebop()
    logger.info("Connexion au drone...")
    if not bebop.connect(10):
        logger.error("Echec connexion drone")
        return
    logger.info("Drone connecté !")
    ctrl_thread = threading.Thread(target=drone_control_thread, args=(bebop,), daemon=True)
    ctrl_thread.start()

    # -- Lance ffmpeg en pipe --
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', f'udp://@{BEBOP_IP}:{BEBOP_PORT}',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]
    try:
        pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    except FileNotFoundError:
        logger.error("ffmpeg non trouvé ! Ajoute ffmpeg à ton PATH système.")
        return

    detector = ImprovedGloveDetector()
    window_name = "Bebop 2 - Detection Gant Bicolore (RAM)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    screenshot_count = 0

    try:
        while True:
            raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
            if len(raw_frame) != WIDTH * HEIGHT * 3:
                logger.error("Problème lecture frame video, arrêt.")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
            detector.frame_count += 1
            processed_frame, detected = detector.detect_glove(frame)

            # Affichage
            cv2.imshow(window_name, processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Arrêt demandé par l'utilisateur.")
                break
            elif key == ord('r'):
                detector.frame_count = 0
                detector.detection_count = 0
                detector.error_count = 0
                logger.info("Statistiques réinitialisées.")
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}_{screenshot_count}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot sauvegardé : {screenshot_name}")
                screenshot_count += 1

            # FPS Log
            if detector.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - detector.fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                logger.info(f"FPS: {fps:.1f}")
                detector.fps_start_time = now

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
