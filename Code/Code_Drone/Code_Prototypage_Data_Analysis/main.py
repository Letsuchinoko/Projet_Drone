import cv2
import numpy as np
import time
import subprocess
import threading
import sys
import logging
import os
import pyparrot
from pyparrot.Bebop import Bebop
from collections import deque

# === PARAMÈTRES OPTIMISÉS POUR FLUIDITÉ ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_optimized_glove.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT ULTRA-OPTIMISÉ POUR FLUIDITÉ ===
class FastGloveDetector:
    def __init__(self):
        # Configuration minimaliste pour performance
        self.detection_history = deque(maxlen=10)  # Réduit
        self.stable_detections = deque(maxlen=3)   # Très réduit
        self.confidence_threshold = 2              # 2 sur 3
        
        # Paramètres de détection optimisés selon votre image
        self.min_area = 200
        self.max_area = 80000
        self.min_contour_points = 6
        
        # Kernels pré-calculés ultra-légers
        self.kernel_small = np.ones((2, 2), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((8, 8), np.uint8)
        
        # Stats simplifiées
        self.frame_count = 0
        self.detection_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def detect_glove_fast(self, frame):
        """Détection ultra-rapide focalisée sur le gant orange/rouge"""
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            # Pas de redimensionnement pour garder la vitesse
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur ultra-optimisé basé sur votre image
            mask = self._create_optimized_glove_mask(hsv)
            
            # Morphologie minimale
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # Détection rapide
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour_fast(contours)
            
            # Validation ultra-simple
            detected = best_contour is not None
            
            # Stabilisation minimale
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            # Historique
            self.detection_history.append(stable_detection)
            if stable_detection:
                self.detection_count += 1
            
            # Dessin optimisé
            if stable_detection and best_contour is not None:
                self._draw_fast_detection(original_frame, best_contour)
            
            # Overlay minimal
            result_frame = self._add_minimal_overlay(original_frame, stable_detection, mask)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return original_frame, False

    def _create_optimized_glove_mask(self, hsv):
        """Masque couleur optimisé basé sur votre gant orange/rouge"""
        try:
            h, w = hsv.shape[:2]
            
            # Masques orange optimisés (basé sur votre image)
            # Orange principal du gant
            orange_main_lower = np.array([10, 140, 140])
            orange_main_upper = np.array([22, 255, 255])
            mask_orange_main = cv2.inRange(hsv, orange_main_lower, orange_main_upper)
            
            # Orange légèrement désaturé (ombres)
            orange_shadow_lower = np.array([12, 100, 100])
            orange_shadow_upper = np.array([20, 200, 200])
            mask_orange_shadow = cv2.inRange(hsv, orange_shadow_lower, orange_shadow_upper)
            
            # Orange très vif (reflets)
            orange_bright_lower = np.array([8, 150, 180])
            orange_bright_upper = np.array([18, 255, 255])
            mask_orange_bright = cv2.inRange(hsv, orange_bright_lower, orange_bright_upper)
            
            # Masques rouge (parties rouges du gant)
            # Rouge principal
            red_main_lower1 = np.array([0, 140, 140])
            red_main_upper1 = np.array([8, 255, 255])
            mask_red_main1 = cv2.inRange(hsv, red_main_lower1, red_main_upper1)
            
            red_main_lower2 = np.array([172, 140, 140])
            red_main_upper2 = np.array([180, 255, 255])
            mask_red_main2 = cv2.inRange(hsv, red_main_lower2, red_main_upper2)
            
            # Rouge désaturé
            red_desat_lower1 = np.array([0, 80, 120])
            red_desat_upper1 = np.array([10, 180, 220])
            mask_red_desat1 = cv2.inRange(hsv, red_desat_lower1, red_desat_upper1)
            
            red_desat_lower2 = np.array([170, 80, 120])
            red_desat_upper2 = np.array([180, 180, 220])
            mask_red_desat2 = cv2.inRange(hsv, red_desat_lower2, red_desat_upper2)
            
            # Combinaison des masques orange
            mask_orange = cv2.bitwise_or(mask_orange_main, 
                         cv2.bitwise_or(mask_orange_shadow, mask_orange_bright))
            
            # Combinaison des masques rouge
            mask_red = cv2.bitwise_or(mask_red_main1, 
                      cv2.bitwise_or(mask_red_main2, 
                      cv2.bitwise_or(mask_red_desat1, mask_red_desat2)))
            
            # Masque final gant
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # Exclusion peau très ciblée (pour éviter faux positifs main)
            skin_lower = np.array([5, 50, 120])
            skin_upper = np.array([15, 120, 200])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Érosion de la peau pour garder le gant
            mask_skin_eroded = cv2.erode(mask_skin, self.kernel_small, iterations=1)
            
            # Application de l'exclusion
            mask_final = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_skin_eroded))
            
            # Bordures réduites
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 10
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            # Nettoyage léger
            mask_final = cv2.medianBlur(mask_final, 3)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Mask creation error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _select_best_contour_fast(self, contours):
        """Sélection ultra-rapide du meilleur contour"""
        if not contours:
            return None
            
        try:
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base ultra-rapides
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Sélection par aire (simple et rapide)
                if area > best_area:
                    # Validation forme basique
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Ratio acceptable pour une main/gant
                    if 0.2 <= aspect_ratio <= 4.0:
                        best_area = area
                        best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Fast contour selection error: {e}")
            return None

    def _draw_fast_detection(self, frame, contour):
        """Dessin ultra-rapide"""
        try:
            # Contour principal
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Centre simple
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            
            # Texte minimal
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT: {int(area)}", (x, max(y - 10, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Fast drawing error: {e}")

    def _add_minimal_overlay(self, frame, detected, mask):
        """Overlay ultra-minimal pour performance"""
        try:
            h, w = frame.shape[:2]
            
            # Status simple
            status = "🎯 GANT DETECTE" if detected else "🔍 RECHERCHE GANT"
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Stats essentielles
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Det: {detection_rate:.1f}%"
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # FPS temps réel
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Historique compact
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-10:]])
            cv2.putText(frame, f"Hist: {history}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Masque simplifié (optionnel)
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (120, 90))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                    
                    mask_x, mask_y = w - 130, 90
                    frame[mask_y:mask_y+90, mask_x:mask_x+120] = mask_colored
                    
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+120, mask_y+90), (255, 255, 255), 1)
                    cv2.putText(frame, "Masque", (mask_x, mask_y + 105), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except Exception:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Minimal overlay error: {e}")
            return frame

# === CONTRÔLE DRONE ULTRA-SIMPLE ===
def simple_drone_control(bebop):
    """Contrôle drone simplifié"""
    logger.info("Contrôle drone démarré.")
    print("\n[Commandes drone simples]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f/b/g/d = avant/arrière/gauche/droite\n"
          "  h/m = haut/bas | a/c = rotation gauche/droite\n")
    
    while True:
        try:
            key = input("> ").strip().lower()
        except EOFError:
            break
            
        if key == 't':
            bebop.safe_takeoff(10)
            print("✈️ Décollage")
        elif key == 'l':
            bebop.safe_land(10)
            print("🛬 Atterrissage")
        elif key == 'e':
            bebop.safe_land(10)
            bebop.disconnect()
            print("🔚 Arrêt")
            break
        elif key == 'f':
            bebop.fly_direct(roll=0, pitch=25, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-25, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'g':
            bebop.fly_direct(roll=-25, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'd':
            bebop.fly_direct(roll=25, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=0.3)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=0.3)
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-35, vertical_movement=0, duration=0.3)
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=35, vertical_movement=0, duration=0.3)

def main():
    """Fonction principale ultra-optimisée pour fluidité"""
    logger.info("=== BEBOP 2 OPTIMIZED GLOVE CAPTURE ===")
    logger.info("🚀 Version ultra-fluide pour capture parfaite du gant")
    
    bebop = None
    pipe = None
    detector = None
    start_time = time.time()
    
    try:
        # === CONNEXION DRONE ===
        logger.info("📡 Connexion au drone...")
        bebop = Bebop()
        if not bebop.connect(10):
            logger.error("❌ Échec connexion drone")
            return False

        logger.info("✅ Drone connecté!")
        
        # === FLUX VIDÉO ===
        logger.info("📹 Démarrage flux vidéo...")
        bebop.start_video_stream()
        time.sleep(2)  # Temps minimal
        
        # === CONTRÔLE DRONE ===
        ctrl_thread = threading.Thread(target=simple_drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG ULTRA-OPTIMISÉ ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # Commande FFmpeg optimisée pour latence minimale
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',           # Pas de buffer
            '-flags', 'low_delay',           # Latence minimale
            '-avioflags', 'direct',          # I/O direct
            '-analyzeduration', '500000',    # Analyse ultra-réduite (0.5s)
            '-probesize', '500000',          # Probe ultra-réduit (0.5MB)
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg ultra-rapide: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=512*1024)  # Buffer minimal
            logger.info("✅ Pipeline ultra-rapide initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR ULTRA-OPTIMISÉ ===
        detector = FastGloveDetector()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - Capture Gant Ultra-Fluide"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 50)
        logger.info("🎮 COMMANDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset")
        logger.info("=" * 50)
        
        # === BOUCLE PRINCIPALE ULTRA-RAPIDE ===
        logger.info("🎬 Démarrage capture ultra-fluide...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        skip_counter = 0
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Skip frames pour fluidité maximale (traiter 1 frame sur 2)
                skip_counter += 1
                if skip_counter % 2 != 0:
                    continue
                
                # Détection ultra-rapide
                processed_frame, detected = detector.detect_glove_fast(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Comptage FPS
                fps_counter += 1
                if fps_counter % 60 == 0:  # Tous les 60 frames affichées
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    logger.info(f"📊 FPS: {display_fps:.1f} | "
                               f"Détections: {detector.detection_count}/{detector.frame_count} "
                               f"({(detector.detection_count/max(detector.frame_count,1))*100:.1f}%)")
                    last_fps_log = current_time
                
                # Gestion touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                elif key == ord('s'):
                    screenshot_name = f"gant_capture_{int(time.time())}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord('r'):
                    detector.__init__()
                    logger.info("🔄 Reset détecteur")
                elif key == ord('d'):
                    # Debug masque
                    if hasattr(detector, '_last_mask'):
                        debug_name = f"debug_mask_{int(time.time())}.png"
                        cv2.imwrite(debug_name, detector._last_mask)
                        logger.info(f"🔍 Masque debug sauvé: {debug_name}")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle: {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        return False
        
    finally:
        # === NETTOYAGE ===
        logger.info("🧹 Nettoyage...")
        
        if detector:
            total_runtime = time.time() - start_time
            detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
            
            logger.info("=" * 50)
            logger.info("📊 STATS FINALES:")
            logger.info(f"  ⏱️ Durée: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames: {detector.frame_count}")
            logger.info(f"  ⚡ FPS: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections: {detector.detection_count} ({detection_rate:.1f}%)")
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            logger.info("=" * 50)
        
        if pipe:
            try:
                pipe.terminate()
                logger.info("✅ Pipeline fermé")
            except:
                pass
        
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Interface fermée")
        except:
            pass
        
        if bebop:
            try:
                bebop.disconnect()
                logger.info("✅ Drone déconnecté")
            except:
                pass
        
        logger.info("🎉 Terminé!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception: {e}")
        sys.exit(1)