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

# === PARAMÈTRES OPTIMISÉS POUR PERFORMANCE ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_optimized.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR OPTIMISÉ POUR VITESSE ===
class FastGloveDetector:
    def __init__(self):
        # Configuration optimisée pour vitesse
        self.detection_history = deque(maxlen=8)  # Réduit
        self.stable_detections = deque(maxlen=3)   # Réduit
        self.confidence_threshold = 2
        
        # Paramètres de détection
        self.min_area = 150
        self.max_area = 50000
        self.min_contour_points = 5
        
        # Zoom adaptatif simplifié
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.15  # Plus rapide
        self.zoom_min = 1.0
        self.zoom_max = 4.0  # Réduit pour moins de calculs
        
        # Cache pour optimisation
        self.frame_skip = 1  # Skip frames pour performance
        self.process_every_n = 2  # Traiter 1 frame sur 2
        self.frame_counter = 0
        
        # Stats simplifiées
        self.frame_count = 0
        self.detection_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.processing_times = deque(maxlen=10)  # Réduit
        
        # Cache dernière détection pour interpolation
        self.last_detection = None
        self.last_contour = None
        
        logger.info("⚡ Détecteur Fast Glove initialisé pour performance maximale")

    def fast_detect_glove(self, frame):
        """Détection ultra-rapide avec optimisations agressives"""
        if frame is None:
            return frame, False
            
        start_time = time.time()
        original_frame = frame.copy()
        self.frame_count += 1
        self.frame_counter += 1
        
        try:
            # === OPTIMISATION 1: SKIP FRAMES ===
            if self.frame_counter % self.process_every_n != 0:
                # Utiliser la dernière détection pour interpolation
                if self.last_detection and self.last_contour is not None:
                    result_frame = self._draw_fast_detection(original_frame, self.last_contour)
                    return result_frame, True
                else:
                    return self._add_fast_overlay(original_frame, False), False
            
            # === OPTIMISATION 2: PRÉTRAITEMENT MINIMAL ===
            # Redimensionnement pour vitesse si nécessaire
            process_frame = frame
            scale_factor = 1.0
            
            if self.zoom_factor > 2.0:
                # Traitement sur image réduite pour zoom élevé
                new_width = int(WIDTH * 0.8)
                new_height = int(HEIGHT * 0.8)
                process_frame = cv2.resize(frame, (new_width, new_height))
                scale_factor = WIDTH / new_width
            
            # === OPTIMISATION 3: DÉTECTION RAPIDE ===
            detected, contour, area = self._ultra_fast_detection(process_frame)
            
            # Remapping si nécessaire
            if detected and contour is not None and scale_factor != 1.0:
                contour = (contour * scale_factor).astype(np.int32)
                area = cv2.contourArea(contour)
            
            # === OPTIMISATION 4: VALIDATION SIMPLIFIÉE ===
            if detected:
                if self._quick_validation(contour, area):
                    self.detection_count += 1
                    self.last_detection = True
                    self.last_contour = contour
                    self._update_zoom_fast(area)
                else:
                    detected = False
                    contour = None
            
            # === OPTIMISATION 5: RENDU RAPIDE ===
            self.detection_history.append(detected)
            self.stable_detections.append(detected)
            
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            if stable_detection and contour is not None:
                result_frame = self._draw_fast_detection(original_frame, contour)
            else:
                result_frame = self._add_fast_overlay(original_frame, False)
                self.last_detection = False
                self.last_contour = None
            
            # Stats performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Fast detection error: {e}")
            return original_frame, False

    def _ultra_fast_detection(self, frame):
        """Détection ultra-rapide avec algorithmes optimisés"""
        try:
            # === CONVERSION COULEUR RAPIDE ===
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Pas de flou pour gagner du temps
            
            # === MASQUES COULEUR OPTIMISÉS ===
            # Orange simplifié
            orange_lower = np.array([10, 160, 160])
            orange_upper = np.array([20, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge simplifié
            red_lower1 = np.array([0, 160, 160])
            red_upper1 = np.array([8, 255, 255])
            red_lower2 = np.array([172, 160, 160])
            red_upper2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # === MORPHOLOGIE MINIMALE ===
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_glove = cv2.morphologyEx(mask_glove, cv2.MORPH_CLOSE, kernel)
            mask_glove = cv2.morphologyEx(mask_glove, cv2.MORPH_OPEN, 
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            
            # === DÉTECTION CONTOURS RAPIDE ===
            contours, _ = cv2.findContours(mask_glove, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return False, None, 0
            
            # Sélection rapide du plus grand contour valide
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > self.min_area and area < self.max_area and len(contour) >= self.min_contour_points:
                    if area > best_area:
                        best_area = area
                        best_contour = contour
            
            return best_contour is not None, best_contour, best_area
            
        except Exception as e:
            logger.debug(f"Ultra fast detection error: {e}")
            return False, None, 0

    def _quick_validation(self, contour, area):
        """Validation rapide et simple"""
        try:
            if contour is None or area < self.min_area:
                return False
            
            # Validation géométrique basique
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filtres simples
            if not (0.3 <= aspect_ratio <= 3.0):
                return False
            
            # Validation aire
            if area > self.max_area:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Quick validation error: {e}")
            return False

    def _update_zoom_fast(self, area):
        """Mise à jour zoom rapide"""
        try:
            # Fonction de zoom simplifiée
            if area < 1000:
                self.target_zoom = min(self.zoom_max, 3.0)
            elif area < 2500:
                self.target_zoom = 2.0
            elif area < 5000:
                self.target_zoom = 1.5
            else:
                self.target_zoom = 1.0
            
            # Application zoom lissé
            self.zoom_factor += (self.target_zoom - self.zoom_factor) * self.zoom_smooth_factor
            self.zoom_factor = np.clip(self.zoom_factor, self.zoom_min, self.zoom_max)
            
        except Exception as e:
            logger.debug(f"Fast zoom update error: {e}")

    def _draw_fast_detection(self, frame, contour):
        """Dessin rapide de la détection"""
        try:
            area = cv2.contourArea(contour)
            
            # Couleur selon taille
            if area > 3000:
                color = (0, 255, 0)      # Vert - proche
                distance_text = "PROCHE"
            elif area > 1500:
                color = (0, 255, 255)    # Jaune - moyen
                distance_text = "MOYEN"
            else:
                color = (0, 150, 255)    # Orange - loin
                distance_text = "LOIN"
            
            # Contour simple
            cv2.drawContours(frame, [contour], -1, color, 2)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            
            # Centre simple
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            
            # Texte minimal
            cv2.putText(frame, f"GANT {distance_text}", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Aire: {int(area)}", (x, max(y - 30, 40)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return self._add_fast_overlay(frame, True)
            
        except Exception as e:
            logger.debug(f"Fast drawing error: {e}")
            return frame

    def _add_fast_overlay(self, frame, detected):
        """Overlay rapide avec informations essentielles"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            if detected:
                status = "🎯 GANT DÉTECTÉ"
                color = (0, 255, 0)
            else:
                status = "🔍 RECHERCHE"
                color = (0, 255, 255)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Zoom barre simple
            zoom_width = int(200 * (self.zoom_factor - 1.0) / (self.zoom_max - 1.0))
            cv2.rectangle(frame, (10, 50), (210, 65), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 50), (10 + zoom_width, 65), (0, 255, 255), -1)
            cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x", (220, 62),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS simple
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
            # Temps de traitement
            if self.processing_times:
                avg_time = np.mean(self.processing_times) * 1000
                cv2.putText(frame, f"Proc: {avg_time:.1f}ms", (w - 150, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Stats basiques
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            cv2.putText(frame, f"Det: {detection_rate:.1f}%", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Fast overlay error: {e}")
            return frame

    def reset(self):
        """Reset rapide"""
        self.detection_history.clear()
        self.stable_detections.clear()
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.last_detection = None
        self.last_contour = None
        logger.info("🔄 Détecteur reset")

# === CONTRÔLE DRONE SIMPLIFIÉ ===
def simple_drone_control(bebop):
    """Contrôle drone basique pour performance"""
    logger.info("🎮 Contrôle drone simplifié démarré")
    print("\n[Commandes drone]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f/b/g/d = mouvements | h/m = haut/bas\n")
    
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
            bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=0.2)
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-20, yaw=0, vertical_movement=0, duration=0.2)
        elif key == 'g':
            bebop.fly_direct(roll=-20, pitch=0, yaw=0, vertical_movement=0, duration=0.2)
        elif key == 'd':
            bebop.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.2)
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=15, duration=0.2)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-15, duration=0.2)

# === FONCTION PRINCIPALE OPTIMISÉE ===
def main():
    """Fonction principale optimisée pour vitesse maximale"""
    logger.info("=" * 60)
    logger.info("⚡ BEBOP 2 - DÉTECTION ULTRA-RAPIDE")
    logger.info("🎯 Optimisé pour vitesse et réactivité")
    logger.info("=" * 60)
    
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
        time.sleep(2)
        
        # === CONTRÔLE DRONE ===
        ctrl_thread = threading.Thread(target=simple_drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG OPTIMISÉ ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg optimisé pour vitesse
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '300000',    # Réduit pour vitesse
            '-probesize', '300000',
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info("🚀 FFmpeg ultra-rapide configuré")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=1024*1024)
            logger.info("✅ Pipeline ultra-rapide initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR RAPIDE ===
        detector = FastGloveDetector()
        
        # === INTERFACE RAPIDE ===
        window_name = "Bebop 2 - Détection Ultra-Rapide"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 60)
        logger.info("🎮 COMMANDES RAPIDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset")
        logger.info("  '+'/'-' = Zoom | 'f' = Mode rapide toggle")
        logger.info("=" * 60)
        logger.info("⚡ OPTIMISATIONS ACTIVES:")
        logger.info("  ✓ Skip frames intelligent")
        logger.info("  ✓ Détection ultra-rapide")
        logger.info("  ✓ Rendu minimal")
        logger.info("  ✓ Cache intelligent")
        logger.info("=" * 60)
        
        # === BOUCLE PRINCIPALE ULTRA-RAPIDE ===
        logger.info("🎬 Démarrage détection ultra-rapide...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        fast_mode = True
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # === DÉTECTION ULTRA-RAPIDE ===
                if fast_mode:
                    processed_frame, detected = detector.fast_detect_glove(frame)
                else:
                    # Mode standard si besoin
                    processed_frame, detected = detector.fast_detect_glove(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # === LOGGING PERFORMANCE SIMPLIFIÉ ===
                fps_counter += 1
                if fps_counter % 60 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    
                    detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
                    avg_proc_time = np.mean(detector.processing_times) * 1000 if detector.processing_times else 0
                    
                    logger.info(f"⚡ FPS: {display_fps:.1f} | "
                               f"Détections: {detector.detection_count}/{detector.frame_count} "
                               f"({detection_rate:.1f}%) | "
                               f"Proc: {avg_proc_time:.1f}ms | "
                               f"Zoom: {detector.zoom_factor:.1f}x")
                    
                    last_fps_log = current_time
                
                # === GESTION TOUCHES RAPIDE ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    # Screenshot rapide
                    timestamp = int(time.time())
                    screenshot_name = f"fast_capture_{timestamp}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    # Reset rapide
                    detector.reset()
                    
                elif key == ord('+') or key == ord('='):
                    # Zoom +
                    detector.target_zoom = min(detector.zoom_max, detector.target_zoom + 0.5)
                    logger.info(f"🔍 Zoom: {detector.target_zoom:.1f}x")
                    
                elif key == ord('-'):
                    # Zoom -
                    detector.target_zoom = max(detector.zoom_min, detector.target_zoom - 0.5)
                    logger.info(f"🔍 Zoom: {detector.target_zoom:.1f}x")
                    
                elif key == ord('f'):
                    # Toggle fast mode
                    fast_mode = not fast_mode
                    mode_text = "ULTRA-RAPIDE" if fast_mode else "STANDARD"
                    logger.info(f"⚡ Mode: {mode_text}")

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
        # === NETTOYAGE RAPIDE ===
        logger.info("🧹 Nettoyage...")
        
        if detector:
            total_runtime = time.time() - start_time
            detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
            avg_proc_time = np.mean(detector.processing_times) * 1000 if detector.processing_times else 0
            
            logger.info("=" * 60)
            logger.info("📊 STATS FINALES ULTRA-RAPIDES:")
            logger.info(f"  ⏱️ Durée: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames: {detector.frame_count}")
            logger.info(f"  ⚡ FPS moyen: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections: {detector.detection_count} ({detection_rate:.1f}%)")
            logger.info(f"  ⚙️ Temps proc moy: {avg_proc_time:.1f}ms")
            logger.info(f"  🔍 Zoom final: {detector.zoom_factor:.1f}x")
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            logger.info("=" * 60)
        
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
        
        logger.info("⚡ Session ultra-rapide terminée!")
    
    return True

if __name__ == "__main__":
    try:
        logger.info("⚡ LANCEMENT MODE ULTRA-RAPIDE")
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception: {e}")
        sys.exit(1)