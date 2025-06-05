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

# === PARAMÈTRES OPTIMISÉS ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480
BUFFER_SIZE = 3  # Buffer minimal pour réduire latence
SKIP_FRAMES = 1  # Traiter 1 frame sur 2 pour plus de fluidité

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_optimized_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT ULTRA-OPTIMISÉ ===
class UltraOptimizedGloveDetector:
    def __init__(self):
        # Historique réduit pour moins de latence
        self.detection_history = deque(maxlen=20)
        self.stable_detections = deque(maxlen=3)  # Fenêtre réduite
        self.confidence_threshold = 2  # 2 sur 3 pour réactivité
        
        # Paramètres de détection affinés pour capture complète
        self.min_area = 200  # Plus petit pour capturer les doigts
        self.max_area = 120000  # Plus grand pour gant complet
        self.min_contour_points = 6
        
        # Kernels morphologiques optimisés
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Plus petit
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))  # Plus grand
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))  # Pour connecter les doigts
        
        # Stats simplifiées
        self.frame_count = 0
        self.detection_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Cache ultra-léger
        self.last_valid_contour = None
        self.contour_age = 0
        
        # Paramètres de détection agressive pour capture complète
        self.dilation_iterations = 2
        self.closing_iterations = 1

    def detect_glove_fast(self, frame):
        """Détection ultra-rapide avec capture complète du gant"""
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            h, w = frame.shape[:2]
            
            # Pas de redimensionnement pour garder tous les détails
            work_frame = frame.copy()
            
            # Prétraitement minimal pour vitesse
            work_frame = cv2.medianBlur(work_frame, 3)  # Réduction du blur
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur agressif pour capture complète
            mask = self._create_aggressive_mask(hsv)
            
            # Morphologie agressive pour connecter les doigts
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=self.closing_iterations)
            mask = cv2.dilate(mask, self.kernel_dilate, iterations=self.dilation_iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            
            # Détection de contours avec approximation réduite
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            best_contour = self._select_best_contour_fast(contours, (h, w))
            
            # Validation simplifiée
            detected = best_contour is not None
            
            # Stabilisation ultra-rapide
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            # Mise à jour historique
            self.detection_history.append(stable_detection)
            if stable_detection:
                self.detection_count += 1
                self.last_valid_contour = best_contour.copy() if best_contour is not None else None
                self.contour_age = 0
            else:
                self.contour_age += 1
            
            # Dessin optimisé
            if stable_detection and best_contour is not None:
                self._draw_fast_detection(original_frame, best_contour)
            elif self.last_valid_contour is not None and self.contour_age < 5:
                # Afficher le dernier contour valide pendant quelques frames
                self._draw_ghost_detection(original_frame, self.last_valid_contour)
            
            # Overlay minimal pour performance
            result_frame = self._add_minimal_overlay(original_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return original_frame, False

    def _create_aggressive_mask(self, hsv):
        """Masque couleur agressif pour capture complète"""
        try:
            h, w = hsv.shape[:2]
            
            # Masques orange étendus (plus permissifs)
            orange_lower1 = np.array([5, 80, 80])   # Plus permissif
            orange_upper1 = np.array([30, 255, 255])
            mask_orange1 = cv2.inRange(hsv, orange_lower1, orange_upper1)
            
            orange_lower2 = np.array([8, 100, 60])   # Capture les zones sombres
            orange_upper2 = np.array([25, 200, 200])
            mask_orange2 = cv2.inRange(hsv, orange_lower2, orange_upper2)
            
            orange_lower3 = np.array([12, 60, 100])   # Capture les zones claires
            orange_upper3 = np.array([22, 255, 255])
            mask_orange3 = cv2.inRange(hsv, orange_lower3, orange_upper3)
            
            # Masques rouge étendus
            red_lower1 = np.array([0, 100, 80])
            red_upper1 = np.array([12, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([160, 100, 80])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            red_lower3 = np.array([170, 80, 60])   # Plus permissif
            red_upper3 = np.array([180, 200, 200])
            mask_red3 = cv2.inRange(hsv, red_lower3, red_upper3)
            
            # Combinaison de tous les masques
            mask_orange = cv2.bitwise_or(mask_orange1, cv2.bitwise_or(mask_orange2, mask_orange3))
            mask_red = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_red3))
            mask_gant = cv2.bitwise_or(mask_orange, mask_red)
            
            # Masque peau plus restrictif (pour ne pas trop exclure)
            skin_lower = np.array([0, 40, 100])
            skin_upper = np.array([20, 100, 200])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Exclusion peau avec érosion pour garder plus de gant
            mask_skin_eroded = cv2.erode(mask_skin, self.kernel_open, iterations=1)
            mask_final = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin_eroded))
            
            # Bordures réduites pour capturer plus
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 8  # Bordure réduite
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Aggressive mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _select_best_contour_fast(self, contours, frame_shape):
        """Sélection rapide du meilleur contour"""
        if not contours:
            return None
            
        try:
            h, w = frame_shape
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base rapides
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Score simple basé principalement sur l'aire
                area_score = min(area / 8000.0, 1.0)  # Optimal autour de 8000
                
                # Bonus pour les contours plus au centre
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Distance du centre
                    center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                    position_score = 1.0 - (center_dist / max_dist)
                else:
                    position_score = 0.5
                
                # Score final simplifié
                final_score = area_score * 0.8 + position_score * 0.2
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Fast contour selection error: {e}")
            return None

    def _draw_fast_detection(self, frame, contour):
        """Dessin rapide et efficace"""
        try:
            # Contour principal avec couleur vive
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Centre simplifié
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            
            # Texte principal seulement
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT DETECTE", (x, max(y - 10, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Aire: {int(area)}", (x, max(y - 35, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Fast drawing error: {e}")

    def _draw_ghost_detection(self, frame, contour):
        """Dessin fantôme pour continuité visuelle"""
        try:
            # Contour fantôme semi-transparent
            cv2.drawContours(frame, [contour], -1, (100, 200, 100), 2)
            
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)
            
            cv2.putText(frame, "TRACE", (x, max(y - 10, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
                       
        except Exception as e:
            logger.debug(f"Ghost drawing error: {e}")

    def _add_minimal_overlay(self, frame, detected):
        """Overlay minimal pour performance maximale"""
        try:
            h, w = frame.shape[:2]
            
            # Status simple
            status = "🎯 GANT DETECTE" if detected else "🔍 RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Stats essentielles seulement
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Det: {detection_rate:.1f}%"
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # FPS en temps réel
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Historique compact
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-20:]])
            cv2.putText(frame, f"Hist: {history}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Minimal overlay error: {e}")
            return frame

# === THREAD DE CONTRÔLE DRONE SIMPLIFIÉ ===
def simple_drone_control_thread(bebop):
    """Thread de contrôle drone simplifié"""
    logger.info("Contrôle du drone démarré.")
    print("\n[Commandes drone]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f = avant | b = arrière | g = gauche | d = droite\n"
          "  h = haut | m = bas | a = rot. gauche | c = rot. droite\n")
    
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
            bebop.fly_direct(roll=0, pitch=30, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-30, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'g':
            bebop.fly_direct(roll=-30, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'd':
            bebop.fly_direct(roll=30, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=25, duration=0.3)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-25, duration=0.3)
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-40, vertical_movement=0, duration=0.3)
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=40, vertical_movement=0, duration=0.3)

def main():
    """Fonction principale optimisée pour faible latence"""
    logger.info("=== BEBOP 2 ULTRA-OPTIMIZED DETECTION ===")
    logger.info("🚀 Version ultra-rapide avec capture complète du gant")
    
    bebop = None
    pipe = None
    detector = None
    start_time = time.time()
    
    try:
        # === CONNEXION DRONE ===
        logger.info("📡 Connexion au drone...")
        bebop = Bebop()
        if not bebop.connect(10):
            logger.error("❌ Echec connexion drone")
            return False

        logger.info("✅ Drone connecté!")
        
        # === FLUX VIDÉO ===
        logger.info("📹 Démarrage flux vidéo...")
        bebop.start_video_stream()
        time.sleep(2)  # Temps réduit
        
        # === CONTRÔLE DRONE ===
        ctrl_thread = threading.Thread(target=simple_drone_control_thread, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG OPTIMISÉ ===
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
            '-analyzeduration', '1000000',   # Analyse réduite (1s)
            '-probesize', '1000000',         # Probe réduit (1MB)
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg optimisé: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=1024*1024)  # Buffer réduit
            logger.info("✅ Pipeline optimisé initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR ULTRA-OPTIMISÉ ===
        detector = UltraOptimizedGloveDetector()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - Ultra Detection (Faible Latence)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 50)
        logger.info("🎮 COMMANDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset")
        logger.info("=" * 50)
        
        # === BOUCLE PRINCIPALE ULTRA-OPTIMISÉE ===
        logger.info("🎬 Démarrage boucle ultra-rapide...")
        
        frame_counter = 0
        screenshot_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                frame_counter += 1
                
                # Skip frames pour fluidité (traiter 1 frame sur SKIP_FRAMES)
                if frame_counter % (SKIP_FRAMES + 1) != 0:
                    continue
                
                # Détection ultra-rapide
                processed_frame, detected = detector.detect_glove_fast(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # FPS tracking
                fps_counter += 1
                if fps_counter % 60 == 0:  # Tous les 60 frames affichées
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    logger.info(f"📊 FPS Affichage: {display_fps:.1f} | "
                               f"Détections: {detector.detection_count} | "
                               f"Frames: {detector.frame_count}")
                    last_fps_time = current_time
                
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
            logger.info("=" * 50)
        
        if pipe:
            try:
                pipe.terminate()
                logger.info("✅ Pipeline fermé")
            except:
                pass
        
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Fenêtres fermées")
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