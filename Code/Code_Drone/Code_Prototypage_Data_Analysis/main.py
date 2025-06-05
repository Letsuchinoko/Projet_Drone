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
        """Masque couleur ultra-précis pour éliminer faux positifs"""
        try:
            h, w = hsv.shape[:2]
            
            # === MASQUES ORANGE ULTRA-PRÉCIS ===
            # Orange principal du gant (très saturé)
            orange_main_lower = np.array([12, 160, 160])  # Saturation augmentée
            orange_main_upper = np.array([20, 255, 255])
            mask_orange_main = cv2.inRange(hsv, orange_main_lower, orange_main_upper)
            
            # Orange vif (reflets du gant)
            orange_bright_lower = np.array([10, 180, 180])  # Très saturé
            orange_bright_upper = np.array([18, 255, 255])
            mask_orange_bright = cv2.inRange(hsv, orange_bright_lower, orange_bright_upper)
            
            # Orange moyennement saturé (zones d'ombre du gant)
            orange_shadow_lower = np.array([14, 120, 140])  # Saturation minimale élevée
            orange_shadow_upper = np.array([19, 200, 220])
            mask_orange_shadow = cv2.inRange(hsv, orange_shadow_lower, orange_shadow_upper)
            
            # === MASQUES ROUGE ULTRA-PRÉCIS ===
            # Rouge principal (très saturé)
            red_main_lower1 = np.array([0, 160, 160])   # Saturation élevée
            red_main_upper1 = np.array([6, 255, 255])
            mask_red_main1 = cv2.inRange(hsv, red_main_lower1, red_main_upper1)
            
            red_main_lower2 = np.array([174, 160, 160])  # Saturation élevée
            red_main_upper2 = np.array([180, 255, 255])
            mask_red_main2 = cv2.inRange(hsv, red_main_lower2, red_main_upper2)
            
            # Rouge moyennement saturé
            red_medium_lower1 = np.array([0, 120, 140])
            red_medium_upper1 = np.array([8, 200, 240])
            mask_red_medium1 = cv2.inRange(hsv, red_medium_lower1, red_medium_upper1)
            
            red_medium_lower2 = np.array([172, 120, 140])
            red_medium_upper2 = np.array([180, 200, 240])
            mask_red_medium2 = cv2.inRange(hsv, red_medium_lower2, red_medium_upper2)
            
            # === EXCLUSIONS STRICTES ===
            
            # 1. Exclusion herbe/végétation (vert-jaune)
            grass_lower1 = np.array([25, 40, 40])   # Vert-jaune
            grass_upper1 = np.array([80, 255, 255])
            mask_grass = cv2.inRange(hsv, grass_lower1, grass_upper1)
            
            # 2. Exclusion murs/béton (faible saturation)
            wall_lower = np.array([0, 0, 0])       # Très faible saturation
            wall_upper = np.array([180, 60, 255])   # Seuil saturation strict
            mask_wall = cv2.inRange(hsv, wall_lower, wall_upper)
            
            # 3. Exclusion jaune/doré (confusion possible)
            yellow_lower = np.array([20, 100, 100])  # Jaune pur
            yellow_upper = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            # 4. Exclusion peau stricte
            skin_lower = np.array([5, 60, 120])     # Saturation min élevée
            skin_upper = np.array([15, 140, 220])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # 5. Exclusion bois/marron
            wood_lower = np.array([8, 50, 50])
            wood_upper = np.array([25, 150, 180])
            mask_wood = cv2.inRange(hsv, wood_lower, wood_upper)
            
            # === COMBINAISONS ===
            
            # Masque orange final
            mask_orange = cv2.bitwise_or(mask_orange_main, 
                         cv2.bitwise_or(mask_orange_bright, mask_orange_shadow))
            
            # Masque rouge final
            mask_red = cv2.bitwise_or(mask_red_main1, 
                      cv2.bitwise_or(mask_red_main2, 
                      cv2.bitwise_or(mask_red_medium1, mask_red_medium2)))
            
            # Masque gant complet
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # Masque d'exclusions combiné
            mask_exclusions = cv2.bitwise_or(mask_grass, 
                             cv2.bitwise_or(mask_wall, 
                             cv2.bitwise_or(mask_yellow, 
                             cv2.bitwise_or(mask_skin, mask_wood))))
            
            # Application stricte des exclusions
            mask_exclusions_dilated = cv2.dilate(mask_exclusions, self.kernel_medium, iterations=2)
            mask_final = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_exclusions_dilated))
            
            # === FILTRAGE SPATIAL ===
            
            # Bordures strictes
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 15  # Bordure plus large
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            # Zone centrale privilégiée (gant plus probable au centre)
            center_bonus = np.zeros((h, w), dtype=np.uint8)
            center_h_start, center_h_end = h//4, 3*h//4
            center_w_start, center_w_end = w//4, 3*w//4
            center_bonus[center_h_start:center_h_end, center_w_start:center_w_end] = 255
            
            # Application des filtres spatiaux
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            # Bonus pour zone centrale (dilatation plus forte au centre)
            mask_center = cv2.bitwise_and(mask_final, center_bonus)
            mask_center_enhanced = cv2.dilate(mask_center, self.kernel_medium, iterations=1)
            mask_final = cv2.bitwise_or(mask_final, mask_center_enhanced)
            
            # === POST-TRAITEMENT ===
            
            # Nettoyage morphologique strict
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, self.kernel_medium, iterations=1)
            
            # Suppression des petits artefacts
            mask_final = cv2.medianBlur(mask_final, 5)
            
            # Sauvegarde pour debug
            self._last_mask = mask_final
            self._debug_masks = {
                'orange': mask_orange,
                'red': mask_red,
                'exclusions': mask_exclusions,
                'final': mask_final
            }
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Optimized mask creation error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _select_best_contour_fast(self, contours):
        """Sélection ultra-stricte pour éliminer faux positifs"""
        if not contours:
            return None
            
        try:
            h, w = HEIGHT, WIDTH
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base stricts
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Analyse géométrique stricte
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                aspect_ratio = w_rect / float(h_rect)
                
                # Ratio d'aspect strict pour une main/gant
                if not (0.3 <= aspect_ratio <= 2.5):
                    continue
                
                # Position : éviter les bords et privilégier le centre
                margin = 25
                if (x < margin or y < margin or 
                    (x + w_rect) > (w - margin) or 
                    (y + h_rect) > (h - margin)):
                    continue
                
                # Analyse de forme avancée
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area <= 0:
                    continue
                
                solidity = area / hull_area
                
                # Solidité stricte (éliminer objets trop irréguliers)
                if solidity < 0.4 or solidity > 0.95:  # Pas trop régulier non plus
                    continue
                
                # Test de compacité (éliminer objets trop allongés/fins)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter * perimeter)
                    # Éviter objets trop allongés (comme herbe, branches)
                    if compactness < 0.1:
                        continue
                
                # Centre de masse (pour position)
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Privilégier zone centrale de l'image
                center_x_norm = cx / w
                center_y_norm = cy / h
                
                # Bonus pour position centrale
                position_bonus = 1.0
                if 0.3 <= center_x_norm <= 0.7 and 0.2 <= center_y_norm <= 0.8:
                    position_bonus = 1.5
                
                # Score basé sur l'aire avec bonus position
                area_score = min(area / 3000.0, 1.0) * position_bonus
                
                # Bonus pour forme "main-like"
                shape_bonus = 1.0
                if 0.6 <= aspect_ratio <= 1.4 and 0.5 <= solidity <= 0.8:
                    shape_bonus = 1.3
                
                # Score final
                final_score = area_score * shape_bonus
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Strict contour selection error: {e}")
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
                    # Debug masques détaillé
                    if hasattr(detector, '_debug_masks'):
                        timestamp = int(time.time())
                        
                        # Sauvegarde masque final
                        debug_final = f"debug_final_{timestamp}.png"
                        cv2.imwrite(debug_final, detector._debug_masks['final'])
                        
                        # Sauvegarde masque orange
                        debug_orange = f"debug_orange_{timestamp}.png"
                        cv2.imwrite(debug_orange, detector._debug_masks['orange'])
                        
                        # Sauvegarde masque rouge
                        debug_red = f"debug_red_{timestamp}.png"
                        cv2.imwrite(debug_red, detector._debug_masks['red'])
                        
                        # Sauvegarde masque exclusions
                        debug_exclusions = f"debug_exclusions_{timestamp}.png"
                        cv2.imwrite(debug_exclusions, detector._debug_masks['exclusions'])
                        
                        logger.info(f"🔍 Masques debug sauvés:")
                        logger.info(f"   Final: {debug_final}")
                        logger.info(f"   Orange: {debug_orange}")
                        logger.info(f"   Rouge: {debug_red}")
                        logger.info(f"   Exclusions: {debug_exclusions}")
                    else:
                        logger.info("❌ Pas de données debug disponibles")

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