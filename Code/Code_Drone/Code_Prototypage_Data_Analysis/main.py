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

# === PARAMÈTRES ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_working_zoom.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT BASE + ZOOM SIMPLE ===
class WorkingGloveDetectorWithZoom:
    def __init__(self):
        # Configuration de base qui fonctionne
        self.detection_history = deque(maxlen=10)
        self.stable_detections = deque(maxlen=3)
        self.confidence_threshold = 2
        
        # Paramètres permissifs pour être sûr de détecter
        self.min_area = 150
        self.max_area = 100000
        self.min_contour_points = 6
        
        # Kernels simples
        self.kernel_small = np.ones((2, 2), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        
        # Zoom simple
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_center = (WIDTH//2, HEIGHT//2)
        self.manual_zoom = False
        
        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def detect_glove(self, frame):
        """Détection de base qui fonctionne + zoom optionnel"""
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            # Appliquer zoom si activé
            work_frame = self._apply_simple_zoom(frame) if self.zoom_factor > 1.1 else frame
            
            # Conversion HSV
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur SIMPLE et PERMISSIF (qui marchait avant)
            mask = self._create_working_mask(hsv)
            
            # Morphologie légère
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # Détection contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour, area = self._select_best_contour(contours)
            
            # Remapping si zoom
            if best_contour is not None and self.zoom_factor > 1.1:
                best_contour = self._remap_contour_from_zoom(best_contour)
                area = cv2.contourArea(best_contour)
            
            # Validation simple
            detected = best_contour is not None and area > self.min_area
            
            # Stabilisation
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            # Historique
            self.detection_history.append(stable_detection)
            if stable_detection:
                self.detection_count += 1
                
                # Auto-zoom basé sur l'aire (optionnel)
                if not self.manual_zoom:
                    self._auto_adjust_zoom(area)
            
            # Dessin
            if stable_detection and best_contour is not None:
                self._draw_detection(original_frame, best_contour, area)
            
            # Overlay
            result_frame = self._add_overlay(original_frame, stable_detection, area, mask)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            return original_frame, False

    def _create_working_mask(self, hsv):
        """Masque couleur PERMISSIF qui fonctionne à coup sûr"""
        try:
            h, w = hsv.shape[:2]
            
            # === MASQUES TRÈS PERMISSIFS ===
            
            # Orange très large
            orange_lower = np.array([8, 80, 80])    # Très permissif
            orange_upper = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge très large (2 plages)
            red_lower1 = np.array([0, 80, 80])
            red_upper1 = np.array([12, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([165, 80, 80])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison
            mask_gant = cv2.bitwise_or(mask_orange, mask_red)
            
            # Exclusion peau TRÈS ciblée (pour ne pas trop exclure)
            skin_lower = np.array([0, 40, 100])
            skin_upper = np.array([20, 100, 200])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Érosion légère de la peau
            mask_skin = cv2.erode(mask_skin, self.kernel_small, iterations=1)
            
            # Application
            mask_final = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin))
            
            # Bordures minimales
            border_size = 8
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            # Nettoyage minimal
            mask_final = cv2.medianBlur(mask_final, 3)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Working mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _select_best_contour(self, contours):
        """Sélection simple et permissive"""
        if not contours:
            return None, 0
            
        try:
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres très permissifs
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Juste prendre le plus grand
                if area > best_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    # Ratio très permissif
                    if 0.1 <= aspect_ratio <= 5.0:
                        best_area = area
                        best_contour = contour
            
            return best_contour, best_area
            
        except Exception as e:
            logger.debug(f"Contour selection error: {e}")
            return None, 0

    def _apply_simple_zoom(self, frame):
        """Zoom simple sur toute l'image"""
        try:
            h, w = frame.shape[:2]
            
            if self.zoom_factor <= 1.05:
                return frame
            
            # Zone de crop centrée
            crop_w = int(w / self.zoom_factor)
            crop_h = int(h / self.zoom_factor)
            
            # Centrage
            center_x, center_y = self.zoom_center
            offset_x = max(0, min(center_x - crop_w // 2, w - crop_w))
            offset_y = max(0, min(center_y - crop_h // 2, h - crop_h))
            
            # Crop et redimensionnement
            cropped = frame[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Stocker info pour remapping
            self._zoom_info = {
                'offset_x': offset_x,
                'offset_y': offset_y,
                'crop_w': crop_w,
                'crop_h': crop_h
            }
            
            return zoomed
            
        except Exception as e:
            logger.debug(f"Simple zoom error: {e}")
            return frame

    def _remap_contour_from_zoom(self, contour):
        """Remapping simple du contour zoomé"""
        try:
            if not hasattr(self, '_zoom_info'):
                return contour
            
            info = self._zoom_info
            scale_x = info['crop_w'] / WIDTH
            scale_y = info['crop_h'] / HEIGHT
            
            remapped = contour.copy()
            remapped[:, :, 0] = (contour[:, :, 0] * scale_x + info['offset_x']).astype(np.int32)
            remapped[:, :, 1] = (contour[:, :, 1] * scale_y + info['offset_y']).astype(np.int32)
            
            return remapped
            
        except Exception as e:
            logger.debug(f"Remap error: {e}")
            return contour

    def _auto_adjust_zoom(self, area):
        """Auto-ajustement zoom simple basé sur l'aire"""
        try:
            if area <= 0:
                return
            
            # Zoom simple basé sur l'aire
            if area < 500:
                self.target_zoom = 3.0
            elif area < 1000:
                self.target_zoom = 2.5
            elif area < 2000:
                self.target_zoom = 2.0
            elif area < 4000:
                self.target_zoom = 1.5
            else:
                self.target_zoom = 1.0
            
            # Lissage
            self.zoom_factor += (self.target_zoom - self.zoom_factor) * 0.1
            self.zoom_factor = max(1.0, min(4.0, self.zoom_factor))
            
        except Exception as e:
            logger.debug(f"Auto zoom error: {e}")

    def _draw_detection(self, frame, contour, area):
        """Dessin simple"""
        try:
            # Couleur selon l'aire
            if area > 3000:
                color = (0, 255, 0)      # Vert - proche
                distance = "PROCHE"
            elif area > 1000:
                color = (0, 255, 255)    # Jaune - moyen
                distance = "MOYEN"
            else:
                color = (0, 150, 255)    # Orange - loin
                distance = "LOIN"
            
            # Contour
            cv2.drawContours(frame, [contour], -1, color, 3)
            
            # Rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
                
                # Mettre à jour centre zoom
                self.zoom_center = (cx, cy)
            
            # Texte
            cv2.putText(frame, f"GANT {distance}", (x, max(y - 15, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Aire: {int(area)}", (x, max(y - 40, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")

    def _add_overlay(self, frame, detected, area, mask):
        """Overlay simple avec info zoom"""
        try:
            h, w = frame.shape[:2]
            
            # Status
            status = f"🎯 GANT DETECTE" if detected else "🔍 RECHERCHE GANT"
            if self.zoom_factor > 1.1:
                status += f" (ZOOM {self.zoom_factor:.1f}x)"
            color = (0, 255, 0) if detected else (0, 255, 255)
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Barre de zoom
            if self.zoom_factor > 1.05:
                zoom_bar_width = 200
                zoom_bar_height = 10
                zoom_x, zoom_y = 10, 70
                
                # Fond
                cv2.rectangle(frame, (zoom_x, zoom_y), 
                             (zoom_x + zoom_bar_width, zoom_y + zoom_bar_height), 
                             (50, 50, 50), -1)
                
                # Barre zoom
                zoom_width = int(zoom_bar_width * (self.zoom_factor - 1.0) / 3.0)
                cv2.rectangle(frame, (zoom_x, zoom_y), 
                             (zoom_x + zoom_width, zoom_y + zoom_bar_height), 
                             (0, 255, 255), -1)
                
                cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x", 
                           (zoom_x + zoom_bar_width + 10, zoom_y + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Stats
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Det: {detection_rate:.1f}%"
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Zone de zoom visuelle
            if self.zoom_factor > 1.1 and hasattr(self, '_zoom_info'):
                info = self._zoom_info
                cv2.rectangle(frame, (info['offset_x'], info['offset_y']), 
                             (info['offset_x'] + info['crop_w'], info['offset_y'] + info['crop_h']), 
                             (0, 255, 255), 2)
                cv2.putText(frame, f"ZOOM {self.zoom_factor:.1f}x", 
                           (info['offset_x'], max(info['offset_y'] - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # FPS
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Historique
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-10:]])
            cv2.putText(frame, f"Hist: {history}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Masque debug (petit)
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (100, 75))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                    
                    mask_x, mask_y = w - 110, 70
                    frame[mask_y:mask_y+75, mask_x:mask_x+100] = mask_colored
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+100, mask_y+75), (255, 255, 255), 1)
                except Exception:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

# === CONTRÔLE DRONE ===
def drone_control(bebop):
    logger.info("Contrôle drone démarré.")
    print("\n[Commandes drone]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f/b/g/d = mouvements | h/m = haut/bas | a/c = rotations\n")
    
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
    """Fonction principale - Base qui fonctionne + zoom simple"""
    logger.info("=== BEBOP 2 WORKING BASE + SIMPLE ZOOM ===")
    logger.info("🎯 Base fonctionnelle avec zoom adaptatif simple")
    
    bebop = None
    pipe = None
    detector = None
    start_time = time.time()
    
    try:
        # === CONNEXION ===
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
        ctrl_thread = threading.Thread(target=drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg simple et rapide
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '500000',
            '-probesize', '500000',
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg simple: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=512*1024)
            logger.info("✅ Pipeline simple initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR ===
        detector = WorkingGloveDetectorWithZoom()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - Base Fonctionnelle + Zoom"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 50)
        logger.info("🎮 COMMANDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset")
        logger.info("  'z' = Toggle auto-zoom | '+/-' = Zoom manuel")
        logger.info("  'd' = Debug masque")
        logger.info("=" * 50)
        
        # === BOUCLE PRINCIPALE ===
        logger.info("🎬 Démarrage détection base + zoom...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Détection
                processed_frame, detected = detector.detect_glove(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Stats FPS
                fps_counter += 1
                if fps_counter % 60 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"📊 FPS: {display_fps:.1f} | "
                               f"Détections: {detector.detection_count}/{detector.frame_count} "
                               f"({(detector.detection_count/max(detector.frame_count,1))*100:.1f}%) | "
                               f"Zoom: {detector.zoom_factor:.1f}x")
                    last_fps_log = current_time
                
                # Gestion touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    screenshot_name = f"working_capture_{int(time.time())}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    # Reset
                    detector.__init__()
                    logger.info("🔄 Reset détecteur")
                    
                elif key == ord('z'):
                    # Toggle auto-zoom
                    detector.manual_zoom = not detector.manual_zoom
                    mode = "MANUEL" if detector.manual_zoom else "AUTO"
                    logger.info(f"🔍 Zoom: {mode}")
                    
                elif key == ord('+') or key == ord('='):
                    # Zoom manuel +
                    detector.manual_zoom = True
                    detector.zoom_factor = min(4.0, detector.zoom_factor + 0.5)
                    detector.target_zoom = detector.zoom_factor
                    logger.info(f"🔍 Zoom manuel: {detector.zoom_factor:.1f}x")
                    
                elif key == ord('-'):
                    # Zoom manuel -
                    detector.manual_zoom = True
                    detector.zoom_factor = max(1.0, detector.zoom_factor - 0.5)
                    detector.target_zoom = detector.zoom_factor
                    logger.info(f"🔍 Zoom manuel: {detector.zoom_factor:.1f}x")
                    
                elif key == ord('d'):
                    # Debug masque
                    debug_name = f"debug_mask_{int(time.time())}.png"
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    debug_mask = detector._create_working_mask(hsv)
                    cv2.imwrite(debug_name, debug_mask)
                    logger.info(f"🔍 Masque debug: {debug_name}")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle: {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
            logger.info(f"  🔍 Zoom final: {detector.zoom_factor:.1f}x")
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