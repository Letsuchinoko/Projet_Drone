import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
import logging
import signal
import sys
from collections import deque
import subprocess
import os
import pyparrot

# Configuration
DISPLAY_FPS = 25
CONNECTION_TIMEOUT = 20
WIDTH, HEIGHT = 856, 480  # Résolution Bebop 2

# Variables globales
processing_active = threading.Event()
processing_active.set()

frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time()
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_pipe_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class DirectPipeVision:
    """Vision directe par pipe FFmpeg - zero disk usage"""
    
    def __init__(self, drone_object):
        self.drone_object = drone_object
        self.ffmpeg_process = None
        self.pipe_active = False
        
    def open_video_stream(self):
        """Ouverture du flux vidéo par pipe direct"""
        try:
            # Démarrer le stream sur le drone
            logger.info("Starting Bebop video stream...")
            self.drone_object.start_video_stream()
            time.sleep(3)  # Attendre stabilisation
            
            # Chemin vers le fichier SDP
            sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
            if not os.path.exists(sdp_path):
                logger.error(f"SDP file not found: {sdp_path}")
                return False
            
            logger.info(f"Using SDP file: {sdp_path}")
            
            # Commande FFmpeg pour pipe direct
            ffmpeg_cmd = [
                'ffmpeg',
                '-protocol_whitelist', 'file,rtp,udp',
                '-i', sdp_path,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-'  # Sortie vers stdout
            ]
            
            logger.info(f"Starting FFmpeg pipe: {' '.join(ffmpeg_cmd)}")
            
            # Démarrer le processus FFmpeg
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            # Test de lecture d'une frame
            logger.info("Testing video stream...")
            test_frame = self.ffmpeg_process.stdout.read(WIDTH * HEIGHT * 3)
            
            if len(test_frame) == WIDTH * HEIGHT * 3:
                logger.info("Video stream test: SUCCESS")
                self.pipe_active = True
                return True
            else:
                logger.error(f"Video stream test failed - received {len(test_frame)} bytes, expected {WIDTH * HEIGHT * 3}")
                return False
                
        except Exception as e:
            logger.error(f"Error opening video stream: {e}")
            return False
    
    def read_frame(self):
        """Lecture d'une frame depuis le pipe"""
        if not self.pipe_active or not self.ffmpeg_process:
            return None
        
        try:
            raw_frame = self.ffmpeg_process.stdout.read(WIDTH * HEIGHT * 3)
            
            if len(raw_frame) != WIDTH * HEIGHT * 3:
                logger.warning(f"Incomplete frame: {len(raw_frame)} bytes")
                return None
            
            # Convertir les bytes en image OpenCV
            frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
            return frame
            
        except Exception as e:
            logger.debug(f"Frame reading error: {e}")
            return None
    
    def close_video_stream(self):
        """Fermeture propre du flux vidéo"""
        logger.info("Closing video stream...")
        
        self.pipe_active = False
        
        # Terminer FFmpeg
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                time.sleep(2)
                
                if self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.kill()
                    time.sleep(1)
                
                logger.info("FFmpeg process terminated")
            except Exception as e:
                logger.warning(f"Error terminating FFmpeg: {e}")
        
        # Arrêter le stream drone
        try:
            self.drone_object.stop_video_stream()
            logger.info("Bebop video stream stopped")
        except Exception as e:
            logger.warning(f"Error stopping drone stream: {e}")

class OptimizedGloveDetector:
    """Détecteur de gants optimisé - identique au code qui marche"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=25)
        self.min_area = 400
        self.max_area = 70000
        self.min_contour_points = 10
        
        # Kernels morphologiques
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 7), np.uint8)
        
        # Stabilisation
        self.stable_detections = deque(maxlen=5)
        
        # Cache simple
        self.last_frame_hash = None
        self.last_result = None
        
    def detect_glove(self, frame):
        """Détection de gant - même logique que le code qui marche"""
        if frame is None:
            return frame, False
        
        try:
            # Cache simple
            frame_hash = np.sum(frame) % 1000000  # Éviter les gros nombres
            if frame_hash == self.last_frame_hash and self.last_result is not None:
                return self.last_result
            
            original_frame = frame.copy()
            h, w = original_frame.shape[:2]
            
            # Redimensionnement pour performance
            scale_factor = 1.0
            if w > 640:
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur peau (à exclure)
            skin_lower = np.array([0, 30, 80])
            skin_upper = np.array([25, 130, 255])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Masque orange du gant
            orange_lower = np.array([10, 120, 120])
            orange_upper = np.array([23, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Masque rouge du gant (2 plages car HSV circulaire)
            red_lower1 = np.array([0, 140, 120])
            red_upper1 = np.array([8, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([170, 140, 120])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Fusionne orange + rouge, puis enlève la peau
            mask_gant = cv2.bitwise_or(mask_orange, mask_red)
            mask = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin))
            
            # Nettoyage morphologique
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection de contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours, work_frame.shape)
            
            # Validation
            detected = best_contour is not None
            
            # Stabilisation
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= 2  # Au moins 2 sur 5
            self.detection_history.append(stable_detection)
            
            # Dessin
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_detection(original_frame, best_contour)
                
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Overlay
            result_frame = self._add_overlay(original_frame, stable_detection, mask)
            
            # Cache
            self.last_frame_hash = frame_hash
            self.last_result = (result_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _select_best_contour(self, contours, frame_shape):
        """Sélection du meilleur contour"""
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
            
            # Éviter les bords
            if x < 5 or y < 5 or (x + w_rect) > (w - 5) or (y + h_rect) > (h - 5):
                continue
            
            # Solidité
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.35:
                    continue
            
            # Score
            position_score = 1.0 if y > h * 0.1 else 0.5
            area_score = min(area / 4000.0, 1.0)
            score = area_score * position_score
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour
    
    def _draw_detection(self, frame, contour):
        """Dessiner la détection"""
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
        """Overlay d'informations"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "GANT DETECTE" if detected else "RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Statistiques
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
            
            detection_rate = (detections / max(frames, 1)) * 100
            stats_text = f"Frames: {frames} | Detections: {detections} ({detection_rate:.1f}%)"
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Historique
            history_text = "Hist: " + "".join(["●" if x else "○" for x in list(self.detection_history)[-25:]])
            cv2.putText(frame, history_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Masque miniature
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (160, 120))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                    frame[10:130, w-170:w-10] = mask_colored
                    cv2.rectangle(frame, (w-170, 10), (w-10, 130), (255, 255, 255), 1)
                    cv2.putText(frame, "Masque", (w-160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except Exception:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def monitor_thread():
    """Thread de monitoring"""
    logger.info("Monitor thread started")
    last_frame_count = 0
    
    while processing_active.is_set():
        time.sleep(10)
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_stats = frame_stats.copy()
        
        frame_diff = current_stats['frame_count'] - last_frame_count
        last_frame_count = current_stats['frame_count']
        
        if frame_diff > 0:
            fps = frame_diff / 10.0
            detection_rate = (current_stats['detection_count'] / max(current_stats['frame_count'], 1)) * 100
            
            logger.info(f"MONITOR - Frames: {current_stats['frame_count']} (+{frame_diff}), "
                       f"FPS: {fps:.1f}, Det: {detection_rate:.1f}%")
        else:
            logger.warning(f"No new frames in last 10s - Total: {current_stats['frame_count']}")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - shutting down")
    processing_active.clear()

def main():
    """Fonction principale avec pipe direct"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Direct Pipe Detection System")
    logger.info("Features: Direct FFmpeg pipe + Zero disk usage + Optimized detection")
    
    bebop = None
    vision = None
    monitor_thread_obj = None
    start_time = time.time()
    
    try:
        # Vérification FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("FFmpeg found and working")
            else:
                logger.error("FFmpeg not working properly")
                return False
        except Exception as e:
            logger.error(f"FFmpeg not found: {e}")
            return False
        
        # Connexion au drone
        logger.info("Connecting to Bebop 2...")
        bebop = Bebop()
        
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            return False
        
        logger.info("Drone connected successfully")
        
        # Initialisation de la vision par pipe
        vision = DirectPipeVision(bebop)
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream via direct pipe...")
        if not vision.open_video_stream():
            logger.error("Failed to open video stream")
            return False
        
        open_time = time.time() - start_time
        logger.info(f"Video stream opened successfully in {open_time:.1f}s")
        
        # Démarrage du monitoring
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        monitor_thread_obj.start()
        logger.info("Monitor thread started")
        
        # Informations système
        logger.info("=" * 60)
        logger.info("SYSTEM STATUS:")
        logger.info(f"  Mode:              Direct Pipe (Zero Disk)")
        logger.info(f"  Resolution:        {WIDTH}x{HEIGHT}")
        logger.info(f"  Target FPS:        {DISPLAY_FPS}")
        logger.info("=" * 60)
        logger.info("CONTROLS:")
        logger.info("  'q' or ESC     = Quit")
        logger.info("  'r'            = Reset statistics")
        logger.info("  's'            = Screenshot")
        logger.info("=" * 60)
        
        # Boucle d'affichage principale
        detector = OptimizedGloveDetector()
        window_name = "Bebop 2 - Direct Pipe Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        fps_counter = 0
        fps_start = time.time()
        screenshot_count = 0
        
        logger.info("Starting main display loop...")
        
        while processing_active.is_set():
            try:
                # Lecture directe de la frame
                frame = vision.read_frame()
                
                if frame is None:
                    # Pas de frame disponible
                    time.sleep(0.01)
                    continue
                
                # Mise à jour des stats
                with stats_lock:
                    frame_stats['frame_count'] += 1
                    frame_stats['last_frame_time'] = time.time()
                
                # Traitement de détection
                processed_frame, detected = detector.detect_glove(frame)
                
                # Calcul FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - fps_start
                    if elapsed > 0:
                        current_fps = fps_counter / elapsed
                        logger.info(f"Display FPS: {current_fps:.1f}")
                    fps_start = time.time()
                    fps_counter = 0
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("User requested quit")
                    processing_active.clear()
                    break
                elif key == ord('r'):
                    # Reset statistiques
                    with stats_lock:
                        frame_stats['frame_count'] = 0
                        frame_stats['detection_count'] = 0
                        frame_stats['error_count'] = 0
                    detector.detection_history.clear()
                    detector.stable_detections.clear()
                    logger.info("Statistics reset")
                elif key == ord('s'):
                    # Screenshot
                    screenshot_name = f"screenshot_{int(time.time())}_{screenshot_count}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"Screenshot saved: {screenshot_name}")
                    screenshot_count += 1
                
                # Limitation FPS
                time.sleep(1.0 / DISPLAY_FPS)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.1)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Nettoyage final
        logger.info("Starting system cleanup...")
        processing_active.clear()
        
        # Arrêter le monitoring
        if monitor_thread_obj:
            try:
                monitor_thread_obj.join(timeout=5)
                logger.info("Monitor thread stopped")
            except Exception as e:
                logger.warning(f"Error stopping monitor: {e}")
        
        # Fermeture de la vision
        if vision:
            try:
                vision.close_video_stream()
                logger.info("Vision system closed")
            except Exception as e:
                logger.warning(f"Error closing vision: {e}")
        
        # Déconnexion du drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("Drone disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting drone: {e}")
        
        # Fermeture OpenCV
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.warning(f"Error closing OpenCV: {e}")
        
        # Statistiques finales
        total_runtime = time.time() - start_time
        
        with stats_lock:
            final_stats = frame_stats.copy()
        
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS:")
        logger.info(f"  Total Runtime:        {total_runtime:.1f}s")
        logger.info(f"  Frames Processed:     {final_stats['frame_count']}")
        logger.info(f"  Average Frame Rate:   {final_stats['frame_count']/max(total_runtime,1):.1f} fps")
        logger.info(f"  Total Detections:     {final_stats['detection_count']}")
        logger.info(f"  Detection Rate:       {(final_stats['detection_count']/max(final_stats['frame_count'],1))*100:.1f}%")
        logger.info(f"  Processing Errors:    {final_stats['error_count']}")
        logger.info("=" * 60)
        logger.info("System shutdown completed successfully")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        logger.info(f"Program exiting with code {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)