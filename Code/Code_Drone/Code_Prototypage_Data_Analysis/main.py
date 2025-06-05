import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from queue import Queue, Empty
import logging
import signal
import sys
from collections import deque
import subprocess
import socket
import struct

# Configuration
DISPLAY_FPS = 25
CONNECTION_TIMEOUT = 20
FRAME_TIMEOUT = 1.0

# Variables globales pour le flux direct
current_frame = None
frame_lock = threading.RLock()
processing_active = threading.Event()
processing_active.set()

frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'stream_restarts': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_direct_stream.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class DirectStreamVision:
    """Capture directe du flux vidéo sans fichiers temporaires"""
    
    def __init__(self, bebop):
        self.bebop = bebop
        self.video_process = None
        self.is_streaming = False
        self.stream_thread = None
        
    def start_stream(self):
        """Démarre la capture directe du flux vidéo"""
        try:
            if self.is_streaming:
                return True
            
            logger.info("Starting direct video stream capture...")
            
            # Commande FFmpeg pour capturer le flux RTP directement
            cmd = [
                'ffmpeg',
                '-i', 'rtp://192.168.42.1:55004',  # Adresse RTP du Bebop
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-an',  # Pas d'audio
                '-'  # Sortie vers stdout
            ]
            
            # Alternative avec pipe nommé local si disponible
            # cmd = [
            #     'ffmpeg',
            #     '-i', 'udp://192.168.42.1:55004',
            #     '-f', 'rawvideo',
            #     '-pix_fmt', 'bgr24',
            #     '-an',
            #     '-'
            # ]
            
            self.video_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            if self.video_process:
                self.is_streaming = True
                self.stream_thread = threading.Thread(target=self._stream_reader, daemon=True)
                self.stream_thread.start()
                logger.info("Direct video stream started successfully")
                return True
            else:
                logger.error("Failed to start video process")
                return False
                
        except Exception as e:
            logger.error(f"Error starting direct stream: {e}")
            return False
    
    def _stream_reader(self):
        """Thread de lecture du flux vidéo direct"""
        global current_frame
        
        logger.info("Stream reader thread started")
        frame_width = 856  # Largeur du Bebop 2
        frame_height = 480  # Hauteur du Bebop 2
        frame_size = frame_width * frame_height * 3  # 3 canaux BGR
        
        try:
            while self.is_streaming and processing_active.is_set():
                if not self.video_process or self.video_process.poll() is not None:
                    logger.warning("Video process terminated unexpectedly")
                    break
                
                try:
                    # Lire une frame complète
                    raw_frame = self.video_process.stdout.read(frame_size)
                    
                    if len(raw_frame) != frame_size:
                        logger.debug(f"Incomplete frame received: {len(raw_frame)}/{frame_size}")
                        continue
                    
                    # Convertir en array NumPy
                    frame_array = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = frame_array.reshape((frame_height, frame_width, 3))
                    
                    # Validation de base
                    if frame is None or frame.size == 0:
                        continue
                    
                    # Test de corruption simple
                    if np.mean(frame) < 5 or np.mean(frame) > 250:
                        continue
                    
                    # Mettre à jour la frame globale
                    with frame_lock:
                        current_frame = frame.copy()
                    
                    # Mettre à jour les statistiques
                    with stats_lock:
                        frame_stats['frame_count'] += 1
                        frame_stats['last_frame_time'] = time.time()
                    
                except Exception as e:
                    logger.debug(f"Frame reading error: {e}")
                    time.sleep(0.01)
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in stream reader: {e}")
        finally:
            logger.info("Stream reader thread terminated")
    
    def stop_stream(self):
        """Arrête la capture du flux vidéo"""
        try:
            self.is_streaming = False
            
            if self.video_process:
                try:
                    self.video_process.terminate()
                    self.video_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.video_process.kill()
                    self.video_process.wait()
                self.video_process = None
            
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=3)
            
            logger.info("Direct video stream stopped")
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")

class SimpleGloveDetector:
    """Détecteur de gants simplifié et optimisé"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.min_area = 500
        self.max_area = 30000
        
        # Kernels morphologiques
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Stabilisation
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3
        
    def detect_glove(self, frame):
        """Détection de gants optimisée"""
        if frame is None:
            return frame, False
            
        try:
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement pour les performances
            scale_factor = 1.0
            if w > 640:
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement simple
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur pour gants orange/rouge
            mask = self._create_color_mask(hsv)
            
            # Morphologie
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours, work_frame.shape)
            
            # Validation
            detected = best_contour is not None
            
            # Historique
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            self.detection_history.append(stable_detection)
            
            # Dessin
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_detection(original_frame, best_contour)
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            result_frame = self._add_overlay(original_frame, stable_detection)
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _create_color_mask(self, hsv):
        """Masque couleur pour gants orange/rouge"""
        try:
            # Orange
            orange_lower = np.array([10, 100, 100])
            orange_upper = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([170, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison
            return cv2.bitwise_or(mask_orange, mask_red)
            
        except Exception as e:
            logger.debug(f"Color mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_best_contour(self, contours, frame_shape):
        """Sélection du meilleur contour"""
        if not contours:
            return None
            
        try:
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_area <= area <= self.max_area:
                    if area > best_area:
                        best_area = area
                        best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Contour selection error: {e}")
            return None
    
    def _draw_detection(self, frame, contour):
        """Dessin de la détection"""
        try:
            # Contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Centre
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            
            # Texte
            cv2.putText(frame, "GANT DETECTE", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_overlay(self, frame, detected):
        """Overlay d'informations"""
        try:
            h, w = frame.shape[:2]
            
            # Status
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Statistiques
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
                restarts = frame_stats['stream_restarts']
                detection_rate = (detections / max(frames, 1)) * 100
            
            stats_text = f"Frames: {frames} | Det: {detection_rate:.1f}% | Err: {errors} | Restart: {restarts}"
            cv2.putText(frame, stats_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Historique
            history_symbols = ["●" if d else "○" for d in list(self.detection_history)[-15:]]
            history_text = "Hist: " + "".join(history_symbols)
            cv2.putText(frame, history_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Confiance
            confidence = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0
            confidence_text = f"Confiance: {confidence:.1%}"
            conf_color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
            cv2.putText(frame, confidence_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mode de capture
            cv2.putText(frame, "MODE: DIRECT STREAM", (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def display_thread():
    """Thread d'affichage avec capture directe"""
    detector = SimpleGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Direct Stream Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while processing_active.is_set():
        try:
            # Récupérer la frame actuelle
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = None
            
            if frame is None:
                # Afficher un écran d'attente
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Attente du flux direct...", (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, blank_frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            # Traitement de la détection
            processed_frame, detected = detector.detect_glove(frame)
            
            # Calcul FPS
            fps_counter += 1
            current_time = time.time()
            
            if fps_counter % 30 == 0:
                fps_elapsed = current_time - fps_start_time
                current_fps = fps_counter / fps_elapsed if fps_elapsed > 0 else 0
                logger.info(f"Display FPS: {current_fps:.1f}")
                fps_start_time = current_time
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
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                    frame_stats['stream_restarts'] = 0
                detector.detection_history.clear()
                detector.stable_detections.clear()
                logger.info("Statistics reset")
            elif key == ord('s'):
                screenshot_name = f"screenshot_direct_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def monitor_thread():
    """Thread de monitoring simple"""
    logger.info("Monitor thread started")
    last_frame_count = 0
    
    while processing_active.is_set():
        time.sleep(5)
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            detections = frame_stats['detection_count']
            errors = frame_stats['error_count']
            last_received_time = frame_stats['last_frame_time']
        
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        time_since_last_frame = time.time() - last_received_time
        
        if frame_diff > 0:
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            
            logger.info(f"MONITOR - Frames: {current_frames}, FPS: {avg_fps:.1f}, "
                       f"Detections: {detection_rate:.1f}%, Errors: {errors}")
        else:
            logger.warning(f"No new frames - last frame {time_since_last_frame:.1f}s ago")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale avec flux direct"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Direct Stream Detection System")
    
    bebop = None
    direct_vision = None
    threads = []
    
    try:
        # Connexion au drone
        bebop = Bebop()
        logger.info("Connecting to Bebop 2...")
        
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            return False
        
        logger.info("Drone connected successfully")
        
        # Initialiser la vision directe
        direct_vision = DirectStreamVision(bebop)
        
        # Démarrer les threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        
        threads = [display_thread_obj, monitor_thread_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.1)
        
        logger.info("All threads started successfully")
        
        # Démarrer le flux direct
        if direct_vision.start_stream():
            logger.info("Direct stream detection system is now active")
            logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot")
        else:
            logger.error("Failed to start direct stream")
            return False
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Direct Stream Detection", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Display window was closed")
                        break
                except:
                    pass
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
    
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        return False
    
    finally:
        # Nettoyage
        logger.info("Starting cleanup...")
        processing_active.clear()
        
        # Arrêter le flux direct
        if direct_vision:
            direct_vision.stop_stream()
        
        # Attendre les threads
        for thread in threads:
            try:
                thread.join(timeout=5)
                logger.debug(f"Thread {thread.name} terminated")
            except Exception as e:
                logger.debug(f"Error joining thread: {e}")
        
        # Déconnexion du drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("Drone disconnected")
            except Exception as e:
                logger.debug(f"Error disconnecting drone: {e}")
        
        # Fermeture OpenCV
        cv2.destroyAllWindows()
        
        logger.info("Cleanup completed successfully")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        logger.info(f"Program exiting with code {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        sys.exit(1)