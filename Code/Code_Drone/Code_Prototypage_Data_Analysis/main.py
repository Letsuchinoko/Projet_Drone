import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from queue import Queue, Empty
import logging
import os
import signal
import sys
from collections import deque

# Configuration optimisée
DISPLAY_FPS = 25
MAX_QUEUE_SIZE = 3
CONNECTION_TIMEOUT = 20
FRAME_TIMEOUT = 1.0

# Variables globales pour la frame unique
current_frame = None
frame_lock = threading.RLock()
processing_active = threading.Event()
processing_active.set()

frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'callback_calls': 0,
    'frame_updates': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_callback_intercept.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def intercepted_vision_callback(args):
    """
    Callback intercepté qui récupère la frame directement depuis PyParrot
    avant qu'elle soit écrite sur disque
    """
    global current_frame
    
    try:
        with stats_lock:
            frame_stats['callback_calls'] += 1
        
        # Le callback de PyParrot reçoit les arguments dans args
        # On doit examiner la structure pour récupérer la frame
        
        # Méthode 1: Si args contient directement la frame
        if hasattr(args, 'frame') and args.frame is not None:
            frame = args.frame
        # Méthode 2: Si args est un dictionnaire
        elif isinstance(args, dict) and 'frame' in args:
            frame = args['frame']
        # Méthode 3: Si args contient les données d'image
        elif hasattr(args, 'data') and args.data is not None:
            # Convertir les données brutes en frame OpenCV
            try:
                # Supposons que data contient les données d'image
                frame_data = args.data
                if isinstance(frame_data, bytes):
                    # Convertir bytes en array numpy puis en image
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    frame = None
            except Exception as e:
                logger.debug(f"Frame conversion error: {e}")
                frame = None
        else:
            # Fallback: lire depuis le fichier le plus récent (comme avant)
            frame = read_latest_image_file()
        
        # Validation de la frame
        if frame is not None and frame.size > 0:
            try:
                # Tests de validité
                h, w = frame.shape[:2]
                if h >= 240 and w >= 320:
                    # Test de corruption simple
                    mean_val = np.mean(frame)
                    if 10 <= mean_val <= 245:
                        # Frame valide - mise à jour
                        with frame_lock:
                            current_frame = frame.copy()
                        
                        with stats_lock:
                            frame_stats['frame_count'] += 1
                            frame_stats['frame_updates'] += 1
                            frame_stats['last_frame_time'] = time.time()
                        
                        logger.debug(f"Frame updated: {w}x{h}")
                    else:
                        logger.debug(f"Frame rejected (mean: {mean_val:.1f})")
                else:
                    logger.debug(f"Frame rejected (size: {w}x{h})")
            except Exception as e:
                logger.debug(f"Frame validation error: {e}")
                
    except Exception as e:
        logger.debug(f"Callback error: {e}")

def read_latest_image_file():
    """
    Fallback: lecture du fichier image le plus récent
    (utilisé si l'interception directe échoue)
    """
    try:
        import glob
        images_dir = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
        
        if not os.path.exists(images_dir):
            return None
        
        pattern = os.path.join(images_dir, "image_*.png")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Prendre le fichier le plus récent
        latest_file = max(files, key=os.path.getmtime)
        
        # Vérifications de sécurité
        current_time = time.time()
        file_mtime = os.path.getmtime(latest_file)
        file_size = os.path.getsize(latest_file)
        
        # Éviter les fichiers trop récents (en cours d'écriture) ou trop petits
        if (current_time - file_mtime) < 0.05 or file_size < 5000:
            return None
        
        # Lecture avec retry
        for attempt in range(2):
            try:
                frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                time.sleep(0.01)
            except:
                if attempt == 0:
                    time.sleep(0.02)
                continue
        
        return None
        
    except Exception as e:
        logger.debug(f"File fallback error: {e}")
        return None

class OptimizedGloveDetector:
    """Détecteur de gants optimisé pour une seule frame"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.min_area = 600
        self.max_area = 35000
        
        # Kernels morphologiques
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Stabilisation
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3
        
        # Cache pour éviter les recalculs
        self.last_frame_id = None
        self.last_result = None
        
    def detect_glove(self, frame):
        """Détection optimisée avec cache"""
        if frame is None:
            return frame, False
            
        try:
            # ID unique pour cette frame (basé sur le contenu)
            frame_id = hash(frame.tobytes())
            
            # Si c'est la même frame que la dernière fois, retourner le cache
            if frame_id == self.last_frame_id and self.last_result is not None:
                return self.last_result
            
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement pour performances
            scale_factor = 1.0
            if w > 640:
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur
            mask = self._create_color_mask(hsv)
            
            # Morphologie
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours)
            
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
            
            # Overlay
            result_frame = self._add_overlay(original_frame, stable_detection, mask)
            
            # Cache du résultat
            self.last_frame_id = frame_id
            self.last_result = (result_frame, stable_detection)
            
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
            orange_lower = np.array([10, 120, 120])
            orange_upper = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge
            red_lower1 = np.array([0, 120, 120])
            red_upper1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([170, 120, 120])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            return cv2.bitwise_or(mask_orange, mask_red)
            
        except Exception as e:
            logger.debug(f"Color mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_best_contour(self, contours):
        """Sélection du meilleur contour"""
        if not contours:
            return None
            
        try:
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_area <= area <= self.max_area:
                    # Score simple basé sur l'aire
                    score = min(area / 3000.0, 1.0)
                    
                    if score > best_score:
                        best_score = score
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
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT (A:{int(area)})", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_overlay(self, frame, detected, mask=None):
        """Overlay d'informations"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Statistiques
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
                callbacks = frame_stats['callback_calls']
                updates = frame_stats['frame_updates']
                detection_rate = (detections / max(frames, 1)) * 100
                update_rate = (updates / max(callbacks, 1)) * 100
            
            stats_text = f"Frames: {frames} | Det: {detection_rate:.1f}% | Err: {errors}"
            cv2.putText(frame, stats_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            callback_text = f"Callbacks: {callbacks} | Updates: {updates} ({update_rate:.1f}%)"
            cv2.putText(frame, callback_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
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
            
            # Mode
            cv2.putText(frame, "MODE: CALLBACK INTERCEPT", (w - 250, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Masque miniature
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (120, 90))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    mask_x, mask_y = w - 130, 45
                    frame[mask_y:mask_y+90, mask_x:mask_x+120] = mask_colored
                    
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+120, mask_y+90), (255, 255, 255), 1)
                    cv2.putText(frame, "Masque", (mask_x, mask_y + 105), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                except:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def display_thread():
    """Thread d'affichage utilisant la frame unique"""
    detector = OptimizedGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Callback Intercept Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    last_display_time = time.time()
    
    while processing_active.is_set():
        try:
            current_time = time.time()
            
            # Limiter le FPS d'affichage
            if (current_time - last_display_time) < (1.0 / DISPLAY_FPS):
                time.sleep(0.005)
                continue
            
            last_display_time = current_time
            
            # Récupérer la frame actuelle
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = None
            
            if frame is None:
                # Écran d'attente
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Attente des frames...", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                with stats_lock:
                    callbacks = frame_stats['callback_calls']
                    updates = frame_stats['frame_updates']
                
                cv2.putText(blank_frame, f"Callbacks: {callbacks} | Updates: {updates}", 
                           (180, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
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
                    frame_stats['callback_calls'] = 0
                    frame_stats['frame_updates'] = 0
                detector.detection_history.clear()
                detector.stable_detections.clear()
                detector.last_frame_id = None
                detector.last_result = None
                logger.info("Statistics and cache reset")
            elif key == ord('s'):
                screenshot_name = f"screenshot_intercept_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            elif key == ord('c'):
                with stats_lock:
                    stats = frame_stats.copy()
                logger.info(f"Current stats: {stats}")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def monitor_thread():
    """Thread de monitoring simple"""
    logger.info("Monitor thread started")
    last_frame_count = 0
    last_callback_count = 0
    
    while processing_active.is_set():
        time.sleep(5)
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            detections = frame_stats['detection_count']
            errors = frame_stats['error_count']
            callbacks = frame_stats['callback_calls']
            updates = frame_stats['frame_updates']
            last_received_time = frame_stats['last_frame_time']
        
        frame_diff = current_frames - last_frame_count
        callback_diff = callbacks - last_callback_count
        last_frame_count = current_frames
        last_callback_count = callbacks
        
        time_since_last_frame = time.time() - last_received_time
        
        if frame_diff > 0:
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            update_rate = (updates / max(callbacks, 1)) * 100
            callback_fps = callback_diff / 5
            
            logger.info(f"MONITOR - Frames: {current_frames} (+{frame_diff}), FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}, "
                       f"Callbacks: {callbacks} (+{callback_diff}, {callback_fps:.1f}/s), "
                       f"Update rate: {update_rate:.1f}%")
        else:
            logger.warning(f"No new frames - callbacks: {callbacks} (+{callback_diff}), "
                         f"last frame {time_since_last_frame:.1f}s ago")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale avec interception de callback"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Callback Intercept Detection System")
    
    bebop = None
    vision = None
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
        
        # Configuration de la vision avec callback intercepté
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(intercepted_vision_callback)
        
        # Démarrer les threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        
        threads = [display_thread_obj, monitor_thread_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.1)
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream with intercepted callback...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        logger.info("Video stream opened successfully")
        logger.info("Callback intercept detection system is now active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot, 'c'=Show stats")
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Callback Intercept Detection", cv2.WND_PROP_VISIBLE) < 1:
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
        
        # Attendre les threads
        for thread in threads:
            try:
                thread.join(timeout=5)
                logger.debug(f"Thread {thread.name} terminated")
            except Exception as e:
                logger.debug(f"Error joining thread: {e}")
        
        # Fermeture du flux vidéo
        if vision:
            try:
                vision.close_video()
                logger.info("Video stream closed")
            except Exception as e:
                logger.debug(f"Error closing video: {e}")
        
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