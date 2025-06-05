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
import glob

# Configuration optimisée
DISPLAY_FPS = 25
MAX_QUEUE_SIZE = 3
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
CONNECTION_TIMEOUT = 20
FRAME_TIMEOUT = 1.0
CACHE_SIZE = 10  # Nombre maximum d'images en cache RAM

# Cache RAM pour les images
class ImageCache:
    def __init__(self, max_size=CACHE_SIZE):
        self.max_size = max_size
        self.cache = {}  # {filename: {'frame': np.array, 'timestamp': float, 'mtime': float}}
        self.lock = threading.RLock()
        self.access_order = deque(maxlen=max_size)
        
    def add_image(self, filename, frame, file_mtime):
        """Ajoute une image au cache"""
        with self.lock:
            # Si le cache est plein, supprimer la plus ancienne
            if len(self.cache) >= self.max_size and filename not in self.cache:
                if self.access_order:
                    oldest = self.access_order.popleft()
                    if oldest in self.cache:
                        del self.cache[oldest]
            
            # Ajouter/mettre à jour l'image
            self.cache[filename] = {
                'frame': frame.copy(),
                'timestamp': time.time(),
                'mtime': file_mtime
            }
            
            # Mettre à jour l'ordre d'accès
            if filename in self.access_order:
                self.access_order.remove(filename)
            self.access_order.append(filename)
    
    def get_latest_frame(self):
        """Récupère la frame la plus récente du cache"""
        with self.lock:
            if not self.cache:
                return None
                
            # Trouver l'image avec le mtime le plus récent
            latest_file = max(self.cache.items(), key=lambda x: x[1]['mtime'])
            return latest_file[1]['frame'].copy()
    
    def has_image(self, filename, file_mtime):
        """Vérifie si l'image est déjà en cache et à jour"""
        with self.lock:
            if filename not in self.cache:
                return False
            return abs(self.cache[filename]['mtime'] - file_mtime) < 0.001
    
    def clear(self):
        """Vide le cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self):
        """Retourne les statistiques du cache"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'files': list(self.cache.keys())
            }

# Variables globales
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
image_cache = ImageCache(CACHE_SIZE)
processing_active = threading.Event()
processing_active.set()
frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'last_processed_file': None,
    'cache_hits': 0,
    'cache_misses': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_ram_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedGloveDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=20)
        self.min_area = 800
        self.max_area = 50000
        self.min_contour_points = 15
        
        # Kernels morphologiques optimisés
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Stabilisation des détections
        self.stable_detections = deque(maxlen=7)
        self.confidence_threshold = 4
        
        # Filtrage temporel
        self.last_detection_center = None
        self.max_movement_threshold = 100
        
    def detect_glove(self, frame):
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        try:
            # Redimensionnement adaptatif
            scale_factor = 1.0
            if w > 800:
                scale_factor = 800.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement amélioré
            work_frame = cv2.bilateralFilter(work_frame, 9, 80, 80)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masques couleur améliorés
            mask = self._create_advanced_color_mask(hsv)
            
            # Post-traitement morphologique amélioré
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.erode(mask, self.kernel_erode, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection et sélection du meilleur contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour_advanced(contours, work_frame.shape)
            
            # Validation temporelle
            detected = self._validate_detection_temporal(best_contour, scale_factor)
            
            # Mise à jour de l'historique
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            self.detection_history.append(stable_detection)
            
            # Dessin des résultats
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_advanced_detection(original_frame, best_contour)
                with stats_lock:
                    frame_stats['detection_count'] += 1
                    
            result_frame = self._add_advanced_overlay(original_frame, stable_detection, mask)
            return result_frame, stable_detection
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _create_advanced_color_mask(self, hsv):
        """Création d'un masque couleur avancé avec exclusion de la peau"""
        h, w = hsv.shape[:2]
        
        # Masque peau amélioré
        skin_masks = []
        skin_lower1 = np.array([0, 20, 70])
        skin_upper1 = np.array([25, 120, 255])
        skin_masks.append(cv2.inRange(hsv, skin_lower1, skin_upper1))
        
        skin_lower2 = np.array([0, 25, 50])
        skin_upper2 = np.array([15, 100, 200])
        skin_masks.append(cv2.inRange(hsv, skin_lower2, skin_upper2))
        
        mask_skin = cv2.bitwise_or(skin_masks[0], skin_masks[1])
        
        # Masque orange optimisé
        orange_lower = np.array([8, 140, 140])
        orange_upper = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Masque rouge optimisé
        red_lower1 = np.array([0, 150, 140])
        red_upper1 = np.array([6, 255, 255])
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        
        red_lower2 = np.array([174, 150, 140])
        red_upper2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combinaison et exclusion de la peau
        mask_gant = cv2.bitwise_or(mask_orange, mask_red)
        mask_skin_dilated = cv2.dilate(mask_skin, self.kernel_close, iterations=1)
        mask_final = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin_dilated))
        
        return mask_final
    
    def _select_best_contour_advanced(self, contours, frame_shape):
        """Sélection avancée du meilleur contour"""
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
            if not (0.3 <= aspect_ratio <= 3.0):
                continue
                
            margin = 10
            if (x < margin or y < margin or 
                (x + w_rect) > (w - margin) or 
                (y + h_rect) > (h - margin)):
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.45:
                    continue
            else:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.15:
                    continue
            
            # Scoring
            area_score = min(area / 5000.0, 1.0)
            position_score = 1.0 if y > h * 0.15 else 0.6
            solidity_score = min(solidity * 2, 1.0)
            aspect_bonus = 1.0 if 0.7 <= aspect_ratio <= 1.4 else 0.8
            
            score = area_score * position_score * solidity_score * aspect_bonus
            
            if score > best_score:
                best_score = score
                best_contour = contour
                
        return best_contour
    
    def _validate_detection_temporal(self, contour, scale_factor):
        """Validation temporelle"""
        if contour is None:
            self.last_detection_center = None
            return False
            
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        current_center = (cx, cy)
        
        if self.last_detection_center is not None:
            distance = np.sqrt((cx - self.last_detection_center[0])**2 + 
                             (cy - self.last_detection_center[1])**2)
            
            if distance > self.max_movement_threshold:
                return False
        
        self.last_detection_center = current_center
        return True
    
    def _draw_advanced_detection(self, frame, contour):
        """Dessin avancé de la détection"""
        try:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            info_y = max(y - 15, 30)
            cv2.putText(frame, f"GANT DETECTE", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Aire: {int(area)} | Sol: {solidity:.2f}", 
                       (x, info_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_advanced_overlay(self, frame, detected, mask=None):
        """Overlay avancé avec informations du cache"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Statistiques détaillées avec cache
            with stats_lock:
                detection_rate = (frame_stats['detection_count'] / max(frame_stats['frame_count'], 1)) * 100
                cache_stats = image_cache.get_stats()
                cache_hit_rate = (frame_stats['cache_hits'] / max(frame_stats['cache_hits'] + frame_stats['cache_misses'], 1)) * 100
                
                stats_text = (f"Frames: {frame_stats['frame_count']} | "
                            f"Detections: {frame_stats['detection_count']} ({detection_rate:.1f}%) | "
                            f"Erreurs: {frame_stats['error_count']}")
                
                cache_text = (f"Cache: {cache_stats['size']}/{cache_stats['max_size']} | "
                            f"Hit Rate: {cache_hit_rate:.1f}%")
            
            cv2.putText(frame, stats_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, cache_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Historique visuel
            history_symbols = []
            for detection in list(self.detection_history)[-20:]:
                history_symbols.append("●" if detection else "○")
            
            history_text = "Historique: " + "".join(history_symbols)
            cv2.putText(frame, history_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Niveau de confiance
            confidence = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0
            confidence_text = f"Confiance: {confidence:.1%}"
            conf_color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
            cv2.putText(frame, confidence_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def ram_vision_callback(args):
    """Callback optimisé avec cache RAM"""
    try:
        current_time = time.time()
        
        # Scanner les fichiers
        pattern = os.path.join(IMAGES_DIR, "image_*.png")
        files = glob.glob(pattern)
        
        if not files:
            return
        
        # Trier par temps de modification
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Traiter les fichiers les plus récents
        for latest_file in files[:3]:
            try:
                # Vérifications de base
                stat_info = os.stat(latest_file)
                file_size = stat_info.st_size
                file_mtime = stat_info.st_mtime
                filename = os.path.basename(latest_file)
                
                if file_size < 3000:
                    continue
                
                if (current_time - file_mtime) < 0.02:
                    continue
                
                # Vérifier le cache
                if image_cache.has_image(filename, file_mtime):
                    with stats_lock:
                        frame_stats['cache_hits'] += 1
                    continue
                
                # Lecture de l'image
                frame = None
                for attempt in range(2):
                    try:
                        frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                        if frame is not None:
                            break
                        time.sleep(0.005)
                    except:
                        if attempt == 0:
                            time.sleep(0.01)
                        continue
                
                if frame is None:
                    continue
                
                # Validations
                h, w = frame.shape[:2]
                if h < 240 or w < 320:
                    continue
                
                if np.all(frame == 0) or np.all(frame == 255):
                    continue
                
                gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.var(gray_test) < 100:
                    continue
                
                # Ajouter au cache
                image_cache.add_image(filename, frame, file_mtime)
                
                with stats_lock:
                    frame_stats['cache_misses'] += 1
                
                logger.debug(f"Image cached: {filename} ({file_size} bytes)")
                break
                
            except Exception as e:
                logger.debug(f"File processing error for {latest_file}: {e}")
                continue
        
        # Récupérer la frame la plus récente du cache pour traitement
        latest_frame = image_cache.get_latest_frame()
        if latest_frame is not None:
            # Gestion de la queue
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            
            try:
                frame_queue.put_nowait(latest_frame)
                with stats_lock:
                    frame_stats['frame_count'] += 1
                    frame_stats['last_frame_time'] = current_time
            except:
                logger.debug("Failed to add frame to queue")
                
    except Exception as e:
        logger.debug(f"RAM vision callback error: {e}")

def cleanup_thread():
    """Thread de nettoyage optimisé - ne touche plus aux images en cours d'utilisation"""
    logger.info("Cleanup thread started")
    cleanup_interval = 30  # Nettoyage moins fréquent
    
    while processing_active.is_set():
        try:
            # Nettoyage du répertoire d'images (fichiers très anciens seulement)
            files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            
            if len(files) > 100:  # Seuil plus élevé
                current_time = time.time()
                old_files = []
                
                for file_path in files:
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        # Supprimer seulement les fichiers de plus de 30 secondes
                        if (current_time - file_mtime) > 30:
                            old_files.append(file_path)
                    except:
                        continue
                
                # Supprimer les anciens fichiers
                removed_count = 0
                for file_path in old_files:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except:
                        continue
                
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} old image files")
            
            # Nettoyage du cache si nécessaire
            cache_stats = image_cache.get_stats()
            if cache_stats['size'] > cache_stats['max_size'] * 0.8:
                logger.debug(f"Cache usage: {cache_stats['size']}/{cache_stats['max_size']}")
                        
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        
        # Attente interruptible
        for _ in range(cleanup_interval):
            if not processing_active.is_set():
                break
            time.sleep(1)
    
    logger.info("Cleanup thread terminated")

def display_thread():
    """Thread d'affichage"""
    detector = AdvancedGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Detection Gant (RAM Cache)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    last_fps_log = time.time()
    
    while processing_active.is_set():
        try:
            try:
                frame = frame_queue.get(timeout=FRAME_TIMEOUT)
            except Empty:
                logger.debug("No frame received within timeout")
                continue
            
            if frame is None:
                continue
            
            # Traitement de la détection
            processed_frame, detected = detector.detect_glove(frame)
            
            # Calcul FPS
            fps_counter += 1
            current_time = time.time()
            
            if fps_counter % 30 == 0 or (current_time - last_fps_log) > 5:
                fps_elapsed = current_time - fps_start_time
                current_fps = fps_counter / fps_elapsed if fps_elapsed > 0 else 0
                
                if fps_counter % 30 == 0:
                    logger.info(f"Display FPS: {current_fps:.1f}")
                
                last_fps_log = current_time
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
                    frame_stats['cache_hits'] = 0
                    frame_stats['cache_misses'] = 0
                detector.detection_history.clear()
                detector.stable_detections.clear()
                image_cache.clear()
                logger.info("Statistics, detection history and cache reset")
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            elif key == ord('c'):
                cache_stats = image_cache.get_stats()
                logger.info(f"Cache stats: {cache_stats}")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def connection_monitor_thread():
    """Thread de monitoring avec statistiques de cache"""
    logger.info("Connection monitor started")
    last_frame_count = 0
    check_interval = 3
    consecutive_failures = 0
    
    while processing_active.is_set():
        time.sleep(check_interval)
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            errors = frame_stats['error_count']
            detections = frame_stats['detection_count']
            last_received_time = frame_stats['last_frame_time']
            cache_hits = frame_stats['cache_hits']
            cache_misses = frame_stats['cache_misses']
        
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        time_since_last_frame = time.time() - last_received_time
        
        if frame_diff == 0 or time_since_last_frame > 5:
            consecutive_failures += 1
            logger.warning(f"Stream issue detected (failure #{consecutive_failures}). "
                         f"Last frame: {time_since_last_frame:.1f}s ago")
            
            if consecutive_failures >= 3:
                logger.error("Multiple consecutive stream failures. Manual restart may be required.")
        else:
            if consecutive_failures > 0:
                logger.info("Stream recovered")
            consecutive_failures = 0
            
            avg_fps = frame_diff / check_interval
            detection_rate = (detections / max(current_frames, 1)) * 100
            cache_hit_rate = (cache_hits / max(cache_hits + cache_misses, 1)) * 100
            cache_stats = image_cache.get_stats()
            
            logger.info(f"MONITOR - Frames: {current_frames}, FPS: {avg_fps:.1f}, "
                       f"Detections: {detection_rate:.1f}%, Errors: {errors}, "
                       f"Cache: {cache_stats['size']}/{cache_stats['max_size']} ({cache_hit_rate:.1f}% hits)")
    
    logger.info("Connection monitor terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale avec cache RAM"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Advanced Bebop 2 glove detection system with RAM cache")
    
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        return False
    
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
        
        # Configuration de la vision avec callback RAM
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(ram_vision_callback)
        
        # Démarrage des threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        cleanup_thread_obj = threading.Thread(target=cleanup_thread, daemon=True)
        monitor_thread_obj = threading.Thread(target=connection_monitor_thread, daemon=True)
        
        threads = [display_thread_obj, cleanup_thread_obj, monitor_thread_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.1)
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        logger.info("Video stream opened successfully")
        logger.info("Advanced detection system with RAM cache is now active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot, 'c'=Cache stats")
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Detection Gant (RAM Cache)", cv2.WND_PROP_VISIBLE) < 1:
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
        # Nettoyage complet
        logger.info("Starting comprehensive cleanup...")
        processing_active.clear()
        
        # Attendre un peu pour que les threads se terminent proprement
        time.sleep(2)
        
        # Vider le cache
        try:
            image_cache.clear()
            logger.info("RAM cache cleared")
        except Exception as e:
            logger.debug(f"Error clearing cache: {e}")
        
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
        
        # Fermeture des fenêtres OpenCV
        cv2.destroyAllWindows()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            try:
                thread.join(timeout=5)
                logger.debug(f"Thread {thread.name} terminated")
            except Exception as e:
                logger.debug(f"Error joining thread: {e}")
        
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