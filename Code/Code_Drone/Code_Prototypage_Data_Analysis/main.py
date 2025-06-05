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
CACHE_SIZE = 8  # Réduit pour éviter la surcharge mémoire
WATCHDOG_TIMEOUT = 8  # Plus tolérant
MAX_RESTART_ATTEMPTS = 5  # Nombre max de redémarrages automatiques
RESTART_COOLDOWN = 10  # Délai entre redémarrages

# Variables globales
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processing_active = threading.Event()
processing_active.set()
restart_event = threading.Event()
vision_active = threading.Event()

frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'last_processed_file': None,
    'cache_hits': 0,
    'cache_misses': 0,
    'restart_count': 0,
    'last_restart_time': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_auto_restart.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class RobustImageCache:
    """Cache RAM robuste avec gestion d'erreurs avancée"""
    def __init__(self, max_size=CACHE_SIZE):
        self.max_size = max_size
        self.cache = {}
        self.lock = threading.RLock()
        self.access_order = deque(maxlen=max_size)
        self.error_count = 0
        
    def add_image(self, filename, frame, file_mtime):
        """Ajoute une image au cache avec gestion d'erreurs"""
        try:
            with self.lock:
                # Validation de la frame
                if frame is None or frame.size == 0:
                    return False
                
                # Si le cache est plein, supprimer la plus ancienne
                if len(self.cache) >= self.max_size and filename not in self.cache:
                    if self.access_order:
                        oldest = self.access_order.popleft()
                        if oldest in self.cache:
                            del self.cache[oldest]
                
                # Copie sécurisée de la frame
                try:
                    frame_copy = frame.copy()
                except Exception as e:
                    logger.debug(f"Frame copy error: {e}")
                    return False
                
                # Ajouter au cache
                self.cache[filename] = {
                    'frame': frame_copy,
                    'timestamp': time.time(),
                    'mtime': file_mtime,
                    'shape': frame_copy.shape
                }
                
                # Mettre à jour l'ordre d'accès
                if filename in self.access_order:
                    self.access_order.remove(filename)
                self.access_order.append(filename)
                
                return True
                
        except Exception as e:
            self.error_count += 1
            logger.debug(f"Cache add error: {e}")
            return False
    
    def get_latest_frame(self):
        """Récupère la frame la plus récente du cache"""
        try:
            with self.lock:
                if not self.cache:
                    return None
                    
                # Trouver l'image avec le mtime le plus récent
                latest_item = max(self.cache.items(), key=lambda x: x[1]['mtime'])
                frame = latest_item[1]['frame']
                
                # Validation avant retour
                if frame is None or frame.size == 0:
                    return None
                    
                return frame.copy()
                
        except Exception as e:
            self.error_count += 1
            logger.debug(f"Cache get error: {e}")
            return None
    
    def has_image(self, filename, file_mtime):
        """Vérifie si l'image est déjà en cache et à jour"""
        try:
            with self.lock:
                if filename not in self.cache:
                    return False
                return abs(self.cache[filename]['mtime'] - file_mtime) < 0.001
        except:
            return False
    
    def clear(self):
        """Vide le cache"""
        try:
            with self.lock:
                self.cache.clear()
                self.access_order.clear()
                logger.debug("Cache cleared")
        except Exception as e:
            logger.debug(f"Cache clear error: {e}")
    
    def get_stats(self):
        """Retourne les statistiques du cache"""
        try:
            with self.lock:
                total_memory = 0
                for item in self.cache.values():
                    if 'shape' in item:
                        total_memory += np.prod(item['shape']) * 3  # 3 bytes per pixel
                
                return {
                    'size': len(self.cache),
                    'max_size': self.max_size,
                    'memory_mb': total_memory / (1024 * 1024),
                    'error_count': self.error_count,
                    'files': list(self.cache.keys())
                }
        except:
            return {'size': 0, 'max_size': self.max_size, 'memory_mb': 0, 'error_count': self.error_count}

# Cache global
image_cache = RobustImageCache(CACHE_SIZE)

class AdvancedGloveDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=15)  # Réduit pour plus de réactivité
        self.min_area = 600          # Légèrement réduit
        self.max_area = 40000        
        self.min_contour_points = 12 
        
        # Kernels morphologiques optimisés
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Réduit
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Stabilisation des détections
        self.stable_detections = deque(maxlen=5)  # Plus réactif
        self.confidence_threshold = 3  # Sur 5 détections
        
        # Filtrage temporel
        self.last_detection_center = None
        self.max_movement_threshold = 120
        
    def detect_glove(self, frame):
        if frame is None:
            return frame, False
            
        try:
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement plus agressif pour les performances
            scale_factor = 1.0
            if w > 640:  # Réduit de 800 à 640
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement simplifié pour les performances
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)  # Plus simple que bilateral
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masques couleur optimisés
            mask = self._create_color_mask(hsv)
            
            # Post-traitement morphologique simplifié
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection de contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours, work_frame.shape)
            
            # Validation temporelle
            detected = self._validate_detection(best_contour, scale_factor)
            
            # Mise à jour de l'historique
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            self.detection_history.append(stable_detection)
            
            # Dessin des résultats
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
        """Masque couleur simplifié"""
        try:
            # Masque orange
            orange_lower = np.array([10, 120, 120])
            orange_upper = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Masque rouge
            red_lower1 = np.array([0, 120, 120])
            red_upper1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([170, 120, 120])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison
            mask_final = cv2.bitwise_or(mask_orange, mask_red)
            
            return mask_final
        except Exception as e:
            logger.debug(f"Color mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_best_contour(self, contours, frame_shape):
        """Sélection simplifiée du meilleur contour"""
        if not contours:
            return None
            
        try:
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
                
                # Filtres de base
                aspect_ratio = w_rect / float(h_rect)
                if not (0.4 <= aspect_ratio <= 2.5):
                    continue
                    
                # Score simple basé sur l'aire et la position
                area_score = min(area / 3000.0, 1.0)
                position_score = 1.0 if y > h * 0.1 else 0.7
                
                score = area_score * position_score
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
                    
            return best_contour
        except Exception as e:
            logger.debug(f"Contour selection error: {e}")
            return None
    
    def _validate_detection(self, contour, scale_factor):
        """Validation temporelle simplifiée"""
        if contour is None:
            self.last_detection_center = None
            return False
            
        try:
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
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False
    
    def _draw_detection(self, frame, contour):
        """Dessin simplifié"""
        try:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            
            cv2.putText(frame, "GANT", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_overlay(self, frame, detected):
        """Overlay simplifié"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 DETECTE" if detected else "🔍 RECHERCHE"
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Statistiques
            with stats_lock:
                detection_rate = (frame_stats['detection_count'] / max(frame_stats['frame_count'], 1)) * 100
                cache_stats = image_cache.get_stats()
                restart_count = frame_stats['restart_count']
            
            stats_text = f"Frames: {frame_stats['frame_count']} | Det: {detection_rate:.1f}% | Restart: {restart_count}"
            cv2.putText(frame, stats_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cache_text = f"Cache: {cache_stats['size']}/{cache_stats['max_size']} | Mem: {cache_stats['memory_mb']:.1f}MB"
            cv2.putText(frame, cache_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            # Historique visuel compact
            history_symbols = ["●" if d else "○" for d in list(self.detection_history)[-10:]]
            history_text = "Hist: " + "".join(history_symbols)
            cv2.putText(frame, history_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Indicateur de redémarrage
            if restart_event.is_set():
                cv2.putText(frame, "REDEMARRAGE...", (w//2 - 80, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def robust_vision_callback(args):
    """Callback robuste avec gestion d'erreurs avancée"""
    try:
        if not vision_active.is_set():
            return
            
        current_time = time.time()
        
        # Scanner les fichiers avec gestion d'erreurs
        try:
            pattern = os.path.join(IMAGES_DIR, "image_*.png")
            files = glob.glob(pattern)
        except Exception as e:
            logger.debug(f"File scanning error: {e}")
            return
        
        if not files:
            return
        
        # Trier par temps de modification
        try:
            files.sort(key=os.path.getmtime, reverse=True)
        except Exception as e:
            logger.debug(f"File sorting error: {e}")
            return
        
        # Traiter les fichiers les plus récents
        for latest_file in files[:2]:  # Réduit à 2 pour les performances
            try:
                # Vérifications de base avec timeouts
                stat_info = os.stat(latest_file)
                file_size = stat_info.st_size
                file_mtime = stat_info.st_mtime
                filename = os.path.basename(latest_file)
                
                # Filtres plus stricts
                if file_size < 5000:  # Augmenté pour éviter les fichiers corrompus
                    continue
                
                if (current_time - file_mtime) < 0.05:  # Plus conservateur
                    continue
                
                # Vérifier le cache
                if image_cache.has_image(filename, file_mtime):
                    with stats_lock:
                        frame_stats['cache_hits'] += 1
                    continue
                
                # Lecture robuste de l'image
                frame = None
                for attempt in range(3):  # Plus de tentatives
                    try:
                        frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                        if frame is not None and frame.size > 0:
                            break
                        time.sleep(0.01)
                    except Exception as e:
                        logger.debug(f"Read attempt {attempt+1} failed: {e}")
                        if attempt < 2:
                            time.sleep(0.02)
                        continue
                
                if frame is None:
                    continue
                
                # Validations étendues
                h, w = frame.shape[:2]
                if h < 240 or w < 320 or h > 2000 or w > 2000:
                    continue
                
                # Test de corruption plus strict
                mean_val = np.mean(frame)
                if mean_val < 10 or mean_val > 245:
                    continue
                
                # Test de variance plus strict
                gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.var(gray_test) < 200:  # Augmenté
                    continue
                
                # Test de pixels valides
                if np.count_nonzero(frame) < (frame.size * 0.1):  # Au moins 10% de pixels non-noirs
                    continue
                
                # Ajouter au cache
                if image_cache.add_image(filename, frame, file_mtime):
                    with stats_lock:
                        frame_stats['cache_misses'] += 1
                    logger.debug(f"Image cached: {filename} ({file_size} bytes)")
                else:
                    logger.debug(f"Failed to cache image: {filename}")
                    continue
                
                break
                
            except Exception as e:
                logger.debug(f"File processing error for {latest_file}: {e}")
                continue
        
        # Récupérer la frame la plus récente du cache
        latest_frame = image_cache.get_latest_frame()
        if latest_frame is not None and vision_active.is_set():
            # Gestion robuste de la queue
            try:
                if frame_queue.full():
                    # Vider la queue rapidement
                    cleared = 0
                    while not frame_queue.empty() and cleared < MAX_QUEUE_SIZE:
                        try:
                            frame_queue.get_nowait()
                            cleared += 1
                        except Empty:
                            break
                
                frame_queue.put_nowait(latest_frame)
                with stats_lock:
                    frame_stats['frame_count'] += 1
                    frame_stats['last_frame_time'] = current_time
                    
            except Exception as e:
                logger.debug(f"Queue management error: {e}")
                
    except Exception as e:
        logger.debug(f"Vision callback critical error: {e}")

def vision_restart_manager(bebop, vision_obj_ref):
    """Gestionnaire de redémarrage automatique du flux vidéo"""
    logger.info("Vision restart manager started")
    
    while processing_active.is_set():
        restart_event.wait()  # Attendre un signal de redémarrage
        
        if not processing_active.is_set():
            break
        
        current_time = time.time()
        with stats_lock:
            if (current_time - frame_stats['last_restart_time']) < RESTART_COOLDOWN:
                logger.info(f"Restart cooldown active, waiting...")
                restart_event.clear()
                time.sleep(RESTART_COOLDOWN)
                continue
            
            if frame_stats['restart_count'] >= MAX_RESTART_ATTEMPTS:
                logger.error(f"Maximum restart attempts ({MAX_RESTART_ATTEMPTS}) reached")
                restart_event.clear()
                time.sleep(30)  # Pause plus longue
                frame_stats['restart_count'] = 0  # Reset après pause
                continue
            
            frame_stats['restart_count'] += 1
            frame_stats['last_restart_time'] = current_time
        
        logger.warning(f"Attempting vision restart #{frame_stats['restart_count']}")
        
        try:
            # Arrêter le flux actuel
            vision_active.clear()
            time.sleep(2)
            
            if vision_obj_ref[0]:
                try:
                    vision_obj_ref[0].close_video()
                    logger.info("Video stream closed for restart")
                except Exception as e:
                    logger.debug(f"Error closing video: {e}")
            
            # Vider le cache et la queue
            image_cache.clear()
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    break
            
            time.sleep(3)  # Pause pour stabilisation
            
            # Redémarrer le flux - vérifier si le drone est connecté
            drone_connected = False
            try:
                # Tenter de vérifier l'état du drone avec une méthode sûre
                drone_connected = hasattr(bebop, '_drone_connection') or bebop is not None
                # Alternative: on peut aussi tenter un ping simple
                if hasattr(bebop, 'ask_for_state_update'):
                    bebop.ask_for_state_update()
                    drone_connected = True
            except Exception as e:
                logger.debug(f"Drone connection check failed: {e}")
                drone_connected = False
            
            if drone_connected:
                try:
                    vision_obj_ref[0] = DroneVision(bebop, is_bebop=True)
                    vision_obj_ref[0].set_user_callback_function(robust_vision_callback)
                    
                    if vision_obj_ref[0].open_video():
                        vision_active.set()
                        logger.info("Video stream restarted successfully")
                        restart_event.clear()
                        
                        # Attendre un peu pour voir si ça fonctionne
                        time.sleep(5)
                        continue
                    else:
                        logger.error("Failed to restart video stream")
                        
                except Exception as e:
                    logger.error(f"Vision restart error: {e}")
            else:
                logger.error("Drone not connected, cannot restart video stream")
            
            # Si on arrive ici, le redémarrage a échoué
            logger.error("Vision restart failed")
            time.sleep(10)  # Pause avant nouvelle tentative
            
        except Exception as e:
            logger.error(f"Critical restart manager error: {e}")
            time.sleep(10)
    
    logger.info("Vision restart manager terminated")

def enhanced_monitor_thread():
    """Thread de monitoring avec redémarrage automatique"""
    logger.info("Enhanced connection monitor started")
    last_frame_count = 0
    check_interval = 4  # Plus fréquent
    consecutive_failures = 0
    performance_issues = 0
    
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
            restart_count = frame_stats['restart_count']
        
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        time_since_last_frame = time.time() - last_received_time
        
        # Détection de problèmes
        stream_issue = frame_diff == 0 or time_since_last_frame > WATCHDOG_TIMEOUT
        performance_issue = frame_diff > 0 and (frame_diff / check_interval) < 8  # Moins de 8 FPS
        
        if stream_issue:
            consecutive_failures += 1
            logger.warning(f"Stream issue detected (failure #{consecutive_failures}). "
                         f"Last frame: {time_since_last_frame:.1f}s ago")
            
            # Déclencher redémarrage après 2 échecs consécutifs
            if consecutive_failures >= 2 and not restart_event.is_set():
                logger.warning("Triggering automatic restart")
                restart_event.set()
                
        elif performance_issue:
            performance_issues += 1
            logger.warning(f"Performance issue detected (#{performance_issues}). "
                         f"Low FPS: {frame_diff / check_interval:.1f}")
            
            # Redémarrage sur problèmes de performance persistants
            if performance_issues >= 3 and not restart_event.is_set():
                logger.warning("Triggering restart due to performance issues")
                restart_event.set()
                performance_issues = 0
                
        else:
            if consecutive_failures > 0:
                logger.info("Stream recovered")
            consecutive_failures = 0
            performance_issues = max(0, performance_issues - 1)  # Décrémentation graduelle
            
            # Statistiques normales
            avg_fps = frame_diff / check_interval
            detection_rate = (detections / max(current_frames, 1)) * 100
            cache_hit_rate = (cache_hits / max(cache_hits + cache_misses, 1)) * 100
            cache_stats = image_cache.get_stats()
            
            logger.info(f"MONITOR - Frames: {current_frames}, FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}, "
                       f"Cache: {cache_stats['size']}/{cache_stats['max_size']} "
                       f"({cache_hit_rate:.1f}% hits), Restarts: {restart_count}")
    
    logger.info("Enhanced connection monitor terminated")

def display_thread():
    """Thread d'affichage robuste"""
    detector = AdvancedGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Detection Auto-Restart"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    last_fps_log = time.time()
    
    while processing_active.is_set():
        try:
            try:
                frame = frame_queue.get(timeout=FRAME_TIMEOUT)
            except Empty:
                # Afficher un écran de veille
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Attente du flux video...", (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, blank_frame)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            if frame is None:
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
                    frame_stats['cache_hits'] = 0
                    frame_stats['cache_misses'] = 0
                    frame_stats['restart_count'] = 0
                detector.detection_history.clear()
                detector.stable_detections.clear()
                image_cache.clear()
                logger.info("All statistics and cache reset")
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            elif key == ord('c'):
                cache_stats = image_cache.get_stats()
                logger.info(f"Cache stats: {cache_stats}")
            elif key == ord('v'):
                # Redémarrage manuel du flux vidéo
                logger.info("Manual video restart requested")
                restart_event.set()
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def cleanup_thread():
    """Thread de nettoyage très conservateur"""
    logger.info("Conservative cleanup thread started")
    cleanup_interval = 60  # Nettoyage toutes les minutes
    
    while processing_active.is_set():
        try:
            # Nettoyage très conservateur - seulement les très vieux fichiers
            files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            
            if len(files) > 200:  # Seuil très élevé
                current_time = time.time()
                very_old_files = []
                
                for file_path in files:
                    try:
                        file_mtime = os.path.getmtime(file_path)
                        # Supprimer seulement les fichiers de plus de 2 minutes
                        if (current_time - file_mtime) > 120:
                            very_old_files.append(file_path)
                    except:
                        continue
                
                # Supprimer seulement une partie des anciens fichiers
                files_to_remove = very_old_files[:len(very_old_files)//2]
                removed_count = 0
                
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        time.sleep(0.001)  # Petite pause entre suppressions
                    except:
                        continue
                
                if removed_count > 0:
                    logger.info(f"Conservative cleanup: removed {removed_count} very old files")
                        
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        
        # Attente interruptible
        for _ in range(cleanup_interval):
            if not processing_active.is_set():
                break
            time.sleep(1)
    
    logger.info("Cleanup thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating graceful shutdown")
    processing_active.clear()
    restart_event.set()  # Débloquer le restart manager

def main():
    """Fonction principale avec redémarrage automatique"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Advanced Bebop 2 detection with auto-restart")
    
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        return False
    
    bebop = None
    vision_obj_ref = [None]  # Liste pour permettre modification dans les threads
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
        
        # Configuration initiale de la vision
        vision_obj_ref[0] = DroneVision(bebop, is_bebop=True)
        vision_obj_ref[0].set_user_callback_function(robust_vision_callback)
        
        # Démarrage des threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        cleanup_thread_obj = threading.Thread(target=cleanup_thread, daemon=True, name="Cleanup")
        monitor_thread_obj = threading.Thread(target=enhanced_monitor_thread, daemon=True, name="Monitor")
        restart_manager_obj = threading.Thread(target=vision_restart_manager, 
                                             args=(bebop, vision_obj_ref), 
                                             daemon=True, name="RestartManager")
        
        threads = [display_thread_obj, cleanup_thread_obj, monitor_thread_obj, restart_manager_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.2)  # Délai plus long entre démarrages
        
        logger.info("All threads started successfully")
        
        # Ouverture initiale du flux vidéo
        logger.info("Opening initial video stream...")
        try:
            if vision_obj_ref[0].open_video():
                vision_active.set()
                logger.info("Initial video stream opened successfully")
            else:
                logger.warning("Failed to open initial video stream, will retry automatically")
                restart_event.set()  # Déclencher un redémarrage immédiat
        except Exception as e:
            logger.warning(f"Initial video stream error: {e}, will retry automatically")
            restart_event.set()
        
        logger.info("Auto-restart detection system is now active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset all, 's'=Screenshot, 'c'=Cache stats, 'v'=Manual restart")
        
        # Boucle principale avec monitoring de santé
        last_health_check = time.time()
        health_check_interval = 30  # Vérification toutes les 30 secondes
        
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                current_time = time.time()
                
                # Vérification de santé périodique
                if (current_time - last_health_check) > health_check_interval:
                    try:
                        # Vérifier si la fenêtre est toujours ouverte
                        if cv2.getWindowProperty("Bebop 2 - Detection Auto-Restart", cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Display window was closed")
                            break
                        
                        # Vérifier la santé du drone
                        if bebop and not bebop.is_alive():
                            logger.warning("Drone connection lost")
                            break
                        
                        # Statistiques de santé
                        with stats_lock:
                            frames = frame_stats['frame_count']
                            restarts = frame_stats['restart_count']
                            time_since_last_frame = current_time - frame_stats['last_frame_time']
                        
                        cache_stats = image_cache.get_stats()
                        
                        logger.info(f"HEALTH CHECK - Frames: {frames}, Restarts: {restarts}, "
                                  f"Last frame: {time_since_last_frame:.1f}s ago, "
                                  f"Cache: {cache_stats['size']}/{cache_stats['max_size']}, "
                                  f"Errors: {cache_stats['error_count']}")
                        
                        last_health_check = current_time
                        
                    except Exception as e:
                        logger.debug(f"Health check error: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
    
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        return False
    
    finally:
        # Nettoyage complet et ordonné
        logger.info("Starting comprehensive cleanup...")
        processing_active.clear()
        restart_event.set()  # Débloquer tous les threads en attente
        vision_active.clear()
        
        # Attendre que les threads se terminent proprement
        logger.info("Waiting for threads to terminate...")
        for thread in threads:
            try:
                thread.join(timeout=10)
                logger.debug(f"Thread {thread.name} terminated successfully")
            except Exception as e:
                logger.debug(f"Error joining thread {thread.name}: {e}")
        
        # Vider le cache et la queue
        try:
            image_cache.clear()
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    break
            logger.info("Cache and queue cleared")
        except Exception as e:
            logger.debug(f"Error clearing cache/queue: {e}")
        
        # Fermeture du flux vidéo
        if vision_obj_ref[0]:
            try:
                vision_obj_ref[0].close_video()
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
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
        except Exception as e:
            logger.debug(f"Error closing OpenCV windows: {e}")
        
        logger.info("Comprehensive cleanup completed successfully")
    
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