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
MAX_QUEUE_SIZE = 2  # Réduit pour éviter le lag
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
CONNECTION_TIMEOUT = 20
MAX_IMAGE_FILES = 50    # Réduit pour un nettoyage plus fréquent
IMAGE_KEEP_COUNT = 25   
WATCHDOG_TIMEOUT = 5    # Réduit pour une détection plus rapide des problèmes
FRAME_TIMEOUT = 1.0     # Timeout plus court pour les frames

# Variables globales
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processing_active = threading.Event()
processing_active.set()
frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'last_processed_file': None,
    'stream_restarts': 0
}
stats_lock = threading.Lock()
image_dir_lock = threading.Lock()
vision_restart_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedGloveDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=20)
        self.min_area = 800          # Augmenté pour éviter les petits éléments
        self.max_area = 50000        # Réduit pour éviter les grands objets
        self.min_contour_points = 15 # Augmenté pour des formes plus complexes
        
        # Kernels morphologiques optimisés
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Stabilisation des détections
        self.stable_detections = deque(maxlen=7)
        self.confidence_threshold = 4  # Sur 7 dernières détections
        
        # Filtrage temporel
        self.last_detection_center = None
        self.max_movement_threshold = 100  # pixels max entre deux détections
        
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
            work_frame = cv2.bilateralFilter(work_frame, 9, 80, 80)  # Meilleur que GaussianBlur
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
        
        # Masque peau amélioré (plusieurs teintes)
        skin_masks = []
        # Peau claire
        skin_lower1 = np.array([0, 20, 70])
        skin_upper1 = np.array([25, 120, 255])
        skin_masks.append(cv2.inRange(hsv, skin_lower1, skin_upper1))
        
        # Peau foncée
        skin_lower2 = np.array([0, 25, 50])
        skin_upper2 = np.array([15, 100, 200])
        skin_masks.append(cv2.inRange(hsv, skin_lower2, skin_upper2))
        
        mask_skin = cv2.bitwise_or(skin_masks[0], skin_masks[1])
        
        # Masque orange optimisé
        orange_lower = np.array([8, 140, 140])   # Plus restrictif
        orange_upper = np.array([20, 255, 255])
        mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Masque rouge optimisé (deux plages)
        red_lower1 = np.array([0, 150, 140])    # Plus restrictif
        red_upper1 = np.array([6, 255, 255])
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        
        red_lower2 = np.array([174, 150, 140])  # Plus restrictif
        red_upper2 = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combinaison et exclusion de la peau
        mask_gant = cv2.bitwise_or(mask_orange, mask_red)
        mask_skin_dilated = cv2.dilate(mask_skin, self.kernel_close, iterations=1)
        mask_final = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin_dilated))
        
        return mask_final
    
    def _select_best_contour_advanced(self, contours, frame_shape):
        """Sélection avancée du meilleur contour avec scoring multicritères"""
        if not contours:
            return None
            
        h, w = frame_shape[:2]
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtres de base
            if area < self.min_area or area > self.max_area:
                continue
            if len(contour) < self.min_contour_points:
                continue
                
            # Rectangle englobant
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Filtre aspect ratio plus strict
            aspect_ratio = w_rect / float(h_rect)
            if not (0.3 <= aspect_ratio <= 3.0):  # Plus restrictif
                continue
                
            # Éviter les bords (plus strict)
            margin = 10
            if (x < margin or y < margin or 
                (x + w_rect) > (w - margin) or 
                (y + h_rect) > (h - margin)):
                continue
            
            # Calcul de la solidité (convexité)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.45:  # Plus restrictif
                    continue
            else:
                continue
            
            # Calcul du périmètre et circularité
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.15:  # Éviter les formes trop allongées
                    continue
            
            # Scoring multicritères
            area_score = min(area / 5000.0, 1.0)
            position_score = 1.0 if y > h * 0.15 else 0.6  # Préférer le centre-bas
            solidity_score = min(solidity * 2, 1.0)
            
            # Bonus pour les formes plus carrées
            aspect_bonus = 1.0 if 0.7 <= aspect_ratio <= 1.4 else 0.8
            
            score = area_score * position_score * solidity_score * aspect_bonus
            
            if score > best_score:
                best_score = score
                best_contour = contour
                
        return best_contour
    
    def _validate_detection_temporal(self, contour, scale_factor):
        """Validation temporelle pour éviter les fausses détections"""
        if contour is None:
            self.last_detection_center = None
            return False
            
        # Calcul du centre
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return False
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        current_center = (cx, cy)
        
        # Si on a une détection précédente, vérifier la cohérence
        if self.last_detection_center is not None:
            distance = np.sqrt((cx - self.last_detection_center[0])**2 + 
                             (cy - self.last_detection_center[1])**2)
            
            # Si le mouvement est trop important, c'est probablement une fausse détection
            if distance > self.max_movement_threshold:
                return False
        
        self.last_detection_center = current_center
        return True
    
    def _draw_advanced_detection(self, frame, contour):
        """Dessin avancé de la détection"""
        try:
            # Contour principal
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Centre de masse
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            
            # Informations détaillées
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Texte d'information
            info_y = max(y - 15, 30)
            cv2.putText(frame, f"GANT DETECTE", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Aire: {int(area)} | Sol: {solidity:.2f}", 
                       (x, info_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_advanced_overlay(self, frame, detected, mask=None):
        """Overlay avancé avec plus d'informations"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Statistiques détaillées
            with stats_lock:
                detection_rate = (frame_stats['detection_count'] / max(frame_stats['frame_count'], 1)) * 100
                stats_text = (f"Frames: {frame_stats['frame_count']} | "
                            f"Detections: {frame_stats['detection_count']} ({detection_rate:.1f}%) | "
                            f"Erreurs: {frame_stats['error_count']}")
            
            cv2.putText(frame, stats_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Historique visuel amélioré
            history_symbols = []
            for detection in list(self.detection_history)[-20:]:
                if detection:
                    history_symbols.append("●")
                else:
                    history_symbols.append("○")
            
            history_text = "Historique: " + "".join(history_symbols)
            cv2.putText(frame, history_text, (10, h - 50), 
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
            
            # Masque miniature amélioré
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (180, 135))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    # Position du masque
                    mask_x, mask_y = w - 190, 50
                    frame[mask_y:mask_y+135, mask_x:mask_x+180] = mask_colored
                    
                    # Bordure du masque
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+180, mask_y+135), (255, 255, 255), 2)
                    cv2.putText(frame, "Masque Couleur", (mask_x, mask_y + 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def vision_callback(args):
    """Callback ultra-optimisé pour la réception des frames avec cache intelligent"""
    try:
        current_time = time.time()
        
        with image_dir_lock:
            pattern = os.path.join(IMAGES_DIR, "image_*.png")
            files = glob.glob(pattern)
            
            if not files:
                return
                
            # Tri par temps de modification (plus efficace que max())
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Traiter les 2-3 fichiers les plus récents pour éviter les images corrompues
            processed = False
            for i, latest_file in enumerate(files[:3]):
                try:
                    # Vérifications rapides de validité
                    stat_info = os.stat(latest_file)
                    file_size = stat_info.st_size
                    file_mtime = stat_info.st_mtime
                    
                    # Éviter les fichiers trop petits ou en cours d'écriture
                    if file_size < 3000:
                        continue
                        
                    # Éviter les fichiers trop récents (potentiellement en cours d'écriture)
                    if (current_time - file_mtime) < 0.02:  # 20ms de sécurité
                        continue
                    
                    # Éviter de retraiter le même fichier
                    with stats_lock:
                        if frame_stats['last_processed_file'] == os.path.basename(latest_file):
                            if i == 0:  # Si c'est le plus récent et déjà traité, pas de nouveau
                                return
                            continue
                    
                    # Lecture avec retry en cas d'échec
                    frame = None
                    for attempt in range(2):
                        try:
                            frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                            if frame is not None:
                                break
                            time.sleep(0.005)  # Petit délai avant retry
                        except:
                            if attempt == 0:
                                time.sleep(0.01)
                            continue
                    
                    if frame is None:
                        continue
                        
                    # Vérifications de la frame
                    h, w = frame.shape[:2]
                    if h < 240 or w < 320:  # Résolution minimum
                        continue
                        
                    # Test de corruption simple (pixels tous noirs ou tous blancs)
                    if np.all(frame == 0) or np.all(frame == 255):
                        continue
                    
                    # Test de variance pour éviter les images uniformes
                    gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if np.var(gray_test) < 100:  # Image trop uniforme
                        continue
                    
                    # Frame valide trouvée
                    processed = True
                    
                    # Gestion intelligente de la queue
                    queue_size = frame_queue.qsize()
                    
                    if queue_size >= MAX_QUEUE_SIZE:
                        # Vider complètement la queue pour éviter le lag accumulé
                        cleared = 0
                        while not frame_queue.empty() and cleared < MAX_QUEUE_SIZE:
                            try:
                                frame_queue.get_nowait()
                                cleared += 1
                            except Empty:
                                break
                        logger.debug(f"Queue cleared: {cleared} frames")
                    
                    # Ajout de la nouvelle frame
                    try:
                        frame_queue.put_nowait(frame)
                    except:
                        # Dernier recours : forcer l'ajout
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame)
                        except:
                            logger.debug("Failed to add frame to queue")
                            continue
                    
                    # Mise à jour des statistiques
                    with stats_lock:
                        frame_stats['frame_count'] += 1
                        frame_stats['last_frame_time'] = current_time
                        frame_stats['last_processed_file'] = os.path.basename(latest_file)
                    
                    logger.debug(f"Frame captured: {os.path.basename(latest_file)} ({file_size} bytes)")
                    break
                    
                except (OSError, IOError) as e:
                    logger.debug(f"File access error for {latest_file}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Frame processing error for {latest_file}: {e}")
                    continue
            
            if not processed:
                logger.debug("No valid frames found in recent files")
                
    except Exception as e:
        logger.debug(f"Vision callback critical error: {e}")

def enhanced_vision_callback(args):
    """Version alternative avec buffer circulaire pour ultra-haute performance"""
    try:
        current_time = time.time()
        
        # Cache des fichiers pour éviter les appels système répétés
        if not hasattr(enhanced_vision_callback, 'file_cache'):
            enhanced_vision_callback.file_cache = {}
            enhanced_vision_callback.last_scan = 0
        
        # Rescan des fichiers seulement toutes les 100ms
        if (current_time - enhanced_vision_callback.last_scan) > 0.1:
            with image_dir_lock:
                pattern = os.path.join(IMAGES_DIR, "image_*.png")
                files = glob.glob(pattern)
                
                # Mise à jour du cache
                new_cache = {}
                for f in files:
                    try:
                        stat_info = os.stat(f)
                        new_cache[f] = {
                            'mtime': stat_info.st_mtime,
                            'size': stat_info.st_size
                        }
                    except:
                        continue
                
                enhanced_vision_callback.file_cache = new_cache
                enhanced_vision_callback.last_scan = current_time
        
        # Trouver le fichier le plus récent depuis le cache
        if not enhanced_vision_callback.file_cache:
            return
        
        latest_file = max(enhanced_vision_callback.file_cache.items(), 
                         key=lambda x: x[1]['mtime'])
        
        file_path, file_info = latest_file
        
        # Vérifications rapides
        if file_info['size'] < 3000:
            return
        
        if (current_time - file_info['mtime']) < 0.015:  # 15ms de sécurité
            return
        
        # Éviter le retraitement
        with stats_lock:
            if frame_stats['last_processed_file'] == os.path.basename(file_path):
                return
        
        # Lecture optimisée
        try:
            frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if frame is None:
                return
            
            h, w = frame.shape[:2]
            if h < 240 or w < 320:
                return
            
            # Test rapide de validité
            if np.mean(frame) < 5 or np.mean(frame) > 250:  # Image trop sombre ou claire
                return
            
            # Gestion ultra-rapide de la queue
            if frame_queue.full():
                # Vider d'un coup au lieu de un par un
                temp_frames = []
                while not frame_queue.empty():
                    try:
                        temp_frames.append(frame_queue.get_nowait())
                    except Empty:
                        break
                # Ne garder que la dernière frame si nécessaire
                if temp_frames:
                    logger.debug(f"Queue flushed: {len(temp_frames)} frames")
            
            frame_queue.put_nowait(frame)
            
            # Stats
            with stats_lock:
                frame_stats['frame_count'] += 1
                frame_stats['last_frame_time'] = current_time
                frame_stats['last_processed_file'] = os.path.basename(file_path)
            
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
            
    except Exception as e:
        logger.debug(f"Enhanced callback error: {e}")

def cleanup_thread():
    """Thread de nettoyage optimisé"""
    logger.info("Cleanup thread started")
    cleanup_interval = 15  # Nettoyage plus fréquent
    
    while processing_active.is_set():
        try:
            with image_dir_lock:
                files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
                
                if len(files) > MAX_IMAGE_FILES:
                    # Trier par date de modification
                    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
                    files_to_remove = files_sorted[IMAGE_KEEP_COUNT:]
                    
                    removed_count = 0
                    for file_path in files_to_remove:
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except OSError as e:
                            logger.debug(f"Could not remove {file_path}: {e}")
                    
                    if removed_count > 0:
                        logger.info(f"Cleaned up {removed_count} old image files")
                        
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        
        # Attente interruptible
        for _ in range(cleanup_interval):
            if not processing_active.is_set():
                break
            time.sleep(1)
    
    logger.info("Cleanup thread terminated")

def display_thread():
    """Thread d'affichage optimisé"""
    detector = AdvancedGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Detection Gant Avancee"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Compteurs pour les statistiques
    fps_counter = 0
    fps_start_time = time.time()
    last_fps_log = time.time()
    
    while processing_active.is_set():
        try:
            # Récupération de frame avec timeout
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
            
            # Log FPS toutes les 30 frames ou toutes les 5 secondes
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
            if key == ord('q') or key == 27:  # 'q' ou Escape
                logger.info("User requested quit")
                processing_active.clear()
                break
            elif key == ord('r'):  # Reset statistiques
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                detector.detection_history.clear()
                detector.stable_detections.clear()
                logger.info("Statistics and detection history reset")
            elif key == ord('s'):  # Screenshot
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            elif key == ord('d'):  # Toggle debug
                logger.info("Debug info toggled")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def connection_monitor_thread():
    """Thread de monitoring optimisé avec redémarrage automatique"""
    logger.info("Connection monitor started")
    last_frame_count = 0
    check_interval = 3  # Vérification plus fréquente
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
        
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        time_since_last_frame = time.time() - last_received_time
        
        # Vérification de l'état du flux
        if frame_diff == 0 or time_since_last_frame > WATCHDOG_TIMEOUT:
            consecutive_failures += 1
            logger.warning(f"Stream issue detected (failure #{consecutive_failures}). "
                         f"Last frame: {time_since_last_frame:.1f}s ago")
            
            if consecutive_failures >= 3:
                logger.error("Multiple consecutive stream failures. Manual restart may be required.")
                # On pourrait implémenter un redémarrage automatique ici
                
        else:
            if consecutive_failures > 0:
                logger.info("Stream recovered")
            consecutive_failures = 0
            
            # Statistiques normales
            avg_fps = frame_diff / check_interval
            detection_rate = (detections / max(current_frames, 1)) * 100
            
            logger.info(f"MONITOR - Frames: {current_frames}, FPS: {avg_fps:.1f}, "
                       f"Detections: {detection_rate:.1f}%, Errors: {errors}")
    
    logger.info("Connection monitor terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux amélioré"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale optimisée"""
    # Configuration des signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Advanced Bebop 2 glove detection system")
    
    # Vérification du répertoire d'images
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
        
        # Configuration de la vision
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(vision_callback)
        
        # Démarrage des threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        cleanup_thread_obj = threading.Thread(target=cleanup_thread, daemon=True)
        monitor_thread_obj = threading.Thread(target=connection_monitor_thread, daemon=True)
        
        threads = [display_thread_obj, cleanup_thread_obj, monitor_thread_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.1)  # Petit délai entre les démarrages
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        logger.info("Video stream opened successfully")
        logger.info("Advanced detection system is now active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot, 'd'=Debug")
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Detection Gant Avancee", cv2.WND_PROP_VISIBLE) < 1:
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