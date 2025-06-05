import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import logging
import os
import signal
import sys
from collections import deque
import glob

# Configuration
DISPLAY_FPS = 25
CONNECTION_TIMEOUT = 20
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
POLLING_RATE = 0.05  # 50ms entre chaque vérification = 20Hz max

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
    'polling_cycles': 0,
    'successful_reads': 0,
    'failed_reads': 0,
    'files_found': 0,
    'files_cleaned': 0,
    'last_file_processed': '',
    'duplicate_skips': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_active_polling.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def active_polling_thread():
    """
    Thread qui surveille activement le dossier d'images
    """
    global current_frame
    
    logger.info("Active polling thread started")
    
    while processing_active.is_set():
        try:
            with stats_lock:
                frame_stats['polling_cycles'] += 1
            
            # Lecture active du fichier le plus récent
            frame = read_latest_with_polling()
            
            if frame is not None:
                with frame_lock:
                    current_frame = frame.copy()
                
                with stats_lock:
                    frame_stats['frame_count'] += 1
                    frame_stats['successful_reads'] += 1
                    frame_stats['last_frame_time'] = time.time()
            else:
                with stats_lock:
                    frame_stats['failed_reads'] += 1
            
            # Attendre avant le prochain cycle
            time.sleep(POLLING_RATE)
            
        except Exception as e:
            logger.debug(f"Polling thread error: {e}")
            with stats_lock:
                frame_stats['failed_reads'] += 1
            time.sleep(POLLING_RATE)
    
    logger.info("Active polling thread terminated")

def read_latest_with_polling():
    """
    Lecture active avec nettoyage intelligent
    """
    try:
        if not os.path.exists(IMAGES_DIR):
            return None
        
        # Scanner tous les fichiers image
        pattern = os.path.join(IMAGES_DIR, "image_*.png")
        files = glob.glob(pattern)
        
        with stats_lock:
            frame_stats['files_found'] = len(files)
        
        if not files:
            return None
        
        # Trier par date de modification (plus récent en premier)
        files.sort(key=os.path.getmtime, reverse=True)
        
        current_time = time.time()
        frame = None
        
        # Essayer les 3 fichiers les plus récents
        for i, latest_file in enumerate(files[:3]):
            try:
                stat_info = os.stat(latest_file)
                file_size = stat_info.st_size
                file_mtime = stat_info.st_mtime
                filename = os.path.basename(latest_file)
                
                # Filtres de sécurité
                if file_size < 3000:
                    continue
                    
                # Éviter fichiers trop récents (en cours d'écriture)
                if (current_time - file_mtime) < 0.02:
                    continue
                
                # Éviter de relire le même fichier
                with stats_lock:
                    if frame_stats['last_file_processed'] == filename:
                        frame_stats['duplicate_skips'] += 1
                        if i == 0:  # Si c'est le plus récent, pas de nouveau contenu
                            continue
                
                # Lecture rapide avec validation
                try:
                    frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                    if frame is not None and frame.size > 0:
                        h, w = frame.shape[:2]
                        if h >= 240 and w >= 320:
                            mean_val = np.mean(frame)
                            if 10 <= mean_val <= 245:
                                # Frame valide !
                                with stats_lock:
                                    frame_stats['last_file_processed'] = filename
                                
                                logger.debug(f"Frame loaded: {filename} ({w}x{h}, {file_size} bytes)")
                                break
                    frame = None
                except Exception as e:
                    logger.debug(f"Read error {latest_file}: {e}")
                    frame = None
                    continue
                    
            except Exception as e:
                logger.debug(f"File stat error {latest_file}: {e}")
                continue
        
        # Nettoyage intelligent : garder seulement les 10 plus récents
        if len(files) > 10:
            files_to_remove = files[10:]
            removed_count = 0
            
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except:
                    continue
            
            if removed_count > 0:
                with stats_lock:
                    frame_stats['files_cleaned'] += removed_count
                logger.debug(f"Cleaned {removed_count} old files")
        
        return frame
        
    except Exception as e:
        logger.debug(f"Polling read error: {e}")
        return None

class SmartGloveDetector:
    """Détecteur de gants intelligent optimisé"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=8)
        self.min_area = 1000
        self.max_area = 30000
        self.min_contour_points = 20
        
        # Kernels morphologiques
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Stabilisation
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3
        
        # Anti-faux positifs
        self.last_detection_center = None
        self.max_movement = 70
        self.detection_cooldown = 0
        
        # Cache
        self.last_frame_hash = None
        self.last_detection_result = None
        
    def detect_glove(self, frame):
        """Détection intelligente avec cache"""
        if frame is None:
            return frame, False
            
        try:
            # Cache simple
            frame_hash = hash(frame.tobytes()[::1000])
            
            if frame_hash == self.last_frame_hash and self.last_detection_result is not None:
                return self.last_detection_result
            
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement pour performance
            scale_factor = 1.0
            if w > 600:
                scale_factor = 600.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur intelligent
            mask = self._create_smart_mask(hsv)
            
            # Morphologie
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.erode(mask, self.kernel_erode, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours, work_frame.shape)
            
            # Validation
            detected = self._validate_detection(best_contour, scale_factor)
            
            # Cooldown
            if self.detection_cooldown > 0:
                self.detection_cooldown -= 1
                detected = False
            
            # Stabilisation
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
            
            # Cache
            self.last_frame_hash = frame_hash
            self.last_detection_result = (result_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _create_smart_mask(self, hsv):
        """Masque couleur intelligent"""
        try:
            h, w = hsv.shape[:2]
            
            # Exclusion peau
            skin_lower = np.array([0, 20, 70])
            skin_upper = np.array([25, 120, 255])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Orange gants
            orange_lower = np.array([8, 140, 140])
            orange_upper = np.array([22, 240, 240])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge gants (pas trop saturé)
            red_lower1 = np.array([0, 120, 120])
            red_upper1 = np.array([8, 200, 220])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([172, 120, 120])
            red_upper2 = np.array([180, 200, 220])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison
            mask_gants = cv2.bitwise_or(mask_orange, mask_red)
            
            # Exclusion peau
            mask_skin_dilated = cv2.dilate(mask_skin, self.kernel_close, iterations=1)
            mask_final = cv2.bitwise_and(mask_gants, cv2.bitwise_not(mask_skin_dilated))
            
            # Exclusion bords
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 15
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_best_contour(self, contours, frame_shape):
        """Sélection du meilleur contour"""
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
                
                aspect_ratio = w_rect / float(h_rect)
                if not (0.4 <= aspect_ratio <= 2.5):
                    continue
                
                # Éviter bords
                margin = 20
                if (x < margin or y < margin or 
                    (x + w_rect) > (w - margin) or 
                    (y + h_rect) > (h - margin)):
                    continue
                
                # Solidité
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.5:
                        continue
                else:
                    continue
                
                # Score simple
                area_score = min(area / 3000.0, 1.0)
                solidity_score = min(solidity * 2, 1.0)
                
                score = area_score * solidity_score
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Contour selection error: {e}")
            return None
    
    def _validate_detection(self, contour, scale_factor):
        """Validation temporelle"""
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
                
                if distance > self.max_movement:
                    self.detection_cooldown = 3
                    return False
            
            self.last_detection_center = current_center
            return True
            
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False
    
    def _draw_detection(self, frame, contour):
        """Dessin de détection"""
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
            cv2.putText(frame, f"GANT DETECTE (A:{int(area)})", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_overlay(self, frame, detected, mask=None):
        """Overlay d'informations"""
        try:
            h, w = frame.shape[:2]
            
            # Status
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE GANT..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Statistiques
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
                polling_cycles = frame_stats['polling_cycles']
                success_reads = frame_stats['successful_reads']
                failed_reads = frame_stats['failed_reads']
                files_found = frame_stats['files_found']
                cleaned = frame_stats['files_cleaned']
                duplicates = frame_stats['duplicate_skips']
                
                detection_rate = (detections / max(frames, 1)) * 100
                read_success_rate = (success_reads / max(polling_cycles, 1)) * 100
            
            # Performance
            perf_text = f"Frames: {frames} | Det: {detection_rate:.1f}% | Err: {errors}"
            cv2.putText(frame, perf_text, (10, h - 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Polling
            polling_text = f"Polling: {polling_cycles} cycles | {read_success_rate:.1f}% success | Files: {files_found}"
            cv2.putText(frame, polling_text, (10, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # I/O
            io_text = f"I/O: {success_reads} reads | {failed_reads} fails | {duplicates} skips | {cleaned} cleaned"
            cv2.putText(frame, io_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            
            # Détecteur
            confidence = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0
            detector_text = f"Confiance: {confidence:.1%} | Cooldown: {self.detection_cooldown}"
            cv2.putText(frame, detector_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Historique
            history = ["●" if d else "○" for d in list(self.detection_history)[-15:]]
            history_text = "Hist: " + "".join(history)
            cv2.putText(frame, history_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mode
            cv2.putText(frame, "MODE: ACTIVE POLLING (No Callback Dependency)", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Timestamp et FPS
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            polling_rate_text = f"Poll: {1/POLLING_RATE:.0f}Hz"
            cv2.putText(frame, polling_rate_text, (w - 120, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Masque miniature
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (140, 100))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    mask_x, mask_y = w - 150, 60
                    frame[mask_y:mask_y+100, mask_x:mask_x+140] = mask_colored
                    
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+140, mask_y+100), (255, 255, 255), 2)
                    cv2.putText(frame, "Masque", (mask_x, mask_y + 115), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def display_thread():
    """Thread d'affichage"""
    detector = SmartGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Active Polling Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    last_display_time = time.time()
    
    no_frame_count = 0
    
    while processing_active.is_set():
        try:
            current_time = time.time()
            
            # Limitation FPS
            if (current_time - last_display_time) < (1.0 / DISPLAY_FPS):
                time.sleep(0.01)
                continue
            
            last_display_time = current_time
            
            # Récupération frame
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = None
            
            if frame is None:
                no_frame_count += 1
                
                # Écran d'attente
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Polling actif en cours...", (160, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                with stats_lock:
                    cycles = frame_stats['polling_cycles']
                    reads = frame_stats['successful_reads']
                    fails = frame_stats['failed_reads']
                    files = frame_stats['files_found']
                    success_rate = (reads / max(cycles, 1)) * 100
                
                cv2.putText(blank_frame, f"Cycles: {cycles} | Reads: {reads} | Success: {success_rate:.1f}%", 
                           (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank_frame, f"Files found: {files} | Fails: {fails}", 
                           (180, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
                cv2.putText(blank_frame, f"No frame cycles: {no_frame_count}", 
                           (190, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                
                cv2.imshow(window_name, blank_frame)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            no_frame_count = 0
            
            # Traitement détection
            processed_frame, detected = detector.detect_glove(frame)
            
            # FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_elapsed = current_time - fps_start_time
                current_fps = fps_counter / fps_elapsed if fps_elapsed > 0 else 0
                logger.info(f"Display FPS: {current_fps:.1f}")
                fps_start_time = current_time
                fps_counter = 0
            
            # Affichage
            cv2.imshow(window_name, processed_frame)
            
            # Touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("User requested quit")
                processing_active.clear()
                break
            elif key == ord('r'):
                # Reset complet
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                    frame_stats['polling_cycles'] = 0
                    frame_stats['successful_reads'] = 0
                    frame_stats['failed_reads'] = 0
                    frame_stats['files_found'] = 0
                    frame_stats['files_cleaned'] = 0
                    frame_stats['last_file_processed'] = ''
                    frame_stats['duplicate_skips'] = 0
                
                detector.detection_history.clear()
                detector.stable_detections.clear()
                detector.last_frame_hash = None
                detector.last_detection_result = None
                detector.last_detection_center = None
                detector.detection_cooldown = 0
                
                no_frame_count = 0
                logger.info("Complete reset performed")
                
            elif key == ord('s'):
                screenshot_name = f"screenshot_polling_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
                
            elif key == ord('d'):
                # Debug
                with stats_lock:
                    debug_stats = frame_stats.copy()
                logger.info(f"Debug stats: {debug_stats}")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def monitor_thread():
    """Thread de monitoring"""
    logger.info("Monitor thread started")
    last_frame_count = 0
    last_polling_count = 0
    
    while processing_active.is_set():
        time.sleep(5)
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            detections = frame_stats['detection_count']
            errors = frame_stats['error_count']
            polling_cycles = frame_stats['polling_cycles']
            success_reads = frame_stats['successful_reads']
            failed_reads = frame_stats['failed_reads']
            files_found = frame_stats['files_found']
            cleaned = frame_stats['files_cleaned']
            duplicates = frame_stats['duplicate_skips']
            last_received_time = frame_stats['last_frame_time']
        
        frame_diff = current_frames - last_frame_count
        polling_diff = polling_cycles - last_polling_count
        last_frame_count = current_frames
        last_polling_count = polling_cycles
        
        time_since_last_frame = time.time() - last_received_time
        read_success_rate = (success_reads / max(polling_cycles, 1)) * 100
        
        if frame_diff > 0:
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            polling_rate = polling_diff / 5
            
            logger.info(f"MONITOR - Frames: {current_frames} (+{frame_diff}), FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}")
            logger.info(f"         Polling: {polling_cycles} cycles (+{polling_diff}, {polling_rate:.1f}/s), "
                       f"Success: {read_success_rate:.1f}% ({success_reads}/{polling_cycles})")
            logger.info(f"         Files: {files_found} found | {cleaned} cleaned | {duplicates} duplicates skipped")
        else:
            logger.warning(f"No new frames - polling: {polling_cycles} (+{polling_diff}), "
                         f"read success: {read_success_rate:.1f}%, last frame {time_since_last_frame:.1f}s ago")
            logger.warning(f"              Files: {files_found} found, Success reads: {success_reads}, Fails: {failed_reads}")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale avec polling actif"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Active Polling Detection System")
    logger.info(f"Polling rate: {1/POLLING_RATE:.0f}Hz ({POLLING_RATE*1000:.0f}ms interval)")
    
    bebop = None
    vision = None
    threads = []
    
    try:
        # Vérification du répertoire
        if not os.path.exists(IMAGES_DIR):
            logger.error(f"Images directory not found: {IMAGES_DIR}")
            return False
        
        logger.info(f"Images directory verified: {IMAGES_DIR}")
        
        # Nettoyage initial complet
        try:
            initial_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            if initial_files:
                for file_path in initial_files:
                    try:
                        os.remove(file_path)
                    except:
                        continue
                logger.info(f"Initial cleanup: removed {len(initial_files)} old files")
        except Exception as e:
            logger.warning(f"Initial cleanup failed: {e}")
        
        # Connexion au drone
        bebop = Bebop()
        logger.info("Connecting to Bebop 2...")
        
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            return False
        
        logger.info("Drone connected successfully")
        
        # Configuration de la vision (même avec callback minimal)
        vision = DroneVision(bebop, is_bebop=True)
        # Pas de callback - on va utiliser le polling actif
        
        # Démarrage des threads
        polling_thread_obj = threading.Thread(target=active_polling_thread, daemon=True, name="ActivePolling")
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        
        threads = [polling_thread_obj, display_thread_obj, monitor_thread_obj]
        
        for i, thread in enumerate(threads):
            thread.start()
            time.sleep(0.1)
            logger.info(f"Thread {i+1}/{len(threads)} started: {thread.name}")
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream for active polling...")
        start_time = time.time()
        
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        open_time = time.time() - start_time
        logger.info(f"Video stream opened successfully ({open_time:.1f}s)")
        logger.info("Active Polling Detection System is now active")
        logger.info("=" * 60)
        logger.info("SYSTEM INFO:")
        logger.info(f"  Polling Rate: {1/POLLING_RATE:.0f}Hz")
        logger.info(f"  Display FPS:  {DISPLAY_FPS}")
        logger.info(f"  Images Dir:   {IMAGES_DIR}")
        logger.info("=" * 60)
        logger.info("CONTROLS:")
        logger.info("  'q' or ESC  = Quit")
        logger.info("  'r'         = Complete reset")
        logger.info("  's'         = Screenshot")
        logger.info("  'd'         = Debug stats")
        logger.info("=" * 60)
        
        # Boucle principale avec monitoring avancé
        start_time = time.time()
        last_status_time = time.time()
        status_interval = 10  # Status toutes les 10 secondes
        
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                current_time = time.time()
                
                # Vérifier fenêtre OpenCV
                try:
                    if cv2.getWindowProperty("Bebop 2 - Active Polling Detection", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Display window was closed")
                        break
                except:
                    pass
                
                # Status périodique
                if (current_time - last_status_time) >= status_interval:
                    with stats_lock:
                        status_stats = frame_stats.copy()
                    
                    uptime = current_time - start_time
                    avg_frame_rate = status_stats['frame_count'] / max(uptime, 1)
                    avg_polling_rate = status_stats['polling_cycles'] / max(uptime, 1)
                    
                    logger.info(f"STATUS - Uptime: {uptime:.0f}s | Avg Frame Rate: {avg_frame_rate:.1f}/s | "
                               f"Avg Polling: {avg_polling_rate:.1f}/s")
                    logger.info(f"       - Total Frames: {status_stats['frame_count']} | "
                               f"Detections: {status_stats['detection_count']} | "
                               f"Files Cleaned: {status_stats['files_cleaned']}")
                    
                    last_status_time = current_time
                
                # Vérification de santé du système
                with stats_lock:
                    time_since_last_frame = current_time - frame_stats['last_frame_time']
                    polling_cycles = frame_stats['polling_cycles']
                    successful_reads = frame_stats['successful_reads']
                
                # Alerte si pas de frames depuis longtemps
                if time_since_last_frame > 15 and polling_cycles > 100:  # Plus tolérant
                    read_rate = (successful_reads / max(polling_cycles, 1)) * 100
                    logger.warning(f"HEALTH CHECK - No frames for {time_since_last_frame:.1f}s | "
                                 f"Read success rate: {read_rate:.1f}%")
                    
                    # Diagnostic des fichiers
                    try:
                        files_count = len(glob.glob(os.path.join(IMAGES_DIR, "image_*.png")))
                        logger.warning(f"             - Files in directory: {files_count}")
                        
                        if files_count == 0:
                            logger.warning("             - No files found - drone may not be streaming")
                        elif read_rate < 50:
                            logger.warning("             - Low read success rate - possible file corruption")
                    except Exception as e:
                        logger.warning(f"             - File check error: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
    
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Nettoyage complet
        logger.info("Starting comprehensive cleanup...")
        processing_active.clear()
        
        # Attendre les threads
        for thread in threads:
            try:
                thread.join(timeout=8)
                logger.info(f"Thread {thread.name} terminated successfully")
            except Exception as e:
                logger.warning(f"Error joining thread {thread.name}: {e}")
        
        # Fermeture flux vidéo
        if vision:
            try:
                vision.close_video()
                logger.info("Video stream closed successfully")
            except Exception as e:
                logger.warning(f"Error closing video: {e}")
        
        # Déconnexion drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("Drone disconnected successfully")
            except Exception as e:
                logger.warning(f"Error disconnecting drone: {e}")
        
        # Nettoyage final des fichiers
        try:
            final_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            removed_count = 0
            for file_path in final_files:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except:
                    continue
            if removed_count > 0:
                logger.info(f"Final cleanup: removed {removed_count} remaining files")
        except Exception as e:
            logger.warning(f"Final cleanup failed: {e}")
        
        # Fermeture OpenCV
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed successfully")
        except Exception as e:
            logger.warning(f"Error closing OpenCV windows: {e}")
        
        # Statistiques finales détaillées
        with stats_lock:
            final_stats = frame_stats.copy()
        
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS:")
        logger.info(f"  Total Runtime:        {total_time:.1f}s")
        logger.info(f"  Frames Processed:     {final_stats['frame_count']}")
        logger.info(f"  Average Frame Rate:   {final_stats['frame_count']/max(total_time,1):.1f}/s")
        logger.info(f"  Total Detections:     {final_stats['detection_count']}")
        logger.info(f"  Detection Rate:       {(final_stats['detection_count']/max(final_stats['frame_count'],1))*100:.1f}%")
        logger.info(f"  Polling Cycles:       {final_stats['polling_cycles']}")
        logger.info(f"  Average Polling Rate: {final_stats['polling_cycles']/max(total_time,1):.1f}/s")
        logger.info(f"  Successful Reads:     {final_stats['successful_reads']}")
        logger.info(f"  Failed Reads:         {final_stats['failed_reads']}")
        logger.info(f"  Read Success Rate:    {(final_stats['successful_reads']/max(final_stats['polling_cycles'],1))*100:.1f}%")
        logger.info(f"  Files Cleaned:        {final_stats['files_cleaned']}")
        logger.info(f"  Duplicate Skips:      {final_stats['duplicate_skips']}")
        logger.info(f"  Errors:               {final_stats['error_count']}")
        logger.info("=" * 60)
        
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
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)