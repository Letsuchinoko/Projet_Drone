import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import logging
import signal
import sys
from collections import deque
import subprocess
import os
import tempfile
import shutil
from os.path import join
import inspect

# Configuration
DISPLAY_FPS = 25
CONNECTION_TIMEOUT = 20
BUFFER_CHECK_RATE = 0.033  # 30Hz

# Variables globales
current_frame = None
frame_lock = threading.RLock()
processing_active = threading.Event()
processing_active.set()

frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'buffer_checks': 0,
    'buffer_hits': 0,
    'buffer_misses': 0,
    'temp_files_created': 0,
    'temp_files_cleaned': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_no_disk_complete.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class NoDiskVision(DroneVision):
    """Version modifiée de DroneVision qui n'écrit PAS sur disque"""
    
    def __init__(self, drone_object, is_bebop, buffer_size=50):
        # Initialisation de base
        self.fps = 30
        self.buffer_size = buffer_size
        self.drone_object = drone_object
        self.is_bebop = is_bebop
        self.cleanup_old_images = False  # Désactivé !

        # Buffer en mémoire seulement
        self.buffer = [None] * buffer_size
        self.buffer_index = 0

        # Threads
        self.vision_thread = threading.Thread(target=self._memory_only_vision, args=(buffer_size,))
        self.user_vision_thread = None
        self.vision_running = True

        # Index pour le traitement
        self.frame_counter = 0
        self.new_frame = False
        
        # Dossier temporaire en mémoire (tmpfs sur Linux, RAM disk simulé)
        self.temp_dir = tempfile.mkdtemp(prefix="bebop_temp_")
        logger.info(f"Temporary directory created: {self.temp_dir}")

    def set_user_callback_function(self, user_callback_function=None, user_callback_args=None):
        """Set callback function"""
        self.user_vision_thread = threading.Thread(target=self._user_callback,
                                                   args=(user_callback_function, user_callback_args))

    def open_video(self):
        """Ouvre le flux vidéo avec redirection vers dossier temporaire"""
        # Démarrer le stream sur le bebop
        if self.is_bebop:
            self.drone_object.start_video_stream()

        # Configuration des chemins
        fullPath = inspect.getfile(DroneVision)
        shortPathIndex = fullPath.rfind("/")
        if shortPathIndex == -1:
            shortPathIndex = fullPath.rfind("\\")
        
        shortPath = fullPath[0:shortPathIndex]
        self.utilPath = join(shortPath, "utils")
        
        # IMPORTANT : Utiliser le dossier temporaire au lieu du dossier permanent
        self.imagePath = self.temp_dir
        
        logger.info(f"Using temporary image path: {self.imagePath}")
        logger.info(f"Utils path: {self.utilPath}")

        # Pas de nettoyage d'images - le dossier est déjà vide
        logger.info("No disk cleanup needed - using temporary directory")

        # Démarrer FFmpeg avec redirection vers le dossier temporaire
        logger.info("Starting FFmpeg with temporary directory")
        if self.is_bebop:
            cmdStr = f"ffmpeg -protocol_whitelist \"file,rtp,udp\" -i {self.utilPath}/bebop.sdp -r 30 image_%03d.png"
            logger.info(f"FFmpeg command: {cmdStr}")
            self.ffmpeg_process = subprocess.Popen(
                cmdStr, 
                shell=True, 
                cwd=self.imagePath,  # Redirection vers temp
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE
            )
        else:
            self.ffmpeg_process = subprocess.Popen(
                "ffmpeg -i rtsp://192.168.99.1/media/stream2 -r 30 image_%03d.png",
                shell=True, 
                cwd=self.imagePath,  # Redirection vers temp
                stderr=subprocess.PIPE, 
                stdout=subprocess.PIPE
            )

        # Démarrer immédiatement le buffering
        self._start_video_buffering()

        # Vérification du succès comme dans PyParrot original
        from pyparrot.utils.NonBlockingStreamReader import NonBlockingStreamReader
        
        stderr_reader = NonBlockingStreamReader(self.ffmpeg_process.stderr)
        stdout_reader = NonBlockingStreamReader(self.ffmpeg_process.stdout)

        success = False
        timeout_counter = 0
        max_timeout = 50  # 5 secondes

        while not success and timeout_counter < max_timeout:
            line = stderr_reader.readline()
            if line is not None:
                line_str = line.decode("utf-8")
                logger.debug(f"FFmpeg stderr: {line_str.strip()}")
                if "Stream #0:0 -> #0:0 (h264 (native) -> png (native))" in line_str:
                    success = True
                    break
                if "Output file #0 does not contain any stream" in line_str:
                    logger.error("FFmpeg: No stream found")
                    break

            line = stdout_reader.readline()
            if line is not None:
                line_str = line.decode("utf-8")
                logger.debug(f"FFmpeg stdout: {line_str.strip()}")
                if "Output file #0 does not contain any stream" in line_str:
                    logger.error("FFmpeg: No stream found")
                    break
                if "Stream #0:0 -> #0:0 (h264 (native) -> png (native))" in line_str:
                    success = True
                    break

            time.sleep(0.1)
            timeout_counter += 1

        # Cleanup des readers
        stderr_reader.finish_reader()
        stdout_reader.finish_reader()

        if success:
            logger.info("FFmpeg stream opened successfully")
        else:
            logger.error("Failed to open FFmpeg stream")

        return success

    def _start_video_buffering(self):
        """Démarre le thread de buffering"""
        logger.info("Starting memory-only vision thread")
        self.vision_thread.start()

        if self.user_vision_thread is not None:
            self.user_vision_thread.start()

    def _user_callback(self, user_vision_function, user_args):
        """Callback utilisateur modifié"""
        while self.vision_running:
            if self.new_frame:
                user_vision_function(user_args)
                self.new_frame = False

            time.sleep(1.0 / (3.0 * self.fps))

    def _memory_only_vision(self, buffer_size):
        """Version modifiée qui lit depuis temp et nettoie immédiatement"""
        logger.info("Memory-only vision thread started")
        
        self.new_frame = False
        image_index = 1

        # Attendre les premiers fichiers
        found_latest = False
        startup_timeout = 50  # 5 secondes
        startup_counter = 0
        
        while not found_latest and startup_counter < startup_timeout:
            path = f"{self.imagePath}/image_{image_index:03d}.png"
            if os.path.exists(path):
                found_latest = True
                logger.info(f"First image found: {path}")
            else:
                image_index += 1
                time.sleep(0.1)
                startup_counter += 1

        if not found_latest:
            logger.error("No initial images found in temporary directory")
            return

        # Boucle principale de traitement
        while self.vision_running:
            try:
                # Construire le chemin de l'image courante
                path = f"{self.imagePath}/image_{image_index:03d}.png"
                
                if not os.path.exists(path):
                    time.sleep(1.0 / (2.0 * self.fps))
                    continue

                # Lire l'image
                img = cv2.imread(path, 1)

                if img is not None:
                    # Stocker dans le buffer
                    self.buffer_index = (self.buffer_index + 1) % buffer_size
                    self.buffer[self.buffer_index] = img.copy()  # Copie pour sécurité
                    self.new_frame = True
                    
                    # Incrémenter le compteur
                    image_index += 1
                    
                    # NETTOYAGE IMMÉDIAT du fichier temporaire
                    try:
                        os.remove(path)
                        with stats_lock:
                            frame_stats['temp_files_cleaned'] += 1
                    except:
                        pass  # Pas grave si échec de suppression
                    
                    with stats_lock:
                        frame_stats['temp_files_created'] += 1

            except Exception as e:
                logger.debug(f"Vision processing error: {e}")
                continue

            # Pause pour éviter surcharge CPU
            time.sleep(1.0 / (2.0 * self.fps))

        logger.info("Memory-only vision thread terminated")

    def get_latest_valid_picture(self):
        """Retourne la dernière image valide du buffer"""
        return self.buffer[self.buffer_index]

    def close_video(self):
        """Ferme la vision et nettoie le dossier temporaire"""
        logger.info("Closing video and cleaning up...")
        
        # Arrêter les threads
        self.vision_running = False

        # Tuer FFmpeg
        if hasattr(self, 'ffmpeg_process'):
            try:
                self.ffmpeg_process.kill()
                self.ffmpeg_process.terminate()
                time.sleep(2)
                
                if self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.kill()
                    time.sleep(2)
                    
                logger.info("FFmpeg process terminated")
            except Exception as e:
                logger.warning(f"Error terminating FFmpeg: {e}")

        # Nettoyer le dossier temporaire
        try:
            shutil.rmtree(self.temp_dir)
            logger.info(f"Temporary directory cleaned: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning temporary directory: {e}")

        # Arrêter le stream (bebop seulement)
        if self.is_bebop:
            try:
                self.drone_object.stop_video_stream()
                logger.info("Bebop video stream stopped")
            except Exception as e:
                logger.warning(f"Error stopping video stream: {e}")

def direct_buffer_thread(vision_object):
    """Thread qui accède au buffer sans dépendre des fichiers"""
    global current_frame
    
    logger.info("Direct buffer access thread started")
    
    last_buffer_index = -1
    
    while processing_active.is_set():
        try:
            with stats_lock:
                frame_stats['buffer_checks'] += 1
            
            # Accès direct au buffer
            if hasattr(vision_object, 'buffer') and hasattr(vision_object, 'buffer_index'):
                current_buffer_index = vision_object.buffer_index
                
                # Nouvelle frame détectée
                if current_buffer_index != last_buffer_index:
                    frame = vision_object.buffer[current_buffer_index]
                    
                    if frame is not None and validate_frame(frame):
                        with frame_lock:
                            current_frame = frame.copy()
                        
                        with stats_lock:
                            frame_stats['frame_count'] += 1
                            frame_stats['buffer_hits'] += 1
                            frame_stats['last_frame_time'] = time.time()
                        
                        last_buffer_index = current_buffer_index
                        logger.debug(f"New frame from buffer index {current_buffer_index}")
                    else:
                        with stats_lock:
                            frame_stats['buffer_misses'] += 1
                else:
                    # Même frame, pas d'action
                    pass
            else:
                with stats_lock:
                    frame_stats['buffer_misses'] += 1
            
            # Attendre avant prochaine vérification
            time.sleep(BUFFER_CHECK_RATE)
            
        except Exception as e:
            logger.debug(f"Buffer thread error: {e}")
            with stats_lock:
                frame_stats['buffer_misses'] += 1
            time.sleep(BUFFER_CHECK_RATE)
    
    logger.info("Direct buffer access thread terminated")

def validate_frame(frame):
    """Validation rapide d'une frame"""
    try:
        if frame is None or frame.size == 0:
            return False
        
        h, w = frame.shape[:2]
        if h < 240 or w < 320:
            return False
        
        mean_val = np.mean(frame)
        if mean_val < 10 or mean_val > 245:
            return False
        
        return True
        
    except:
        return False

class SmartGloveDetector:
    """Détecteur de gants optimisé"""
    
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
        self.last_frame_id = None
        self.last_detection_result = None
        
    def detect_glove(self, frame):
        """Détection avec cache"""
        if frame is None:
            return frame, False
            
        try:
            # Cache par ID de frame
            frame_id = id(frame)
            
            if frame_id == self.last_frame_id and self.last_detection_result is not None:
                return self.last_detection_result
            
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement
            scale_factor = 1.0
            if w > 600:
                scale_factor = 600.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque intelligent
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
            self.last_frame_id = frame_id
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
            
            # Rouge gants
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
                
                # Score
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
        """Overlay informatif"""
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
                buffer_checks = frame_stats['buffer_checks']
                buffer_hits = frame_stats['buffer_hits']
                temp_created = frame_stats['temp_files_created']
                temp_cleaned = frame_stats['temp_files_cleaned']
                
                detection_rate = (detections / max(frames, 1)) * 100
                buffer_hit_rate = (buffer_hits / max(buffer_checks, 1)) * 100
                cleanup_rate = (temp_cleaned / max(temp_created, 1)) * 100
            
            # Performance
            perf_text = f"Frames: {frames} | Det: {detection_rate:.1f}% | Err: {errors}"
            cv2.putText(frame, perf_text, (10, h - 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Buffer stats
            buffer_text = f"Buffer: {buffer_hit_rate:.1f}% hits | Temp files: {temp_created}"
            cv2.putText(frame, buffer_text, (10, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Cleanup stats
            cleanup_text = f"Cleanup: {cleanup_rate:.1f}% | Created: {temp_created} | Cleaned: {temp_cleaned}"
            cv2.putText(frame, cleanup_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            
            # Détecteur
            confidence = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0
            detector_text = f"Confiance: {confidence:.1%} | Cooldown: {self.detection_cooldown}"
            cv2.putText(frame, detector_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Historique
            history = ["●" if d else "○" for d in list(self.detection_history)[-12:]]
            history_text = "Hist: " + "".join(history)
            cv2.putText(frame, history_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mode ZERO DISK
            cv2.putText(frame, "MODE: ZERO DISK USAGE (Temp + Immediate Cleanup)", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
    
    window_name = "Bebop 2 - Zero Disk Usage"
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
                cv2.putText(blank_frame, "Zero Disk System Active...", (150, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                with stats_lock:
                    checks = frame_stats['buffer_checks']
                    hits = frame_stats['buffer_hits']
                    temp_created = frame_stats['temp_files_created']
                    temp_cleaned = frame_stats['temp_files_cleaned']
                    hit_rate = (hits / max(checks, 1)) * 100
                    cleanup_rate = (temp_cleaned / max(temp_created, 1)) * 100
                
                cv2.putText(blank_frame, f"Buffer: {checks} checks | {hit_rate:.1f}% hits", 
                           (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank_frame, f"Temp files: {temp_created} created | {cleanup_rate:.1f}% cleaned", 
                           (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
                cv2.putText(blank_frame, f"Display cycles sans frame: {no_frame_count}", 
                           (140, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                
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
                    frame_stats['buffer_checks'] = 0
                    frame_stats['buffer_hits'] = 0
                    frame_stats['buffer_misses'] = 0
                    frame_stats['temp_files_created'] = 0
                    frame_stats['temp_files_cleaned'] = 0
                
                detector.detection_history.clear()
                detector.stable_detections.clear()
                detector.last_frame_id = None
                detector.last_detection_result = None
                detector.last_detection_center = None
                detector.detection_cooldown = 0
                
                no_frame_count = 0
                logger.info("Complete reset performed")
                
            elif key == ord('s'):
                screenshot_name = f"screenshot_zerodisk_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
                
            elif key == ord('d'):
                # Debug détaillé
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
    last_buffer_checks = 0
    last_temp_created = 0
    
    while processing_active.is_set():
        time.sleep(5)
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            detections = frame_stats['detection_count']
            errors = frame_stats['error_count']
            buffer_checks = frame_stats['buffer_checks']
            buffer_hits = frame_stats['buffer_hits']
            buffer_misses = frame_stats['buffer_misses']
            temp_created = frame_stats['temp_files_created']
            temp_cleaned = frame_stats['temp_files_cleaned']
            last_received_time = frame_stats['last_frame_time']
        
        frame_diff = current_frames - last_frame_count
        checks_diff = buffer_checks - last_buffer_checks
        temp_diff = temp_created - last_temp_created
        
        last_frame_count = current_frames
        last_buffer_checks = buffer_checks
        last_temp_created = temp_created
        
        time_since_last_frame = time.time() - last_received_time
        buffer_hit_rate = (buffer_hits / max(buffer_checks, 1)) * 100
        cleanup_efficiency = (temp_cleaned / max(temp_created, 1)) * 100
        
        if frame_diff > 0:
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            check_rate = checks_diff / 5
            temp_rate = temp_diff / 5
            
            logger.info(f"MONITOR - Frames: {current_frames} (+{frame_diff}), FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}")
            logger.info(f"         Buffer: {buffer_checks} checks (+{checks_diff}, {check_rate:.1f}/s), "
                       f"Hit rate: {buffer_hit_rate:.1f}%")
            logger.info(f"         Temp files: {temp_created} created (+{temp_diff}, {temp_rate:.1f}/s), "
                       f"Cleanup: {cleanup_efficiency:.1f}% ({temp_cleaned}/{temp_created})")
        else:
            logger.warning(f"No new frames - buffer checks: {buffer_checks} (+{checks_diff}), "
                         f"hit rate: {buffer_hit_rate:.1f}%, last frame {time_since_last_frame:.1f}s ago")
            logger.warning(f"              Temp files: {temp_created} created, {temp_cleaned} cleaned, "
                         f"cleanup efficiency: {cleanup_efficiency:.1f}%")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale avec zéro usage disque"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Zero Disk Usage Detection System")
    logger.info("Features: Temporary directory + Immediate cleanup + Buffer access")
    
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
        
        # Configuration de la vision sans disque
        vision = NoDiskVision(bebop, is_bebop=True, buffer_size=50)
        
        # Démarrage des threads
        buffer_thread_obj = threading.Thread(target=direct_buffer_thread, args=(vision,), daemon=True, name="DirectBuffer")
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        
        threads = [buffer_thread_obj, display_thread_obj, monitor_thread_obj]
        
        for i, thread in enumerate(threads):
            thread.start()
            time.sleep(0.1)
            logger.info(f"Thread {i+1}/{len(threads)} started: {thread.name}")
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream with zero disk usage...")
        start_time = time.time()
        
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        open_time = time.time() - start_time
        logger.info(f"Video stream opened successfully ({open_time:.1f}s)")
        logger.info("Zero Disk Usage Detection System is now active")
        logger.info("=" * 60)
        logger.info("SYSTEM INFO:")
        logger.info(f"  Mode:             Zero Disk Usage")
        logger.info(f"  Temp Directory:   {vision.temp_dir}")
        logger.info(f"  Buffer Size:      {vision.buffer_size} frames")
        logger.info(f"  Immediate Cleanup: Enabled")
        logger.info("=" * 60)
        logger.info("CONTROLS:")
        logger.info("  'q' or ESC  = Quit")
        logger.info("  'r'         = Complete reset")
        logger.info("  's'         = Screenshot")
        logger.info("  'd'         = Debug stats")
        logger.info("=" * 60)
        
        # Boucle principale avec monitoring
        start_time = time.time()
        last_status_time = time.time()
        status_interval = 20  # Status toutes les 20 secondes
        
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                current_time = time.time()
                
                # Vérifier fenêtre OpenCV
                try:
                    if cv2.getWindowProperty("Bebop 2 - Zero Disk Usage", cv2.WND_PROP_VISIBLE) < 1:
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
                    buffer_efficiency = (status_stats['buffer_hits'] / max(status_stats['buffer_checks'], 1)) * 100
                    cleanup_efficiency = (status_stats['temp_files_cleaned'] / max(status_stats['temp_files_created'], 1)) * 100
                    
                    logger.info(f"STATUS - Uptime: {uptime:.0f}s")
                    logger.info(f"       - Frame Rate: {avg_frame_rate:.1f}/s (Total: {status_stats['frame_count']})")
                    logger.info(f"       - Buffer Efficiency: {buffer_efficiency:.1f}%")
                    logger.info(f"       - Temp Files: {status_stats['temp_files_created']} created, {status_stats['temp_files_cleaned']} cleaned")
                    logger.info(f"       - Cleanup Efficiency: {cleanup_efficiency:.1f}%")
                    logger.info(f"       - Detections: {status_stats['detection_count']} | Errors: {status_stats['error_count']}")
                    
                    # Vérification du dossier temporaire
                    try:
                        if hasattr(vision, 'temp_dir') and os.path.exists(vision.temp_dir):
                            temp_files = len([f for f in os.listdir(vision.temp_dir) if f.endswith('.png')])
                            logger.info(f"       - Temp directory status: {temp_files} files remaining")
                            
                            if temp_files > 10:
                                logger.warning(f"       - WARNING: {temp_files} temp files not cleaned!")
                        else:
                            logger.warning("       - Temp directory not accessible")
                    except Exception as e:
                        logger.warning(f"       - Temp directory check error: {e}")
                    
                    last_status_time = current_time
                    
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
        
        # Fermeture de la vision (inclut nettoyage temp)
        if vision:
            try:
                logger.info("Closing zero-disk vision...")
                vision.close_video()
                logger.info("Zero-disk vision closed successfully")
            except Exception as e:
                logger.warning(f"Error closing vision: {e}")
        
        # Déconnexion du drone
        if bebop:
            try:
                logger.info("Disconnecting from Bebop...")
                bebop.disconnect()
                logger.info("Bebop disconnected successfully")
            except Exception as e:
                logger.warning(f"Error disconnecting Bebop: {e}")
        
        # Fermeture OpenCV
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed successfully")
        except Exception as e:
            logger.warning(f"Error closing OpenCV windows: {e}")
        
        # Statistiques finales
        with stats_lock:
            final_stats = frame_stats.copy()
        
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS (ZERO DISK USAGE):")
        logger.info(f"  Total Runtime:          {total_time:.1f}s")
        logger.info(f"  Frames Processed:       {final_stats['frame_count']}")
        logger.info(f"  Average Frame Rate:     {final_stats['frame_count']/max(total_time,1):.1f}/s")
        logger.info(f"  Total Detections:       {final_stats['detection_count']}")
        logger.info(f"  Detection Rate:         {(final_stats['detection_count']/max(final_stats['frame_count'],1))*100:.1f}%")
        logger.info(f"  Buffer Checks:          {final_stats['buffer_checks']}")
        logger.info(f"  Buffer Hits:            {final_stats['buffer_hits']}")
        logger.info(f"  Buffer Hit Rate:        {(final_stats['buffer_hits']/max(final_stats['buffer_checks'],1))*100:.1f}%")
        logger.info(f"  Temp Files Created:     {final_stats['temp_files_created']}")
        logger.info(f"  Temp Files Cleaned:     {final_stats['temp_files_cleaned']}")
        logger.info(f"  Cleanup Efficiency:     {(final_stats['temp_files_cleaned']/max(final_stats['temp_files_created'],1))*100:.1f}%")
        logger.info(f"  Processing Errors:      {final_stats['error_count']}")
        logger.info("=" * 60)
        
        logger.info("Zero Disk Usage system cleanup completed successfully")
    
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