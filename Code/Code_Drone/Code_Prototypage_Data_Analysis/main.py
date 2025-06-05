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
import glob

# Configuration
DISPLAY_FPS = 25
CONNECTION_TIMEOUT = 20
BUFFER_SIZE = 30  # Réduit pour plus de stabilité

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
    'buffer_hits': 0,
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
        logging.FileHandler('bebop_robust_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class RobustNoDiskVision(DroneVision):
    """Version robuste de DroneVision sans écriture disque permanente"""
    
    def __init__(self, drone_object, is_bebop, buffer_size=30):
        # Initialisation simplifiée
        self.fps = 30
        self.buffer_size = buffer_size
        self.drone_object = drone_object
        self.is_bebop = is_bebop
        self.cleanup_old_images = False
        
        # Buffer circulaire en mémoire
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.RLock()
        
        # Threads simplifiés
        self.vision_thread = None
        self.vision_running = False
        
        # Compteurs
        self.frame_counter = 0
        self.last_frame_time = 0
        
        # Dossier temporaire avec nettoyage automatique
        self.temp_dir = tempfile.mkdtemp(prefix="bebop_robust_")
        logger.info(f"Temporary directory created: {self.temp_dir}")
        
        # Processus FFmpeg
        self.ffmpeg_process = None
        
    def open_video(self):
        """Ouverture robuste du flux vidéo"""
        try:
            # Démarrer le stream Bebop
            if self.is_bebop:
                self.drone_object.start_video_stream()
                time.sleep(1)  # Attendre stabilisation
            
            # Configuration des chemins
            fullPath = inspect.getfile(DroneVision)
            shortPathIndex = max(fullPath.rfind("/"), fullPath.rfind("\\"))
            shortPath = fullPath[0:shortPathIndex]
            self.utilPath = join(shortPath, "utils")
            
            logger.info(f"Utils path: {self.utilPath}")
            logger.info(f"Temp directory: {self.temp_dir}")
            
            # Commande FFmpeg simplifiée et robuste
            if self.is_bebop:
                cmd = [
                    "ffmpeg", "-y",  # -y pour overwrite
                    "-protocol_whitelist", "file,rtp,udp",
                    "-i", f"{self.utilPath}/bebop.sdp",
                    "-r", "20",  # FPS réduit pour stabilité
                    "-f", "image2",
                    "-q:v", "8",  # Qualité réduite pour performance
                    f"{self.temp_dir}/frame_%06d.png"
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", "rtsp://192.168.99.1/media/stream2",
                    "-r", "20",
                    "-f", "image2", 
                    "-q:v", "8",
                    f"{self.temp_dir}/frame_%06d.png"
                ]
            
            logger.info(f"Starting FFmpeg: {' '.join(cmd)}")
            
            # Démarrer FFmpeg avec gestion d'erreur
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # Vérification rapide du démarrage
            time.sleep(2)
            if self.ffmpeg_process.poll() is not None:
                stdout, stderr = self.ffmpeg_process.communicate()
                logger.error(f"FFmpeg failed to start. Stderr: {stderr.decode()}")
                return False
            
            # Attendre les premières images
            success = self._wait_for_first_frames()
            
            if success:
                # Démarrer le thread de capture
                self.vision_running = True
                self.vision_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.vision_thread.start()
                logger.info("Video capture started successfully")
                return True
            else:
                logger.error("No frames received from FFmpeg")
                return False
                
        except Exception as e:
            logger.error(f"Error opening video: {e}")
            return False
    
    def _wait_for_first_frames(self, timeout=10):
        """Attendre que les premières images arrivent"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                frames = glob.glob(f"{self.temp_dir}/frame_*.png")
                if len(frames) >= 3:
                    logger.info(f"First frames received: {len(frames)} files")
                    return True
                time.sleep(0.5)
            except Exception as e:
                logger.debug(f"Error checking frames: {e}")
                time.sleep(0.5)
        
        return False
    
    def _capture_loop(self):
        """Boucle de capture principale - simplifiée et robuste"""
        logger.info("Frame capture loop started")
        
        last_processed_frame = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.vision_running and consecutive_errors < max_consecutive_errors:
            try:
                # Trouver les fichiers disponibles
                frame_files = sorted(glob.glob(f"{self.temp_dir}/frame_*.png"))
                
                if not frame_files:
                    time.sleep(0.05)
                    continue
                
                # Prendre le fichier le plus récent non traité
                current_files = []
                for frame_file in frame_files:
                    try:
                        frame_num = int(os.path.basename(frame_file).split('_')[1].split('.')[0])
                        if frame_num > last_processed_frame:
                            current_files.append((frame_num, frame_file))
                    except (ValueError, IndexError):
                        continue
                
                if not current_files:
                    time.sleep(0.05)
                    continue
                
                # Trier et prendre le plus récent
                current_files.sort()
                frame_num, frame_path = current_files[-1]
                
                # Lire l'image
                frame = cv2.imread(frame_path)
                
                if frame is not None and self._validate_frame(frame):
                    # Stocker dans le buffer
                    with self.buffer_lock:
                        self.frame_buffer.append(frame.copy())
                    
                    # Mettre à jour les stats
                    with stats_lock:
                        frame_stats['frame_count'] += 1
                        frame_stats['last_frame_time'] = time.time()
                        frame_stats['buffer_hits'] += 1
                        frame_stats['temp_files_created'] += 1
                    
                    last_processed_frame = frame_num
                    consecutive_errors = 0
                    
                    # Nettoyage immédiat et sélectif
                    self._cleanup_old_frames(frame_files, keep_recent=5)
                    
                    # Debug occasionnel
                    if frame_stats['frame_count'] % 100 == 0:
                        logger.debug(f"Processed {frame_stats['frame_count']} frames")
                
                else:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        logger.warning(f"Consecutive frame errors: {consecutive_errors}")
                
                # Pause adaptative
                time.sleep(0.03)  # ~30 FPS max
                
            except Exception as e:
                consecutive_errors += 1
                logger.debug(f"Capture error #{consecutive_errors}: {e}")
                time.sleep(0.1)
        
        if consecutive_errors >= max_consecutive_errors:
            logger.error("Too many consecutive errors, stopping capture")
        
        logger.info("Frame capture loop terminated")
    
    def _validate_frame(self, frame):
        """Validation rapide et robuste d'une frame"""
        try:
            if frame is None:
                return False
            
            h, w = frame.shape[:2]
            if h < 200 or w < 200:
                return False
            
            # Vérification basique de contenu
            mean_val = np.mean(frame)
            if mean_val < 5 or mean_val > 250:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _cleanup_old_frames(self, frame_files, keep_recent=5):
        """Nettoyage intelligent des anciens fichiers"""
        try:
            if len(frame_files) <= keep_recent:
                return
            
            # Trier par nom (qui contient le numéro)
            frame_files.sort()
            
            # Garder seulement les plus récents
            files_to_delete = frame_files[:-keep_recent]
            
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except OSError:
                    pass  # Fichier déjà supprimé ou en cours d'utilisation
            
            if deleted_count > 0:
                with stats_lock:
                    frame_stats['temp_files_cleaned'] += deleted_count
                    
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
    
    def get_latest_valid_picture(self):
        """Retourne la dernière image valide du buffer"""
        try:
            with self.buffer_lock:
                if self.frame_buffer:
                    return self.frame_buffer[-1].copy()
                return None
        except Exception:
            return None
    
    def close_video(self):
        """Fermeture propre et robuste"""
        logger.info("Closing video system...")
        
        # Arrêter la capture
        self.vision_running = False
        
        # Attendre le thread de capture
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=5)
            logger.info("Capture thread stopped")
        
        # Terminer FFmpeg proprement
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
        
        # Arrêter le stream Bebop
        if self.is_bebop:
            try:
                self.drone_object.stop_video_stream()
                logger.info("Bebop video stream stopped")
            except Exception as e:
                logger.warning(f"Error stopping video stream: {e}")
        
        # Nettoyage final du dossier temporaire
        self._final_cleanup()
    
    def _final_cleanup(self):
        """Nettoyage final robuste"""
        try:
            if os.path.exists(self.temp_dir):
                # Attendre que tous les processus libèrent les fichiers
                time.sleep(1)
                
                # Supprimer tous les fichiers
                for attempt in range(3):
                    try:
                        files = glob.glob(f"{self.temp_dir}/*")
                        for file_path in files:
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                            except OSError:
                                pass
                        
                        # Tenter de supprimer le dossier
                        shutil.rmtree(self.temp_dir)
                        logger.info(f"Temporary directory cleaned: {self.temp_dir}")
                        break
                        
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1)
                        else:
                            logger.info(f"Temp directory will be cleaned by system: {self.temp_dir}")
                            break
        except Exception as e:
            logger.debug(f"Final cleanup error: {e}")

class OptimizedGloveDetector:
    """Détecteur de gants optimisé pour stabilité"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.confidence_threshold = 6  # Sur 10 dernières détections
        
        # Paramètres de détection
        self.min_area = 800
        self.max_area = 25000
        
        # Kernels morphologiques pré-calculés
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Cache pour éviter recalculs
        self.last_frame_hash = None
        self.last_result = None
        
        # Stabilisation
        self.last_detection_center = None
        self.max_movement = 100
        
    def detect_glove(self, frame):
        """Détection robuste avec cache"""
        if frame is None:
            return frame, False
        
        try:
            # Cache simple basé sur la somme des pixels
            frame_hash = np.sum(frame)
            if frame_hash == self.last_frame_hash and self.last_result is not None:
                return self.last_result
            
            original = frame.copy()
            h, w = original.shape[:2]
            
            # Redimensionnement pour performance
            if w > 640:
                scale = 640.0 / w
                work_frame = cv2.resize(frame, (640, int(h * scale)))
            else:
                scale = 1.0
                work_frame = frame.copy()
            
            # Prétraitement léger
            work_frame = cv2.medianBlur(work_frame, 5)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Création du masque couleur
            mask = self._create_color_mask(hsv)
            
            # Morphologie
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)
            
            # Détection de contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._find_best_contour(contours)
            
            # Validation
            detected = self._validate_detection(best_contour, scale)
            
            # Historique pour stabilité
            self.detection_history.append(detected)
            stable_detected = sum(self.detection_history) >= self.confidence_threshold
            
            # Visualisation
            if stable_detected and best_contour is not None:
                if scale != 1.0:
                    best_contour = (best_contour / scale).astype(np.int32)
                self._draw_detection(original, best_contour)
                
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Overlay d'informations
            result_frame = self._add_overlay(original, stable_detected)
            
            # Cache
            self.last_frame_hash = frame_hash
            self.last_result = (result_frame, stable_detected)
            
            return result_frame, stable_detected
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original, False
    
    def _create_color_mask(self, hsv):
        """Masque couleur optimisé"""
        try:
            # Masque pour orange/rouge des gants
            # Orange
            orange_lower = np.array([10, 150, 150])
            orange_upper = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge (deux plages)
            red_lower1 = np.array([0, 120, 120])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 120, 120])
            red_upper2 = np.array([180, 255, 255])
            
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison
            mask_combined = cv2.bitwise_or(mask_orange, mask_red)
            
            # Exclusion des bords
            h, w = hsv.shape[:2]
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 20
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            return cv2.bitwise_and(mask_combined, border_mask)
            
        except Exception:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _find_best_contour(self, contours):
        """Sélection du meilleur contour"""
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if not (self.min_area <= area <= self.max_area):
                continue
            
            # Vérification de la forme
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            if not (0.3 <= aspect_ratio <= 3.0):
                continue
            
            # Score basé sur l'aire
            score = min(area / 5000.0, 1.0)
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour
    
    def _validate_detection(self, contour, scale):
        """Validation de la détection"""
        if contour is None:
            self.last_detection_center = None
            return False
        
        try:
            # Calculer le centre
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return False
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            current_center = (cx, cy)
            
            # Vérifier le mouvement si on a une détection précédente
            if self.last_detection_center is not None:
                distance = np.sqrt((cx - self.last_detection_center[0])**2 + 
                                 (cy - self.last_detection_center[1])**2)
                
                if distance > self.max_movement:
                    return False
            
            self.last_detection_center = current_center
            return True
            
        except Exception:
            return False
    
    def _draw_detection(self, frame, contour):
        """Dessiner la détection"""
        try:
            # Contour principal
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Centre
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            
            # Texte
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT ({int(area)})", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_overlay(self, frame, detected):
        """Overlay d'informations"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Statistiques
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
                hits = frame_stats['buffer_hits']
                created = frame_stats['temp_files_created']
                cleaned = frame_stats['temp_files_cleaned']
            
            # Performance
            detection_rate = (detections / max(frames, 1)) * 100
            cleanup_rate = (cleaned / max(created, 1)) * 100
            
            stats_text = f"Frames: {frames} | Det: {detection_rate:.1f}% | Err: {errors}"
            cv2.putText(frame, stats_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            perf_text = f"Buffer: {hits} hits | Temp: {created} ({cleanup_rate:.0f}% cleaned)"
            cv2.putText(frame, perf_text, (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
            
            # Confiance
            confidence = sum(self.detection_history) / len(self.detection_history) if self.detection_history else 0
            conf_text = f"Confiance: {confidence:.1%} | Historique: {len(self.detection_history)}"
            cv2.putText(frame, conf_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def display_loop(vision):
    """Boucle d'affichage principale"""
    logger.info("Display loop started")
    
    detector = OptimizedGloveDetector()
    window_name = "Bebop 2 - Robust Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start = time.time()
    no_frame_count = 0
    
    while processing_active.is_set():
        try:
            # Récupérer la dernière frame
            frame = vision.get_latest_valid_picture()
            
            if frame is None:
                no_frame_count += 1
                
                # Écran d'attente
                if no_frame_count % 10 == 0:  # Réduire la fréquence des messages
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "En attente du flux video...", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    with stats_lock:
                        stats_copy = frame_stats.copy()
                    
                    cv2.putText(blank, f"Frames recues: {stats_copy['frame_count']}", (180, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(blank, f"Cycles d'attente: {no_frame_count}", (180, 310),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                    
                    cv2.imshow(window_name, blank)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            no_frame_count = 0
            
            # Traitement de détection
            processed_frame, detected = detector.detect_glove(frame)
            
            # Calcul FPS occasionnel
            fps_counter += 1
            if fps_counter % 50 == 0:
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
                # Reset des statistiques
                with stats_lock:
                    for key in frame_stats:
                        if key != 'last_frame_time':
                            frame_stats[key] = 0
                detector.detection_history.clear()
                logger.info("Statistics reset")
            elif key == ord('s'):
                # Screenshot
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            
            # Pause pour limiter l'utilisation CPU
            time.sleep(1.0 / DISPLAY_FPS)
            
        except Exception as e:
            logger.error(f"Display loop error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display loop terminated")

def monitor_thread():
    """Thread de monitoring simplifié"""
    logger.info("Monitor thread started")
    
    last_frame_count = 0
    
    while processing_active.is_set():
        time.sleep(10)  # Monitoring toutes les 10 secondes
        
        if not processing_active.is_set():
            break
        
        with stats_lock:
            current_stats = frame_stats.copy()
        
        frame_diff = current_stats['frame_count'] - last_frame_count
        last_frame_count = current_stats['frame_count']
        
        if frame_diff > 0:
            fps = frame_diff / 10.0
            detection_rate = (current_stats['detection_count'] / max(current_stats['frame_count'], 1)) * 100
            cleanup_rate = (current_stats['temp_files_cleaned'] / max(current_stats['temp_files_created'], 1)) * 100
            
            logger.info(f"MONITOR - Frames: {current_stats['frame_count']} (+{frame_diff}), "
                       f"FPS: {fps:.1f}, Det: {detection_rate:.1f}%, "
                       f"Cleanup: {cleanup_rate:.1f}%")
        else:
            logger.warning(f"No new frames in last 10s - Total: {current_stats['frame_count']}")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux pour arrêt propre"""
    logger.info(f"Signal {sig} received - shutting down")
    processing_active.clear()

def main():
    """Fonction principale robuste"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Robust Detection System")
    logger.info("Features: Zero disk usage + Robust frame capture + Optimized detection")
    
    bebop = None
    vision = None
    threads = []
    start_time = time.time()
    
    try:
        # Connexion au drone
        logger.info("Connecting to Bebop 2...")
        bebop = Bebop()
        
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            return False
        
        logger.info("Drone connected successfully")
        
        # Initialisation de la vision robuste
        vision = RobustNoDiskVision(bebop, is_bebop=True, buffer_size=BUFFER_SIZE)
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        open_time = time.time() - start_time
        logger.info(f"Video stream opened successfully in {open_time:.1f}s")
        
        # Démarrage des threads
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        threads.append(monitor_thread_obj)
        
        # Démarrer le monitoring
        monitor_thread_obj.start()
        logger.info("Monitor thread started")
        
        # Affichage des informations système
        logger.info("=" * 60)
        logger.info("SYSTEM STATUS:")
        logger.info(f"  Mode:              Robust Zero Disk")
        logger.info(f"  Buffer Size:       {BUFFER_SIZE} frames")
        logger.info(f"  Temp Directory:    {vision.temp_dir}")
        logger.info(f"  Target FPS:        {DISPLAY_FPS}")
        logger.info("=" * 60)
        logger.info("CONTROLS:")
        logger.info("  'q' or ESC     = Quit")
        logger.info("  'r'            = Reset statistics")
        logger.info("  's'            = Screenshot")
        logger.info("=" * 60)
        
        # Boucle d'affichage principale (dans le thread principal)
        logger.info("Starting main display loop...")
        display_loop(vision)
        
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
        
        # Attendre les threads
        for thread in threads:
            try:
                thread.join(timeout=5)
                logger.info(f"Thread {thread.name} stopped")
            except Exception as e:
                logger.warning(f"Error stopping thread {thread.name}: {e}")
        
        # Fermeture de la vision
        if vision:
            try:
                vision.close_video()
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
        logger.info(f"  Buffer Hits:          {final_stats['buffer_hits']}")
        logger.info(f"  Temp Files Created:   {final_stats['temp_files_created']}")
        logger.info(f"  Temp Files Cleaned:   {final_stats['temp_files_cleaned']}")
        logger.info(f"  Cleanup Efficiency:   {(final_stats['temp_files_cleaned']/max(final_stats['temp_files_created'],1))*100:.1f}%")
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