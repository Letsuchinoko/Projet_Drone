import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from queue import Queue, Empty
import logging
import os
import glob
import signal
import sys
from collections import deque
import io

# === CONFIGURATION OPTIMISÉE ===
DISPLAY_FPS = 15
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
MAX_QUEUE_SIZE = 2  # Queue très petite pour réduire latence
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_WAIT_TIME = 2.0
CONNECTION_TIMEOUT = 20
VISION_TIMEOUT = 25

# Variables globales thread-safe
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processing_active = threading.Event()
processing_active.set()
connection_stable = threading.Event()
frame_stats = {
    'last_frame_time': time.time(),
    'frame_count': 0,
    'error_count': 0,
    'last_processed_file': None,
    'detection_count': 0
}
stats_lock = threading.Lock()

# Configuration du logging corrigée pour Windows
class UTF8StreamHandler(logging.StreamHandler):
    """Handler personnalisé pour gérer l'UTF-8 sur Windows"""
    def __init__(self, stream=None):
        super().__init__(stream)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            # Remplacer les emojis par du texte pour éviter les erreurs d'encodage
            msg = self._clean_message(msg)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
    
    def _clean_message(self, msg):
        """Remplace les emojis par du texte ASCII"""
        emoji_replacements = {
            '🚁': '[DRONE]',
            '🔗': '[CONNECT]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '🎥': '[VIDEO]',
            '🎬': '[DISPLAY]',
            '🧹': '[CLEANUP]',
            '📡': '[MONITOR]',
            '🎯': '[DETECT]',
            '🛑': '[STOP]',
            '⚠️': '[WARNING]',
            '🪟': '[WINDOW]',
            '🔄': '[REFRESH]',
            '📊': '[STATS]',
            '🧤': '[GLOVE]'
        }
        
        for emoji, replacement in emoji_replacements.items():
            msg = msg.replace(emoji, replacement)
        
        return msg

# Configuration du logging avec handler personnalisé
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(sys.stdout),
        logging.FileHandler('bebop_detection.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class OptimizedFrameProcessor:
    """Processeur de frames ultra-optimisé pour temps réel"""
    
    def __init__(self):
        self.frame_count = 0
        self.detection_history = deque(maxlen=5)  # Historique réduit
        self.last_detection_time = 0
        
        # Paramètres de détection simplifiés
        self.min_area = 300
        self.max_area_ratio = 0.3
        
        # Cache des kernels
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        
        # Variables de performance
        self.skip_frames = 0
        self.process_every_n = 2  # Traiter 1 frame sur 2 pour les performances
    
    def detect_glove_fast(self, image):
        """Détection ultra-rapide optimisée pour le temps réel"""
        try:
            if image is None or image.size == 0:
                return image
            
            self.frame_count += 1
            
            # Skip frames pour améliorer les performances
            if self.skip_frames > 0:
                self.skip_frames -= 1
                return self._add_overlay_info(image, detected=False)
            
            start_time = time.time()
            
            # Réduction agressive de la résolution pour le traitement
            h, w = image.shape[:2]
            if w > 480:  # Réduction plus agressive
                scale = 480.0 / w
                new_w, new_h = int(w * scale), int(h * scale)
                work_img = cv2.resize(image, (new_w, new_h))
                scale_back = w / 480.0
            else:
                work_img = image
                scale_back = 1.0
            
            # Conversion HSV rapide
            hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
            
            # Masque couleur simplifié - focus sur rouge/orange principal
            mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))    # Rouge
            mask2 = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([25, 255, 255]))   # Orange
            mask3 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255])) # Rouge haut
            
            combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
            
            # Nettoyage minimal
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_small)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel_medium)
            
            # Détection de contours rapide
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            best_area = 0
            
            # Sélection rapide du meilleur contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                    
                if area > best_area:
                    # Vérifications basiques
                    x, y, w_rect, h_rect = cv2.boundingRect(contour)
                    aspect_ratio = w_rect / float(h_rect)
                    
                    # Filtres simples
                    if 0.2 <= aspect_ratio <= 5.0 and y > work_img.shape[0] * 0.1:
                        best_contour = contour
                        best_area = area
            
            detected = best_contour is not None
            
            # Dessiner la détection sur l'image originale
            if detected:
                # Redimensionner le contour si nécessaire
                if scale_back != 1.0:
                    best_contour = (best_contour * scale_back).astype(np.int32)
                
                self._draw_detection_simple(image, best_contour)
                self.last_detection_time = time.time()
                
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Ajuster la fréquence de traitement selon les performances
            processing_time = time.time() - start_time
            if processing_time > 0.05:  # Si > 50ms, skip plus de frames
                self.skip_frames = min(3, self.skip_frames + 1)
            elif processing_time < 0.02:  # Si < 20ms, peut traiter plus souvent
                self.skip_frames = max(0, self.skip_frames - 1)
            
            # Mise à jour historique
            self.detection_history.append(detected)
            
            return self._add_overlay_info(image, detected, processing_time)
            
        except Exception as e:
            logger.error(f"Detection error frame {self.frame_count}: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return image
    
    def _draw_detection_simple(self, image, contour):
        """Affichage simplifié pour les performances"""
        try:
            # Contour vert
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Point central
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)
            
            # Texte simple
            cv2.putText(image, "GANT DETECTE", (x, max(y - 10, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Draw error: {e}")
    
    def _add_overlay_info(self, image, detected=False, proc_time=None):
        """Ajoute les informations overlay"""
        try:
            h, w = image.shape[:2]
            
            # Status de détection
            status = "DETECTE" if detected else "RECHERCHE"
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(image, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Statistiques
            with stats_lock:
                stats_text = f"Frames: {frame_stats['frame_count']} | Detections: {frame_stats['detection_count']}"
            
            cv2.putText(image, stats_text, (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Temps de traitement si disponible
            if proc_time:
                time_text = f"Proc: {proc_time*1000:.1f}ms"
                cv2.putText(image, time_text, (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(image, timestamp, (w - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return image
            
        except Exception:
            return image

def robust_vision_callback(args):
    """Callback robuste avec gestion d'erreurs améliorée"""
    global frame_stats
    
    try:
        # Throttling intelligent
        now = time.time()
        with stats_lock:
            time_since_last = now - frame_stats['last_frame_time']
            if time_since_last < DISPLAY_INTERVAL * 0.7:
                return
        
        # Recherche sécurisée du fichier le plus récent
        try:
            pattern = os.path.join(IMAGES_DIR, "image_*.png")
            files = glob.glob(pattern)
            
            if not files:
                return
            
            # Trouve le fichier le plus récent par timestamp
            latest_file = max(files, key=lambda f: os.path.getmtime(f))
            
        except (OSError, ValueError, TypeError) as e:
            logger.debug(f"File search error: {e}")
            return
        
        # Éviter le retraitement du même fichier
        with stats_lock:
            if latest_file == frame_stats['last_processed_file']:
                return
        
        # Chargement robuste avec vérifications
        try:
            # Vérifier la taille du fichier
            if not os.path.exists(latest_file):
                return
                
            file_size = os.path.getsize(latest_file)
            if file_size < 500:  # Fichier trop petit
                return
            
            # Attendre un peu si le fichier semble en cours d'écriture
            time.sleep(0.01)
            
            # Charger l'image
            frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return
            
            # Vérifier les dimensions
            h, w = frame.shape[:2]
            if h < 100 or w < 100:  # Image trop petite
                return
            
            # Gestion de la queue - éviter l'accumulation
            try:
                # Vider complètement la queue pour éviter la latence
                while True:
                    frame_queue.get_nowait()
            except Empty:
                pass
            
            # Ajouter la nouvelle frame
            try:
                frame_queue.put_nowait(frame.copy())
            except:
                # Queue pleine, on ignore cette frame
                return
            
            # Mise à jour des statistiques
            with stats_lock:
                frame_stats['last_frame_time'] = now
                frame_stats['last_processed_file'] = latest_file
                frame_stats['frame_count'] += 1
            
            connection_stable.set()
            
        except Exception as e:
            logger.debug(f"Frame loading error {latest_file}: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
                
    except Exception as e:
        logger.warning(f"Vision callback error: {e}")

def optimized_display_thread():
    """Thread d'affichage ultra-optimisé"""
    processor = OptimizedFrameProcessor()
    logger.info("[DISPLAY] Thread started")
    
    window_name = "Bebop 2 - Detection Gant Rouge/Orange [OPTIMIZED]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
    
    no_frame_warnings = 0
    last_log_time = time.time()
    
    try:
        while processing_active.is_set():
            try:
                # Récupération frame avec timeout adaptatif
                try:
                    frame = frame_queue.get(timeout=1.0)
                    no_frame_warnings = 0
                except Empty:
                    no_frame_warnings += 1
                    if no_frame_warnings >= 3:  # 3 secondes sans frame
                        now = time.time()
                        if now - last_log_time > 5:  # Log max toutes les 5s
                            logger.warning("[WARNING] No frames received for several seconds")
                            last_log_time = now
                    continue
                
                if frame is not None:
                    # Redimensionnement pour l'affichage si nécessaire
                    display_frame = frame.copy()
                    h, w = display_frame.shape[:2]
                    
                    # Limitation de taille d'affichage
                    if w > 800:
                        scale = 800.0 / w
                        new_w, new_h = int(w * scale), int(h * scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))
                    
                    # Traitement de détection
                    processed_frame = processor.detect_glove_fast(display_frame)
                    
                    if processed_frame is not None:
                        # Affichage
                        cv2.imshow(window_name, processed_frame)
                        
                        # Gestion des touches
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' ou ESC
                            logger.info("[STOP] User requested stop")
                            processing_active.clear()
                            break
                        elif key == ord('r'):  # Reset statistiques
                            with stats_lock:
                                frame_stats['frame_count'] = 0
                                frame_stats['error_count'] = 0
                                frame_stats['detection_count'] = 0
                            logger.info("[STATS] Statistics reset")
                        elif key == ord('s'):  # Screenshot
                            screenshot_name = f"detection_screenshot_{int(time.time())}.png"
                            cv2.imwrite(screenshot_name, processed_frame)
                            logger.info(f"[CAPTURE] Screenshot saved: {screenshot_name}")
                
            except Exception as e:
                logger.error(f"Display thread error: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Critical display error: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("[DISPLAY] Thread terminated")

def efficient_cleanup_thread():
    """Thread de nettoyage efficace"""
    logger.info("[CLEANUP] Thread started")
    
    cleanup_interval = 5  # Nettoyage toutes les 5 secondes
    
    while processing_active.is_set():
        try:
            files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            
            if len(files) > 10:  # Garder seulement les 10 plus récentes
                try:
                    # Tri par temps de modification
                    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
                    files_to_remove = files_sorted[10:]
                    
                    removed_count = 0
                    for file_path in files_to_remove:
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except OSError:
                            pass
                    
                    if removed_count > 0:
                        logger.debug(f"[CLEANUP] Removed {removed_count} old files")
                        
                except Exception as e:
                    logger.debug(f"Cleanup sorting error: {e}")
                    
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        
        # Attente interruptible
        for _ in range(cleanup_interval * 10):  # Check toutes les 0.1s
            if not processing_active.is_set():
                break
            time.sleep(0.1)
    
    logger.info("[CLEANUP] Thread terminated")

def connection_monitor_thread():
    """Monitoring de connexion léger"""
    logger.info("[MONITOR] Connection monitoring started")
    
    last_frame_count = 0
    check_interval = 10  # Check toutes les 10 secondes
    
    while processing_active.is_set():
        time.sleep(check_interval)
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            errors = frame_stats['error_count']
            detections = frame_stats['detection_count']
        
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        
        if frame_diff == 0:
            logger.warning("[MONITOR] No new frames received")
            connection_stable.clear()
        else:
            connection_stable.set()
            avg_fps = frame_diff / check_interval
            detection_rate = (detections / max(current_frames, 1)) * 100
            
            logger.info(f"[STATS] Frames: {current_frames}, FPS: {avg_fps:.1f}, "
                       f"Detections: {detection_rate:.1f}%, Errors: {errors}")
    
    logger.info("[MONITOR] Connection monitoring terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux amélioré"""
    logger.info("[STOP] Stop signal received")
    processing_active.clear()

def main():
    """Fonction principale robuste"""
    # Configuration des signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("[DRONE] Starting Bebop 2 detection system")
    
    # Vérifications préliminaires
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"[ERROR] Images directory not found: {IMAGES_DIR}")
        return False
    
    bebop = None
    vision = None
    threads = []
    
    try:
        # Initialisation du drone
        bebop = Bebop()
        logger.info("[CONNECT] Connecting to Bebop 2...")
        
        # Tentatives de connexion avec retry intelligent
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                success = bebop.connect(CONNECTION_TIMEOUT)
                if success:
                    logger.info("[OK] Drone connection established")
                    break
                else:
                    logger.warning(f"[ERROR] Connection attempt {attempt + 1}/{max_attempts} failed")
                    if attempt < max_attempts - 1:
                        time.sleep(3)
            except Exception as e:
                logger.error(f"Connection error attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(3)
        else:
            logger.error("[ERROR] Failed to connect after multiple attempts")
            return False
        
        # Initialisation vision avec buffer optimisé
        vision = DroneVision(bebop, is_bebop=True, buffer_size=256)
        vision.set_user_callback_function(robust_vision_callback)
        
        # Démarrage des threads
        thread_functions = [
            ("Display", optimized_display_thread),
            ("Cleanup", efficient_cleanup_thread),
            ("Monitor", connection_monitor_thread)
        ]
        
        for name, func in thread_functions:
            thread = threading.Thread(target=func, name=name, daemon=True)
            thread.start()
            threads.append(thread)
            logger.info(f"[OK] {name} thread started")
        
        # Ouverture du flux vidéo avec retry
        logger.info("[VIDEO] Opening video stream...")
        
        video_attempts = 3
        for attempt in range(video_attempts):
            try:
                if vision.open_video():
                    logger.info("[OK] Video stream opened successfully")
                    break
                else:
                    logger.warning(f"[ERROR] Video opening failed (attempt {attempt + 1}/{video_attempts})")
                    time.sleep(3)
            except Exception as e:
                logger.error(f"Video opening error: {e}")
                time.sleep(3)
        else:
            logger.error("[ERROR] Failed to open video stream")
            return False
        
        # Messages d'information
        logger.info("[DETECT] Detection system active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot, Ctrl+C=Emergency stop")
        
        # Attente de stabilisation
        time.sleep(2)
        
        # Boucle principale non-bloquante
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérification fenêtre OpenCV
                try:
                    window_name = "Bebop 2 - Detection Gant Rouge/Orange [OPTIMIZED]"
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("[WINDOW] Window closed by user")
                        break
                except:
                    pass
                    
        except KeyboardInterrupt:
            logger.info("[STOP] Keyboard interrupt detected")
            
    except Exception as e:
        logger.error(f"[ERROR] Critical error: {e}")
        return False
        
    finally:
        logger.info("[REFRESH] Starting cleanup process...")
        processing_active.clear()
        
        # Délai pour permettre aux threads de se terminer proprement
        time.sleep(1)
        
        # Nettoyage vision
        if vision:
            try:
                vision.close_video()
                logger.info("[VIDEO] Video stream closed")
            except Exception as e:
                logger.warning(f"Video closing error: {e}")
        
        # Nettoyage drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("[CONNECT] Drone disconnected")
            except Exception as e:
                logger.warning(f"Disconnection error: {e}")
        
        # Nettoyage OpenCV
        cv2.destroyAllWindows()
        
        # Attendre les threads avec timeout
        for thread in threads:
            try:
                thread.join(timeout=3)
            except:
                pass
        
        logger.info("[OK] Cleanup completed - Program terminated properly")
        
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"[ERROR] Unhandled exception: {e}")
        sys.exit(1)