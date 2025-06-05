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
CONNECTION_TIMEOUT = 20
FRAME_TIMEOUT = 1.0
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"

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
    'successful_reads': 0,
    'failed_reads': 0,
    'last_file_processed': None
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_optimized_fallback.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def smart_vision_callback(args):
    """
    Callback qui se contente de déclencher une lecture intelligente de fichier
    """
    global current_frame
    
    try:
        with stats_lock:
            frame_stats['callback_calls'] += 1
        
        # Déclencher une lecture de fichier optimisée
        frame = read_latest_image_smart()
        
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
            
            with stats_lock:
                frame_stats['frame_count'] += 1
                frame_stats['last_frame_time'] = time.time()
                frame_stats['successful_reads'] += 1
        else:
            with stats_lock:
                frame_stats['failed_reads'] += 1
                
    except Exception as e:
        logger.debug(f"Callback error: {e}")
        with stats_lock:
            frame_stats['failed_reads'] += 1

def read_latest_image_smart():
    """
    Lecture intelligente et robuste du fichier image le plus récent
    """
    try:
        if not os.path.exists(IMAGES_DIR):
            return None
        
        # Scanner les fichiers avec pattern optimisé
        pattern = os.path.join(IMAGES_DIR, "image_*.png")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Trier par temps de modification (plus récent en premier)
        files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        
        current_time = time.time()
        
        # Essayer les 3 fichiers les plus récents
        for latest_file in files[:3]:
            try:
                # Informations sur le fichier
                stat_info = os.stat(latest_file)
                file_size = stat_info.st_size
                file_mtime = stat_info.st_mtime
                filename = os.path.basename(latest_file)
                
                # Filtres de sécurité
                if file_size < 3000:  # Trop petit
                    continue
                    
                if (current_time - file_mtime) < 0.02:  # Trop récent (en cours d'écriture)
                    continue
                
                # Éviter de relire le même fichier
                with stats_lock:
                    if frame_stats['last_file_processed'] == filename:
                        continue
                
                # Lecture avec plusieurs tentatives
                frame = None
                for attempt in range(3):
                    try:
                        frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                        if frame is not None and frame.size > 0:
                            break
                        time.sleep(0.005)  # Petite pause avant retry
                    except Exception as e:
                        logger.debug(f"Read attempt {attempt+1} failed: {e}")
                        if attempt < 2:
                            time.sleep(0.01)
                        continue
                
                if frame is None:
                    continue
                
                # Validations étendues
                h, w = frame.shape[:2]
                if h < 240 or w < 320:
                    continue
                
                # Test de corruption
                mean_val = np.mean(frame)
                if mean_val < 10 or mean_val > 245:
                    continue
                
                # Test de variance (éviter images uniformes)
                gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if np.var(gray_test) < 150:
                    continue
                
                # Test de pixels non-nuls
                if np.count_nonzero(frame) < (frame.size * 0.05):
                    continue
                
                # Frame valide trouvée !
                with stats_lock:
                    frame_stats['last_file_processed'] = filename
                
                logger.debug(f"Frame loaded: {filename} ({w}x{h}, {file_size} bytes)")
                return frame
                
            except (OSError, IOError) as e:
                logger.debug(f"File access error for {latest_file}: {e}")
                continue
            except Exception as e:
                logger.debug(f"File processing error for {latest_file}: {e}")
                continue
        
        # Aucun fichier valide trouvé
        return None
        
    except Exception as e:
        logger.debug(f"Smart read critical error: {e}")
        return None

class FastGloveDetector:
    """Détecteur de gants ultra-optimisé"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=8)
        self.min_area = 800
        self.max_area = 40000
        
        # Kernels morphologiques
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Stabilisation
        self.stable_detections = deque(maxlen=4)
        self.confidence_threshold = 2
        
        # Cache pour éviter les recalculs
        self.last_frame_hash = None
        self.last_detection_result = None
        
    def detect_glove(self, frame):
        """Détection ultra-rapide avec cache"""
        if frame is None:
            return frame, False
            
        try:
            # Hash de la frame pour détecter les changements
            frame_hash = hash(frame.tobytes()[:10000])  # Hash partiel pour la vitesse
            
            # Si c'est la même frame, retourner le cache
            if frame_hash == self.last_frame_hash and self.last_detection_result is not None:
                return self.last_detection_result
            
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement agressif pour les performances
            scale_factor = 1.0
            target_width = 480  # Encore plus petit
            if w > target_width:
                scale_factor = target_width / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement minimal
            work_frame = cv2.GaussianBlur(work_frame, (3, 3), 0)  # Blur plus léger
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur simplifié
            mask = self._create_fast_mask(hsv)
            
            # Morphologie minimale
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_contour_fast(contours)
            
            # Validation simple
            detected = best_contour is not None
            
            # Historique court
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            self.detection_history.append(stable_detection)
            
            # Dessin si détection
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_simple(original_frame, best_contour)
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Overlay minimal
            result_frame = self._add_simple_overlay(original_frame, stable_detection)
            
            # Cache du résultat
            self.last_frame_hash = frame_hash
            self.last_detection_result = (result_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _create_fast_mask(self, hsv):
        """Masque couleur ultra-rapide"""
        try:
            # Orange simplifié
            orange_lower = np.array([8, 100, 100])
            orange_upper = np.array([25, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge simplifié (une seule plage)
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
            
            red_lower2 = np.array([170, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            return cv2.bitwise_or(mask_orange, mask_red)
            
        except Exception as e:
            logger.debug(f"Mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_contour_fast(self, contours):
        """Sélection rapide du meilleur contour"""
        if not contours:
            return None
            
        try:
            # Prendre simplement le contour avec la plus grande aire valide
            best_contour = None
            best_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area <= area <= self.max_area and area > best_area:
                    best_area = area
                    best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Contour selection error: {e}")
            return None
    
    def _draw_simple(self, frame, contour):
        """Dessin simple et rapide"""
        try:
            # Contour simple
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Texte simple
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT ({int(area)})", (x, max(y - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_simple_overlay(self, frame, detected):
        """Overlay simplifié et rapide"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 DETECTE" if detected else "🔍 RECHERCHE"
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Statistiques essentielles
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                callbacks = frame_stats['callback_calls']
                success_reads = frame_stats['successful_reads']
                failed_reads = frame_stats['failed_reads']
                detection_rate = (detections / max(frames, 1)) * 100
                read_success_rate = (success_reads / max(callbacks, 1)) * 100
            
            # Ligne 1: Stats de base
            stats_text = f"Frames: {frames} | Det: {detection_rate:.1f}%"
            cv2.putText(frame, stats_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Ligne 2: Stats de lecture
            read_text = f"Callbacks: {callbacks} | Reads: {success_reads}/{success_reads + failed_reads} ({read_success_rate:.1f}%)"
            cv2.putText(frame, read_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            
            # Ligne 3: Historique minimal
            history = ["●" if d else "○" for d in list(self.detection_history)[-10:]]
            history_text = "Hist: " + "".join(history)
            cv2.putText(frame, history_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def display_thread():
    """Thread d'affichage optimisé"""
    detector = FastGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Optimized Fallback Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    last_display_time = time.time()
    
    no_frame_count = 0
    
    while processing_active.is_set():
        try:
            current_time = time.time()
            
            # Limiter le FPS d'affichage
            if (current_time - last_display_time) < (1.0 / DISPLAY_FPS):
                time.sleep(0.01)
                continue
            
            last_display_time = current_time
            
            # Récupérer la frame actuelle
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = None
            
            if frame is None:
                no_frame_count += 1
                
                # Écran d'attente avec plus d'infos
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Attente des frames...", (200, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                with stats_lock:
                    callbacks = frame_stats['callback_calls']
                    success_reads = frame_stats['successful_reads']
                    failed_reads = frame_stats['failed_reads']
                    total_reads = success_reads + failed_reads
                    success_rate = (success_reads / max(total_reads, 1)) * 100
                
                cv2.putText(blank_frame, f"Callbacks: {callbacks}", 
                           (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank_frame, f"Lectures reussies: {success_reads}/{total_reads} ({success_rate:.1f}%)", 
                           (160, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank_frame, f"Tentatives sans frame: {no_frame_count}", 
                           (190, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                
                cv2.imshow(window_name, blank_frame)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            # Reset du compteur si on a une frame
            no_frame_count = 0
            
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
                    frame_stats['successful_reads'] = 0
                    frame_stats['failed_reads'] = 0
                    frame_stats['last_file_processed'] = None
                detector.detection_history.clear()
                detector.stable_detections.clear()
                detector.last_frame_hash = None
                detector.last_detection_result = None
                no_frame_count = 0
                logger.info("Complete reset performed")
            elif key == ord('s'):
                screenshot_name = f"screenshot_fallback_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def monitor_thread():
    """Thread de monitoring optimisé"""
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
            success_reads = frame_stats['successful_reads']
            failed_reads = frame_stats['failed_reads']
            last_received_time = frame_stats['last_frame_time']
        
        frame_diff = current_frames - last_frame_count
        callback_diff = callbacks - last_callback_count
        last_frame_count = current_frames
        last_callback_count = callbacks
        
        time_since_last_frame = time.time() - last_received_time
        total_reads = success_reads + failed_reads
        read_success_rate = (success_reads / max(total_reads, 1)) * 100
        
        if frame_diff > 0:
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            callback_fps = callback_diff / 5
            
            logger.info(f"MONITOR - Frames: {current_frames} (+{frame_diff}), FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}")
            logger.info(f"         Callbacks: {callbacks} (+{callback_diff}, {callback_fps:.1f}/s), "
                       f"Read success: {read_success_rate:.1f}% ({success_reads}/{total_reads})")
        else:
            logger.warning(f"No new frames - callbacks: {callbacks} (+{callback_diff}), "
                         f"read success: {read_success_rate:.1f}%, last frame {time_since_last_frame:.1f}s ago")
    
    logger.info("Monitor thread terminated")

def cleanup_files_gentle():
    """Nettoyage très doux des fichiers anciens"""
    try:
        if not os.path.exists(IMAGES_DIR):
            return
        
        files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
        
        if len(files) > 150:  # Seuil très élevé
            current_time = time.time()
            old_files = []
            
            for file_path in files:
                try:
                    file_mtime = os.path.getmtime(file_path)
                    # Supprimer seulement les fichiers de plus de 3 minutes
                    if (current_time - file_mtime) > 180:
                        old_files.append(file_path)
                except:
                    continue
            
            # Supprimer seulement la moitié des anciens fichiers
            files_to_remove = old_files[:len(old_files)//2]
            removed_count = 0
            
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    removed_count += 1
                    time.sleep(0.001)  # Pause entre suppressions
                except:
                    continue
            
            if removed_count > 0:
                logger.info(f"Gentle cleanup: removed {removed_count} very old files")
                        
    except Exception as e:
        logger.debug(f"Cleanup error: {e}")

def cleanup_thread():
    """Thread de nettoyage ultra-conservateur"""
    logger.info("Gentle cleanup thread started")
    
    while processing_active.is_set():
        try:
            cleanup_files_gentle()
        except Exception as e:
            logger.debug(f"Cleanup thread error: {e}")
        
        # Attente de 2 minutes entre nettoyages
        for _ in range(120):
            if not processing_active.is_set():
                break
            time.sleep(1)
    
    logger.info("Cleanup thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale avec fallback optimisé"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Optimized Fallback Detection System")
    
    bebop = None
    vision = None
    threads = []
    
    try:
        # Vérification du répertoire d'images
        if not os.path.exists(IMAGES_DIR):
            logger.error(f"Images directory not found: {IMAGES_DIR}")
            return False
        
        # Connexion au drone
        bebop = Bebop()
        logger.info("Connecting to Bebop 2...")
        
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            return False
        
        logger.info("Drone connected successfully")
        
        # Configuration de la vision avec callback optimisé
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(smart_vision_callback)
        
        # Démarrer les threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        cleanup_thread_obj = threading.Thread(target=cleanup_thread, daemon=True, name="Cleanup")
        
        threads = [display_thread_obj, monitor_thread_obj, cleanup_thread_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.1)
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream with optimized fallback...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        logger.info("Video stream opened successfully")
        logger.info("Optimized fallback detection system is now active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Complete reset, 's'=Screenshot")
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Optimized Fallback Detection", cv2.WND_PROP_VISIBLE) < 1:
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