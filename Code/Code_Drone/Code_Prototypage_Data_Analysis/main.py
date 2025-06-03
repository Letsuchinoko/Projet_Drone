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

# === CONFIGURATION OPTIMISÉE ===
DISPLAY_FPS = 20
MAX_QUEUE_SIZE = 3
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
CONNECTION_TIMEOUT = 15
VISION_TIMEOUT = 20
MAX_WAIT_TIME = 2.0
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS

# Variables globales thread-safe
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processing_active = threading.Event()
processing_active.set()
connection_stable = threading.Event()
connection_stable.set()
frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'last_processed_file': None
}
stats_lock = threading.Lock()

# Configuration logging simplifiée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedGloveDetector:
    """Détecteur de gants amélioré avec filtres robustes"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=10)
        self.min_area = 800  # Augmenté pour éviter les faux positifs
        self.max_area = 50000  # Limite supérieure
        self.min_contour_points = 20  # Contour minimum pour être valide
        
        # Kernels pour morphologie
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 7), np.uint8)
        
        # Historique pour stabilité
        self.stable_detections = deque(maxlen=5)
        
    def detect_glove(self, frame):
        """Détection de gants avec filtres améliorés"""
        if frame is None:
            return frame, False
        
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        try:
            # Réduction de résolution pour traitement
            scale_factor = 1.0
            if w > 640:
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Lissage pour réduire le bruit
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            
            # Conversion HSV
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masques couleur améliorés pour rouge/orange
            # Rouge bas (0-15)
            mask_red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
            # Rouge haut (165-180)  
            mask_red2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
            # Orange (15-35)
            mask_orange = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
            
            # Combinaison des masques
            mask_combined = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_orange))
            
            # Nettoyage morphologique
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, self.kernel_open)
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection de contours
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sélection du meilleur contour
            best_contour = self._select_best_contour(contours, work_frame.shape)
            
            detected = best_contour is not None
            
            # Stabilisation des détections
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= 3  # Au moins 3/5 détections
            
            # Mise à jour historique
            self.detection_history.append(stable_detection)
            
            # Dessin sur l'image originale
            if stable_detection and best_contour is not None:
                # Redimensionnement du contour si nécessaire
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                
                self._draw_detection(original_frame, best_contour)
                
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Ajout des informations overlay
            result_frame = self._add_overlay(original_frame, stable_detection, mask_combined)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _select_best_contour(self, contours, frame_shape):
        """Sélection intelligente du meilleur contour"""
        if not contours:
            return None
        
        h, w = frame_shape[:2]
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrage par aire
            if area < self.min_area or area > self.max_area:
                continue
            
            # Filtrage par nombre de points
            if len(contour) < self.min_contour_points:
                continue
            
            # Rectangle englobant
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Filtres géométriques
            aspect_ratio = w_rect / float(h_rect)
            if not (0.3 <= aspect_ratio <= 3.0):  # Forme raisonnable
                continue
            
            # Éviter les détections en bordure d'image
            if x < 10 or y < 10 or (x + w_rect) > (w - 10) or (y + h_rect) > (h - 10):
                continue
            
            # Calcul de la convexité
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.5:  # Forme trop irrégulière
                    continue
            
            # Score basé sur l'aire et la position (éviter le haut de l'image)
            position_score = 1.0 if y > h * 0.2 else 0.5
            area_score = min(area / 5000.0, 1.0)  # Normalisation
            
            score = area_score * position_score
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        return best_contour
    
    def _draw_detection(self, frame, contour):
        """Dessin de la détection"""
        try:
            # Contour principal
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Centre de masse
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Informations
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT DETECTE", (x, max(y - 10, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Aire: {int(area)}", (x, max(y - 35, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            logger.debug(f"Drawing error: {e}")
    
    def _add_overlay(self, frame, detected, mask=None):
        """Ajout des informations overlay"""
        try:
            h, w = frame.shape[:2]
            
            # Statut de détection
            status = "GANT DETECTE" if detected else "RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Statistiques
            with stats_lock:
                detection_rate = (frame_stats['detection_count'] / max(frame_stats['frame_count'], 1)) * 100
                stats_text = f"Frames: {frame_stats['frame_count']} | Detections: {frame_stats['detection_count']} ({detection_rate:.1f}%)"
            
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Historique de détection
            history_text = "Historique: " + "".join(["●" if x else "○" for x in list(self.detection_history)[-10:]])
            cv2.putText(frame, history_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Affichage du masque en petit (debug)
            if mask is not None and mask.size > 0:
                mask_small = cv2.resize(mask, (160, 120))
                mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                frame[10:130, w-170:w-10] = mask_colored
                cv2.rectangle(frame, (w-170, 10), (w-10, 130), (255, 255, 255), 1)
                cv2.putText(frame, "Masque", (w-160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def vision_callback(args):
    """Callback robuste pour la vision"""
    try:
        # Recherche du fichier le plus récent
        import glob
        pattern = os.path.join(IMAGES_DIR, "image_*.png")
        files = glob.glob(pattern)
        
        if not files:
            return
        
        latest_file = max(files, key=os.path.getmtime)
        
        # Vérification de l'existence et de la taille
        if not os.path.exists(latest_file) or os.path.getsize(latest_file) < 1000:
            return
        
        # Petit délai pour s'assurer que l'écriture est terminée
        time.sleep(0.02)
        
        # Chargement de l'image
        frame = cv2.imread(latest_file)
        if frame is None:
            return
        
        # Validation de l'image
        h, w = frame.shape[:2]
        if h < 200 or w < 200:
            return
        
        # Ajout à la queue (non-bloquant)
        try:
            # Vider la queue si elle est pleine (éviter la latence)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            
            frame_queue.put_nowait(frame)
            
        except:
            pass  # Queue pleine, on ignore cette frame
        
        # Mise à jour des stats
        with stats_lock:
            frame_stats['frame_count'] += 1
            frame_stats['last_frame_time'] = time.time()
            
    except Exception as e:
        logger.debug(f"Vision callback error: {e}")

def display_thread():
    """Thread d'affichage principal"""
    detector = ImprovedGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Detection Gant Amelioree"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while processing_active.is_set():
        try:
            # Récupération de la frame
            try:
                frame = frame_queue.get(timeout=2.0)
            except Empty:
                logger.warning("No frames received for 2 seconds")
                continue
            
            if frame is None:
                continue
            
            # Traitement de détection
            processed_frame, detected = detector.detect_glove(frame)
            
            # Affichage FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_elapsed = time.time() - fps_start_time
                current_fps = 30 / fps_elapsed if fps_elapsed > 0 else 0
                logger.info(f"FPS: {current_fps:.1f}")
                fps_start_time = time.time()
            
            # Affichage
            cv2.imshow(window_name, processed_frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # ESC
                logger.info("User requested quit")
                processing_active.clear()
                break
            elif key == ord('r'):  # Reset stats
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                logger.info("Statistics reset")
            elif key == ord('s'):  # Screenshot
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def connection_monitor_thread():
    """Thread de monitoring de connexion"""
    logger.info("Connection monitor started")
    
    last_frame_count = 0
    check_interval = 10
    
    while processing_active.is_set():
        time.sleep(check_interval)
        
        with stats_lock:
            current_frames = frame_stats['frame_count']
            errors = frame_stats['error_count']
            detections = frame_stats['detection_count']
        
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        
        if frame_diff == 0:
            logger.warning("No new frames received in monitoring")
            connection_stable.clear()
        else:
            connection_stable.set()
            avg_fps = frame_diff / check_interval
            detection_rate = (detections / max(current_frames, 1)) * 100
            
            logger.info(f"STATS - Frames: {current_frames}, FPS: {avg_fps:.1f}, "
                       f"Detections: {detection_rate:.1f}%, Errors: {errors}")
    
    logger.info("Connection monitor terminated")

def cleanup_thread():
    """Thread de nettoyage des fichiers"""
    logger.info("Cleanup thread started")
    
    while processing_active.is_set():
        try:
            import glob
            files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            
            if len(files) > 15:  # Garder les 15 plus récents
                files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
                files_to_remove = files_sorted[15:]
                
                removed_count = 0
                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except OSError:
                        pass
                
                if removed_count > 0:
                    logger.debug(f"Cleaned up {removed_count} old files")
        
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        
        # Attente de 10 secondes
        for _ in range(100):
            if not processing_active.is_set():
                break
            time.sleep(0.1)
    
    logger.info("Cleanup thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info("Stop signal received")
    processing_active.clear()

def main():
    """Fonction principale"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 glove detection system")
    
    # Vérifications
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
        
        # Initialisation de la vision
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(vision_callback)
        
        # Démarrage des threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        cleanup_thread_obj = threading.Thread(target=cleanup_thread, daemon=True)
        monitor_thread_obj = threading.Thread(target=connection_monitor_thread, daemon=True)
        
        threads = [display_thread_obj, cleanup_thread_obj, monitor_thread_obj]
        
        for thread in threads:
            thread.start()
        
        logger.info("All threads started (display, cleanup, monitor)")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        logger.info("Video stream opened successfully")
        logger.info("Detection system active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot")
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérification de l'état de la fenêtre
                try:
                    if cv2.getWindowProperty("Bebop 2 - Detection Gant Amelioree", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed")
                        break
                except:
                    pass
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return False
        
    finally:
        logger.info("Starting cleanup...")
        processing_active.clear()
        
        # Attendre un peu pour que les threads se terminent
        time.sleep(2)
        
        # Fermeture de la vision
        if vision:
            try:
                vision.close_video()
                logger.info("Video stream closed")
            except:
                pass
        
        # Déconnexion du drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("Drone disconnected")
            except:
                pass
        
        # Nettoyage OpenCV
        cv2.destroyAllWindows()
        
        # Attendre les threads
        for thread in threads:
            try:
                thread.join(timeout=3)
            except:
                pass
        
        logger.info("Cleanup completed")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)