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

# === CONFIGURATION ===
DISPLAY_FPS = 5
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
MAX_QUEUE_SIZE = 2  # Réduit pour éviter l'accumulation
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_WAIT_TIME = 5.0  # Temps max d'attente pour une nouvelle image

# Variables globales
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
last_display_time = 0
processing_active = True
last_processed_file = None
frame_received = False

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameProcessor:
    """Classe pour gérer le traitement des frames en mémoire"""
    
    def __init__(self):
        self.frame_count = 0
    
    def detect_gant(self, image):
        """Détection du gant rouge/orange optimisée"""
        try:
            if image is None or image.size == 0:
                return None
                
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_h, img_w = image.shape[:2]

            # Plages HSV pour détecter rouge/orange
            lower_ranges = [
                np.array([0, 80, 40]),   # Rouge bas
                np.array([10, 100, 50]), # Orange
                np.array([170, 50, 40])  # Rouge haut
            ]
            
            upper_ranges = [
                np.array([10, 255, 255]),
                np.array([25, 255, 255]),
                np.array([180, 255, 255])
            ]

            # Création du masque combiné
            masks = [cv2.inRange(hsv, lower, upper) 
                    for lower, upper in zip(lower_ranges, upper_ranges)]
            
            mask = masks[0]
            for m in masks[1:]:
                mask = cv2.bitwise_or(mask, m)

            # Nettoyage morphologique
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Détection des contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = self._find_best_glove_contour(contours, img_w, img_h)
            
            if best_contour is not None:
                self._draw_detection(image, best_contour)
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection: {e}")
            return image
    
    def _find_best_glove_contour(self, contours, img_w, img_h):
        """Trouve le meilleur contour correspondant à un gant"""
        best_contour = None
        max_score = 0

        for contour in contours:
            try:
                area = cv2.contourArea(contour)
                if area < 600:
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                if center_y < img_h * 0.25 and img_w * 0.3 < center_x < img_w * 0.7:
                    continue
                if area > img_w * img_h * 0.6:
                    continue
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue

                aspect_ratio = w / float(h)
                solidity = float(area) / hull_area
                complexity = area / perimeter

                if not (0.25 <= aspect_ratio <= 2.5):
                    continue
                if solidity > 0.995:
                    continue
                if complexity > 35:
                    continue

                score = area * (1 - solidity) * complexity
                if score > max_score:
                    max_score = score
                    best_contour = contour
                    
            except Exception as e:
                continue

        return best_contour
    
    def _draw_detection(self, image, contour):
        """Dessine la détection sur l'image"""
        try:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image, "Gant detecte", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except:
            pass

def vision_callback(args):
    """Callback appelé pour chaque frame reçue du drone"""
    global last_display_time, last_processed_file, frame_received
    
    try:
        # Vérification du timing pour limiter à 5 FPS
        now = time.time()
        if now - last_display_time < DISPLAY_INTERVAL:
            return
        
        # Lire la dernière image du dossier
        raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
        if not raw_files:
            return
            
        # Trier par date de modification (plus rapide)
        try:
            latest_file = max(raw_files, key=os.path.getmtime)
        except (OSError, ValueError):
            return
        
        # Éviter de retraiter la même image
        if latest_file == last_processed_file:
            return
            
        # Charger l'image en mémoire avec gestion d'erreur robuste
        try:
            frame = cv2.imread(latest_file)
            if frame is not None and frame.size > 0:
                # Vider la queue si elle est pleine et ajouter la nouvelle frame
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except Empty:
                        break
                
                frame_queue.put_nowait(frame.copy())
                last_display_time = now
                last_processed_file = latest_file
                frame_received = True
                
        except Exception as e:
            # Image corrompue ou en cours d'écriture, on ignore
            logger.debug(f"Erreur lecture image {latest_file}: {e}")
            return
                
    except Exception as e:
        logger.warning(f"Erreur dans vision_callback: {e}")

def display_thread():
    """Thread dédié à l'affichage et au traitement des frames"""
    global processing_active, frame_received
    
    processor = FrameProcessor()
    logger.info("Thread d'affichage démarré")
    
    last_frame_time = time.time()
    no_frame_warning_shown = False
    
    while processing_active:
        try:
            # Récupération d'une frame avec timeout court
            try:
                frame = frame_queue.get(timeout=0.5)
                last_frame_time = time.time()
                no_frame_warning_shown = False
            except Empty:
                # Vérifier si on reçoit encore des frames
                if time.time() - last_frame_time > MAX_WAIT_TIME:
                    if not no_frame_warning_shown:
                        logger.warning("⚠️ Aucune nouvelle frame depuis 5 secondes")
                        no_frame_warning_shown = True
                continue
            
            if frame is not None:
                try:
                    # Redimensionnement pour l'affichage
                    height, width = frame.shape[:2]
                    if width > 800:
                        new_width = 800
                        new_height = int(height * new_width / width)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Détection du gant
                    processed_frame = processor.detect_gant(frame)
                    
                    if processed_frame is not None:
                        # Affichage avec gestion d'erreur
                        cv2.imshow("🧤 Détection Gant Rouge - Bebop 2", processed_frame)
                        
                        # Vérification de la touche 'q' pour quitter
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            logger.info("Arrêt demandé par l'utilisateur")
                            processing_active = False
                            break
                        elif key == 27:  # ESC
                            logger.info("Arrêt demandé par ESC")
                            processing_active = False
                            break
                            
                except Exception as e:
                    logger.error(f"Erreur affichage frame: {e}")
                    time.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Erreur dans display_thread: {e}")
            time.sleep(0.1)
    
    logger.info("Thread d'affichage terminé")

def cleanup_old_images():
    """Nettoie les anciennes images pour éviter l'accumulation"""
    while processing_active:
        try:
            raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            if len(raw_files) > 10:  # Garder seulement les 10 plus récentes
                try:
                    # Tri plus robuste
                    files_with_time = []
                    for f in raw_files:
                        try:
                            files_with_time.append((f, os.path.getmtime(f)))
                        except OSError:
                            continue
                    
                    if files_with_time:
                        files_with_time.sort(key=lambda x: x[1])
                        files_to_delete = files_with_time[:-10]
                        
                        for f, _ in files_to_delete:
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                                
                except Exception as e:
                    logger.debug(f"Erreur lors du tri des fichiers: {e}")
                        
        except Exception as e:
            logger.debug(f"Erreur lors du nettoyage: {e}")
            
        time.sleep(5)  # Nettoie toutes les 5 secondes

def monitor_connection():
    """Surveille la connexion et redémarre si nécessaire"""
    global processing_active, frame_received
    
    while processing_active:
        time.sleep(10)  # Vérification toutes les 10 secondes
        
        if not frame_received:
            logger.warning("⚠️ Aucune frame reçue, possible problème de connexion")
        else:
            frame_received = False  # Reset pour la prochaine vérification

def main():
    """Fonction principale"""
    global processing_active
    
    # Vérification du dossier d'images
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"❌ Le dossier d'images n'existe pas: {IMAGES_DIR}")
        return
    
    # Initialisation du drone
    bebop = Bebop()
    bebop.drone_ip = "192.168.42.1"
    
    logger.info("Connexion au drone Bebop 2...")
    
    try:
        if not bebop.connect(10):
            logger.error("❌ Impossible de se connecter au drone")
            return
        
        logger.info("✅ Connecté au drone")
        
        # Initialisation de la vision avec paramètres optimisés
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(vision_callback)
        
        # Démarrage des threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        display_thread_obj.start()
        
        cleanup_thread_obj = threading.Thread(target=cleanup_old_images, daemon=True)
        cleanup_thread_obj.start()
        
        monitor_thread_obj = threading.Thread(target=monitor_connection, daemon=True)
        monitor_thread_obj.start()
        
        # Ouverture du flux vidéo
        if vision.open_video():
            logger.info("🎥 Flux vidéo ouvert - Détection en cours...")
            logger.info("💡 Appuyez sur 'q' ou 'ESC' dans la fenêtre vidéo pour quitter")
            logger.info("💡 Ou utilisez Ctrl+C dans le terminal")
            
            # Attendre un peu pour que les premières images arrivent
            time.sleep(3)
            
            try:
                # Boucle principale avec gestion d'interruption
                while processing_active:
                    time.sleep(0.2)
                    
                    # Vérifier si la fenêtre OpenCV est fermée
                    if cv2.getWindowProperty("🧤 Détection Gant Rouge - Bebop 2", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Fenêtre fermée par l'utilisateur")
                        break
                    
            except KeyboardInterrupt:
                logger.info("⏹ Arrêt manuel détecté")
                
        else:
            logger.error("❌ Impossible d'ouvrir le flux vidéo")
            
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
        
    finally:
        logger.info("🔄 Nettoyage en cours...")
        processing_active = False
        
        # Attendre un peu que les threads se terminent
        time.sleep(1)
        
        # Nettoyage
        try:
            vision.close_video()
        except:
            pass
            
        try:
            bebop.disconnect()
        except:
            pass
            
        cv2.destroyAllWindows()
        logger.info("✅ Programme terminé proprement")

if __name__ == "__main__":
    main()