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
MAX_QUEUE_SIZE = 3
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"

# Variables globales
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
last_display_time = 0
processing_active = True
last_processed_file = None

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
                logger.warning(f"Erreur lors de l'analyse du contour: {e}")
                continue

        return best_contour
    
    def _draw_detection(self, image, contour):
        """Dessine la détection sur l'image"""
        try:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image, "Gant detecte", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            logger.warning(f"Erreur lors du dessin: {e}")

def vision_callback(args):
    """Callback appelé pour chaque frame reçue du drone"""
    global last_display_time, last_processed_file
    
    try:
        # Vérification du timing pour limiter à 5 FPS
        now = time.time()
        if now - last_display_time < DISPLAY_INTERVAL:
            return
        
        # Méthode alternative : lire la dernière image du dossier mais la charger en mémoire
        raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
        if not raw_files:
            return
            
        # Trier par date de modification
        fichiers = []
        for f in raw_files:
            try:
                fichiers.append((f, os.path.getmtime(f)))
            except FileNotFoundError:
                continue

        if not fichiers:
            return
            
        fichiers = [f[0] for f in sorted(fichiers, key=lambda x: x[1])]
        latest_file = fichiers[-1]
        
        # Éviter de retraiter la même image
        if latest_file == last_processed_file:
            return
            
        # Charger l'image en mémoire
        frame = cv2.imread(latest_file)
        if frame is not None and frame.size > 0:
            try:
                # Ajouter la frame à la queue (non-bloquant)
                frame_queue.put_nowait(frame.copy())
                last_display_time = now
                last_processed_file = latest_file
                logger.debug(f"Frame ajoutée à la queue: {latest_file}")
            except:
                # Queue pleine, on ignore cette frame
                pass
                
    except Exception as e:
        logger.error(f"Erreur dans vision_callback: {e}")

def display_thread():
    """Thread dédié à l'affichage et au traitement des frames"""
    global processing_active
    
    processor = FrameProcessor()
    logger.info("Thread d'affichage démarré")
    
    while processing_active:
        try:
            # Récupération d'une frame avec timeout
            frame = frame_queue.get(timeout=1.0)
            
            if frame is not None:
                logger.debug("Traitement d'une frame")
                
                # Redimensionnement pour l'affichage
                height, width = frame.shape[:2]
                if width > 800:
                    new_width = 800
                    new_height = int(height * new_width / width)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Détection du gant
                processed_frame = processor.detect_gant(frame)
                
                if processed_frame is not None:
                    # Affichage
                    cv2.imshow("🧤 Détection Gant Rouge - Direct Stream", processed_frame)
                    
                    # Vérification de la touche 'q' pour quitter
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Arrêt demandé par l'utilisateur")
                        processing_active = False
                        break
                        
        except Empty:
            # Timeout normal, on continue
            logger.debug("Timeout - pas de nouvelle frame")
            continue
        except Exception as e:
            logger.error(f"Erreur dans display_thread: {e}")
            time.sleep(0.1)
    
    logger.info("Thread d'affichage terminé")

def cleanup_old_images():
    """Nettoie les anciennes images pour éviter l'accumulation"""
    while processing_active:
        try:
            raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            if len(raw_files) > 15:  # Garder seulement les 15 plus récentes
                fichiers = []
                for f in raw_files:
                    try:
                        fichiers.append((f, os.path.getmtime(f)))
                    except FileNotFoundError:
                        continue
                
                fichiers = [f[0] for f in sorted(fichiers, key=lambda x: x[1])]
                
                # Supprimer les plus anciennes
                for f in fichiers[:-15]:
                    try:
                        os.remove(f)
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage: {e}")
            
        time.sleep(10)  # Nettoie toutes les 10 secondes

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
        
        # Initialisation de la vision
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(vision_callback)
        
        # Démarrage du thread d'affichage
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        display_thread_obj.start()
        
        # Démarrage du thread de nettoyage
        cleanup_thread_obj = threading.Thread(target=cleanup_old_images, daemon=True)
        cleanup_thread_obj.start()
        
        # Ouverture du flux vidéo
        if vision.open_video():
            logger.info("🎥 Flux vidéo ouvert - Détection en cours...")
            logger.info("Appuyez sur 'q' dans la fenêtre vidéo pour quitter")
            
            # Attendre un peu pour que les premières images arrivent
            time.sleep(2)
            
            try:
                # Boucle principale
                while processing_active:
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                logger.info("⏹ Arrêt manuel détecté")
                
        else:
            logger.error("❌ Impossible d'ouvrir le flux vidéo")
            
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
        
    finally:
        logger.info("🔄 Nettoyage en cours...")
        processing_active = False
        
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