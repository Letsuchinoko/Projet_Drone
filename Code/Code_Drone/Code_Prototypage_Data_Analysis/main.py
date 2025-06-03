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

# === CONFIGURATION OPTIMISÉE ===
DISPLAY_FPS = 10  # Augmenté pour plus de fluidité
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
MAX_QUEUE_SIZE = 3  # Queue plus petite pour réduire la latence
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
MAX_WAIT_TIME = 3.0  # Temps réduit pour réagir plus vite
CONNECTION_TIMEOUT = 15  # Timeout de connexion augmenté
VISION_TIMEOUT = 20  # Timeout pour l'ouverture de la vision

# Variables globales thread-safe
frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processing_active = threading.Event()
processing_active.set()
connection_stable = threading.Event()
frame_stats = {
    'last_frame_time': time.time(),
    'frame_count': 0,
    'error_count': 0,
    'last_processed_file': None
}
stats_lock = threading.Lock()

# Configuration du logging améliorée
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_detection.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class FrameProcessor:
    """Classe optimisée pour le traitement des frames"""
    
    def __init__(self):
        self.frame_count = 0
        self.detection_history = deque(maxlen=10)  # Historique pour stabiliser
        self.last_detection_time = 0
        
        # Paramètres de détection optimisés
        self.min_area = 400  # Aire minimale réduite
        self.max_area_ratio = 0.4  # Ratio maximum de l'image
        
        # Kernels morphologiques pré-calculés
        self.small_kernel = np.ones((3, 3), np.uint8)
        self.medium_kernel = np.ones((5, 5), np.uint8)
        self.large_kernel = np.ones((7, 7), np.uint8)
    
    def detect_gant(self, image):
        """Détection du gant rouge/orange ultra-optimisée"""
        try:
            if image is None or image.size == 0:
                return None
            
            self.frame_count += 1
            start_time = time.time()
            
            # Redimensionnement intelligent pour optimiser les performances
            original_h, original_w = image.shape[:2]
            scale_factor = 1.0
            
            if original_w > 640:  # Réduire pour accélérer le traitement
                scale_factor = 640.0 / original_w
                new_w = int(original_w * scale_factor)
                new_h = int(original_h * scale_factor)
                working_image = cv2.resize(image, (new_w, new_h))
            else:
                working_image = image.copy()
            
            # Conversion HSV optimisée
            hsv = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
            img_h, img_w = working_image.shape[:2]

            # Masque rouge/orange optimisé avec des plages plus larges
            mask_combined = self._create_color_mask(hsv)
            
            # Nettoyage morphologique adaptatif
            mask_cleaned = self._clean_mask(mask_combined, img_w * img_h)
            
            # Détection des contours avec algorithme optimisé
            contours = self._find_contours(mask_cleaned)
            
            # Sélection du meilleur contour
            best_contour = self._select_best_contour(contours, img_w, img_h)
            
            # Mise à l'échelle du contour si nécessaire
            if best_contour is not None and scale_factor != 1.0:
                best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_detection(image, best_contour, scale_factor)
            elif best_contour is not None:
                self._draw_detection(image, best_contour)
            
            # Mise à jour des statistiques
            processing_time = time.time() - start_time
            self._update_detection_stats(best_contour is not None, processing_time)
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur détection frame {self.frame_count}: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return image
    
    def _create_color_mask(self, hsv):
        """Création optimisée du masque couleur"""
        # Plages HSV étendues pour une meilleure détection
        ranges = [
            # Rouge bas (0-15°)
            (np.array([0, 60, 30]), np.array([15, 255, 255])),
            # Orange (15-30°)
            (np.array([10, 80, 50]), np.array([30, 255, 255])),
            # Rouge haut (165-180°)
            (np.array([165, 60, 30]), np.array([180, 255, 255]))
        ]
        
        masks = []
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            masks.append(mask)
        
        # Combinaison optimisée des masques
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def _clean_mask(self, mask, image_area):
        """Nettoyage morphologique adaptatif"""
        # Choix du kernel basé sur la taille de l'image
        if image_area > 300000:  # Grande image
            kernel = self.large_kernel
        elif image_area > 100000:  # Image moyenne
            kernel = self.medium_kernel
        else:  # Petite image
            kernel = self.small_kernel
        
        # Nettoyage en deux passes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilatation légère pour combler les trous
        mask = cv2.dilate(mask, self.small_kernel, iterations=1)
        
        return mask
    
    def _find_contours(self, mask):
        """Détection optimisée des contours"""
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Pré-filtrage rapide par aire
        return [c for c in contours if cv2.contourArea(c) >= self.min_area]
    
    def _select_best_contour(self, contours, img_w, img_h):
        """Sélection intelligente du meilleur contour"""
        if not contours:
            return None
        
        max_area = img_w * img_h * self.max_area_ratio
        candidates = []
        
        for contour in contours:
            try:
                area = cv2.contourArea(contour)
                if area > max_area:
                    continue
                
                # Calculs géométriques
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2
                
                # Filtres de position (éviter le ciel et les bords)
                if center_y < img_h * 0.2:  # Trop haut
                    continue
                if center_x < img_w * 0.1 or center_x > img_w * 0.9:  # Trop sur les bords
                    continue
                
                # Calculs de forme
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                
                # Métriques de qualité
                aspect_ratio = w / float(h)
                solidity = area / hull_area
                compactness = (perimeter * perimeter) / area
                
                # Critères de sélection ajustés
                if not (0.3 <= aspect_ratio <= 3.0):
                    continue
                if solidity > 0.98:  # Trop rectangulaire
                    continue
                if compactness > 25:  # Trop complexe
                    continue
                
                # Score composite
                position_score = 1.0 - abs(center_x / img_w - 0.5)  # Favorise le centre
                shape_score = (1.0 - solidity) * 2  # Favorise les formes irrégulières
                size_score = min(area / 2000, 1.0)  # Favorise les tailles moyennes
                
                total_score = area * (position_score + shape_score + size_score)
                
                candidates.append((contour, total_score, area))
                
            except Exception:
                continue
        
        if not candidates:
            return None
        
        # Retourner le meilleur candidat
        return max(candidates, key=lambda x: x[1])[0]
    
    def _draw_detection(self, image, contour, scale_factor=1.0):
        """Affichage amélioré de la détection"""
        try:
            # Contour principal
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Centre de masse
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            
            # Texte informatif
            area = cv2.contourArea(contour)
            text = f"Gant detecte (A:{int(area)})"
            
            # Position du texte optimisée
            text_y = max(y - 10, 20)
            cv2.putText(image, text, (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(image, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        except Exception as e:
            logger.debug(f"Erreur affichage: {e}")
    
    def _update_detection_stats(self, detected, processing_time):
        """Mise à jour des statistiques de détection"""
        self.detection_history.append(detected)
        if detected:
            self.last_detection_time = time.time()
        
        # Log périodique des performances
        if self.frame_count % 100 == 0:
            detection_rate = sum(self.detection_history) / len(self.detection_history)
            logger.info(f"Frame {self.frame_count}: Détection {detection_rate:.1%}, "
                       f"Temps traitement: {processing_time:.3f}s")

def enhanced_vision_callback(args):
    """Callback optimisé pour la réception des frames"""
    global frame_stats
    
    try:
        # Limitation de fréquence plus intelligente
        now = time.time()
        with stats_lock:
            if now - frame_stats['last_frame_time'] < DISPLAY_INTERVAL * 0.8:
                return
        
        # Recherche du fichier le plus récent
        try:
            raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            if not raw_files:
                return
            
            # Tri optimisé par temps de modification
            latest_file = max(raw_files, key=lambda f: os.path.getmtime(f))
            
        except (OSError, ValueError) as e:
            logger.debug(f"Erreur accès fichiers: {e}")
            return
        
        # Éviter le retraitement
        with stats_lock:
            if latest_file == frame_stats['last_processed_file']:
                return
        
        # Chargement sécurisé de l'image
        try:
            # Vérifier que le fichier est complet
            file_size = os.path.getsize(latest_file)
            if file_size < 1000:  # Fichier trop petit, probablement incomplet
                return
            
            frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return
            
            # Vider la queue si nécessaire (mode LIFO pour réduire la latence)
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    break
            
            # Ajouter la nouvelle frame
            frame_queue.put_nowait(frame.copy())
            
            # Mise à jour des stats
            with stats_lock:
                frame_stats['last_frame_time'] = now
                frame_stats['last_processed_file'] = latest_file
                frame_stats['frame_count'] += 1
            
            # Marquer la connexion comme stable
            connection_stable.set()
            
        except Exception as e:
            logger.debug(f"Erreur chargement {latest_file}: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
                
    except Exception as e:
        logger.warning(f"Erreur vision_callback: {e}")

def display_thread():
    """Thread d'affichage haute performance"""
    processor = FrameProcessor()
    logger.info("🎬 Thread d'affichage démarré")
    
    window_name = "🧤 Détection Gant Rouge - Bebop 2 [OPTIMISÉ]"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    no_frame_count = 0
    max_no_frame = int(MAX_WAIT_TIME * 2)  # 2 checks par seconde
    
    while processing_active.is_set():
        try:
            # Récupération frame avec timeout court
            try:
                frame = frame_queue.get(timeout=0.5)
                no_frame_count = 0
            except Empty:
                no_frame_count += 1
                if no_frame_count >= max_no_frame:
                    logger.warning("⚠️ Aucune frame depuis plusieurs secondes")
                    no_frame_count = 0  # Reset pour éviter le spam
                continue
            
            if frame is not None:
                try:
                    # Redimensionnement pour l'affichage
                    display_frame = frame.copy()
                    height, width = display_frame.shape[:2]
                    
                    if width > 1024:  # Limitation taille d'affichage
                        scale = 1024.0 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    # Traitement de détection
                    processed_frame = processor.detect_gant(display_frame)
                    
                    if processed_frame is not None:
                        # Ajout d'informations de debug
                        info_text = f"FPS: {DISPLAY_FPS} | Frames: {frame_stats['frame_count']}"
                        cv2.putText(processed_frame, info_text, (10, processed_frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Affichage
                        cv2.imshow(window_name, processed_frame)
                        
                        # Gestion des touches
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' ou ESC
                            logger.info("🛑 Arrêt demandé par l'utilisateur")
                            processing_active.clear()
                            break
                        elif key == ord('r'):  # 'r' pour reset stats
                            with stats_lock:
                                frame_stats['frame_count'] = 0
                                frame_stats['error_count'] = 0
                            logger.info("📊 Statistiques remises à zéro")
                            
                except Exception as e:
                    logger.error(f"Erreur traitement frame: {e}")
                    time.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Erreur display_thread: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("🎬 Thread d'affichage terminé")

def cleanup_thread():
    """Thread de nettoyage optimisé"""
    logger.info("🧹 Thread de nettoyage démarré")
    
    while processing_active.is_set():
        try:
            raw_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            
            if len(raw_files) > 15:  # Garde les 15 plus récentes
                try:
                    # Tri par temps de modification
                    files_sorted = sorted(raw_files, key=os.path.getmtime, reverse=True)
                    files_to_delete = files_sorted[15:]  # Supprimer les plus anciennes
                    
                    deleted_count = 0
                    for file_path in files_to_delete:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except OSError:
                            pass
                    
                    if deleted_count > 0:
                        logger.debug(f"🧹 {deleted_count} fichiers nettoyés")
                        
                except Exception as e:
                    logger.debug(f"Erreur tri fichiers: {e}")
                    
        except Exception as e:
            logger.debug(f"Erreur nettoyage: {e}")
        
        # Nettoyage toutes les 3 secondes
        for _ in range(30):  # 30 * 0.1 = 3 secondes
            if not processing_active.is_set():
                break
            time.sleep(0.1)
    
    logger.info("🧹 Thread de nettoyage terminé")

def connection_monitor():
    """Monitoring de connexion amélioré"""
    logger.info("📡 Monitoring de connexion démarré")
    
    last_frame_count = 0
    stable_periods = 0
    
    while processing_active.is_set():
        time.sleep(5)  # Check toutes les 5 secondes
        
        with stats_lock:
            current_frame_count = frame_stats['frame_count']
            error_count = frame_stats['error_count']
        
        # Vérifier la progression des frames
        frame_diff = current_frame_count - last_frame_count
        last_frame_count = current_frame_count
        
        if frame_diff == 0:
            logger.warning("📡 Aucune nouvelle frame reçue")
            connection_stable.clear()
        else:
            stable_periods += 1
            connection_stable.set()
            
            # Log périodique des stats
            if stable_periods % 12 == 0:  # Toutes les minutes
                fps_estimate = frame_diff / 5.0
                logger.info(f"📊 Stats: {current_frame_count} frames, "
                           f"~{fps_estimate:.1f} FPS, {error_count} erreurs")
    
    logger.info("📡 Monitoring de connexion terminé")

def signal_handler(sig, frame):
    """Gestionnaire de signaux pour arrêt propre"""
    logger.info("🛑 Signal d'arrêt reçu")
    processing_active.clear()

def main():
    """Fonction principale optimisée"""
    # Installation du gestionnaire de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚁 Démarrage du système de détection Bebop 2")
    
    # Vérifications préliminaires
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"❌ Dossier d'images inexistant: {IMAGES_DIR}")
        return False
    
    # Variables pour le nettoyage
    bebop = None
    vision = None
    threads = []
    
    try:
        # Initialisation du drone avec paramètres optimisés
        bebop = Bebop()
        bebop.drone_ip = "192.168.42.1"
        
        logger.info("🔗 Connexion au Bebop 2...")
        
        # Connexion avec retry
        connection_attempts = 3
        for attempt in range(connection_attempts):
            try:
                if bebop.connect(CONNECTION_TIMEOUT):
                    logger.info("✅ Connexion drone établie")
                    break
                else:
                    logger.warning(f"❌ Tentative {attempt + 1}/{connection_attempts} échouée")
                    if attempt < connection_attempts - 1:
                        time.sleep(2)
            except Exception as e:
                logger.error(f"Erreur connexion tentative {attempt + 1}: {e}")
                if attempt < connection_attempts - 1:
                    time.sleep(2)
        else:
            logger.error("❌ Impossible de se connecter après plusieurs tentatives")
            return False
        
        # Initialisation de la vision
        vision = DroneVision(bebop, is_bebop=True, buffer_size=512)
        vision.set_user_callback_function(enhanced_vision_callback)
        
        # Démarrage des threads workers
        thread_configs = [
            ("🎬 Affichage", display_thread),
            ("🧹 Nettoyage", cleanup_thread),
            ("📡 Monitoring", connection_monitor)
        ]
        
        for name, target in thread_configs:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            threads.append(thread)
            logger.info(f"✅ {name} démarré")
        
        # Ouverture du flux vidéo
        logger.info("🎥 Ouverture du flux vidéo...")
        
        video_opened = False
        for attempt in range(3):
            try:
                if vision.open_video():
                    video_opened = True
                    logger.info("🎥 Flux vidéo ouvert avec succès")
                    break
                else:
                    logger.warning(f"❌ Ouverture vidéo échouée (tentative {attempt + 1}/3)")
                    time.sleep(2)
            except Exception as e:
                logger.error(f"Erreur ouverture vidéo: {e}")
                time.sleep(2)
        
        if not video_opened:
            logger.error("❌ Impossible d'ouvrir le flux vidéo")
            return False
        
        # Messages d'information
        logger.info("🎯 Détection active - Contrôles:")
        logger.info("   • 'q' ou ESC: Quitter")
        logger.info("   • 'r': Reset statistiques")
        logger.info("   • Ctrl+C: Arrêt d'urgence")
        
        # Attente du démarrage
        time.sleep(3)
        
        # Boucle principale non-bloquante
        try:
            while processing_active.is_set():
                time.sleep(0.5)
                
                # Vérification de la fenêtre OpenCV
                try:
                    if cv2.getWindowProperty("🧤 Détection Gant Rouge - Bebop 2 [OPTIMISÉ]", 
                                           cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("🪟 Fenêtre fermée")
                        break
                except:
                    pass  # Fenêtre pas encore créée
                    
        except KeyboardInterrupt:
            logger.info("⏹ Interruption clavier détectée")
            
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        return False
        
    finally:
        logger.info("🔄 Nettoyage final en cours...")
        processing_active.clear()
        
        # Attendre un peu que les threads se terminent
        time.sleep(2)
        
        # Nettoyage vision
        if vision:
            try:
                vision.close_video()
                logger.info("🎥 Flux vidéo fermé")
            except Exception as e:
                logger.warning(f"Erreur fermeture vidéo: {e}")
        
        # Nettoyage drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("🔗 Drone déconnecté")
            except Exception as e:
                logger.warning(f"Erreur déconnexion: {e}")
        
        # Nettoyage OpenCV
        cv2.destroyAllWindows()
        
        # Attendre les threads (timeout)
        for thread in threads:
            try:
                thread.join(timeout=2)
            except:
                pass
        
        logger.info("✅ Nettoyage terminé - Programme arrêté proprement")
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)