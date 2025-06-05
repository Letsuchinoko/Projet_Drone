import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from queue import Queue, Empty
import logging
import signal
import sys
from collections import deque

# Configuration optimisée
DISPLAY_FPS = 25
CONNECTION_TIMEOUT = 20
FRAME_TIMEOUT = 1.0

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
    'successful_updates': 0,
    'rejected_frames': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_no_disk.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def direct_frame_callback(args):
    """
    Callback qui intercepte directement les frames sans passer par les fichiers
    """
    global current_frame
    
    try:
        with stats_lock:
            frame_stats['callback_calls'] += 1
        
        # Méthodes d'extraction de frame selon la structure de args
        frame = None
        
        # Méthode 1: args est un objet avec attribut frame
        if hasattr(args, 'frame'):
            frame = args.frame
        
        # Méthode 2: args est un dict avec clé frame
        elif isinstance(args, dict):
            if 'frame' in args:
                frame = args['frame']
            elif 'image' in args:
                frame = args['image']
            elif 'data' in args:
                # Données brutes à décoder
                try:
                    if isinstance(args['data'], bytes):
                        nparr = np.frombuffer(args['data'], np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except:
                    frame = None
        
        # Méthode 3: args a un attribut data
        elif hasattr(args, 'data'):
            try:
                if isinstance(args.data, bytes):
                    nparr = np.frombuffer(args.data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                elif hasattr(args.data, 'shape'):  # Déjà un array numpy
                    frame = args.data
            except:
                frame = None
        
        # Méthode 4: args est directement une frame
        elif hasattr(args, 'shape'):  # C'est un array numpy
            frame = args
        
        # Validation et mise à jour
        if frame is not None and validate_frame(frame):
            with frame_lock:
                current_frame = frame.copy()
            
            with stats_lock:
                frame_stats['frame_count'] += 1
                frame_stats['successful_updates'] += 1
                frame_stats['last_frame_time'] = time.time()
            
            logger.debug(f"Frame updated directly: {frame.shape}")
        else:
            with stats_lock:
                frame_stats['rejected_frames'] += 1
            logger.debug("Frame rejected or None")
                
    except Exception as e:
        logger.debug(f"Direct callback error: {e}")
        with stats_lock:
            frame_stats['rejected_frames'] += 1

def validate_frame(frame):
    """Validation rapide d'une frame"""
    try:
        if frame is None or frame.size == 0:
            return False
        
        h, w = frame.shape[:2]
        if h < 240 or w < 320:
            return False
        
        # Test de corruption basique
        mean_val = np.mean(frame)
        if mean_val < 10 or mean_val > 245:
            return False
        
        # Test de variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.var(gray) < 100:
            return False
        
        return True
        
    except:
        return False

class SmartGloveDetector:
    """Détecteur de gants intelligent avec anti-faux positifs"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=8)
        self.min_area = 1000  # Augmenté pour éviter les petits objets
        self.max_area = 35000
        self.min_contour_points = 20  # Plus strict
        
        # Kernels morphologiques
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Stabilisation renforcée
        self.stable_detections = deque(maxlen=6)
        self.confidence_threshold = 4  # Plus strict
        
        # Anti-faux positifs
        self.last_detection_center = None
        self.max_movement = 80  # Movement max entre détections
        self.detection_cooldown = 0  # Cooldown après fausse détection
        
        # Cache optimisé
        self.last_frame_hash = None
        self.last_detection_result = None
        
    def detect_glove(self, frame):
        """Détection intelligente avec anti-faux positifs"""
        if frame is None:
            return frame, False
            
        try:
            # Cache hash pour éviter les recalculs
            frame_hash = hash(frame.tobytes()[::1000])  # Hash sparse pour vitesse
            
            if frame_hash == self.last_frame_hash and self.last_detection_result is not None:
                return self.last_detection_result
            
            original_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Redimensionnement intelligent
            scale_factor = 1.0
            if w > 600:
                scale_factor = 600.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement amélioré
            work_frame = cv2.bilateralFilter(work_frame, 9, 75, 75)  # Meilleur pour préserver les contours
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque couleur intelligent avec exclusion de peau
            mask = self._create_smart_mask(hsv)
            
            # Morphologie renforcée
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.erode(mask, self.kernel_erode, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection de contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_smart_contour(contours, work_frame.shape)
            
            # Validation temporelle et géométrique
            detected = self._validate_smart_detection(best_contour, scale_factor)
            
            # Gestion du cooldown
            if self.detection_cooldown > 0:
                self.detection_cooldown -= 1
                detected = False
            
            # Historique de stabilisation
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            self.detection_history.append(stable_detection)
            
            # Dessin si détection valide
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_smart_detection(original_frame, best_contour)
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Overlay informatif
            result_frame = self._add_smart_overlay(original_frame, stable_detection, mask)
            
            # Cache du résultat
            self.last_frame_hash = frame_hash
            self.last_detection_result = (result_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Smart detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _create_smart_mask(self, hsv):
        """Masque couleur intelligent avec exclusion de peau et objets courants"""
        try:
            h, w = hsv.shape[:2]
            
            # Exclusion de la peau (plusieurs teintes)
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
            
            # Masque orange pour gants (plus précis)
            orange_lower = np.array([8, 150, 150])  # Plus restrictif
            orange_upper = np.array([22, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Masque rouge pour gants (éviter rouge vif/saturé des objets)
            red_lower1 = np.array([0, 120, 120])  # Éviter rouge trop saturé
            red_upper1 = np.array([8, 200, 220])   # Limiter saturation et valeur
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([172, 120, 120])
            red_upper2 = np.array([180, 200, 220])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Combinaison gants
            mask_gants = cv2.bitwise_or(mask_orange, mask_red)
            
            # Exclusion des zones de peau élargie
            mask_skin_dilated = cv2.dilate(mask_skin, self.kernel_close, iterations=2)
            mask_final = cv2.bitwise_and(mask_gants, cv2.bitwise_not(mask_skin_dilated))
            
            # Exclusion des bords (éviter objets hors champ)
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 15
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Smart mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_smart_contour(self, contours, frame_shape):
        """Sélection intelligente avec scoring multicritères"""
        if not contours:
            return None
            
        try:
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
                
                # Filtres géométriques
                aspect_ratio = w_rect / float(h_rect)
                if not (0.4 <= aspect_ratio <= 2.5):  # Forme pas trop allongée
                    continue
                
                # Éviter les contours trop près des bords
                margin = 20
                if (x < margin or y < margin or 
                    (x + w_rect) > (w - margin) or 
                    (y + h_rect) > (h - margin)):
                    continue
                
                # Calcul de la solidité (forme compacte)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.5:  # Forme trop creuse
                        continue
                else:
                    continue
                
                # Calcul de la circularité
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.2:  # Forme trop irrégulière
                        continue
                
                # Test de position (préférer centre-bas)
                center_y = y + h_rect // 2
                position_bonus = 1.0
                if center_y > h * 0.3:  # Dans les 70% inférieurs
                    position_bonus = 1.2
                
                # Score composite
                area_score = min(area / 4000.0, 1.0)
                shape_score = min(solidity * 2, 1.0)
                circularity_score = min(circularity * 5, 1.0)
                
                total_score = area_score * shape_score * circularity_score * position_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Smart contour selection error: {e}")
            return None
    
    def _validate_smart_detection(self, contour, scale_factor):
        """Validation temporelle et cohérence spatiale"""
        if contour is None:
            self.last_detection_center = None
            return False
        
        try:
            # Calcul du centre
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return False
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            current_center = (cx, cy)
            
            # Validation de mouvement
            if self.last_detection_center is not None:
                distance = np.sqrt((cx - self.last_detection_center[0])**2 + 
                                 (cy - self.last_detection_center[1])**2)
                
                # Si mouvement trop important, c'est suspect
                if distance > self.max_movement:
                    self.detection_cooldown = 3  # Cooldown de 3 frames
                    return False
            
            self.last_detection_center = current_center
            return True
            
        except Exception as e:
            logger.debug(f"Smart validation error: {e}")
            return False
    
    def _draw_smart_detection(self, frame, contour):
        """Dessin amélioré avec informations détaillées"""
        try:
            # Contour principal
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Centre de masse avec croix
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
                
                # Croix pour meilleure visibilité
                cv2.line(frame, (cx-15, cy), (cx+15, cy), (255, 255, 255), 2)
                cv2.line(frame, (cx, cy-15), (cx, cy+15), (255, 255, 255), 2)
            
            # Informations détaillées
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Texte principal
            info_y = max(y - 15, 30)
            cv2.putText(frame, "GANT DETECTE", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Détails techniques
            details = f"A:{int(area)} Sol:{solidity:.2f}"
            cv2.putText(frame, details, (x, info_y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Smart drawing error: {e}")
    
    def _add_smart_overlay(self, frame, detected, mask=None):
        """Overlay intelligent avec diagnostics"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            status = "🟢 GANT DETECTE" if detected else "🔍 RECHERCHE GANT..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Statistiques détaillées
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
                callbacks = frame_stats['callback_calls']
                updates = frame_stats['successful_updates']
                rejected = frame_stats['rejected_frames']
                detection_rate = (detections / max(frames, 1)) * 100
                update_rate = (updates / max(callbacks, 1)) * 100
            
            # Ligne 1: Performance
            perf_text = f"Frames: {frames} | Det: {detection_rate:.1f}% | Err: {errors}"
            cv2.putText(frame, perf_text, (10, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Ligne 2: Callbacks
            callback_text = f"Callbacks: {callbacks} | Updates: {updates} ({update_rate:.1f}%) | Rejected: {rejected}"
            cv2.putText(frame, callback_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # Ligne 3: État détection
            confidence = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0
            cooldown_text = f"Confiance: {confidence:.1%} | Cooldown: {self.detection_cooldown}"
            cv2.putText(frame, cooldown_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Ligne 4: Historique
            history_symbols = ["●" if d else "○" for d in list(self.detection_history)[-15:]]
            history_text = "Historique: " + "".join(history_symbols)
            cv2.putText(frame, history_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Mode et timestamp
            cv2.putText(frame, "MODE: DIRECT RAM (NO DISK)", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Masque miniature amélioré
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (150, 100))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    mask_x, mask_y = w - 160, 50
                    frame[mask_y:mask_y+100, mask_x:mask_x+150] = mask_colored
                    
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+150, mask_y+100), (255, 255, 255), 2)
                    cv2.putText(frame, "Masque Anti-FP", (mask_x, mask_y + 115), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Smart overlay error: {e}")
            return frame

class NoSaveVision(DroneVision):
    """Version modifiée de DroneVision qui n'enregistre pas sur disque"""
    
    def __init__(self, bebop, is_bebop=True):
        super().__init__(bebop, is_bebop=is_bebop)
        # Désactiver l'enregistrement sur disque
        self.save_pictures = False
        
    def save_frame(self, frame):
        """Override pour désactiver l'enregistrement"""
        # Ne rien faire - pas d'enregistrement sur disque
        pass

def display_thread():
    """Thread d'affichage optimisé"""
    detector = SmartGloveDetector()
    logger.info("Display thread started")
    
    window_name = "Bebop 2 - Smart Detection (No Disk)"
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
                
                # Écran d'attente informatif
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Attente frames directes...", (180, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                with stats_lock:
                    callbacks = frame_stats['callback_calls']
                    updates = frame_stats['successful_updates']
                    rejected = frame_stats['rejected_frames']
                    update_rate = (updates / max(callbacks, 1)) * 100
                
                cv2.putText(blank_frame, f"Callbacks: {callbacks}", 
                           (240, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank_frame, f"Updates: {updates} ({update_rate:.1f}%)", 
                           (220, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
                cv2.putText(blank_frame, f"Rejected: {rejected}", 
                           (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                cv2.putText(blank_frame, f"No frame cycles: {no_frame_count}", 
                           (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
                
                cv2.imshow(window_name, blank_frame)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            # Reset du compteur si on a une frame
            no_frame_count = 0
            
            # Traitement de la détection intelligente
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
                # Reset complet
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                    frame_stats['callback_calls'] = 0
                    frame_stats['successful_updates'] = 0
                    frame_stats['rejected_frames'] = 0
                detector.detection_history.clear()
                detector.stable_detections.clear()
                detector.last_frame_hash = None
                detector.last_detection_result = None
                detector.last_detection_center = None
                detector.detection_cooldown = 0
                no_frame_count = 0
                logger.info("Complete system reset performed")
            elif key == ord('s'):
                screenshot_name = f"screenshot_smart_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
            elif key == ord('d'):
                # Debug: afficher statistiques détaillées
                with stats_lock:
                    stats = frame_stats.copy()
                logger.info(f"Debug stats: {stats}")
                logger.info(f"Detector state: cooldown={detector.detection_cooldown}, center={detector.last_detection_center}")
                
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def monitor_thread():
    """Thread de monitoring simple"""
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
            updates = frame_stats['successful_updates']
            rejected = frame_stats['rejected_frames']
            last_received_time = frame_stats['last_frame_time']
        
        frame_diff = current_frames - last_frame_count
        callback_diff = callbacks - last_callback_count
        last_frame_count = current_frames
        last_callback_count = callbacks
        
        time_since_last_frame = time.time() - last_received_time
        update_rate = (updates / max(callbacks, 1)) * 100
        
        if frame_diff > 0:
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            callback_fps = callback_diff / 5
            
            logger.info(f"MONITOR - Frames: {current_frames} (+{frame_diff}), FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}")
            logger.info(f"         Callbacks: {callbacks} (+{callback_diff}, {callback_fps:.1f}/s), "
                       f"Direct update: {update_rate:.1f}% ({updates}/{callbacks}), Rejected: {rejected}")
        else:
            logger.warning(f"No new frames - callbacks: {callbacks} (+{callback_diff}), "
                         f"direct update rate: {update_rate:.1f}%, last frame {time_since_last_frame:.1f}s ago")
    
    logger.info("Monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux"""
    logger.info(f"Signal {sig} received - initiating shutdown")
    processing_active.clear()

def main():
    """Fonction principale sans enregistrement disque"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Smart Detection System (No Disk Recording)")
    
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
        
        # Configuration de la vision SANS enregistrement
        vision = NoSaveVision(bebop, is_bebop=True)
        vision.set_user_callback_function(direct_frame_callback)
        
        # Démarrer les threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="Display")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
        
        threads = [display_thread_obj, monitor_thread_obj]
        
        for thread in threads:
            thread.start()
            time.sleep(0.1)
        
        logger.info("All threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream without disk recording...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        
        logger.info("Video stream opened successfully")
        logger.info("Smart detection system is now active (No disk recording)")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Complete reset, 's'=Screenshot, 'd'=Debug stats")
        
        # Boucle principale
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Smart Detection (No Disk)", cv2.WND_PROP_VISIBLE) < 1:
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