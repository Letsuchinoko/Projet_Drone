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
    'last_file_processed': '',
    'files_cleaned': 0
}
stats_lock = threading.Lock()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_hybrid_final.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def hybrid_callback(args):
    """
    Callback hybride : déclenche une lecture fichier + nettoyage immédiat
    """
    global current_frame
    
    try:
        with stats_lock:
            frame_stats['callback_calls'] += 1
        
        # Lecture immédiate du fichier le plus récent
        frame = read_and_cleanup_latest()
        
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
                
    except Exception as e:
        logger.debug(f"Hybrid callback error: {e}")
        with stats_lock:
            frame_stats['failed_reads'] += 1

def read_and_cleanup_latest():
    """
    Lit le fichier le plus récent ET nettoie immédiatement les anciens
    """
    try:
        if not os.path.exists(IMAGES_DIR):
            return None
        
        # Scanner TOUS les fichiers image
        pattern = os.path.join(IMAGES_DIR, "image_*.png")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Trier par date de modification
        files.sort(key=os.path.getmtime, reverse=True)
        
        current_time = time.time()
        frame = None
        
        # Essayer de lire le fichier le plus récent
        for i, latest_file in enumerate(files[:2]):  # Essayer les 2 plus récents
            try:
                stat_info = os.stat(latest_file)
                file_size = stat_info.st_size
                file_mtime = stat_info.st_mtime
                filename = os.path.basename(latest_file)
                
                # Filtres de sécurité
                if file_size < 3000:
                    continue
                    
                if (current_time - file_mtime) < 0.03:  # 30ms de sécurité
                    continue
                
                # Éviter de relire le même fichier
                with stats_lock:
                    if frame_stats['last_file_processed'] == filename:
                        if i == 0:  # Si c'est le plus récent, continuer au suivant
                            continue
                
                # Lecture avec retry
                for attempt in range(2):
                    try:
                        frame = cv2.imread(latest_file, cv2.IMREAD_COLOR)
                        if frame is not None and frame.size > 0:
                            # Validation rapide
                            h, w = frame.shape[:2]
                            if h >= 240 and w >= 320:
                                mean_val = np.mean(frame)
                                if 10 <= mean_val <= 245:
                                    # Frame valide !
                                    with stats_lock:
                                        frame_stats['last_file_processed'] = filename
                                    logger.debug(f"Frame read: {filename} ({w}x{h})")
                                    break
                        frame = None
                        time.sleep(0.01)
                    except:
                        frame = None
                        if attempt == 0:
                            time.sleep(0.02)
                        continue
                
                if frame is not None:
                    break
                    
            except Exception as e:
                logger.debug(f"File read error: {e}")
                continue
        
        # NETTOYAGE IMMÉDIAT ET AGRESSIF des anciens fichiers
        cleanup_immediately(files)
        
        return frame
        
    except Exception as e:
        logger.debug(f"Read and cleanup error: {e}")
        return None

def cleanup_immediately(all_files):
    """
    Nettoyage immédiat et agressif : garde seulement les 3 plus récents
    """
    try:
        if len(all_files) <= 3:
            return
        
        # Garder seulement les 3 plus récents
        files_to_remove = all_files[3:]
        
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
            logger.debug(f"Immediate cleanup: removed {removed_count} files")
                    
    except Exception as e:
        logger.debug(f"Immediate cleanup error: {e}")

class UltraSmartGloveDetector:
    """Détecteur de gants ultra-intelligent"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=6)
        self.min_area = 1200  # Plus strict
        self.max_area = 30000
        self.min_contour_points = 25  # Très strict
        
        # Kernels morphologiques optimisés
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Stabilisation ultra-stricte
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 4  # 4/5 détections
        
        # Anti-faux positifs renforcé
        self.last_detection_center = None
        self.max_movement = 60  # Très strict
        self.detection_cooldown = 0
        self.min_stability_frames = 10  # Minimum de frames stables avant première détection
        self.frame_counter = 0
        
        # Cache optimisé
        self.last_frame_hash = None
        self.last_detection_result = None
        
    def detect_glove(self, frame):
        """Détection ultra-intelligente"""
        if frame is None:
            return frame, False
            
        try:
            self.frame_counter += 1
            
            # Cache pour éviter recalculs
            frame_hash = hash(frame.tobytes()[::1500])
            
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
            
            # Prétraitement de qualité
            work_frame = cv2.bilateralFilter(work_frame, 11, 80, 80)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Masque ultra-intelligent
            mask = self._create_ultra_smart_mask(hsv)
            
            # Morphologie renforcée
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.erode(mask, self.kernel_erode, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection avec scoring avancé
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_ultra_smart_contour(contours, work_frame.shape)
            
            # Validation multi-niveaux
            detected = self._validate_ultra_smart(best_contour, scale_factor)
            
            # Cooldown et stabilité
            if self.detection_cooldown > 0:
                self.detection_cooldown -= 1
                detected = False
            
            # Période de stabilisation initiale
            if self.frame_counter < self.min_stability_frames:
                detected = False
            
            # Historique de stabilisation
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            self.detection_history.append(stable_detection)
            
            # Dessin sophistiqué
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_ultra_detection(original_frame, best_contour)
                with stats_lock:
                    frame_stats['detection_count'] += 1
            
            # Overlay complet
            result_frame = self._add_ultra_overlay(original_frame, stable_detection, mask)
            
            # Cache
            self.last_frame_hash = frame_hash
            self.last_detection_result = (result_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Ultra detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False
    
    def _create_ultra_smart_mask(self, hsv):
        """Masque ultra-intelligent avec exclusions multiples"""
        try:
            h, w = hsv.shape[:2]
            
            # === EXCLUSIONS ===
            
            # Exclusion peau (3 variations)
            skin_masks = []
            
            # Peau claire
            skin_lower1 = np.array([0, 15, 60])
            skin_upper1 = np.array([28, 120, 255])
            skin_masks.append(cv2.inRange(hsv, skin_lower1, skin_upper1))
            
            # Peau medium
            skin_lower2 = np.array([0, 25, 45])
            skin_upper2 = np.array([20, 100, 200])
            skin_masks.append(cv2.inRange(hsv, skin_lower2, skin_upper2))
            
            # Peau foncée
            skin_lower3 = np.array([0, 30, 30])
            skin_upper3 = np.array([15, 80, 150])
            skin_masks.append(cv2.inRange(hsv, skin_lower3, skin_upper3))
            
            mask_skin = cv2.bitwise_or(skin_masks[0], cv2.bitwise_or(skin_masks[1], skin_masks[2]))
            
            # Exclusion rouge vif (objets, logos, etc.)
            red_bright_lower = np.array([0, 200, 200])
            red_bright_upper = np.array([10, 255, 255])
            mask_red_bright1 = cv2.inRange(hsv, red_bright_lower, red_bright_upper)
            
            red_bright_lower2 = np.array([170, 200, 200])
            red_bright_upper2 = np.array([180, 255, 255])
            mask_red_bright2 = cv2.inRange(hsv, red_bright_lower2, red_bright_upper2)
            
            mask_red_bright = cv2.bitwise_or(mask_red_bright1, mask_red_bright2)
            
            # === INCLUSIONS (GANTS) ===
            
            # Orange gants (très précis)
            orange_lower = np.array([10, 140, 140])
            orange_upper = np.array([24, 220, 240])  # Éviter trop saturé
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Rouge gants (modéré, pas vif)
            red_glove_lower1 = np.array([0, 100, 100])
            red_glove_upper1 = np.array([8, 190, 210])  # Éviter trop saturé
            mask_red_glove1 = cv2.inRange(hsv, red_glove_lower1, red_glove_upper1)
            
            red_glove_lower2 = np.array([172, 100, 100])
            red_glove_upper2 = np.array([180, 190, 210])
            mask_red_glove2 = cv2.inRange(hsv, red_glove_lower2, red_glove_upper2)
            
            mask_red_glove = cv2.bitwise_or(mask_red_glove1, mask_red_glove2)
            
            # Combinaison gants
            mask_gants = cv2.bitwise_or(mask_orange, mask_red_glove)
            
            # Application des exclusions
            mask_exclusions = cv2.bitwise_or(mask_skin, mask_red_bright)
            mask_exclusions_dilated = cv2.dilate(mask_exclusions, self.kernel_close, iterations=2)
            
            mask_final = cv2.bitwise_and(mask_gants, cv2.bitwise_not(mask_exclusions_dilated))
            
            # Exclusion des bords renforcée
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = 20
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Ultra mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    def _select_ultra_smart_contour(self, contours, frame_shape):
        """Sélection ultra-intelligente avec scoring complexe"""
        if not contours:
            return None
            
        try:
            h, w = frame_shape[:2]
            best_contour = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base stricts
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Géométrie de base
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Aspect ratio strict
                aspect_ratio = w_rect / float(h_rect)
                if not (0.5 <= aspect_ratio <= 2.0):  # Plus strict
                    continue
                
                # Éviter les bords avec marge importante
                margin = 25
                if (x < margin or y < margin or 
                    (x + w_rect) > (w - margin) or 
                    (y + h_rect) > (h - margin)):
                    continue
                
                # Calculs géométriques avancés
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                
                solidity = area / hull_area
                if solidity < 0.6:  # Très strict
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.25:  # Plus strict
                    continue
                
                # Test de compactité
                rect_area = w_rect * h_rect
                extent = area / rect_area
                if extent < 0.4:  # Forme doit remplir son rectangle
                    continue
                
                # Scoring multicritères avancé
                area_score = min(area / 3000.0, 1.0)
                solidity_score = min(solidity * 1.5, 1.0)
                circularity_score = min(circularity * 4, 1.0)
                extent_score = min(extent * 2, 1.0)
                
                # Bonus position (préférer centre-bas)
                center_y = y + h_rect // 2
                position_bonus = 1.0
                if center_y > h * 0.4:
                    position_bonus = 1.3
                
                # Bonus pour forme "gant-like" (légèrement plus large que haut)
                shape_bonus = 1.0
                if 1.0 <= aspect_ratio <= 1.6:
                    shape_bonus = 1.2
                
                # Score final
                total_score = (area_score * solidity_score * circularity_score * 
                             extent_score * position_bonus * shape_bonus)
                
                if total_score > best_score:
                    best_score = total_score
                    best_contour = contour
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"Ultra contour selection error: {e}")
            return None
    
    def _validate_ultra_smart(self, contour, scale_factor):
        """Validation ultra-stricte avec cohérence temporelle"""
        if contour is None:
            self.last_detection_center = None
            return False
        
        try:
            # Centre de masse
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return False
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            current_center = (cx, cy)
            
            # Validation de mouvement ultra-stricte
            if self.last_detection_center is not None:
                distance = np.sqrt((cx - self.last_detection_center[0])**2 + 
                                 (cy - self.last_detection_center[1])**2)
                
                # Mouvement trop important = faux positif
                if distance > self.max_movement:
                    self.detection_cooldown = 5  # Cooldown de 5 frames
                    logger.debug(f"Movement too large: {distance:.1f}px > {self.max_movement}px")
                    return False
            
            self.last_detection_center = current_center
            return True
            
        except Exception as e:
            logger.debug(f"Ultra validation error: {e}")
            return False
    
    def _draw_ultra_detection(self, frame, contour):
        """Dessin ultra-détaillé"""
        try:
            # Contour épais
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 4)
            
            # Rectangle avec coins arrondis visuels
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 3)
            
            # Centre avec croix elaborate
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Centre principal
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 3)
                
                # Croix de précision
                cv2.line(frame, (cx-20, cy), (cx+20, cy), (255, 255, 255), 3)
                cv2.line(frame, (cx, cy-20), (cx, cy+20), (255, 255, 255), 3)
            
            # Informations techniques détaillées
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Texte principal avec fond
            info_y = max(y - 20, 35)
            text = "GANT VALIDE DETECTE"
            
            # Fond pour le texte
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, info_y - text_size[1] - 5), 
                         (x + text_size[0] + 10, info_y + 5), (0, 0, 0), -1)
            
            cv2.putText(frame, text, (x + 5, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Détails techniques
            details = f"A:{int(area)} S:{solidity:.2f} C:{circularity:.2f}"
            cv2.putText(frame, details, (x, info_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Ultra drawing error: {e}")
    
    def _add_ultra_overlay(self, frame, detected, mask=None):
        """Overlay ultra-complet avec toutes les infos"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal avec animation
            if detected:
                status = "🟢 GANT DETECTE ✓"
                color = (0, 255, 0)
            else:
                status = "🔍 ANALYSE EN COURS..."
                color = (0, 255, 255)
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            
            # Statistiques complètes
            with stats_lock:
                frames = frame_stats['frame_count']
                detections = frame_stats['detection_count']
                errors = frame_stats['error_count']
                callbacks = frame_stats['callback_calls']
                success_reads = frame_stats['successful_reads']
                failed_reads = frame_stats['failed_reads']
                cleaned = frame_stats['files_cleaned']
                detection_rate = (detections / max(frames, 1)) * 100
                read_success_rate = (success_reads / max(callbacks, 1)) * 100
            
            # Performance système
            perf_text = f"Performance: {frames} frames | {detection_rate:.1f}% detections | {errors} erreurs"
            cv2.putText(frame, perf_text, (10, h - 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Callbacks et lecture
            io_text = f"I/O: {callbacks} callbacks | {read_success_rate:.1f}% lectures | {cleaned} fichiers nettoyes"
            cv2.putText(frame, io_text, (10, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            # État détecteur
            confidence = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0
            detector_text = f"Detecteur: {confidence:.1%} confiance | cooldown: {self.detection_cooldown} | frame: {self.frame_counter}"
            cv2.putText(frame, detector_text, (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            # Historique visuel étendu
            history_symbols = []
            for detection in list(self.detection_history)[-20:]:
                if detection:
                    history_symbols.append("●")
                else:
                    history_symbols.append("○")
            
            history_text = "Historique (20): " + "".join(history_symbols)
            cv2.putText(frame, history_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Stabilité actuelle
            stable_symbols = ["●" if d else "○" for d in list(self.stable_detections)]
            stable_text = f"Stabilite ({self.confidence_threshold}/{len(self.stable_detections)}): " + "".join(stable_symbols)
            cv2.putText(frame, stable_text, (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            
            # Mode et timestamp
            cv2.putText(frame, "MODE: HYBRIDE ULTRA-SMART (Minimal Disk)", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # FPS en temps réel
            fps_text = f"Traitement: {DISPLAY_FPS} FPS"
            cv2.putText(frame, fps_text, (w - 200, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Masque miniature ultra-détaillé
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (160, 120))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    # Position optimisée
                    mask_x, mask_y = w - 170, 60
                    frame[mask_y:mask_y+120, mask_x:mask_x+160] = mask_colored
                    
                    # Bordure élégante
                    cv2.rectangle(frame, (mask_x-2, mask_y-2), (mask_x+162, mask_y+122), (255, 255, 255), 2)
                    cv2.rectangle(frame, (mask_x-1, mask_y-1), (mask_x+161, mask_y+121), (0, 0, 0), 1)
                    
                    # Titre du masque
                    cv2.putText(frame, "Masque Ultra-Smart", (mask_x, mask_y + 135), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                except:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Ultra overlay error: {e}")
            return frame

def display_thread():
    """Thread d'affichage ultra-optimisé"""
    detector = UltraSmartGloveDetector()
    logger.info("Ultra display thread started")
    
    window_name = "Bebop 2 - Ultra Smart Hybrid Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    fps_counter = 0
    fps_start_time = time.time()
    last_display_time = time.time()
    
    no_frame_count = 0
    
    while processing_active.is_set():
        try:
            current_time = time.time()
            
            # Limitation FPS stricte
            if (current_time - last_display_time) < (1.0 / DISPLAY_FPS):
                time.sleep(0.005)
                continue
            
            last_display_time = current_time
            
            # Récupération frame thread-safe
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = None
            
            if frame is None:
                no_frame_count += 1
                
                # Écran d'attente ultra-informatif
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Attente flux hybride...", (170, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                
                with stats_lock:
                    callbacks = frame_stats['callback_calls']
                    success_reads = frame_stats['successful_reads']
                    failed_reads = frame_stats['failed_reads']
                    cleaned = frame_stats['files_cleaned']
                    total_attempts = success_reads + failed_reads
                    success_rate = (success_reads / max(total_attempts, 1)) * 100
                
                # Statistiques d'attente détaillées
                cv2.putText(blank_frame, f"Callbacks recus: {callbacks}", 
                           (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(blank_frame, f"Lectures reussies: {success_reads}/{total_attempts} ({success_rate:.1f}%)", 
                           (140, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
                cv2.putText(blank_frame, f"Fichiers nettoyes: {cleaned}", 
                           (210, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
                cv2.putText(blank_frame, f"Cycles sans frame: {no_frame_count}", 
                           (190, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                
                # Instructions
                cv2.putText(blank_frame, "Appuyez sur 'q' pour quitter", 
                           (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow(window_name, blank_frame)
                
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:
                    processing_active.clear()
                    break
                continue
            
            # Reset compteur si frame reçue
            no_frame_count = 0
            
            # Traitement ultra-intelligent
            processed_frame, detected = detector.detect_glove(frame)
            
            # Calcul FPS avec moyennage
            fps_counter += 1
            
            if fps_counter % 30 == 0:
                fps_elapsed = current_time - fps_start_time
                current_fps = fps_counter / fps_elapsed if fps_elapsed > 0 else 0
                logger.info(f"Display FPS: {current_fps:.1f}")
                fps_start_time = current_time
                fps_counter = 0
            
            # Affichage final
            cv2.imshow(window_name, processed_frame)
            
            # Gestion touches étendues
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("User requested quit")
                processing_active.clear()
                break
            elif key == ord('r'):
                # Reset ultra-complet
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                    frame_stats['callback_calls'] = 0
                    frame_stats['successful_reads'] = 0
                    frame_stats['failed_reads'] = 0
                    frame_stats['last_file_processed'] = ''
                    frame_stats['files_cleaned'] = 0
                
                # Reset détecteur
                detector.detection_history.clear()
                detector.stable_detections.clear()
                detector.last_frame_hash = None
                detector.last_detection_result = None
                detector.last_detection_center = None
                detector.detection_cooldown = 0
                detector.frame_counter = 0
                
                no_frame_count = 0
                logger.info("Ultra-complete system reset performed")
                
            elif key == ord('s'):
                screenshot_name = f"screenshot_ultra_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Ultra screenshot saved: {screenshot_name}")
                
            elif key == ord('d'):
                # Debug ultra-détaillé
                with stats_lock:
                    debug_stats = frame_stats.copy()
                
                logger.info(f"=== ULTRA DEBUG STATS ===")
                logger.info(f"System: {debug_stats}")
                logger.info(f"Detector: frames={detector.frame_counter}, cooldown={detector.detection_cooldown}")
                logger.info(f"Stability: {list(detector.stable_detections)}")
                logger.info(f"History: {list(detector.detection_history)}")
                logger.info(f"Center: {detector.last_detection_center}")
                
            elif key == ord('c'):
                # Nettoyage manuel
                try:
                    files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
                    if len(files) > 5:
                        cleanup_immediately(files)
                        logger.info(f"Manual cleanup triggered: {len(files)} files found")
                except Exception as e:
                    logger.error(f"Manual cleanup error: {e}")
                
        except Exception as e:
            logger.error(f"Ultra display thread error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    logger.info("Ultra display thread terminated")

def monitor_thread():
    """Thread de monitoring ultra-détaillé"""
    logger.info("Ultra monitor thread started")
    last_frame_count = 0
    last_callback_count = 0
    last_cleaned_count = 0
    
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
            cleaned = frame_stats['files_cleaned']
            last_received_time = frame_stats['last_frame_time']
        
        # Calculs des différentiels
        frame_diff = current_frames - last_frame_count
        callback_diff = callbacks - last_callback_count
        cleaned_diff = cleaned - last_cleaned_count
        
        last_frame_count = current_frames
        last_callback_count = callbacks
        last_cleaned_count = cleaned
        
        time_since_last_frame = time.time() - last_received_time
        total_reads = success_reads + failed_reads
        read_success_rate = (success_reads / max(total_reads, 1)) * 100
        
        # Diagnostics selon l'état
        if frame_diff > 0:
            # Fonctionnement normal
            avg_fps = frame_diff / 5
            detection_rate = (detections / max(current_frames, 1)) * 100
            callback_fps = callback_diff / 5
            cleanup_rate = cleaned_diff / 5
            
            logger.info(f"ULTRA MONITOR - Frames: {current_frames} (+{frame_diff}), FPS: {avg_fps:.1f}, "
                       f"Det: {detection_rate:.1f}%, Err: {errors}")
            logger.info(f"              I/O: {callbacks} callbacks (+{callback_diff}, {callback_fps:.1f}/s), "
                       f"Read: {read_success_rate:.1f}% ({success_reads}/{total_reads})")
            logger.info(f"              Cleanup: {cleaned} total (+{cleaned_diff}, {cleanup_rate:.1f}/s)")
            
        else:
            # Problème détecté
            logger.warning(f"ISSUE DETECTED - No new frames in 5s")
            logger.warning(f"              Callbacks: {callbacks} (+{callback_diff}), "
                         f"Read success: {read_success_rate:.1f}%")
            logger.warning(f"              Last frame: {time_since_last_frame:.1f}s ago")
            logger.warning(f"              Failed reads: {failed_reads}, Cleaned: {cleaned}")
            
            # Diagnostics supplémentaires
            try:
                files_count = len(glob.glob(os.path.join(IMAGES_DIR, "image_*.png")))
                logger.warning(f"              Files in directory: {files_count}")
                
                if os.path.exists(IMAGES_DIR):
                    dir_accessible = os.access(IMAGES_DIR, os.R_OK)
                    logger.warning(f"              Directory accessible: {dir_accessible}")
                else:
                    logger.warning(f"              Directory does not exist: {IMAGES_DIR}")
                    
            except Exception as e:
                logger.warning(f"              Directory check error: {e}")
    
    logger.info("Ultra monitor thread terminated")

def signal_handler(sig, frame):
    """Gestionnaire de signaux amélioré"""
    logger.info(f"Signal {sig} received - initiating ultra shutdown")
    processing_active.clear()

def main():
    """Fonction principale ultra-optimisée"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Bebop 2 Ultra Smart Hybrid Detection System")
    logger.info("Features: Minimal disk usage + Ultra-smart detection + Immediate cleanup")
    
    bebop = None
    vision = None
    threads = []
    
    try:
        # Vérification du répertoire d'images
        if not os.path.exists(IMAGES_DIR):
            logger.error(f"Images directory not found: {IMAGES_DIR}")
            logger.error("Please ensure PyParrot is properly installed")
            return False
        
        logger.info(f"Images directory verified: {IMAGES_DIR}")
        
        # Nettoyage initial
        try:
            initial_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            if initial_files:
                cleanup_immediately(initial_files)
                logger.info(f"Initial cleanup: removed {len(initial_files)} old files")
        except Exception as e:
            logger.warning(f"Initial cleanup failed: {e}")
        
        # Connexion au drone
        bebop = Bebop()
        logger.info("Connecting to Bebop 2...")
        
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            logger.error("Ensure drone is powered on and in WiFi mode")
            return False
        
        logger.info("Drone connected successfully")
        
        # Configuration de la vision hybride
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(hybrid_callback)
        
        # Démarrage des threads
        display_thread_obj = threading.Thread(target=display_thread, daemon=True, name="UltraDisplay")
        monitor_thread_obj = threading.Thread(target=monitor_thread, daemon=True, name="UltraMonitor")
        
        threads = [display_thread_obj, monitor_thread_obj]
        
        for i, thread in enumerate(threads):
            thread.start()
            time.sleep(0.1)
            logger.info(f"Thread {i+1}/{len(threads)} started: {thread.name}")
        
        logger.info("All ultra threads started successfully")
        
        # Ouverture du flux vidéo
        logger.info("Opening video stream with hybrid callback...")
        start_time = time.time()
        
        if not vision.open_video():
            logger.error("Failed to open video stream")
            logger.error("Check drone connection and camera")
            return False
        
        open_time = time.time() - start_time
        logger.info(f"Video stream opened successfully ({open_time:.1f}s)")
        logger.info("Ultra Smart Hybrid Detection System is now active")
        logger.info("=" * 60)
        logger.info("CONTROLS:")
        logger.info("  'q' or ESC  = Quit")
        logger.info("  'r'         = Complete reset (all stats + detector)")
        logger.info("  's'         = Screenshot")
        logger.info("  'd'         = Debug stats")
        logger.info("  'c'         = Manual cleanup")
        logger.info("=" * 60)
        
        # Boucle principale avec monitoring
        start_monitor_time = time.time()
        
        try:
            while processing_active.is_set():
                time.sleep(1)
                
                # Vérifier si la fenêtre est toujours ouverte
                try:
                    if cv2.getWindowProperty("Bebop 2 - Ultra Smart Hybrid Detection", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Display window was closed")
                        break
                except:
                    pass
                
                # Monitoring périodique du système
                current_time = time.time()
                if (current_time - start_monitor_time) > 30:  # Toutes les 30 secondes
                    with stats_lock:
                        uptime_stats = frame_stats.copy()
                    
                    uptime = current_time - start_monitor_time
                    avg_fps = uptime_stats['frame_count'] / max(uptime, 1)
                    
                    logger.info(f"SYSTEM UPTIME: {uptime:.0f}s, Avg FPS: {avg_fps:.1f}, "
                               f"Total detections: {uptime_stats['detection_count']}")
                    
                    start_monitor_time = current_time
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
    
    except Exception as e:
        logger.error(f"Critical error in ultra main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        # Nettoyage ultra-complet
        logger.info("Starting ultra-comprehensive cleanup...")
        processing_active.clear()
        
        # Attendre les threads avec timeout progressif
        for i, thread in enumerate(threads):
            try:
                timeout = 5 + i  # Timeout progressif
                thread.join(timeout=timeout)
                logger.info(f"Thread {thread.name} terminated successfully")
            except Exception as e:
                logger.warning(f"Error joining thread {thread.name}: {e}")
        
        # Fermeture du flux vidéo
        if vision:
            try:
                vision.close_video()
                logger.info("Video stream closed successfully")
            except Exception as e:
                logger.warning(f"Error closing video: {e}")
        
        # Déconnexion du drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("Drone disconnected successfully")
            except Exception as e:
                logger.warning(f"Error disconnecting drone: {e}")
        
        # Nettoyage final des fichiers
        try:
            final_files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
            if final_files:
                cleanup_immediately(final_files)
                logger.info(f"Final cleanup: removed {len(final_files)} remaining files")
        except Exception as e:
            logger.warning(f"Final cleanup failed: {e}")
        
        # Fermeture OpenCV
        try:
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed successfully")
        except Exception as e:
            logger.warning(f"Error closing OpenCV windows: {e}")
        
        # Statistiques finales
        with stats_lock:
            final_stats = frame_stats.copy()
        
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS:")
        logger.info(f"  Total frames processed: {final_stats['frame_count']}")
        logger.info(f"  Total detections: {final_stats['detection_count']}")
        logger.info(f"  Detection rate: {(final_stats['detection_count']/max(final_stats['frame_count'],1))*100:.1f}%")
        logger.info(f"  Total callbacks: {final_stats['callback_calls']}")
        logger.info(f"  Read success rate: {(final_stats['successful_reads']/max(final_stats['callback_calls'],1))*100:.1f}%")
        logger.info(f"  Files cleaned: {final_stats['files_cleaned']}")
        logger.info(f"  Errors: {final_stats['error_count']}")
        logger.info("=" * 60)
        
        logger.info("Ultra-comprehensive cleanup completed successfully")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        logger.info(f"Program exiting with code {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception in ultra main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)