import cv2
import numpy as np
import time
import subprocess
import threading
import sys
import logging
import os
import pyparrot
from pyparrot.Bebop import Bebop
from collections import deque

# === PARAMÈTRES OPTIMISÉS POUR IA + LONGUE DISTANCE ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

# === PARAMÈTRES IA ===
AI_OUTPUT_SIZE = (224, 224)  # Taille standard pour IA
SAVE_AI_SAMPLES = True       # Sauvegarder échantillons pour dataset
AI_SAMPLES_DIR = "ai_dataset"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_ai_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Créer dossier échantillons IA
if SAVE_AI_SAMPLES and not os.path.exists(AI_SAMPLES_DIR):
    os.makedirs(AI_SAMPLES_DIR)
    logger.info(f"📁 Dossier IA créé: {AI_SAMPLES_DIR}")

# === DÉTECTEUR GANT OPTIMISÉ POUR IA + LONGUE DISTANCE ===
class AIReadyGloveDetector:
    def __init__(self):
        # Configuration de base
        self.detection_history = deque(maxlen=15)
        self.stable_detections = deque(maxlen=4)   # Plus réactif pour IA
        self.confidence_threshold = 3
        
        # === PARAMÈTRES LONGUE DISTANCE AMÉLIORÉS ===
        self.min_area = 80      # Réduit pour détecter plus loin
        self.max_area = 150000  # Augmenté pour gros plans
        self.min_contour_points = 6  # Plus tolérant
        
        # Historique des couleurs détectées
        self.color_balance_history = deque(maxlen=20)
        self.red_orange_ratio_history = deque(maxlen=10)
        
        # Kernels morphologiques optimisés
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # === SYSTÈME DE ZOOM ADAPTATIF LONGUE DISTANCE ===
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.15  # Plus réactif
        self.zoom_min = 1.0
        self.zoom_max = 6.0  # Zoom plus élevé pour longue distance
        
        # Calibrage longue distance
        self.area_reference = 1800  # Réduit pour adaptation distance
        self.area_history = deque(maxlen=15)
        self.quality_scores = deque(maxlen=10)
        
        # Zone de recherche prédictive
        self.search_zone = None
        self.zone_tracking = deque(maxlen=5)
        
        # Statistiques détaillées
        self.frame_count = 0
        self.detection_count = 0
        self.quality_count = 0
        self.zoom_adjustments = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Adaptation éclairage
        self.brightness_history = deque(maxlen=10)
        self.auto_exposure_factor = 1.0
        
        # === NOUVEAUX: EXPORT IA ===
        self.ai_export_count = 0
        self.last_ai_export = 0
        self.ai_export_interval = 0.1  # Export toutes les 100ms pour dataset
        
        logger.info("🧠 Détecteur IA Ready initialisé")
        logger.info(f"📏 Longue distance: aire min={self.min_area}, zoom max={self.zoom_max}x")

    def detect_glove_for_ai(self, frame):
        """Détection optimisée avec export automatique pour IA"""
        if frame is None:
            return frame, False, None, None
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            # === PHASE 1: ANALYSE ÉCLAIRAGE RENFORCÉE ===
            exposure_adjusted_frame = self._enhanced_exposure_correction(frame)
            
            # === PHASE 2: RECHERCHE GLOBALE LONGUE DISTANCE ===
            if sum(self.stable_detections) < 2 or self.zoom_factor < 1.3:
                global_result = self._long_range_global_detection(exposure_adjusted_frame)
                if global_result:
                    detected, contour, area, quality_score = global_result
                    if detected and quality_score > 0.3:  # Plus tolérant
                        self._update_zoom_and_tracking(area, contour)
                        ai_mask, ai_image = self._extract_for_ai(original_frame, contour)
                        return self._finalize_detection(original_frame, detected, contour, area, quality_score), detected, ai_mask, ai_image
            
            # === PHASE 3: DÉTECTION ZOOMÉE LONGUE DISTANCE ===
            zoomed_frame, zoom_info = self._apply_enhanced_zoom(exposure_adjusted_frame)
            
            # Détection bicolore optimisée longue distance
            red_mask, orange_mask, combined_mask = self._create_long_range_masks(zoomed_frame)
            
            # Morphologie adaptative longue distance
            processed_mask = self._enhanced_morphology(combined_mask)
            
            # Analyse de contours avec scoring longue distance
            best_contour, area, quality_score = self._long_range_contour_selection(
                processed_mask, red_mask, orange_mask, zoomed_frame
            )
            
            # Remapping vers coordonnées originales
            if best_contour is not None:
                best_contour = self._remap_contour_to_original(best_contour, zoom_info)
                area = cv2.contourArea(best_contour)
            
            # Validation plus tolérante pour longue distance
            detected = self._validate_long_range_detection(best_contour, area, quality_score)
            
            # Extraction pour IA si détection valide
            ai_mask, ai_image = None, None
            if detected and best_contour is not None:
                ai_mask, ai_image = self._extract_for_ai(original_frame, best_contour)
                
            # Mise à jour système de tracking
            if detected:
                self._update_zoom_and_tracking(area, best_contour)
                self.quality_scores.append(quality_score)
                self.quality_count += 1
            else:
                self._handle_detection_loss()
            
            # Stabilisation avec historique
            self.stable_detections.append(detected)
            final_detected = sum(self.stable_detections) >= self.confidence_threshold
            
            return self._finalize_detection(original_frame, final_detected, best_contour, area, quality_score), final_detected, ai_mask, ai_image
            
        except Exception as e:
            logger.debug(f"AI detection error: {e}")
            return original_frame, False, None, None

    def _enhanced_exposure_correction(self, frame):
        """Correction d'exposition renforcée pour longue distance"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            self.brightness_history.append(mean_brightness)
            
            brightness_avg = np.mean(self.brightness_history) if self.brightness_history else mean_brightness
            
            # Correction plus agressive pour longue distance
            if brightness_avg < 110:  # Sombre
                self.auto_exposure_factor = min(1.6, self.auto_exposure_factor + 0.08)
            elif brightness_avg > 150:  # Trop clair
                self.auto_exposure_factor = max(0.6, self.auto_exposure_factor - 0.08)
            else:
                self.auto_exposure_factor = max(0.9, min(1.1, self.auto_exposure_factor))
            
            # Application correction
            if abs(self.auto_exposure_factor - 1.0) > 0.03:
                # Correction gamma pour améliorer contraste longue distance
                corrected = np.power(frame / 255.0, 1.0/self.auto_exposure_factor) * 255.0
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                
                # Amélioration saturation pour longue distance
                hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Enhanced exposure correction error: {e}")
            return frame

    def _long_range_global_detection(self, frame):
        """Détection globale optimisée longue distance"""
        try:
            # Préprocessing renforcé pour longue distance
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # Amélioration contraste pour petits objets
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Masques couleur longue distance
            red_mask, orange_mask, combined_mask = self._create_long_range_masks(enhanced)
            
            # Morphologie adaptée longue distance
            processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel_medium)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # Sélection contour longue distance
            best_contour, area, quality_score = self._long_range_contour_selection(
                processed_mask, red_mask, orange_mask, enhanced
            )
            
            if best_contour is not None and area > self.min_area and quality_score > 0.25:
                return True, best_contour, area, quality_score
            
            return None
            
        except Exception as e:
            logger.debug(f"Long range global detection error: {e}")
            return None

    def _create_long_range_masks(self, frame):
        """Masques couleur optimisés pour longue distance"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            
            # Analyse adaptative plus sensible
            brightness_factor = np.mean(hsv[:, :, 2]) / 255.0
            saturation_factor = np.mean(hsv[:, :, 1]) / 255.0
            
            # Ajustements plus agressifs pour longue distance
            sat_adjust = max(-35, min(35, int((0.6 - saturation_factor) * 60)))
            val_adjust = max(-30, min(30, int((0.4 - brightness_factor) * 50)))
            
            # === MASQUES ROUGE ÉTENDUS LONGUE DISTANCE ===
            # Rouge principal étendu
            red_lower1 = np.array([0, max(100, 140 + sat_adjust), max(80, 120 + val_adjust)])
            red_upper1 = np.array([12, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            # Rouge teinte haute étendu
            red_lower2 = np.array([168, max(100, 140 + sat_adjust), max(80, 120 + val_adjust)])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            # Rouge-orange transition
            red_orange_lower = np.array([0, max(80, 120 + sat_adjust), max(60, 100 + val_adjust)])
            red_orange_upper = np.array([15, 255, 255])
            mask_red_orange = cv2.inRange(hsv, red_orange_lower, red_orange_upper)
            
            mask_red = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_red_orange))
            
            # === MASQUES ORANGE ÉTENDUS LONGUE DISTANCE ===
            # Orange vif étendu
            orange_bright_lower = np.array([6, max(120, 160 + sat_adjust), max(100, 140 + val_adjust)])
            orange_bright_upper = np.array([22, 255, 255])
            mask_orange_bright = cv2.inRange(hsv, orange_bright_lower, orange_bright_upper)
            
            # Orange moyen étendu
            orange_mid_lower = np.array([8, max(100, 130 + sat_adjust), max(80, 120 + val_adjust)])
            orange_mid_upper = np.array([25, 255, 250])
            mask_orange_mid = cv2.inRange(hsv, orange_mid_lower, orange_mid_upper)
            
            # Orange avec ombres étendues
            orange_shadow_lower = np.array([10, max(60, 100 + sat_adjust//2), max(50, 80 + val_adjust)])
            orange_shadow_upper = np.array([22, 220, 220])
            mask_orange_shadow = cv2.inRange(hsv, orange_shadow_lower, orange_shadow_upper)
            
            mask_orange = cv2.bitwise_or(mask_orange_bright, 
                         cv2.bitwise_or(mask_orange_mid, mask_orange_shadow))
            
            # === ÉQUILIBRAGE INTELLIGENT ===
            red_pixels = np.sum(mask_red > 0)
            orange_pixels = np.sum(mask_orange > 0)
            total_color_pixels = red_pixels + orange_pixels
            
            if total_color_pixels > 0:
                red_ratio = red_pixels / total_color_pixels
                orange_ratio = orange_pixels / total_color_pixels
                self.red_orange_ratio_history.append((red_ratio, orange_ratio))
                
                # Boost plus agressif pour longue distance
                if orange_ratio > 0.8 and red_ratio < 0.2:
                    mask_red = cv2.dilate(mask_red, self.kernel_medium, iterations=1)
                    mask_orange = cv2.erode(mask_orange, self.kernel_small, iterations=1)
                elif red_ratio > 0.8 and orange_ratio < 0.2:
                    mask_orange = cv2.dilate(mask_orange, self.kernel_medium, iterations=1)
            
            # === EXCLUSIONS ADAPTÉES ===
            # Exclusion peau plus tolérante
            skin_lower = np.array([3, 30, 60])
            skin_upper = np.array([18, min(160, 140 - sat_adjust//3), 250])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            mask_skin = cv2.erode(mask_skin, self.kernel_small, iterations=1)
            
            mask_red = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_skin))
            mask_orange = cv2.bitwise_and(mask_orange, cv2.bitwise_not(mask_skin))
            
            # Combinaison finale
            mask_combined = cv2.bitwise_or(mask_red, mask_orange)
            
            # Nettoyage avec préservation petits objets
            mask_combined = cv2.medianBlur(mask_combined, 3)
            
            # Bordures réduites pour longue distance
            border_size = max(4, int(15 / max(self.zoom_factor, 1.0)))
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_combined = cv2.bitwise_and(mask_combined, border_mask)
            
            return mask_red, mask_orange, mask_combined
            
        except Exception as e:
            logger.debug(f"Long range masks error: {e}")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            fallback = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([30, 255, 255]))
            return fallback, fallback, fallback

    def _enhanced_morphology(self, mask):
        """Morphologie optimisée pour longue distance"""
        try:
            # Kernels plus conservateurs pour petits objets
            if self.zoom_factor > 3.0:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                iterations_close = 2
                iterations_open = 1
            elif self.zoom_factor > 2.0:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                iterations_close = 1
                iterations_open = 1
            else:
                kernel_close = self.kernel_medium
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                iterations_close = 1
                iterations_open = 1
            
            # Fermeture conservative
            processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=iterations_close)
            
            # Ouverture très légère pour préserver petits objets
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_open, iterations=iterations_open)
            
            return processed
            
        except Exception as e:
            logger.debug(f"Enhanced morphology error: {e}")
            return mask

    def _long_range_contour_selection(self, mask, red_mask, orange_mask, frame):
        """Sélection contours optimisée longue distance"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, 0, 0
            
            best_contour = None
            best_score = 0
            best_area = 0
            
            # Seuils adaptés longue distance
            min_area_adj = self.min_area * max(0.8, self.zoom_factor ** 1.1)
            max_area_adj = self.max_area * max(1.0, self.zoom_factor ** 1.3)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres plus tolérants
                if area < min_area_adj or area > max_area_adj:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Géométrie plus tolérante
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                if not (0.3 <= aspect_ratio <= 4.0):  # Plus tolérant
                    continue
                
                # === ANALYSE BICOLORE ADAPTÉE ===
                mask_contour = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillPoly(mask_contour, [contour], 255)
                
                red_in_contour = np.sum(cv2.bitwise_and(red_mask, mask_contour) > 0)
                orange_in_contour = np.sum(cv2.bitwise_and(orange_mask, mask_contour) > 0)
                total_color_in_contour = red_in_contour + orange_in_contour
                
                if total_color_in_contour == 0:
                    continue
                
                red_ratio_contour = red_in_contour / total_color_in_contour
                orange_ratio_contour = orange_in_contour / total_color_in_contour
                
                # Score bicolore plus tolérant longue distance
                bicolor_score = 1.0
                if red_ratio_contour < 0.05 or orange_ratio_contour < 0.05:
                    bicolor_score *= 0.4  # Moins pénalisant
                elif red_ratio_contour > 0.95 or orange_ratio_contour > 0.95:
                    bicolor_score *= 0.6  # Moins pénalisant
                else:
                    balance = 1.0 - abs(red_ratio_contour - orange_ratio_contour)
                    bicolor_score *= (0.6 + balance * 0.7)
                
                # === SCORING ADAPTÉ LONGUE DISTANCE ===
                area_score = min(area / (self.area_reference * max(self.zoom_factor, 1.0)), 1.0)
                
                # Forme plus tolérante
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                shape_score = min(extent * 1.3, 1.0)
                
                # Position moins contraignante
                center_x, center_y = x + w//2, y + h//2
                dist_from_center = np.sqrt((center_x - WIDTH//2)**2 + (center_y - HEIGHT//2)**2)
                max_dist = np.sqrt((WIDTH//2)**2 + (HEIGHT//2)**2)
                position_score = 1.0 - (dist_from_center / max_dist) * 0.2  # Moins contraignant
                
                # Qualité couleur
                try:
                    hsv_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                    contour_relative = contour - [x, y]
                    
                    mask_roi = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_roi, [contour_relative], 255)
                    
                    sat_mean = np.mean(hsv_roi[:, :, 1][mask_roi > 0]) if np.any(mask_roi > 0) else 0
                    val_mean = np.mean(hsv_roi[:, :, 2][mask_roi > 0]) if np.any(mask_roi > 0) else 0
                    
                    sat_score = min(sat_mean / 150.0, 1.0)  # Moins strict
                    val_score = min(val_mean / 180.0, 1.0)  # Moins strict
                    
                    color_quality_score = (sat_score + val_score) / 2.0
                except:
                    color_quality_score = 0.5
                
                # Score final ajusté longue distance
                final_score = (
                    area_score * 0.3 +           # Plus de poids à l'aire
                    bicolor_score * 0.3 +        # Équilibré
                    shape_score * 0.2 +          # Moins strict sur forme
                    position_score * 0.1 +       # Moins contraignant
                    color_quality_score * 0.1    # Moins strict
                )
                
                # Bonus tracking plus généreux
                if self.zone_tracking:
                    last_zone = self.zone_tracking[-1]
                    zone_dist = np.sqrt((center_x - last_zone[0])**2 + (center_y - last_zone[1])**2)
                    if zone_dist < 150:  # Zone plus large
                        final_score *= 1.3
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
                    best_area = area
            
            return best_contour, best_area, best_score
            
        except Exception as e:
            logger.debug(f"Long range contour selection error: {e}")
            return None, 0, 0

    def _validate_long_range_detection(self, contour, area, quality_score):
        """Validation adaptée longue distance"""
        try:
            if contour is None or area <= 0:
                return False
            
            # Seuils plus tolérants
            min_quality = 0.25 if self.zoom_factor > 2.0 else 0.3
            min_area_final = self.min_area * max(0.8, (self.zoom_factor ** 1.0))
            
            if quality_score < min_quality:
                return False
            if area < min_area_final:
                return False
            
            # Géométrie plus tolérante
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if not (0.25 <= aspect_ratio <= 4.0):
                return False
            
            # Historique plus tolérant
            if self.area_history:
                area_median = np.median(self.area_history)
                if area > area_median * 4 or area < area_median * 0.2:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Long range validation error: {e}")
            return False

    def _update_zoom_and_tracking(self, area, contour):
        """Zoom et tracking adaptés longue distance"""
        try:
            self.area_history.append(area)
            
            # Calcul zoom optimisé longue distance
            if area < 300:          # Très très loin
                self.target_zoom = min(self.zoom_max, 6.0)
            elif area < 600:        # Très loin
                self.target_zoom = min(self.zoom_max, 4.5)
            elif area < 1200:       # Loin
                self.target_zoom = min(self.zoom_max, 3.5)
            elif area < 2400:       # Moyen-loin
                self.target_zoom = 2.5
            elif area < 4800:       # Moyen
                self.target_zoom = 1.8
            elif area < 8000:       # Proche
                self.target_zoom = 1.3
            else:                   # Très proche
                self.target_zoom = 1.0
            
            # Tracking adapté
            if contour is not None:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.zone_tracking.append((cx, cy))
                    
                    if len(self.zone_tracking) >= 3:
                        dx = self.zone_tracking[-1][0] - self.zone_tracking[-3][0]
                        dy = self.zone_tracking[-1][1] - self.zone_tracking[-3][1]
                        
                        pred_x = cx + dx // 2
                        pred_y = cy + dy // 2
                        
                        zone_size = max(100, int(180 / max(self.zoom_factor, 1.0)))
                        self.search_zone = (pred_x, pred_y, zone_size, zone_size)
            
            self.zoom_adjustments += 1
            
        except Exception as e:
            logger.debug(f"Zoom and tracking update error: {e}")

    def _handle_detection_loss(self):
        """Gestion perte détection longue distance"""
        try:
            if sum(self.stable_detections) == 0:
                self.target_zoom = max(self.zoom_min, self.target_zoom * 0.95)
                
                if self.search_zone and self.target_zoom < 1.5:
                    x, y, w, h = self.search_zone
                    self.search_zone = (x, y, min(w * 1.2, WIDTH//2), min(h * 1.2, HEIGHT//2))
                
                if self.target_zoom < 1.2:
                    self.search_zone = None
                    self.zone_tracking.clear()
                    
        except Exception as e:
            logger.debug(f"Detection loss handling error: {e}")

    def _apply_enhanced_zoom(self, frame):
        """Zoom amélioré pour longue distance"""
        try:
            h, w = frame.shape[:2]
            
            self.zoom_factor += (self.target_zoom - self.zoom_factor) * self.zoom_smooth_factor
            self.zoom_factor = np.clip(self.zoom_factor, self.zoom_min, self.zoom_max)

            if self.zoom_factor <= 1.1:
                return frame, {'zoom': 1.0, 'offset_x': 0, 'offset_y': 0, 'crop_w': w, 'crop_h': h}
            
            # Zone de focus intelligente
            if self.search_zone:
                center_x, center_y, zone_w, zone_h = self.search_zone
                center_x = max(zone_w//2, min(center_x, w - zone_w//2))
                center_y = max(zone_h//2, min(center_y, h - zone_h//2))
            else:
                center_x, center_y = w // 2, int(h * 0.45)
            
            # Calcul zone de crop
            crop_w = int(w / self.zoom_factor)
            crop_h = int(h / self.zoom_factor)
            
            offset_x = max(0, min(center_x - crop_w // 2, w - crop_w))
            offset_y = max(0, min(center_y - crop_h // 2, h - crop_h))
            
            cropped = frame[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
            
            # Interpolation haute qualité pour IA
            if self.zoom_factor > 4.0:
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
            elif self.zoom_factor > 2.0:
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
            else:
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            zoom_info = {
                'zoom': self.zoom_factor,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'crop_w': crop_w,
                'crop_h': crop_h
            }
            
            return zoomed, zoom_info
            
        except Exception as e:
            logger.debug(f"Enhanced zoom error: {e}")
            return frame, {'zoom': 1.0, 'offset_x': 0, 'offset_y': 0, 'crop_w': w, 'crop_h': h}

    def _remap_contour_to_original(self, contour, zoom_info):
        """Remapping précis pour IA"""
        try:
            if zoom_info['zoom'] <= 1.1:
                return contour
            
            scale_x = zoom_info['crop_w'] / WIDTH
            scale_y = zoom_info['crop_h'] / HEIGHT
            
            remapped_contour = contour.copy().astype(np.float32)
            remapped_contour[:, :, 0] = remapped_contour[:, :, 0] * scale_x + zoom_info['offset_x']
            remapped_contour[:, :, 1] = remapped_contour[:, :, 1] * scale_y + zoom_info['offset_y']
            
            return np.round(remapped_contour).astype(np.int32)
            
        except Exception as e:
            logger.debug(f"Contour remapping error: {e}")
            return contour

    def _extract_for_ai(self, frame, contour):
        """NOUVEAU: Extraction gant sur fond noir pour IA"""
        try:
            if contour is None:
                return None, None
            
            h, w = frame.shape[:2]
            
            # === CRÉATION MASQUE GANT ===
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Dilatation légère pour capturer bordures
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # === EXTRACTION GANT SUR FOND NOIR ===
            glove_extracted = np.zeros_like(frame)
            glove_extracted[mask > 0] = frame[mask > 0]
            
            # === REDIMENSIONNEMENT POUR IA ===
            # Trouver bounding rect pour crop intelligent
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Ajouter padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_rect = min(w - x, w_rect + 2*padding)
            h_rect = min(h - y, h_rect + 2*padding)
            
            # Crop région d'intérêt
            roi_extracted = glove_extracted[y:y+h_rect, x:x+w_rect]
            roi_mask = mask[y:y+h_rect, x:x+w_rect]
            
            # Redimensionner pour IA (carré)
            ai_size = AI_OUTPUT_SIZE[0]
            
            # Créer image carrée avec fond noir
            square_size = max(roi_extracted.shape[:2])
            square_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
            square_mask = np.zeros((square_size, square_size), dtype=np.uint8)
            
            # Centrer l'image
            y_offset = (square_size - roi_extracted.shape[0]) // 2
            x_offset = (square_size - roi_extracted.shape[1]) // 2
            
            square_img[y_offset:y_offset+roi_extracted.shape[0], 
                      x_offset:x_offset+roi_extracted.shape[1]] = roi_extracted
            square_mask[y_offset:y_offset+roi_mask.shape[0], 
                       x_offset:x_offset+roi_mask.shape[1]] = roi_mask
            
            # Redimensionner à la taille IA
            ai_image = cv2.resize(square_img, AI_OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            ai_mask = cv2.resize(square_mask, AI_OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            
            # === SAUVEGARDE ÉCHANTILLON IA ===
            if SAVE_AI_SAMPLES:
                current_time = time.time()
                if current_time - self.last_ai_export > self.ai_export_interval:
                    timestamp = int(current_time * 1000)
                    
                    # Nom fichier avec métadonnées
                    area = cv2.contourArea(contour)
                    quality = self.quality_scores[-1] if self.quality_scores else 0
                    
                    filename = f"glove_{timestamp}_area{int(area)}_q{quality:.2f}_zoom{self.zoom_factor:.1f}x.png"
                    filepath = os.path.join(AI_SAMPLES_DIR, filename)
                    
                    cv2.imwrite(filepath, ai_image)
                    
                    self.ai_export_count += 1
                    self.last_ai_export = current_time
                    
                    if self.ai_export_count % 50 == 0:
                        logger.info(f"🧠 IA samples: {self.ai_export_count} échantillons sauvés")
            
            return ai_mask, ai_image
            
        except Exception as e:
            logger.debug(f"AI extraction error: {e}")
            return None, None

    def _finalize_detection(self, frame, detected, contour, area, quality_score):
        """Finalisation avec interface IA"""
        try:
            self.detection_history.append(detected)
            if detected:
                self.detection_count += 1
            
            # Visualisation
            if detected and contour is not None:
                self._draw_ai_detection(frame, contour, area, quality_score)
            
            # Interface IA
            result_frame = self._create_ai_overlay(frame, detected, area, quality_score)
            
            return result_frame
            
        except Exception as e:
            logger.debug(f"Finalization error: {e}")
            return frame

    def _draw_ai_detection(self, frame, contour, area, quality_score):
        """Visualisation optimisée pour IA"""
        try:
            # Couleurs selon distance estimée
            if area > 5000:
                color = (0, 255, 0)      # Vert - très proche
                distance_text = "TRÈS PROCHE"
            elif area > 2000:
                color = (0, 255, 100)    # Vert-jaune - proche
                distance_text = "PROCHE"
            elif area > 800:
                color = (0, 255, 255)    # Jaune - moyen
                distance_text = "MOYEN"
            elif area > 300:
                color = (0, 150, 255)    # Orange - loin
                distance_text = "LOIN"
            else:
                color = (0, 100, 255)    # Rouge - très loin
                distance_text = "TRÈS LOIN"
            
            # Contour principal
            thickness = max(2, int(4 * min(self.zoom_factor, 2.0)))
            cv2.drawContours(frame, [contour], -1, color, thickness)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre avec croix IA
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Point central avec indicateur IA
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
                cv2.putText(frame, "AI", (cx-8, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Croix directionnelle
                cross_size = 20
                cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (255, 255, 255), 3)
                cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (255, 255, 255), 3)
            
            # Informations détaillées pour IA
            info_y = max(y - 25, 35)
            cv2.putText(frame, f"GANT {distance_text}", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            info_y += 30
            cv2.putText(frame, f"IA Ready | Q: {quality_score:.2f} | Aire: {int(area)}", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            info_y += 25
            cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x | Samples: {self.ai_export_count}", 
                       (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                       
        except Exception as e:
            logger.debug(f"AI drawing error: {e}")

    def _create_ai_overlay(self, frame, detected, area, quality_score):
        """Interface utilisateur IA enrichie"""
        try:
            h, w = frame.shape[:2]
            
            # === STATUS PRINCIPAL IA ===
            if detected:
                if quality_score > 0.6:
                    status = f"🧠 IA READY - GANT CAPTURÉ (Q:{quality_score:.2f})"
                    status_color = (0, 255, 0)
                else:
                    status = f"🧠 IA READY - GANT DÉTECTÉ (Q:{quality_score:.2f})"
                    status_color = (0, 255, 255)
            else:
                status = f"🔍 RECHERCHE GANT LONGUE DISTANCE (Zoom {self.zoom_factor:.1f}x)"
                status_color = (100, 100, 255)
            
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # === BARRE DE ZOOM LONGUE DISTANCE ===
            self._draw_enhanced_zoom_bar(frame, 10, 60)
            
            # === COMPTEUR ÉCHANTILLONS IA ===
            ai_text = f"🧠 Échantillons IA: {self.ai_export_count}"
            cv2.putText(frame, ai_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
            
            # === ANALYSE COULEURS ===
            if self.red_orange_ratio_history:
                avg_ratios = np.mean(self.red_orange_ratio_history, axis=0)
                red_ratio, orange_ratio = avg_ratios
                
                color_text = f"🎨 R/O: {red_ratio:.2f}/{orange_ratio:.2f}"
                cv2.putText(frame, color_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # === STATISTIQUES LONGUE DISTANCE ===
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            quality_rate = (self.quality_count / max(self.detection_count, 1)) * 100 if self.detection_count > 0 else 0
            
            stats_y = h - 120
            stats = f"Frames: {self.frame_count} | Détections: {detection_rate:.1f}% | Qualité: {quality_rate:.1f}%"
            cv2.putText(frame, stats, (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Paramètres longue distance
            stats_y += 20
            ld_text = f"Longue Distance | Min aire: {self.min_area} | Zoom max: {self.zoom_max:.1f}x"
            cv2.putText(frame, ld_text, (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            
            # === FPS ===
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # === HISTORIQUE DÉTECTION ===
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-15:]])
            cv2.putText(frame, f"Hist: {history}", (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # === ZONE DE TRACKING ===
            if self.search_zone and self.zoom_factor > 1.5:
                self._draw_ai_tracking_zone(frame)
            
            return frame
            
        except Exception as e:
            logger.debug(f"AI overlay error: {e}")
            return frame

    def _draw_enhanced_zoom_bar(self, frame, x, y):
        """Barre de zoom longue distance"""
        try:
            bar_w, bar_h = 250, 15
            
            # Fond
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
            
            # Niveau zoom
            zoom_progress = (self.zoom_factor - self.zoom_min) / (self.zoom_max - self.zoom_min)
            zoom_w = int(bar_w * zoom_progress)
            
            # Couleur selon niveau longue distance
            if self.zoom_factor > 4.0:
                zoom_color = (0, 0, 255)    # Rouge - zoom très élevé
            elif self.zoom_factor > 3.0:
                zoom_color = (0, 100, 255)  # Orange - zoom élevé
            elif self.zoom_factor > 2.0:
                zoom_color = (0, 200, 255)  # Jaune
            else:
                zoom_color = (100, 255, 200)  # Vert
            
            cv2.rectangle(frame, (x, y), (x + zoom_w, y + bar_h), zoom_color, -1)
            
            # Texte longue distance
            cv2.putText(frame, f"Zoom LD: {self.zoom_factor:.1f}x / {self.zoom_max:.1f}x", 
                       (x + bar_w + 15, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Enhanced zoom bar error: {e}")

    def _draw_ai_tracking_zone(self, frame):
        """Zone de tracking pour IA"""
        try:
            if self.search_zone:
                cx, cy, zw, zh = self.search_zone
                
                # Rectangle zone de recherche IA
                cv2.rectangle(frame, (cx - zw//2, cy - zh//2), (cx + zw//2, cy + zh//2), 
                             (255, 100, 255), 3)
                
                # Point central IA
                cv2.circle(frame, (cx, cy), 6, (255, 100, 255), -1)
                
                # Texte IA
                cv2.putText(frame, "IA TRACK", (cx - 35, cy - zh//2 - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 2)
                           
                # Trajectoire prédite IA
                if len(self.zone_tracking) >= 2:
                    points = list(self.zone_tracking)[-5:]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (255, 150, 255), 2)
                        
        except Exception as e:
            logger.debug(f"AI tracking zone error: {e}")


# === PAS DE CONTRÔLE DRONE - DÉTECTION PURE ===
def main():
    """Fonction principale - Mode Détection Pure pour IA"""
    logger.info("=== BEBOP 2 AI READY DETECTION - LONGUE DISTANCE ===")
    logger.info("🧠 Détection gant pour IA - Longue distance optimisée")
    logger.info("🚫 Mode détection pure - AUCUNE commande drone")
    
    bebop = None
    pipe = None
    detector = None
    start_time = time.time()
    
    try:
        # === CONNEXION DRONE (FLUX UNIQUEMENT) ===
        logger.info("📡 Connexion au drone...")
        bebop = Bebop()
        if not bebop.connect(10):
            logger.error("❌ Échec connexion drone")
            return False

        logger.info("✅ Drone connecté! (mode détection pure)")
        
        # === FLUX VIDÉO UNIQUEMENT ===
        logger.info("📹 Démarrage flux vidéo...")
        bebop.start_video_stream()
        time.sleep(2)
        
        # === PIPELINE FFMPEG OPTIMISÉ IA ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg optimisé pour IA et longue distance
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '1500000',  # Plus élevé pour stabilité
            '-probesize', '1500000',
            '-i', sdp_path,
            '-vf', 'eq=saturation=1.2:gamma=0.9:contrast=1.1',  # Optimisé longue distance
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg IA longue distance: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=4*1024*1024)
            logger.info("✅ Pipeline IA initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR IA LONGUE DISTANCE ===
        detector = AIReadyGloveDetector()
        
        # === INTERFACE IA ===
        window_name = "Bebop 2 - IA Ready Detection (Longue Distance)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 80)
        logger.info("🧠 MODE IA READY - DÉTECTION PURE:")
        logger.info("  🚫 Aucune commande drone (flux stable)")
        logger.info("  📏 Optimisé longue distance (aire min: 80)")
        logger.info("  🔍 Zoom max: 6.0x pour objets distants")
        logger.info("  💾 Échantillons IA auto-sauvés")
        logger.info("=" * 80)
        logger.info("🎮 COMMANDES INTERFACE:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset détecteur")
        logger.info("  'z' = Reset zoom | '+/-' = Zoom manuel | 'd' = Debug IA")
        logger.info("  'c' = Calibrage couleurs | 'e' = Reset exposition")
        logger.info("  'i' = Info IA | 'x' = Export manuel IA")
        logger.info("=" * 80)
        
        # === BOUCLE PRINCIPALE IA ===
        logger.info("🎬 Démarrage détection IA longue distance...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        manual_ai_exports = 0
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.warning("⚠️ Frame incomplète...")
                    continue
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # === DÉTECTION IA PURE ===
                processed_frame, detected, ai_mask, ai_image = detector.detect_glove_for_ai(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Affichage IA en fenêtre séparée si détecté
                if detected and ai_image is not None:
                    cv2.imshow("AI Sample (Gant Détouré)", ai_image)
                
                # Logs périodiques IA
                fps_counter += 1
                if fps_counter % 120 == 0:  # Toutes les 4 secondes
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 120 / elapsed if elapsed > 0 else 0
                    
                    det_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
                    qual_avg = np.mean(detector.quality_scores) if detector.quality_scores else 0
                    
                    logger.info(f"🧠 IA | FPS: {display_fps:.1f} | "
                               f"Détections: {det_rate:.1f}% | "
                               f"Qualité: {qual_avg:.2f} | "
                               f"Zoom: {detector.zoom_factor:.1f}x | "
                               f"Samples: {detector.ai_export_count}")
                    last_fps_log = current_time
                
                # === GESTION TOUCHES IA ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt mode IA")
                    break
                    
                elif key == ord('s'):
                    timestamp = int(time.time())
                    screenshot_name = f"ai_detection_{timestamp}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot IA: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('x') and detected and ai_image is not None:
                    # Export manuel échantillon IA
                    timestamp = int(time.time() * 1000)
                    manual_name = f"manual_ai_{timestamp}_{manual_ai_exports:03d}.png"
                    manual_path = os.path.join(AI_SAMPLES_DIR, manual_name)
                    cv2.imwrite(manual_path, ai_image)
                    manual_ai_exports += 1
                    logger.info(f"🧠 Export manuel IA: {manual_name}")
                    
                elif key == ord('r'):
                    old_count = detector.detection_count
                    old_ai_count = detector.ai_export_count
                    detector = AIReadyGloveDetector()
                    logger.info(f"🔄 Détecteur IA reset (détections: {old_count}, samples: {old_ai_count})")
                    
                elif key == ord('z'):
                    detector.zoom_factor = 1.0
                    detector.target_zoom = 1.0
                    detector.search_zone = None
                    detector.zone_tracking.clear()
                    logger.info("🔍 Zoom IA reset")
                    
                elif key == ord('+') or key == ord('='):
                    detector.target_zoom = min(detector.zoom_max, detector.target_zoom + 0.5)
                    logger.info(f"🔍 Zoom IA manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('-'):
                    detector.target_zoom = max(detector.zoom_min, detector.target_zoom - 0.5)
                    logger.info(f"🔍 Zoom IA manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('c'):
                    detector.red_orange_ratio_history.clear()
                    detector.color_balance_history.clear()
                    logger.info("🎨 Calibrage couleurs IA reset")
                    
                elif key == ord('e'):
                    detector.auto_exposure_factor = 1.0
                    detector.brightness_history.clear()
                    logger.info("💡 Exposition IA reset")
                    
                elif key == ord('i'):
                    # Infos IA détaillées
                    logger.info("🧠 INFOS IA DÉTAILLÉES:")
                    logger.info(f"   === DÉTECTION IA ===")
                    logger.info(f"   Frames total: {detector.frame_count}")
                    logger.info(f"   Détections: {detector.detection_count}")
                    logger.info(f"   Échantillons IA: {detector.ai_export_count}")
                    logger.info(f"   Exports manuels: {manual_ai_exports}")
                    logger.info(f"   Dossier: {AI_SAMPLES_DIR}")
                    
                    logger.info(f"   === LONGUE DISTANCE ===")
                    logger.info(f"   Aire min: {detector.min_area}")
                    logger.info(f"   Zoom: {detector.zoom_factor:.2f}x -> {detector.target_zoom:.2f}x")
                    logger.info(f"   Zoom max: {detector.zoom_max:.1f}x")
                    logger.info(f"   Exposition: {detector.auto_exposure_factor:.2f}")
                    
                    if detector.area_history:
                        areas = list(detector.area_history)[-5:]
                        logger.info(f"   Aires récentes: {areas}")
                        logger.info(f"   Aire médiane: {np.median(detector.area_history):.0f}")
                    
                elif key == ord('d'):
                    # Debug IA complet
                    logger.info("🔍 DEBUG IA COMPLET:")
                    logger.info(f"   Paramètres IA: {AI_OUTPUT_SIZE}")
                    logger.info(f"   Sauvegarde: {SAVE_AI_SAMPLES}")
                    logger.info(f"   Intervalle export: {detector.ai_export_interval:.2f}s")
                    
                    if detector.red_orange_ratio_history:
                        avg_ratios = np.mean(detector.red_orange_ratio_history, axis=0)
                        logger.info(f"   Équilibrage R/O: {avg_ratios[0]:.2f}/{avg_ratios[1]:.2f}")
                    if detector.quality_scores:
                        logger.info(f"   Qualité min/moy/max: {min(detector.quality_scores):.2f}/"
                                   f"{np.mean(detector.quality_scores):.2f}/"
                                   f"{max(detector.quality_scores):.2f}")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle IA: {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique IA: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # === NETTOYAGE ET STATS FINALES IA ===
        logger.info("🧹 Nettoyage IA...")
        
        if detector:
            total_runtime = time.time() - start_time
            detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
            quality_rate = (detector.quality_count / max(detector.detection_count, 1)) * 100 if detector.detection_count > 0 else 0
            avg_quality = np.mean(detector.quality_scores) if detector.quality_scores else 0
            
            logger.info("=" * 80)
            logger.info("🧠 STATS FINALES IA LONGUE DISTANCE:")
            logger.info(f"  ⏱️ Durée session: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames traitées: {detector.frame_count}")
            logger.info(f"  ⚡ FPS moyen: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections totales: {detector.detection_count} ({detection_rate:.1f}%)")
            logger.info(f"  ⭐ Détections qualité: {detector.quality_count} ({quality_rate:.1f}%)")
            logger.info(f"  📈 Qualité moyenne: {avg_quality:.2f}")
            logger.info(f"  🔍 Zoom final: {detector.zoom_factor:.1f}x (max: {detector.zoom_max:.1f}x)")
            logger.info(f"  📈 Ajustements zoom: {detector.zoom_adjustments}")
            logger.info(f"  💡 Exposition finale: {detector.auto_exposure_factor:.2f}")
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            
            # === STATS IA SPÉCIFIQUES ===
            logger.info(f"  🧠 DATASET IA:")
            logger.info(f"      📁 Dossier: {AI_SAMPLES_DIR}")
            logger.info(f"      💾 Échantillons auto: {detector.ai_export_count}")
            logger.info(f"      ✋ Exports manuels: {manual_ai_exports}")
            logger.info(f"      📊 Total échantillons: {detector.ai_export_count + manual_ai_exports}")
            logger.info(f"      🔢 Taille IA: {AI_OUTPUT_SIZE}")
            
            # === STATS LONGUE DISTANCE ===
            logger.info(f"  📏 LONGUE DISTANCE:")
            logger.info(f"      🎯 Aire minimum: {detector.min_area} (vs 200 standard)")
            logger.info(f"      🔍 Zoom maximum: {detector.zoom_max:.1f}x (vs 4.5x standard)")
            logger.info(f"      📐 Seuils tolérants: aspect 0.25-4.0 (vs 0.4-3.0)")
            
            if detector.red_orange_ratio_history:
                final_ratios = np.mean(detector.red_orange_ratio_history, axis=0)
                logger.info(f"  🎨 Équilibrage final R/O: {final_ratios[0]:.2f}/{final_ratios[1]:.2f}")
            if detector.area_history:
                min_area = min(detector.area_history)
                max_area = max(detector.area_history)
                med_area = np.median(detector.area_history)
                logger.info(f"  📊 Aires détectées: min={min_area:.0f}, med={med_area:.0f}, max={max_area:.0f}")
            
            logger.info("=" * 80)
            
            # === RAPPORT IA FINAL ===
            total_samples = detector.ai_export_count + manual_ai_exports
            if total_samples > 0:
                logger.info("🧠 RAPPORT DATASET IA:")
                logger.info(f"   ✅ Dataset prêt avec {total_samples} échantillons")
                logger.info(f"   📁 Emplacement: {os.path.abspath(AI_SAMPLES_DIR)}")
                logger.info(f"   📐 Format: {AI_OUTPUT_SIZE[0]}x{AI_OUTPUT_SIZE[1]} pixels, fond noir")
                logger.info(f"   🎯 Optimisé pour reconnaissance de gestes")
                logger.info(f"   📈 Qualité moyenne échantillons: {avg_quality:.2f}")
                logger.info("   🚀 Prêt pour entraînement IA!")
            else:
                logger.info("⚠️ Aucun échantillon IA généré - vérifiez la détection")
        
        # Fermeture fenêtres IA
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Interfaces IA fermées")
        except:
            pass
        
        if pipe:
            try:
                pipe.terminate()
                logger.info("✅ Pipeline IA fermé")
            except:
                pass
        
        if bebop:
            try:
                bebop.disconnect()
                logger.info("✅ Drone déconnecté (mode IA)")
            except:
                pass
        
        logger.info("🎉 Session IA Ready terminée - Dataset prêt!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie IA: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception finale IA: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)