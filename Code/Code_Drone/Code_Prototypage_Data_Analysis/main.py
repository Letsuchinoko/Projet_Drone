import cv2
import numpy as np
import time
import subprocess
import sys
import logging
import os
import pyparrot
from pyparrot.Bebop import Bebop
from collections import deque

# === PARAMÈTRES OPTIMISÉS DETECTION BICOLORE PURE ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_pure_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT OPTIMISÉ - VERSION PURE ===
class OptimizedBicolorGloveDetector:
    def __init__(self):
        # Configuration de base
        self.detection_history = deque(maxlen=15)
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3
        
        # Paramètres de détection affinés
        self.min_area = 200
        self.max_area = 120000
        self.min_contour_points = 8
        
        # Historique des couleurs détectées
        self.color_balance_history = deque(maxlen=20)
        self.red_orange_ratio_history = deque(maxlen=10)
        
        # Kernels morphologiques optimisés
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # === SYSTÈME DE ZOOM ADAPTATIF AMÉLIORÉ ===
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.12
        self.zoom_min = 1.0
        self.zoom_max = 4.5
        
        # Calibrage amélioré
        self.area_reference = 2800
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

    def detect_glove_optimized(self, frame):
        """Détection optimisée avec analyse bicolore équilibrée"""
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            # === PHASE 1: ANALYSE ÉCLAIRAGE ===
            exposure_adjusted_frame = self._adaptive_exposure_correction(frame)
            
            # === PHASE 2: RECHERCHE GLOBALE OU ZOOMÉE ===
            if sum(self.stable_detections) < 2 or self.zoom_factor < 1.3:
                # Recherche globale améliorée
                global_result = self._enhanced_global_detection(exposure_adjusted_frame)
                if global_result:
                    detected, contour, area, quality_score = global_result
                    if detected and quality_score > 0.4:
                        self._update_zoom_and_tracking(area, contour)
                        return self._finalize_detection(original_frame, detected, contour, area, quality_score)
            
            # === PHASE 3: DÉTECTION ZOOMÉE OPTIMISÉE ===
            zoomed_frame, zoom_info = self._apply_predictive_zoom(exposure_adjusted_frame)
            
            # Détection bicolore équilibrée
            red_mask, orange_mask, combined_mask = self._create_balanced_color_masks(zoomed_frame)
            
            # Morphologie adaptative
            processed_mask = self._advanced_morphology(combined_mask)
            
            # Analyse de contours avec scoring
            best_contour, area, quality_score = self._intelligent_contour_selection(
                processed_mask, red_mask, orange_mask, zoomed_frame
            )
            
            # Remapping vers coordonnées originales
            if best_contour is not None:
                best_contour = self._remap_contour_to_original(best_contour, zoom_info)
                area = cv2.contourArea(best_contour)
            
            # Validation avec critères de qualité
            detected = self._validate_detection(best_contour, area, quality_score)
            
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
            
            return self._finalize_detection(original_frame, final_detected, best_contour, area, quality_score)
            
        except Exception as e:
            logger.debug(f"Optimized detection error: {e}")
            return original_frame, False

    def _adaptive_exposure_correction(self, frame):
        """Correction d'exposition adaptative pour améliorer les couleurs"""
        try:
            # Analyse de luminosité
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            self.brightness_history.append(mean_brightness)
            
            # Calcul facteur d'exposition
            target_brightness = 128
            brightness_avg = np.mean(self.brightness_history) if self.brightness_history else mean_brightness
            
            if brightness_avg < 100:  # Sombre
                self.auto_exposure_factor = min(1.4, self.auto_exposure_factor + 0.05)
            elif brightness_avg > 160:  # Trop clair
                self.auto_exposure_factor = max(0.7, self.auto_exposure_factor - 0.05)
            else:
                self.auto_exposure_factor = max(0.95, min(1.05, self.auto_exposure_factor))
            
            # Application correction douce
            if abs(self.auto_exposure_factor - 1.0) > 0.05:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.auto_exposure_factor, 0, 255)
                corrected_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                return corrected_frame
            
            return frame
            
        except Exception as e:
            logger.debug(f"Exposure correction error: {e}")
            return frame

    def _enhanced_global_detection(self, frame):
        """Détection globale améliorée avec analyse bicolore"""
        try:
            # Préprocessing amélioré
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # Masques couleur équilibrés
            red_mask, orange_mask, combined_mask = self._create_balanced_color_masks(blurred)
            
            # Morphologie légère pour recherche globale
            processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel_medium)
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, self.kernel_small)
            
            # Sélection contour avec analyse qualité
            best_contour, area, quality_score = self._intelligent_contour_selection(
                processed_mask, red_mask, orange_mask, blurred
            )
            
            if best_contour is not None and area > self.min_area and quality_score > 0.3:
                return True, best_contour, area, quality_score
            
            return None
            
        except Exception as e:
            logger.debug(f"Enhanced global detection error: {e}")
            return None

    def _create_balanced_color_masks(self, frame):
        """Création de masques couleur équilibrés rouge/orange"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            
            # Analyse adaptative des couleurs présentes
            brightness_factor = np.mean(hsv[:, :, 2]) / 255.0
            saturation_factor = np.mean(hsv[:, :, 1]) / 255.0
            
            # Ajustements dynamiques
            sat_adjust = max(-25, min(25, int((0.7 - saturation_factor) * 50)))
            val_adjust = max(-20, min(20, int((0.5 - brightness_factor) * 40)))
            
            # === MASQUES ROUGE OPTIMISÉS ===
            # Rouge principal (teinte basse)
            red_lower1 = np.array([0, max(140, 160 + sat_adjust), max(120, 140 + val_adjust)])
            red_upper1 = np.array([8, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            # Rouge principal (teinte haute)
            red_lower2 = np.array([172, max(140, 160 + sat_adjust), max(120, 140 + val_adjust)])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            # Rouge avec orange
            red_orange_lower = np.array([0, max(120, 140 + sat_adjust), max(100, 120 + val_adjust)])
            red_orange_upper = np.array([12, 255, 255])
            mask_red_orange = cv2.inRange(hsv, red_orange_lower, red_orange_upper)
            
            # Combinaison rouge
            mask_red = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_red_orange))
            
            # === MASQUES ORANGE OPTIMISÉS ===
            # Orange vif
            orange_bright_lower = np.array([8, max(160, 180 + sat_adjust), max(140, 160 + val_adjust)])
            orange_bright_upper = np.array([18, 255, 255])
            mask_orange_bright = cv2.inRange(hsv, orange_bright_lower, orange_bright_upper)
            
            # Orange moyen
            orange_mid_lower = np.array([10, max(130, 150 + sat_adjust), max(120, 140 + val_adjust)])
            orange_mid_upper = np.array([22, 255, 245])
            mask_orange_mid = cv2.inRange(hsv, orange_mid_lower, orange_mid_upper)
            
            # Orange avec ombres
            orange_shadow_lower = np.array([12, max(100, 120 + sat_adjust//2), max(80, 100 + val_adjust)])
            orange_shadow_upper = np.array([20, 200, 200])
            mask_orange_shadow = cv2.inRange(hsv, orange_shadow_lower, orange_shadow_upper)
            
            # Combinaison orange
            mask_orange = cv2.bitwise_or(mask_orange_bright, 
                         cv2.bitwise_or(mask_orange_mid, mask_orange_shadow))
            
            # === ÉQUILIBRAGE ROUGE/ORANGE ===
            # Calcul des proportions
            red_pixels = np.sum(mask_red > 0)
            orange_pixels = np.sum(mask_orange > 0)
            total_color_pixels = red_pixels + orange_pixels
            
            if total_color_pixels > 0:
                red_ratio = red_pixels / total_color_pixels
                orange_ratio = orange_pixels / total_color_pixels
                self.red_orange_ratio_history.append((red_ratio, orange_ratio))
                
                # Équilibrage dynamique si orange domine trop
                if orange_ratio > 0.75 and red_ratio < 0.25:
                    # Boost du rouge, réduction orange
                    mask_red = cv2.dilate(mask_red, self.kernel_small, iterations=1)
                    mask_orange = cv2.erode(mask_orange, self.kernel_small, iterations=1)
                    logger.debug("Équilibrage: boost rouge, réduction orange")
                    
                elif red_ratio > 0.75 and orange_ratio < 0.25:
                    # Boost de l'orange
                    mask_orange = cv2.dilate(mask_orange, self.kernel_small, iterations=1)
                    logger.debug("Équilibrage: boost orange")
            
            # === EXCLUSIONS INTELLIGENTES ===
            # Exclusion peau adaptée
            skin_lower = np.array([5, 50, 80])
            skin_upper = np.array([15, min(140, 120 - sat_adjust//2), 240])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            mask_skin = cv2.erode(mask_skin, self.kernel_small, iterations=1)
            
            # Application exclusions
            mask_red = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_skin))
            mask_orange = cv2.bitwise_and(mask_orange, cv2.bitwise_not(mask_skin))
            
            # === COMBINAISON FINALE PONDÉRÉE ===
            # Pondération pour équilibrer
            avg_ratios = np.mean(self.red_orange_ratio_history, axis=0) if self.red_orange_ratio_history else (0.5, 0.5)
            
            if len(avg_ratios) == 2:
                red_weight = 1.0 + max(0, 0.4 - avg_ratios[0]) * 2  # Boost si rouge sous-représenté
                orange_weight = 1.0 + max(0, 0.4 - avg_ratios[1]) * 2
                
                # Application pondération
                if red_weight > 1.1:
                    mask_red = cv2.dilate(mask_red, self.kernel_small, iterations=1)
                if orange_weight > 1.1:
                    mask_orange = cv2.dilate(mask_orange, self.kernel_small, iterations=1)
            
            # Combinaison finale
            mask_combined = cv2.bitwise_or(mask_red, mask_orange)
            
            # Nettoyage final
            mask_combined = cv2.medianBlur(mask_combined, 3)
            
            # Bordures
            border_size = max(8, int(20 / max(self.zoom_factor, 1.0)))
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_combined = cv2.bitwise_and(mask_combined, border_mask)
            
            return mask_red, mask_orange, mask_combined
            
        except Exception as e:
            logger.debug(f"Balanced color masks error: {e}")
            # Fallback
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            fallback = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([25, 255, 255]))
            return fallback, fallback, fallback

    def _advanced_morphology(self, mask):
        """Morphologie avancée adaptée au zoom"""
        try:
            # Sélection kernels selon zoom
            if self.zoom_factor > 2.5:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
                iterations_close = 2
                iterations_open = 1
            elif self.zoom_factor > 1.5:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                iterations_close = 2
                iterations_open = 1
            else:
                kernel_close = self.kernel_medium
                kernel_open = self.kernel_small
                iterations_close = 1
                iterations_open = 1
            
            # Fermeture pour connecter les zones
            processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=iterations_close)
            
            # Ouverture pour nettoyer
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_open, iterations=iterations_open)
            
            # Dilatation finale légère pour robustesse
            if self.zoom_factor > 2.0:
                processed = cv2.dilate(processed, self.kernel_small, iterations=1)
            
            return processed
            
        except Exception as e:
            logger.debug(f"Advanced morphology error: {e}")
            return mask

    def _intelligent_contour_selection(self, mask, red_mask, orange_mask, frame):
        """Sélection intelligente avec analyse bicolore"""
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, 0, 0
            
            best_contour = None
            best_score = 0
            best_area = 0
            
            # Ajustement seuils selon zoom
            min_area_adj = self.min_area * max(1.0, self.zoom_factor ** 1.3)
            max_area_adj = self.max_area * max(1.0, self.zoom_factor ** 1.5)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base
                if area < min_area_adj or area > max_area_adj:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Analyse géométrique
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                if not (0.4 <= aspect_ratio <= 3.0):
                    continue
                
                # === ANALYSE BICOLORE DANS LE CONTOUR ===
                mask_contour = np.zeros(mask.shape, dtype=np.uint8)
                cv2.fillPoly(mask_contour, [contour], 255)
                
                # Pixels rouge et orange dans le contour
                red_in_contour = np.sum(cv2.bitwise_and(red_mask, mask_contour) > 0)
                orange_in_contour = np.sum(cv2.bitwise_and(orange_mask, mask_contour) > 0)
                total_color_in_contour = red_in_contour + orange_in_contour
                
                if total_color_in_contour == 0:
                    continue
                
                # Ratio rouge/orange dans le contour
                red_ratio_contour = red_in_contour / total_color_in_contour
                orange_ratio_contour = orange_in_contour / total_color_in_contour
                
                # Score bicolore (optimal entre 30% et 70% pour chaque couleur)
                bicolor_score = 1.0
                if red_ratio_contour < 0.1 or orange_ratio_contour < 0.1:
                    bicolor_score *= 0.3  # Pénalité pour quasi-monocolore
                elif red_ratio_contour > 0.9 or orange_ratio_contour > 0.9:
                    bicolor_score *= 0.5  # Pénalité pour trop monocolore
                else:
                    # Bonus pour équilibre bicolore
                    balance = 1.0 - abs(red_ratio_contour - orange_ratio_contour)
                    bicolor_score *= (0.7 + balance * 0.6)
                
                # === SCORE GÉOMÉTRIQUE ===
                # Aire normalisée
                area_score = min(area / (self.area_reference * max(self.zoom_factor, 1.0)), 1.0)
                
                # Forme (proximité rectangle)
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                shape_score = min(extent * 1.5, 1.0)  # Bonus pour formes pleines
                
                # Position (centré est mieux)
                center_x, center_y = x + w//2, y + h//2
                dist_from_center = np.sqrt((center_x - WIDTH//2)**2 + (center_y - HEIGHT//2)**2)
                max_dist = np.sqrt((WIDTH//2)**2 + (HEIGHT//2)**2)
                position_score = 1.0 - (dist_from_center / max_dist) * 0.3
                
                # === SCORE QUALITÉ COULEUR ===
                # Analyse saturation et valeur dans le contour
                hsv_roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
                contour_relative = contour - [x, y]  # Ajustement coordonnées
                
                mask_roi = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_roi, [contour_relative], 255)
                
                # Saturation moyenne
                sat_mean = np.mean(hsv_roi[:, :, 1][mask_roi > 0]) if np.any(mask_roi > 0) else 0
                sat_score = min(sat_mean / 180.0, 1.0)  # Bonus pour saturation élevée
                
                # Valeur (luminosité)
                val_mean = np.mean(hsv_roi[:, :, 2][mask_roi > 0]) if np.any(mask_roi > 0) else 0
                val_score = min(val_mean / 200.0, 1.0)
                
                color_quality_score = (sat_score + val_score) / 2.0
                
                # === SCORE FINAL ===
                final_score = (
                    area_score * 0.25 +
                    bicolor_score * 0.35 +  # Poids important pour bicolore
                    shape_score * 0.15 +
                    position_score * 0.1 +
                    color_quality_score * 0.15
                )
                
                # Bonus historique si proche des détections précédentes
                if self.zone_tracking:
                    last_zone = self.zone_tracking[-1]
                    zone_dist = np.sqrt((center_x - last_zone[0])**2 + (center_y - last_zone[1])**2)
                    if zone_dist < 100:  # Proche de la dernière détection
                        final_score *= 1.2
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
                    best_area = area
            
            return best_contour, best_area, best_score
            
        except Exception as e:
            logger.debug(f"Intelligent contour selection error: {e}")
            return None, 0, 0

    def _validate_detection(self, contour, area, quality_score):
        """Validation avec critères de qualité stricts"""
        try:
            if contour is None or area <= 0:
                return False
            
            # Seuils adaptatifs
            min_quality = 0.35 if self.zoom_factor > 2.0 else 0.4
            min_area_final = self.min_area * max(1.0, (self.zoom_factor ** 1.2))
            
            # Validation de base
            if quality_score < min_quality:
                return False
            if area < min_area_final:
                return False
            
            # Validation géométrique avancée
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if not (0.3 <= aspect_ratio <= 2.8):
                return False
            
            # Validation par rapport à l'historique
            if self.area_history:
                area_median = np.median(self.area_history)
                if area > area_median * 3 or area < area_median * 0.3:
                    return False  # Changement trop brusque
            
            return True
            
        except Exception as e:
            logger.debug(f"Detection validation error: {e}")
            return False

    def _update_zoom_and_tracking(self, area, contour):
        """Mise à jour zoom et tracking prédictif"""
        try:
            # Mise à jour historique
            self.area_history.append(area)
            
            # Calcul zoom optimal avec lissage amélioré
            if area < 600:          # Très loin
                self.target_zoom = min(self.zoom_max, 4.0)
            elif area < 1200:       # Loin
                self.target_zoom = min(self.zoom_max, 3.0)
            elif area < 2400:       # Moyen-loin
                self.target_zoom = 2.2
            elif area < 4800:       # Moyen
                self.target_zoom = 1.6
            elif area < 8000:       # Proche
                self.target_zoom = 1.2
            else:                   # Très proche
                self.target_zoom = 1.0
            
            # Tracking zone prédictif
            if contour is not None:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.zone_tracking.append((cx, cy))
                    
                    # Prédiction position future basée sur mouvement
                    if len(self.zone_tracking) >= 3:
                        # Calcul vélocité
                        dx = self.zone_tracking[-1][0] - self.zone_tracking[-3][0]
                        dy = self.zone_tracking[-1][1] - self.zone_tracking[-3][1]
                        
                        # Prédiction
                        pred_x = cx + dx // 2
                        pred_y = cy + dy // 2
                        
                        # Zone de recherche prédictive
                        zone_size = max(80, int(150 / max(self.zoom_factor, 1.0)))
                        self.search_zone = (pred_x, pred_y, zone_size, zone_size)
            
            self.zoom_adjustments += 1
            
        except Exception as e:
            logger.debug(f"Zoom and tracking update error: {e}")

    def _handle_detection_loss(self):
        """Gestion de la perte de détection"""
        try:
            # Zoom out progressif
            if sum(self.stable_detections) == 0:
                self.target_zoom = max(self.zoom_min, self.target_zoom * 0.92)
                
                # Élargissement zone de recherche
                if self.search_zone and self.target_zoom < 1.5:
                    x, y, w, h = self.search_zone
                    self.search_zone = (x, y, min(w * 1.1, WIDTH//2), min(h * 1.1, HEIGHT//2))
                
                # Reset zone si zoom très faible
                if self.target_zoom < 1.15:
                    self.search_zone = None
                    self.zone_tracking.clear()
                    
        except Exception as e:
            logger.debug(f"Detection loss handling error: {e}")

    def _apply_predictive_zoom(self, frame):
        """Application zoom avec prédiction améliorée"""
        try:
            h, w = frame.shape[:2]
            
            # Lissage zoom
            self.zoom_factor += (self.target_zoom - self.zoom_factor) * self.zoom_smooth_factor
            self.zoom_factor = np.clip(self.zoom_factor, self.zoom_min, self.zoom_max)
            
            if self.zoom_factor <= 1.08:
                return frame, {'zoom': 1.0, 'offset_x': 0, 'offset_y': 0, 'crop_w': w, 'crop_h': h}
            
            # Zone de focus intelligente
            if self.search_zone:
                center_x, center_y, zone_w, zone_h = self.search_zone
                # Contraintes dans les limites de l'image
                center_x = max(zone_w//2, min(center_x, w - zone_w//2))
                center_y = max(zone_h//2, min(center_y, h - zone_h//2))
            else:
                # Centre par défaut avec léger décalage vers le haut (position naturelle main)
                center_x, center_y = w // 2, int(h * 0.45)
            
            # Calcul zone de crop
            crop_w = int(w / self.zoom_factor)
            crop_h = int(h / self.zoom_factor)
            
            # Positionnement crop centré sur zone de focus
            offset_x = max(0, min(center_x - crop_w // 2, w - crop_w))
            offset_y = max(0, min(center_y - crop_h // 2, h - crop_h))
            
            # Extraction et redimensionnement
            cropped = frame[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
            
            # Interpolation adaptée au niveau de zoom
            if self.zoom_factor > 3.0:
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
            logger.debug(f"Predictive zoom error: {e}")
            return frame, {'zoom': 1.0, 'offset_x': 0, 'offset_y': 0, 'crop_w': w, 'crop_h': h}

    def _remap_contour_to_original(self, contour, zoom_info):
        """Remapping optimisé vers coordonnées originales"""
        try:
            if zoom_info['zoom'] <= 1.08:
                return contour
            
            # Facteurs de conversion précis
            scale_x = zoom_info['crop_w'] / WIDTH
            scale_y = zoom_info['crop_h'] / HEIGHT
            
            # Remapping avec arrondi approprié
            remapped_contour = contour.copy().astype(np.float32)
            remapped_contour[:, :, 0] = remapped_contour[:, :, 0] * scale_x + zoom_info['offset_x']
            remapped_contour[:, :, 1] = remapped_contour[:, :, 1] * scale_y + zoom_info['offset_y']
            
            return np.round(remapped_contour).astype(np.int32)
            
        except Exception as e:
            logger.debug(f"Contour remapping error: {e}")
            return contour

    def _finalize_detection(self, frame, detected, contour, area, quality_score):
        """Finalisation avec visualisation enrichie"""
        try:
            # Mise à jour historiques
            self.detection_history.append(detected)
            if detected:
                self.detection_count += 1
            
            # Visualisation
            if detected and contour is not None:
                self._draw_advanced_detection(frame, contour, area, quality_score)
            
            # Interface enrichie
            result_frame = self._create_advanced_overlay(frame, detected, area, quality_score)
            
            return result_frame
            
        except Exception as e:
            logger.debug(f"Finalization error: {e}")
            return frame

    def _draw_advanced_detection(self, frame, contour, area, quality_score):
        """Visualisation avancée de la détection"""
        try:
            # Couleurs selon qualité et distance
            if quality_score > 0.7:
                if area > 3000:
                    color = (0, 255, 0)      # Vert - excellente qualité proche
                    quality_text = "PARFAIT"
                else:
                    color = (0, 255, 100)    # Vert-jaune - excellente qualité loin
                    quality_text = "EXCELLENT"
            elif quality_score > 0.5:
                color = (0, 255, 255)        # Jaune - bonne qualité
                quality_text = "BON"
            else:
                color = (0, 150, 255)        # Orange - qualité acceptable
                quality_text = "ACCEPTABLE"
            
            # Contour principal avec épaisseur adaptée
            thickness = max(2, int(3 * min(self.zoom_factor, 2.0)))
            cv2.drawContours(frame, [contour], -1, color, thickness)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre avec croix
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Point central
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
                
                # Croix directionnelle
                cross_size = 15
                cv2.line(frame, (cx - cross_size, cy), (cx + cross_size, cy), (255, 255, 255), 2)
                cv2.line(frame, (cx, cy - cross_size), (cx, cy + cross_size), (255, 255, 255), 2)
            
            # Informations détaillées
            info_y = max(y - 20, 30)
            cv2.putText(frame, f"GANT {quality_text}", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            info_y += 25
            cv2.putText(frame, f"Qualite: {quality_score:.2f} | Aire: {int(area)}", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_y += 20
            distance_est = self._estimate_distance(area)
            cv2.putText(frame, f"Distance: ~{distance_est:.1f}m | Zoom: {self.zoom_factor:.1f}x", 
                       (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                       
        except Exception as e:
            logger.debug(f"Advanced drawing error: {e}")

    def _estimate_distance(self, area):
        """Estimation de distance basée sur l'aire"""
        try:
            # Calibrage approximatif (à ajuster selon votre setup)
            if area > 8000:
                return 0.5  # Très proche
            elif area > 4000:
                return 1.0  # Proche
            elif area > 2000:
                return 2.0  # Moyen
            elif area > 1000:
                return 3.5  # Loin
            elif area > 500:
                return 5.0  # Très loin
            else:
                return 7.0  # Très très loin
        except:
            return 0.0

    def _create_advanced_overlay(self, frame, detected, area, quality_score):
        """Interface utilisateur enrichie"""
        try:
            h, w = frame.shape[:2]
            
            # === STATUS PRINCIPAL ===
            if detected:
                if quality_score > 0.6:
                    status = f"🎯 GANT CAPTURÉ (Q:{quality_score:.2f})"
                    status_color = (0, 255, 0)
                else:
                    status = f"🎯 GANT DÉTECTÉ (Q:{quality_score:.2f})"
                    status_color = (0, 255, 255)
            else:
                status = f"🔍 RECHERCHE GANT (Zoom {self.zoom_factor:.1f}x)"
                status_color = (100, 100, 255)
            
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # === BARRE DE ZOOM VISUELLE ===
            self._draw_zoom_bar(frame, 10, 65)
            
            # === BARRE DE QUALITÉ ===
            if detected and quality_score > 0:
                self._draw_quality_bar(frame, 10, 95, quality_score)
            
            # === ANALYSE COULEURS ===
            self._draw_color_analysis(frame, 10, 125)
            
            # === STATISTIQUES ===
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            quality_rate = (self.quality_count / max(self.detection_count, 1)) * 100 if self.detection_count > 0 else 0
            
            stats_y = h - 100
            stats = f"Frames: {self.frame_count} | Détections: {detection_rate:.1f}% | Qualité moy: {quality_rate:.1f}%"
            cv2.putText(frame, stats, (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Zoom et exposition
            stats_y += 20
            expo_text = f"Exposition: {self.auto_exposure_factor:.2f} | Zoom ajust: {self.zoom_adjustments}"
            cv2.putText(frame, expo_text, (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            
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
            cv2.putText(frame, f"Historique: {history}", (10, h - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # === ZONE DE TRACKING ===
            if self.search_zone and self.zoom_factor > 1.3:
                self._draw_tracking_zone(frame)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Advanced overlay error: {e}")
            return frame

    def _draw_zoom_bar(self, frame, x, y):
        """Barre de zoom visuelle"""
        try:
            bar_w, bar_h = 200, 12
            
            # Fond
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
            
            # Niveau zoom
            zoom_progress = (self.zoom_factor - self.zoom_min) / (self.zoom_max - self.zoom_min)
            zoom_w = int(bar_w * zoom_progress)
            
            # Couleur selon niveau
            if self.zoom_factor > 3.0:
                zoom_color = (0, 100, 255)  # Rouge - zoom élevé
            elif self.zoom_factor > 2.0:
                zoom_color = (0, 200, 255)  # Orange
            else:
                zoom_color = (100, 255, 200)  # Vert
            
            cv2.rectangle(frame, (x, y), (x + zoom_w, y + bar_h), zoom_color, -1)
            
            # Texte
            cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x / {self.zoom_max:.1f}x", 
                       (x + bar_w + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Zoom bar error: {e}")

    def _draw_quality_bar(self, frame, x, y, quality_score):
        """Barre de qualité visuelle"""
        try:
            bar_w, bar_h = 150, 10
            
            # Fond
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
            
            # Niveau qualité
            quality_w = int(bar_w * quality_score)
            
            # Couleur selon qualité
            if quality_score > 0.7:
                quality_color = (0, 255, 0)    # Vert
            elif quality_score > 0.5:
                quality_color = (0, 255, 255)  # Jaune
            else:
                quality_color = (0, 150, 255)  # Orange
            
            cv2.rectangle(frame, (x, y), (x + quality_w, y + bar_h), quality_color, -1)
            
            # Texte
            cv2.putText(frame, f"Qualité: {quality_score:.2f}", 
                       (x + bar_w + 10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Quality bar error: {e}")

    def _draw_color_analysis(self, frame, x, y):
        """Analyse des couleurs détectées"""
        try:
            if self.red_orange_ratio_history:
                avg_ratios = np.mean(self.red_orange_ratio_history, axis=0)
                red_ratio, orange_ratio = avg_ratios
                
                # Barres rouge et orange
                bar_w = 80
                red_w = int(bar_w * red_ratio)
                orange_w = int(bar_w * orange_ratio)
                
                # Rouge
                cv2.rectangle(frame, (x, y), (x + red_w, y + 8), (0, 0, 255), -1)
                cv2.putText(frame, f"R:{red_ratio:.2f}", (x, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
                # Orange
                cv2.rectangle(frame, (x + 100, y), (x + 100 + orange_w, y + 8), (0, 165, 255), -1)
                cv2.putText(frame, f"O:{orange_ratio:.2f}", (x + 100, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
                           
        except Exception as e:
            logger.debug(f"Color analysis error: {e}")

    def _draw_tracking_zone(self, frame):
        """Zone de tracking prédictif"""
        try:
            if self.search_zone:
                cx, cy, zw, zh = self.search_zone
                
                # Rectangle zone de recherche
                cv2.rectangle(frame, (cx - zw//2, cy - zh//2), (cx + zw//2, cy + zh//2), 
                             (150, 100, 255), 2)
                
                # Point central
                cv2.circle(frame, (cx, cy), 4, (255, 100, 150), -1)
                
                # Texte
                cv2.putText(frame, "ZONE TRACK", (cx - 35, cy - zh//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 100, 255), 1)
                           
                # Trajectoire prédite
                if len(self.zone_tracking) >= 2:
                    points = list(self.zone_tracking)[-5:]  # 5 derniers points
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (200, 150, 255), 1)
                        
        except Exception as e:
            logger.debug(f"Tracking zone error: {e}")


def main():
    """Fonction principale - Mode Détection Pure"""
    logger.info("=== BEBOP 2 OPTIMIZED BICOLOR DETECTION - PURE MODE ===")
    logger.info("🎯 Détection gant bicolore optimisée rouge/orange")
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
        
        # === PIPELINE FFMPEG OPTIMISÉ ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg avec paramètres optimisés pour qualité couleur
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '1000000',  # Plus élevé pour qualité
            '-probesize', '1000000',
            '-i', sdp_path,
            '-vf', 'eq=saturation=1.1:gamma=0.95',  # Amélioration couleurs
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg optimisé couleurs: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=2*1024*1024)
            logger.info("✅ Pipeline optimisé initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR OPTIMISÉ ===
        detector = OptimizedBicolorGloveDetector()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - Détection Bicolore Pure"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 70)
        logger.info("🎮 COMMANDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset détecteur")
        logger.info("  'z' = Reset zoom | '+/-' = Zoom manuel | 'd' = Debug")
        logger.info("  'c' = Calibrage couleurs | 'e' = Ajust exposition")
        logger.info("=" * 70)
        logger.info("🎯 OPTIMISATIONS:")
        logger.info("  ✓ Équilibrage rouge/orange automatique")
        logger.info("  ✓ Correction exposition adaptative")
        logger.info("  ✓ Zoom prédictif avec tracking")
        logger.info("  ✓ Scoring qualité bicolore")
        logger.info("  ✓ Morphologie adaptée au zoom")
        logger.info("  🚫 AUCUNE commande drone (flux stable)")
        logger.info("=" * 70)
        
        # === BOUCLE PRINCIPALE PURE ===
        logger.info("🎬 Démarrage détection pure...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.warning("⚠️ Frame incomplète, reconnexion...")
                    continue
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Détection optimisée
                processed_frame, detected = detector.detect_glove_optimized(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Logs périodiques
                fps_counter += 1
                if fps_counter % 90 == 0:  # Toutes les 3 secondes
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 90 / elapsed if elapsed > 0 else 0
                    
                    # Stats détaillées
                    det_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
                    qual_avg = np.mean(detector.quality_scores) if detector.quality_scores else 0
                    
                    logger.info(f"📊 FPS: {display_fps:.1f} | "
                               f"Détections: {det_rate:.1f}% | "
                               f"Qualité moy: {qual_avg:.2f} | "
                               f"Zoom: {detector.zoom_factor:.1f}x | "
                               f"Exposition: {detector.auto_exposure_factor:.2f}")
                    last_fps_log = current_time
                
                # Gestion touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    timestamp = int(time.time())
                    screenshot_name = f"pure_capture_{timestamp}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    old_count = detector.detection_count
                    detector = OptimizedBicolorGloveDetector()
                    logger.info(f"🔄 Détecteur reset (détections: {old_count})")
                    
                elif key == ord('z'):
                    detector.zoom_factor = 1.0
                    detector.target_zoom = 1.0
                    detector.search_zone = None
                    detector.zone_tracking.clear()
                    logger.info("🔍 Zoom et tracking reset")
                    
                elif key == ord('+') or key == ord('='):
                    detector.target_zoom = min(detector.zoom_max, detector.target_zoom + 0.5)
                    logger.info(f"🔍 Zoom manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('-'):
                    detector.target_zoom = max(detector.zoom_min, detector.target_zoom - 0.5)
                    logger.info(f"🔍 Zoom manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('c'):
                    # Reset historique couleurs
                    detector.red_orange_ratio_history.clear()
                    detector.color_balance_history.clear()
                    logger.info("🎨 Calibrage couleurs reset")
                    
                elif key == ord('e'):
                    # Reset exposition
                    detector.auto_exposure_factor = 1.0
                    detector.brightness_history.clear()
                    logger.info("💡 Exposition reset")
                    
                elif key == ord('d'):
                    # Debug détaillé
                    logger.info("🔍 INFOS DEBUG PURE:")
                    logger.info(f"   Frames total: {detector.frame_count}")
                    logger.info(f"   Détections: {detector.detection_count}")
                    logger.info(f"   Détections qualité: {detector.quality_count}")
                    logger.info(f"   Zoom: {detector.zoom_factor:.2f}x -> {detector.target_zoom:.2f}x")
                    logger.info(f"   Exposition: {detector.auto_exposure_factor:.2f}")
                    if detector.red_orange_ratio_history:
                        avg_ratios = np.mean(detector.red_orange_ratio_history, axis=0)
                        logger.info(f"   Ratio Rouge/Orange: {avg_ratios[0]:.2f}/{avg_ratios[1]:.2f}")
                    if detector.quality_scores:
                        logger.info(f"   Qualité min/moy/max: {min(detector.quality_scores):.2f}/"
                                   f"{np.mean(detector.quality_scores):.2f}/"
                                   f"{max(detector.quality_scores):.2f}")
                    if detector.area_history:
                        logger.info(f"   Aires récentes: {list(detector.area_history)[-5:]}")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle: {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # === NETTOYAGE ET STATS FINALES ===
        logger.info("🧹 Nettoyage...")
        
        if detector:
            total_runtime = time.time() - start_time
            detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
            quality_rate = (detector.quality_count / max(detector.detection_count, 1)) * 100 if detector.detection_count > 0 else 0
            avg_quality = np.mean(detector.quality_scores) if detector.quality_scores else 0
            
            logger.info("=" * 70)
            logger.info("📊 STATS FINALES MODE PURE:")
            logger.info(f"  ⏱️ Durée: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames: {detector.frame_count}")
            logger.info(f"  ⚡ FPS moyen: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections: {detector.detection_count} ({detection_rate:.1f}%)")
            logger.info(f"  ⭐ Détections qualité: {detector.quality_count} ({quality_rate:.1f}%)")
            logger.info(f"  📈 Qualité moyenne: {avg_quality:.2f}")
            logger.info(f"  🔍 Zoom final: {detector.zoom_factor:.1f}x")
            logger.info(f"  📈 Ajustements zoom: {detector.zoom_adjustments}")
            logger.info(f"  💡 Exposition finale: {detector.auto_exposure_factor:.2f}")
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            if detector.red_orange_ratio_history:
                final_ratios = np.mean(detector.red_orange_ratio_history, axis=0)
                logger.info(f"  🎨 Équilibrage final R/O: {final_ratios[0]:.2f}/{final_ratios[1]:.2f}")
            if detector.area_history:
                logger.info(f"  📏 Aire moyenne: {np.mean(detector.area_history):.0f}")
            logger.info("=" * 70)
        
        if pipe:
            try:
                pipe.terminate()
                logger.info("✅ Pipeline fermé")
            except:
                pass
        
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Interface fermée")
        except:
            pass
        
        if bebop:
            try:
                bebop.disconnect()
                logger.info("✅ Drone déconnecté")
            except:
                pass
        
        logger.info("🎉 Session détection pure terminée!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception finale: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)