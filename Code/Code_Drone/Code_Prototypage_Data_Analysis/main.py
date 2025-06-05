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

# === PARAMÈTRES OPTIMISÉS AVEC ZOOM ADAPTATIF ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_adaptive_zoom.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT AVEC ZOOM ADAPTATIF ===
class AdaptiveZoomGloveDetector:
    def __init__(self):
        # Configuration de base
        self.detection_history = deque(maxlen=10)
        self.stable_detections = deque(maxlen=3)
        self.confidence_threshold = 2
        
        # Paramètres de détection
        self.min_area = 150      # Plus petit pour distance
        self.max_area = 100000   # Plus grand pour zoom
        self.min_contour_points = 6
        
        # Kernels morphologiques
        self.kernel_small = np.ones((2, 2), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.kernel_large = np.ones((8, 8), np.uint8)
        
        # === SYSTÈME DE ZOOM ADAPTATIF ===
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.1  # Lissage du zoom
        self.zoom_min = 1.0
        self.zoom_max = 4.0
        
        # Calibrage distance/aire
        self.area_reference = 3000    # Aire de référence à distance normale
        self.area_history = deque(maxlen=10)
        self.last_detection_area = None
        
        # Zone de recherche adaptative
        self.search_zone = None
        self.zone_expand_factor = 1.5
        
        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.zoom_adjustments = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def detect_glove_with_zoom(self, frame):
        """Détection avec zoom adaptatif intelligent et fallback global"""
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            h, w = frame.shape[:2]
            
            # === STRATÉGIE DOUBLE : ZOOM + GLOBAL ===
            
            # 1. Essayer détection avec zoom si zoom actif
            zoom_detected = False
            zoom_contour = None
            zoom_area = 0
            
            if self.zoom_factor > 1.1:  # Si zoom significatif
                zoomed_frame = self._apply_full_frame_zoom(frame)
                zoom_detected, zoom_contour, zoom_area = self._detect_on_frame(zoomed_frame, self.zoom_factor)
                
                # Remapping des coordonnées si détection en zoom
                if zoom_detected and zoom_contour is not None:
                    zoom_contour = self._remap_contour_from_zoom(zoom_contour, self.zoom_factor)
                    zoom_area = cv2.contourArea(zoom_contour)
            
            # 2. Détection globale (toujours faire en parallèle)
            global_detected, global_contour, global_area = self._detect_on_frame(frame, 1.0)
            
            # === SÉLECTION DU MEILLEUR RÉSULTAT ===
            final_detected = False
            final_contour = None
            final_area = 0
            detection_source = "none"
            
            # Priorité : zoom si bon résultat, sinon global
            if zoom_detected and zoom_area > self.min_area:
                final_detected = True
                final_contour = zoom_contour
                final_area = zoom_area
                detection_source = "zoom"
            elif global_detected and global_area > self.min_area:
                final_detected = True
                final_contour = global_contour
                final_area = global_area
                detection_source = "global"
            
            # === STABILISATION ===
            self.stable_detections.append(final_detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            # === MISE À JOUR ZOOM ===
            if stable_detection and final_area > 0:
                self._update_zoom_from_area(final_area)
                self.area_history.append(final_area)
                self.last_detection_area = final_area
                
                # Mise à jour zone de recherche avec position actuelle
                if final_contour is not None:
                    self._update_search_zone_from_contour_position(final_contour)
            else:
                # Zoom out si pas de détection
                self._zoom_out_gradually()
            
            # === AFFICHAGE ===
            result_frame = self._finalize_detection_with_source(
                original_frame, stable_detection, final_contour, final_area, detection_source
            )
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Adaptive zoom detection error: {e}")
            return original_frame, False

    def _apply_full_frame_zoom(self, frame):
        """Application du zoom sur toute l'image (pas juste une zone)"""
        try:
            h, w = frame.shape[:2]
            
            if self.zoom_factor <= 1.05:
                return frame
            
            # Zone de focus basée sur dernière détection ou centre
            if self.search_zone is not None:
                center_x, center_y = self.search_zone[0], self.search_zone[1]
            else:
                center_x, center_y = w // 2, h // 2
            
            # Calcul de la zone de crop
            crop_w = int(w / self.zoom_factor)
            crop_h = int(h / self.zoom_factor)
            
            # Centrage sur la zone de recherche avec contraintes
            offset_x = max(0, min(center_x - crop_w // 2, w - crop_w))
            offset_y = max(0, min(center_y - crop_h // 2, h - crop_h))
            
            # Crop et redimensionnement vers taille originale
            cropped = frame[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
            zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Stocker les infos de transformation pour remapping
            self._zoom_transform = {
                'zoom': self.zoom_factor,
                'offset_x': offset_x,
                'offset_y': offset_y,
                'crop_w': crop_w,
                'crop_h': crop_h
            }
            
            return zoomed
            
        except Exception as e:
            logger.debug(f"Full frame zoom error: {e}")
            return frame

    def _detect_on_frame(self, frame, zoom_level):
        """Détection sur une frame donnée"""
        try:
            # Conversion HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Création masque adapté au zoom
            mask = self._create_zoom_optimized_mask(hsv, zoom_level)
            
            # Morphologie adaptée
            mask = self._adaptive_morphology_for_zoom(mask, zoom_level)
            
            # Détection contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour, area = self._select_best_contour_zoom(contours, zoom_level)
            
            detected = best_contour is not None and area > self.min_area
            
            return detected, best_contour, area
            
        except Exception as e:
            logger.debug(f"Frame detection error: {e}")
            return False, None, 0

    def _remap_contour_from_zoom(self, contour, zoom_factor):
        """Remapping du contour depuis image zoomée vers originale"""
        try:
            if zoom_factor <= 1.05 or not hasattr(self, '_zoom_transform'):
                return contour
            
            transform = self._zoom_transform
            
            # Facteurs de conversion
            scale_x = transform['crop_w'] / WIDTH
            scale_y = transform['crop_h'] / HEIGHT
            
            # Remapping des points
            remapped_contour = contour.copy()
            remapped_contour[:, :, 0] = (contour[:, :, 0] * scale_x + transform['offset_x']).astype(np.int32)
            remapped_contour[:, :, 1] = (contour[:, :, 1] * scale_y + transform['offset_y']).astype(np.int32)
            
            return remapped_contour
            
        except Exception as e:
            logger.debug(f"Contour remapping error: {e}")
            return contour

    def _create_zoom_optimized_mask(self, hsv, zoom_level=1.0):
        """Masque optimisé selon le niveau de zoom"""
        try:
            h, w = hsv.shape[:2]
            
            # Ajustement des seuils selon le zoom
            sat_boost = min(15, int(8 * zoom_level))
            val_boost = min(10, int(5 * zoom_level))
            
            # === MASQUES ORANGE OPTIMISÉS ===
            orange_main_lower = np.array([12, max(120, 160 - sat_boost), max(120, 160 - val_boost)])
            orange_main_upper = np.array([20, 255, 255])
            mask_orange_main = cv2.inRange(hsv, orange_main_lower, orange_main_upper)
            
            orange_bright_lower = np.array([10, max(140, 180 - sat_boost), max(140, 180 - val_boost)])
            orange_bright_upper = np.array([18, 255, 255])
            mask_orange_bright = cv2.inRange(hsv, orange_bright_lower, orange_bright_upper)
            
            orange_shadow_lower = np.array([14, max(100, 120 - sat_boost//2), max(100, 140 - val_boost)])
            orange_shadow_upper = np.array([19, 200, 220])
            mask_orange_shadow = cv2.inRange(hsv, orange_shadow_lower, orange_shadow_upper)
            
            # === MASQUES ROUGE OPTIMISÉS ===
            red_main_lower1 = np.array([0, max(120, 160 - sat_boost), max(120, 160 - val_boost)])
            red_main_upper1 = np.array([6, 255, 255])
            mask_red_main1 = cv2.inRange(hsv, red_main_lower1, red_main_upper1)
            
            red_main_lower2 = np.array([174, max(120, 160 - sat_boost), max(120, 160 - val_boost)])
            red_main_upper2 = np.array([180, 255, 255])
            mask_red_main2 = cv2.inRange(hsv, red_main_lower2, red_main_upper2)
            
            # Combinaisons
            mask_orange = cv2.bitwise_or(mask_orange_main, 
                         cv2.bitwise_or(mask_orange_bright, mask_orange_shadow))
            
            mask_red = cv2.bitwise_or(mask_red_main1, mask_red_main2)
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # === EXCLUSIONS ADAPTATIVES ===
            # Peau (moins stricte en zoom élevé)
            skin_sat_max = max(80, int(120 - 20 * (zoom_level - 1.0)))
            skin_lower = np.array([5, 60, 120])
            skin_upper = np.array([15, skin_sat_max, 220])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Application exclusions
            mask_skin_eroded = cv2.erode(mask_skin, self.kernel_small, iterations=1)
            mask_final = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_skin_eroded))
            
            # Bordures adaptatives (plus petites en zoom élevé)
            border_size = max(3, int(12 / zoom_level))
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            # Nettoyage adaptatif
            blur_size = max(3, int(5 / zoom_level))
            if blur_size % 2 == 0:
                blur_size += 1
            mask_final = cv2.medianBlur(mask_final, blur_size)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Zoom optimized mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _adaptive_morphology_for_zoom(self, mask, zoom_level=1.0):
        """Morphologie adaptée au niveau de zoom"""
        try:
            # Kernels adaptatifs selon le zoom
            if zoom_level > 2.5:
                # Zoom élevé: kernels plus grands
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                iterations = 1
            elif zoom_level > 1.5:
                # Zoom moyen
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                iterations = 1
            else:
                # Zoom faible/normal
                kernel_close = self.kernel_medium
                kernel_open = self.kernel_small
                iterations = 1
            
            # Application morphologie
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            return mask
            
        except Exception as e:
            logger.debug(f"Adaptive morphology error: {e}")
            return mask

    def _select_best_contour_zoom(self, contours, zoom_level=1.0):
        """Sélection contour optimisée pour zoom"""
        if not contours:
            return None, 0
            
        try:
            best_contour = None
            best_score = 0
            best_area = 0
            
            # Ajustement des seuils selon le zoom
            min_area_adjusted = max(self.min_area, self.min_area * (zoom_level ** 0.5))
            max_area_adjusted = self.max_area * (zoom_level ** 1.2)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < min_area_adjusted or area > max_area_adjusted:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Analyse géométrique
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                if not (0.2 <= aspect_ratio <= 3.0):
                    continue
                
                # Score basé sur l'aire
                ideal_area = self.area_reference * max(0.5, zoom_level * 0.8)
                area_score = min(area / ideal_area, 1.0) if ideal_area > 0 else 0.5
                
                # Bonus pour position (moins important en zoom élevé)
                position_bonus = 1.0
                if zoom_level < 2.0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    frame_center_x = WIDTH // 2
                    frame_center_y = HEIGHT // 2
                    
                    dist_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                    max_dist = np.sqrt(frame_center_x**2 + frame_center_y**2)
                    position_bonus = 1.0 - (dist_from_center / max_dist) * 0.3
                
                final_score = area_score * position_bonus
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
                    best_area = area
            
            return best_contour, best_area
            
        except Exception as e:
            logger.debug(f"Zoom contour selection error: {e}")
            return None, 0

    def _update_search_zone_from_contour_position(self, contour):
        """Mise à jour de la zone de recherche basée sur position du contour"""
        try:
            if contour is None:
                return
            
            # Calculer le centre du contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Mise à jour de la zone avec une certaine inertie
                if self.search_zone is None:
                    self.search_zone = [cx, cy, WIDTH//3, HEIGHT//3]
                else:
                    # Lissage de la position
                    self.search_zone[0] = int(0.7 * self.search_zone[0] + 0.3 * cx)
                    self.search_zone[1] = int(0.7 * self.search_zone[1] + 0.3 * cy)
                    
        except Exception as e:
            logger.debug(f"Search zone update error: {e}")

    def _finalize_detection_with_source(self, frame, detected, contour, area, source):
        """Finalisation avec indication de la source de détection"""
        try:
            # Historique
            self.detection_history.append(detected)
            if detected:
                self.detection_count += 1
            
            # Dessin avec indication de source
            if detected and contour is not None:
                self._draw_detection_with_source(frame, contour, area, source)
            
            # Overlay avec informations étendues
            result_frame = self._add_enhanced_zoom_overlay(frame, detected, area, source)
            
            return result_frame
            
        except Exception as e:
            logger.debug(f"Finalization error: {e}")
            return frame

    def _draw_detection_with_source(self, frame, contour, area, source):
        """Dessin avec indication de la source (zoom/global)"""
        try:
            # Couleur selon la source et distance
            if source == "zoom":
                if area > 4000:
                    color = (0, 255, 0)      # Vert - zoom proche
                    distance_text = "ZOOM PROCHE"
                elif area > 1500:
                    color = (0, 255, 255)    # Jaune - zoom moyen
                    distance_text = "ZOOM MOYEN"
                else:
                    color = (0, 150, 255)    # Orange - zoom loin
                    distance_text = "ZOOM LOIN"
            else:  # global
                color = (255, 0, 255)        # Magenta - détection globale
                distance_text = "GLOBAL"
            
            # Contour principal
            thickness = 4 if source == "zoom" else 2
            cv2.drawContours(frame, [contour], -1, color, thickness)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            
            # Texte avec source et distance
            cv2.putText(frame, f"GANT {distance_text}", (x, max(y - 15, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Aire: {int(area)} | {source.upper()}", 
                       (x, max(y - 40, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Enhanced drawing error: {e}")

    def _add_enhanced_zoom_overlay(self, frame, detected, area, source):
        """Overlay enrichi avec informations de zoom et source"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal avec source
            if detected:
                if source == "zoom":
                    status = f"🎯 GANT DETECTE ZOOM {self.zoom_factor:.1f}x"
                    color = (0, 255, 0)
                else:
                    status = f"🎯 GANT DETECTE (GLOBAL)"
                    color = (255, 0, 255)
            else:
                status = f"🔍 RECHERCHE (Z:{self.zoom_factor:.1f}x)"
                color = (0, 255, 255)
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Indicateur de stratégie active
            strategy_text = f"Stratégie: {'ZOOM+GLOBAL' if self.zoom_factor > 1.1 else 'GLOBAL SEUL'}"
            cv2.putText(frame, strategy_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            # Barre de zoom visuelle améliorée
            zoom_bar_width = 200
            zoom_bar_height = 15
            zoom_x, zoom_y = 10, 90
            
            # Barre de fond
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_bar_width, zoom_y + zoom_bar_height), 
                         (50, 50, 50), -1)
            
            # Barre de zoom actuel
            zoom_width = int(zoom_bar_width * (self.zoom_factor - 1.0) / (self.zoom_max - 1.0))
            zoom_color = (0, 255, 255) if self.zoom_factor > 2.0 else (100, 255, 100)
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_width, zoom_y + zoom_bar_height), 
                         zoom_color, -1)
            
            # Indication target zoom
            target_width = int(zoom_bar_width * (self.target_zoom - 1.0) / (self.zoom_max - 1.0))
            cv2.line(frame, (zoom_x + target_width, zoom_y - 2), 
                    (zoom_x + target_width, zoom_y + zoom_bar_height + 2), (255, 255, 255), 2)
            
            cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x → {self.target_zoom:.1f}x", 
                       (zoom_x + zoom_bar_width + 10, zoom_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stats de performance
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Det: {detection_rate:.1f}%"
            cv2.putText(frame, stats_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Information source et aire
            if area and area > 0:
                source_color = (0, 255, 0) if source == "zoom" else (255, 0, 255)
                area_text = f"Source: {source.upper()} | Aire: {int(area)}"
                cv2.putText(frame, area_text, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, source_color, 1)
            
            # FPS
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Zone de recherche zoom (si active)
            if self.zoom_factor > 1.2 and self.search_zone:
                zone_x, zone_y = self.search_zone[0], self.search_zone[1]
                zone_w, zone_h = self.search_zone[2], self.search_zone[3]
                
                # Calcul de la zone de crop actuelle
                crop_w = int(w / self.zoom_factor)
                crop_h = int(h / self.zoom_factor)
                offset_x = max(0, min(zone_x - crop_w // 2, w - crop_w))
                offset_y = max(0, min(zone_y - crop_h // 2, h - crop_h))
                
                cv2.rectangle(frame, (offset_x, offset_y), (offset_x + crop_w, offset_y + crop_h), 
                             (0, 255, 255), 2)
                cv2.putText(frame, f"ZONE ZOOM {self.zoom_factor:.1f}x", 
                           (offset_x, max(offset_y - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Enhanced zoom overlay error: {e}")
            return frame

    def _global_detection_phase(self, frame):
        """Phase de détection globale (recherche large)"""
        try:
            # Détection rapide sur frame complète
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = self._create_zoom_optimized_mask(hsv)
            
            # Morphologie légère pour recherche globale
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour, area = self._select_best_contour_zoom(contours)
            
            if best_contour is not None and area > self.min_area:
                return True, best_contour, area
            
            return None
            
        except Exception as e:
            logger.debug(f"Global detection error: {e}")
            return None

    def _apply_adaptive_zoom(self, frame):
        """Application du zoom adaptatif intelligent"""
        try:
            h, w = frame.shape[:2]
            
            # Lissage du zoom pour éviter les oscillations
            self.zoom_factor += (self.target_zoom - self.zoom_factor) * self.zoom_smooth_factor
            self.zoom_factor = np.clip(self.zoom_factor, self.zoom_min, self.zoom_max)
            
            if self.zoom_factor <= 1.05:  # Pas de zoom si facteur proche de 1
                return frame, {'zoom': 1.0, 'offset_x': 0, 'offset_y': 0, 'crop_w': w, 'crop_h': h}
            
            # Zone de focus basée sur dernière détection
            if self.search_zone is not None:
                center_x, center_y, zone_w, zone_h = self.search_zone
            else:
                # Centre de l'image par défaut
                center_x, center_y = w // 2, h // 2
                zone_w, zone_h = w // 2, h // 2
            
            # Calcul de la zone de crop
            crop_w = int(w / self.zoom_factor)
            crop_h = int(h / self.zoom_factor)
            
            # Centrage sur la zone de recherche
            offset_x = max(0, min(center_x - crop_w // 2, w - crop_w))
            offset_y = max(0, min(center_y - crop_h // 2, h - crop_h))
            
            # Crop et redimensionnement
            cropped = frame[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
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
            logger.debug(f"Adaptive zoom error: {e}")
            return frame, {'zoom': 1.0, 'offset_x': 0, 'offset_y': 0, 'crop_w': w, 'crop_h': h}

    def _create_zoom_optimized_mask(self, hsv):
        """Masque optimisé pour détection zoomée"""
        try:
            h, w = hsv.shape[:2]
            
            # Ajustement des seuils selon le zoom
            sat_boost = min(20, int(10 * self.zoom_factor))  # Plus de saturation pour zoom élevé
            val_boost = min(15, int(8 * self.zoom_factor))
            
            # === MASQUES ORANGE OPTIMISÉS ===
            orange_main_lower = np.array([12, 160 - sat_boost, 160 - val_boost])
            orange_main_upper = np.array([20, 255, 255])
            mask_orange_main = cv2.inRange(hsv, orange_main_lower, orange_main_upper)
            
            orange_bright_lower = np.array([10, 180 - sat_boost, 180 - val_boost])
            orange_bright_upper = np.array([18, 255, 255])
            mask_orange_bright = cv2.inRange(hsv, orange_bright_lower, orange_bright_upper)
            
            orange_shadow_lower = np.array([14, 120 - sat_boost//2, 140 - val_boost])
            orange_shadow_upper = np.array([19, 200, 220])
            mask_orange_shadow = cv2.inRange(hsv, orange_shadow_lower, orange_shadow_upper)
            
            # === MASQUES ROUGE OPTIMISÉS ===
            red_main_lower1 = np.array([0, 160 - sat_boost, 160 - val_boost])
            red_main_upper1 = np.array([6, 255, 255])
            mask_red_main1 = cv2.inRange(hsv, red_main_lower1, red_main_upper1)
            
            red_main_lower2 = np.array([174, 160 - sat_boost, 160 - val_boost])
            red_main_upper2 = np.array([180, 255, 255])
            mask_red_main2 = cv2.inRange(hsv, red_main_lower2, red_main_upper2)
            
            # Combinaisons
            mask_orange = cv2.bitwise_or(mask_orange_main, 
                         cv2.bitwise_or(mask_orange_bright, mask_orange_shadow))
            
            mask_red = cv2.bitwise_or(mask_red_main1, mask_red_main2)
            
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # === EXCLUSIONS ADAPTATIVES ===
            # Exclusions moins strictes en zoom élevé
            exclusion_strictness = max(0.5, 1.0 - (self.zoom_factor - 1.0) * 0.3)
            
            # Peau (ajustée selon zoom)
            skin_sat_max = int(120 * exclusion_strictness)
            skin_lower = np.array([5, 60, 120])
            skin_upper = np.array([15, skin_sat_max, 220])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Application exclusions
            mask_skin_eroded = cv2.erode(mask_skin, self.kernel_small, iterations=1)
            mask_final = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_skin_eroded))
            
            # Bordures adaptatives (plus petites en zoom)
            border_size = max(5, int(15 / self.zoom_factor))
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            # Nettoyage adaptatif
            blur_size = max(3, int(5 / self.zoom_factor))
            if blur_size % 2 == 0:
                blur_size += 1
            mask_final = cv2.medianBlur(mask_final, blur_size)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Zoom optimized mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _adaptive_morphology_for_zoom(self, mask):
        """Morphologie adaptée au niveau de zoom"""
        try:
            # Kernels adaptatifs selon le zoom
            if self.zoom_factor > 2.5:
                # Zoom élevé: kernels plus grands
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                iterations = 2
            elif self.zoom_factor > 1.5:
                # Zoom moyen
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                iterations = 1
            else:
                # Zoom faible
                kernel_close = self.kernel_medium
                kernel_open = self.kernel_small
                iterations = 1
            
            # Application morphologie
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            return mask
            
        except Exception as e:
            logger.debug(f"Adaptive morphology error: {e}")
            return mask

    def _select_best_contour_zoom(self, contours):
        """Sélection contour optimisée pour zoom"""
        if not contours:
            return None, 0
            
        try:
            best_contour = None
            best_score = 0
            best_area = 0
            
            # Ajustement des seuils selon le zoom
            min_area_adjusted = self.min_area * (self.zoom_factor ** 1.5)
            max_area_adjusted = self.max_area * (self.zoom_factor ** 1.5)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < min_area_adjusted or area > max_area_adjusted:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Analyse géométrique
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                if not (0.3 <= aspect_ratio <= 2.5):
                    continue
                
                # Score basé sur l'aire et la forme
                area_score = min(area / (self.area_reference * self.zoom_factor), 1.0)
                
                # Bonus pour position centrale (important en zoom)
                center_x = x + w // 2
                center_y = y + h // 2
                frame_center_x = WIDTH // 2
                frame_center_y = HEIGHT // 2
                
                dist_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                max_dist = np.sqrt(frame_center_x**2 + frame_center_y**2)
                position_score = 1.0 - (dist_from_center / max_dist) * 0.5
                
                final_score = area_score * position_score
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
                    best_area = area
            
            return best_contour, best_area
            
        except Exception as e:
            logger.debug(f"Zoom contour selection error: {e}")
            return None, 0

    def _remap_contour_to_original(self, contour, zoom_info):
        """Remapping du contour vers les coordonnées originales"""
        try:
            if zoom_info['zoom'] <= 1.05:
                return contour
            
            # Facteurs de conversion
            scale_x = zoom_info['crop_w'] / WIDTH
            scale_y = zoom_info['crop_h'] / HEIGHT
            
            # Remapping des points
            remapped_contour = contour.copy()
            remapped_contour[:, :, 0] = (contour[:, :, 0] * scale_x + zoom_info['offset_x']).astype(np.int32)
            remapped_contour[:, :, 1] = (contour[:, :, 1] * scale_y + zoom_info['offset_y']).astype(np.int32)
            
            return remapped_contour
            
        except Exception as e:
            logger.debug(f"Contour remapping error: {e}")
            return contour

    def _update_zoom_from_area(self, area):
        """Mise à jour du zoom basée sur l'aire détectée"""
        try:
            if area <= 0:
                return
            
            # Calcul du zoom optimal basé sur l'aire
            # Plus l'aire est petite, plus on zoome
            area_ratio = self.area_reference / area
            
            # Fonction de zoom adaptative
            if area < 800:           # Très petit (loin)
                self.target_zoom = min(self.zoom_max, 3.5)
            elif area < 1500:        # Petit (moyennement loin)
                self.target_zoom = min(self.zoom_max, 2.5)
            elif area < 3000:        # Moyen (distance normale)
                self.target_zoom = 1.8
            elif area < 6000:        # Grand (proche)
                self.target_zoom = 1.3
            else:                    # Très grand (très proche)
                self.target_zoom = 1.0
            
            # Mise à jour de la zone de recherche
            self._update_search_zone_from_contour(area)
            
            self.zoom_adjustments += 1
            
        except Exception as e:
            logger.debug(f"Zoom update error: {e}")

    def _zoom_out_gradually(self):
        """Zoom out graduel si pas de détection"""
        try:
            # Réduction progressive du zoom si pas de détection
            if sum(self.stable_detections) == 0:
                self.target_zoom = max(self.zoom_min, self.target_zoom * 0.95)
                
                # Reset de la zone de recherche si zoom faible
                if self.target_zoom < 1.2:
                    self.search_zone = None
                    
        except Exception as e:
            logger.debug(f"Zoom out error: {e}")

    def _update_search_zone_from_contour(self, area):
        """Mise à jour de la zone de recherche"""
        try:
            # Zone de recherche basée sur la dernière détection
            # Ici on pourrait utiliser la position du contour
            # Pour l'instant, on garde le centre avec expansion
            if self.search_zone is None:
                self.search_zone = (WIDTH//2, HEIGHT//2, WIDTH//3, HEIGHT//3)
                
        except Exception as e:
            logger.debug(f"Search zone update error: {e}")

    def _finalize_detection(self, frame, detected, contour, area):
        """Finalisation de la détection avec affichage"""
        try:
            # Historique
            self.detection_history.append(detected)
            if detected:
                self.detection_count += 1
            
            # Dessin
            if detected and contour is not None:
                self._draw_zoom_detection(frame, contour, area)
            
            # Overlay avec informations de zoom
            result_frame = self._add_zoom_overlay(frame, detected, area)
            
            return result_frame, detected
            
        except Exception as e:
            logger.debug(f"Finalization error: {e}")
            return frame, False

    def _draw_zoom_detection(self, frame, contour, area):
        """Dessin avec informations de zoom"""
        try:
            # Couleur selon la distance estimée
            if area > 4000:
                color = (0, 255, 0)      # Vert - proche
                distance_text = "PROCHE"
            elif area > 1500:
                color = (0, 255, 255)    # Jaune - moyen
                distance_text = "MOYEN"
            else:
                color = (0, 150, 255)    # Orange - loin
                distance_text = "LOIN"
            
            # Contour principal
            cv2.drawContours(frame, [contour], -1, color, 3)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            
            # Texte avec distance et aire
            cv2.putText(frame, f"GANT {distance_text}", (x, max(y - 15, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Aire: {int(area)} | Zoom: {self.zoom_factor:.1f}x", 
                       (x, max(y - 40, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            logger.debug(f"Zoom drawing error: {e}")

    def _add_zoom_overlay(self, frame, detected, area):
        """Overlay avec informations de zoom"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            if detected:
                if self.zoom_factor > 2.0:
                    status = f"🎯 GANT DETECTE (ZOOM {self.zoom_factor:.1f}x)"
                else:
                    status = "🎯 GANT DETECTE"
                color = (0, 255, 0)
            else:
                status = f"🔍 RECHERCHE (ZOOM {self.zoom_factor:.1f}x)"
                color = (0, 255, 255)
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Barre de zoom visuelle
            zoom_bar_width = 200
            zoom_bar_height = 15
            zoom_x, zoom_y = 10, 70
            
            # Barre de fond
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_bar_width, zoom_y + zoom_bar_height), 
                         (50, 50, 50), -1)
            
            # Barre de zoom actuel
            zoom_width = int(zoom_bar_width * (self.zoom_factor - 1.0) / (self.zoom_max - 1.0))
            zoom_color = (0, 255, 255) if self.zoom_factor > 1.5 else (100, 255, 100)
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_width, zoom_y + zoom_bar_height), 
                         zoom_color, -1)
            
            cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x", 
                       (zoom_x + zoom_bar_width + 10, zoom_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stats de performance
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Det: {detection_rate:.1f}% | Ajust. zoom: {self.zoom_adjustments}"
            cv2.putText(frame, stats_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Information aire courante
            if area and area > 0:
                area_text = f"Aire actuelle: {int(area)} | Target zoom: {self.target_zoom:.1f}x"
                cv2.putText(frame, area_text, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            
            # FPS
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Historique
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-10:]])
            cv2.putText(frame, f"Hist: {history}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Zone de zoom (si active)
            if self.zoom_factor > 1.2 and self.search_zone:
                zone_x, zone_y, zone_w, zone_h = self.search_zone
                cv2.rectangle(frame, 
                             (zone_x - zone_w//2, zone_y - zone_h//2), 
                             (zone_x + zone_w//2, zone_y + zone_h//2), 
                             (100, 100, 255), 2)
                cv2.putText(frame, "ZONE ZOOM", (zone_x - 40, zone_y - zone_h//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Zoom overlay error: {e}")
            return frame

# === CONTRÔLE DRONE SIMPLE ===
def simple_drone_control(bebop):
    logger.info("Contrôle drone démarré.")
    print("\n[Commandes drone]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f/b/g/d = mouvements | h/m = haut/bas | a/c = rotations\n")
    
    while True:
        try:
            key = input("> ").strip().lower()
        except EOFError:
            break
            
        if key == 't':
            bebop.safe_takeoff(10)
            print("✈️ Décollage")
        elif key == 'l':
            bebop.safe_land(10)
            print("🛬 Atterrissage")
        elif key == 'e':
            bebop.safe_land(10)
            bebop.disconnect()
            print("🔚 Arrêt")
            break
        elif key == 'f':
            bebop.fly_direct(roll=0, pitch=25, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-25, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'g':
            bebop.fly_direct(roll=-25, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'd':
            bebop.fly_direct(roll=25, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=0.3)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=0.3)
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-35, vertical_movement=0, duration=0.3)
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=35, vertical_movement=0, duration=0.3)

def main():
    """Fonction principale avec zoom adaptatif"""
    logger.info("=== BEBOP 2 ADAPTIVE ZOOM DETECTION ===")
    logger.info("🔍 Système de zoom adaptatif pour détection longue distance")
    
    bebop = None
    pipe = None
    detector = None
    start_time = time.time()
    
    try:
        # === CONNEXION DRONE ===
        logger.info("📡 Connexion au drone...")
        bebop = Bebop()
        if not bebop.connect(10):
            logger.error("❌ Échec connexion drone")
            return False

        logger.info("✅ Drone connecté!")
        
        # === FLUX VIDÉO ===
        logger.info("📹 Démarrage flux vidéo...")
        bebop.start_video_stream()
        time.sleep(2)
        
        # === CONTRÔLE DRONE ===
        ctrl_thread = threading.Thread(target=simple_drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg optimisé pour zoom adaptatif
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '800000',    # Légèrement plus pour qualité zoom
            '-probesize', '800000',
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg avec support zoom: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=1024*1024)
            logger.info("✅ Pipeline zoom initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR ZOOM ADAPTATIF ===
        detector = AdaptiveZoomGloveDetector()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - Zoom Adaptatif (3m+ optimisé)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 60)
        logger.info("🎮 COMMANDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset")
        logger.info("  'z' = Reset zoom | '+' = Zoom manuel + | '-' = Zoom manuel -")
        logger.info("=" * 60)
        logger.info("🔍 ZOOM ADAPTATIF:")
        logger.info("  Auto-zoom selon distance gant")
        logger.info("  Plage: 1.0x à 4.0x")
        logger.info("  Optimisé pour 3m+ de distance")
        logger.info("=" * 60)
        
        # === BOUCLE PRINCIPALE ZOOM ADAPTATIF ===
        logger.info("🎬 Démarrage détection avec zoom adaptatif...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        skip_counter = 0
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Skip frames léger pour performance
                skip_counter += 1
                if skip_counter % 2 != 0:
                    continue
                
                # Détection avec zoom adaptatif
                processed_frame, detected = detector.detect_glove_with_zoom(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Stats FPS
                fps_counter += 1
                if fps_counter % 60 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    
                    # Log avec informations de zoom
                    zoom_info = f"Zoom: {detector.zoom_factor:.1f}x (target: {detector.target_zoom:.1f}x)"
                    area_info = f"Aire moy: {np.mean(detector.area_history) if detector.area_history else 0:.0f}"
                    
                    logger.info(f"📊 FPS: {display_fps:.1f} | "
                               f"Détections: {detector.detection_count}/{detector.frame_count} "
                               f"({(detector.detection_count/max(detector.frame_count,1))*100:.1f}%) | "
                               f"{zoom_info} | {area_info}")
                    last_fps_log = current_time
                
                # Gestion touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    timestamp = int(time.time())
                    screenshot_name = f"zoom_capture_{timestamp}_{screenshot_count:03d}.png"
                    
                    # Ajout informations zoom dans le screenshot
                    info_frame = processed_frame.copy()
                    info_text = f"Zoom: {detector.zoom_factor:.1f}x | Frame: {detector.frame_count}"
                    cv2.putText(info_frame, info_text, (10, HEIGHT - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imwrite(screenshot_name, info_frame)
                    logger.info(f"📸 Screenshot avec zoom: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    # Reset complet
                    old_count = detector.detection_count
                    detector.__init__()
                    logger.info(f"🔄 Détecteur reset (détections: {old_count})")
                    
                elif key == ord('z'):
                    # Reset zoom seulement
                    detector.zoom_factor = 1.0
                    detector.target_zoom = 1.0
                    detector.search_zone = None
                    logger.info("🔍 Zoom reset à 1.0x")
                    
                elif key == ord('+') or key == ord('='):
                    # Zoom manuel +
                    detector.target_zoom = min(detector.zoom_max, detector.target_zoom + 0.5)
                    logger.info(f"🔍 Zoom manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('-'):
                    # Zoom manuel -
                    detector.target_zoom = max(detector.zoom_min, detector.target_zoom - 0.5)
                    logger.info(f"🔍 Zoom manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('d'):
                    # Debug informations détaillées
                    logger.info("🔍 INFOS DEBUG ZOOM:")
                    logger.info(f"   Zoom actuel: {detector.zoom_factor:.2f}x")
                    logger.info(f"   Zoom cible: {detector.target_zoom:.2f}x")
                    logger.info(f"   Ajustements zoom: {detector.zoom_adjustments}")
                    logger.info(f"   Aire de référence: {detector.area_reference}")
                    if detector.area_history:
                        logger.info(f"   Aires récentes: {list(detector.area_history)}")
                    if detector.search_zone:
                        logger.info(f"   Zone recherche: {detector.search_zone}")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle principale: {e}")
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # === NETTOYAGE ===
        logger.info("🧹 Nettoyage...")
        
        if detector:
            total_runtime = time.time() - start_time
            detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
            avg_zoom = detector.zoom_factor
            
            logger.info("=" * 60)
            logger.info("📊 STATS FINALES ZOOM ADAPTATIF:")
            logger.info(f"  ⏱️ Durée: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames: {detector.frame_count}")
            logger.info(f"  ⚡ FPS: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections: {detector.detection_count} ({detection_rate:.1f}%)")
            logger.info(f"  🔍 Zoom final: {detector.zoom_factor:.1f}x")
            logger.info(f"  📈 Ajustements zoom: {detector.zoom_adjustments}")
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            if detector.area_history:
                logger.info(f"  📏 Aire moyenne: {np.mean(detector.area_history):.0f}")
            logger.info("=" * 60)
        
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
        
        logger.info("🎉 Session zoom adaptatif terminée!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)