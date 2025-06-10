#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEBOP 2 - SYSTÈME DÉTECTION GANT COMPLET OPTIMISÉ
🎯 Toutes fonctionnalités avancées + Performance maximale
"""

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
import platform
import json

# === PARAMÈTRES OPTIMISÉS AVEC AMÉLIORATIONS COMPLÈTES ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_enhanced_optimized.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === UTILITAIRES DE CLUSTERING SIMPLE OPTIMISÉ ===
class OptimizedKMeans:
    """Implémentation K-means optimisée pour vitesse"""
    
    def __init__(self, n_clusters=2, max_iters=20):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit(self, data):
        """Ajustement optimisé du modèle K-means"""
        try:
            data = np.array(data)
            if len(data) > 1000:
                indices = np.random.choice(len(data), 1000, replace=False)
                data = data[indices]
            
            n_samples, n_features = data.shape
            
            # Initialisation K-means++
            np.random.seed(42)
            centers = []
            centers.append(data[np.random.randint(n_samples)])
            
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centers]) for x in data])
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                i = np.searchsorted(cumulative_probabilities, r)
                centers.append(data[i])
            
            self.cluster_centers_ = np.array(centers)
            
            # Convergence rapide
            for iteration in range(self.max_iters):
                distances = np.sqrt(((data - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
                self.labels_ = np.argmin(distances, axis=0)
                
                new_centers = np.array([data[self.labels_ == i].mean(axis=0) if np.any(self.labels_ == i) 
                                      else self.cluster_centers_[i] for i in range(self.n_clusters)])
                
                if iteration > 5 and np.allclose(self.cluster_centers_, new_centers, rtol=0.01):
                    break
                    
                self.cluster_centers_ = new_centers
            
            return self
            
        except Exception as e:
            logger.debug(f"Optimized K-means error: {e}")
            self.cluster_centers_ = np.array([[10, 180, 180], [0, 200, 200]])
            self.labels_ = np.zeros(len(data), dtype=int)
            return self

# === DÉTECTEUR GANT COMPLET OPTIMISÉ ===
class OptimizedCompleteGloveDetector:
    def __init__(self):
        # Configuration de base
        self.detection_history = deque(maxlen=10)
        self.stable_detections = deque(maxlen=4)
        self.confidence_threshold = 3
        
        # Paramètres de détection optimisés
        self.min_area = 100
        self.max_area = 120000
        self.min_contour_points = 6
        
        # === OPTIMISATION: FRAME PROCESSING ===
        self.frame_skip_counter = 0
        self.process_every_n_frames = 1
        self.adaptive_skip = True
        self.last_processing_time = 0
        
        # === STABILISATION IMAGE OPTIMISÉE ===
        self.stabilization_buffer = deque(maxlen=2)
        self.optical_flow_points = None
        self.stabilization_enabled = True
        
        # === TRACKING TEMPOREL OPTIMISÉ ===
        self.tracking_history = deque(maxlen=15)
        self.velocity_estimation = deque(maxlen=4)
        self.prediction_zone = None
        self.kalman_filter = self._init_kalman_filter()
        
        # === ADAPTATION LUMIÈRE OPTIMISÉE ===
        self.lighting_adaptation = deque(maxlen=3)
        self.exposure_compensation = 0
        self.brightness_history = deque(maxlen=3)
        self.contrast_history = deque(maxlen=3)
        
        # === DÉTECTION MULTI-ÉCHELLE OPTIMISÉE ===
        self.multi_scale_levels = [1.0, 0.8, 0.6]
        self.scale_weights = [1.0, 0.8, 0.6]
        self.scale_detection_cache = {}
        self.use_multi_scale = True
        
        # === ZONES DE CONFIANCE OPTIMISÉES ===
        self.confidence_zones = {
            'high': deque(maxlen=5),
            'medium': deque(maxlen=8),
            'low': deque(maxlen=10)
        }
        
        # === FILTRAGE ADAPTATIF COULEURS ===
        self.color_calibration = {
            'orange_ranges': [
                ([10, 180, 180], [18, 255, 255]),
                ([12, 200, 200], [20, 255, 255]),
                ([14, 140, 160], [19, 220, 240])
            ],
            'red_ranges': [
                ([0, 180, 180], [8, 255, 255]),
                ([172, 180, 180], [180, 255, 255]),
                ([0, 200, 200], [6, 255, 255])
            ]
        }
        
        # Auto-calibration optimisée
        self.color_samples = deque(maxlen=50)
        self.background_model = None
        
        # === ZOOM ADAPTATIF OPTIMISÉ ===
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.08
        self.zoom_min = 1.0
        self.zoom_max = 5.0
        
        # Prédiction de zoom
        self.zoom_prediction = deque(maxlen=3)
        self.zoom_stability_counter = 0
        
        # === VALIDATION OPTIMISÉE ===
        self.validation_cascade = {
            'geometry_threshold': 0.3,
            'color_threshold': 0.35,
            'motion_threshold': 0.2,
            'temporal_threshold': 0.25
        }
        
        # === PERFORMANCE ET STATS ===
        self.frame_count = 0
        self.detection_count = 0
        self.quality_scores = deque(maxlen=20)
        self.false_positive_rejection = 0
        self.processing_times = deque(maxlen=15)
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # === OPTIMISATIONS DYNAMIQUES ===
        self.performance_mode = 'balanced'
        self.auto_performance_adjustment = True
        self.target_fps = 15
        
        # === GESTION D'ERREURS ROBUSTE ===
        self.error_recovery_counter = 0
        self.last_successful_detection = None
        
        logger.info("🚀 Détecteur Enhanced Optimisé initialisé - Toutes fonctionnalités + Performance")

    def _init_kalman_filter(self):
        """Initialisation du filtre de Kalman optimisé"""
        try:
            kalman = cv2.KalmanFilter(4, 2)
            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
            kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
            kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
            return kalman
        except Exception as e:
            logger.debug(f"Kalman filter init error: {e}")
            return None

    def enhanced_detect_glove(self, frame):
        """Détection complète optimisée avec ajustements dynamiques"""
        if frame is None:
            return frame, False, {}
            
        start_time = time.time()
        original_frame = frame.copy()
        self.frame_count += 1
        self.frame_skip_counter += 1
        
        # === OPTIMISATION DYNAMIQUE: SKIP FRAMES ADAPTATIF ===
        if self.adaptive_skip and self.frame_skip_counter % self.process_every_n_frames != 0:
            if self.last_successful_detection:
                result_frame = self._render_cached_detection(original_frame)
                return result_frame, self.last_successful_detection.get('detected', False), self.last_successful_detection
            else:
                return self._render_enhanced_detection(original_frame, {'detected': False}), False, {'detected': False}
        
        try:
            # === PHASE 1: PRÉTRAITEMENT OPTIMISÉ ===
            
            # 1.1 Stabilisation conditionnelle
            if self.stabilization_enabled and self.performance_mode != 'fast':
                stabilized_frame = self._optimized_stabilization(frame)
            else:
                stabilized_frame = frame
            
            # 1.2 Adaptation lumière rapide
            light_adapted_frame = self._fast_lighting_adaptation(stabilized_frame)
            
            # 1.3 Amélioration contraste conditionnelle
            if self.performance_mode == 'quality':
                enhanced_frame = self._adaptive_contrast_enhancement(light_adapted_frame)
            else:
                enhanced_frame = light_adapted_frame
            
            # === PHASE 2: DÉTECTION ADAPTATIVE ===
            
            # 2.1 Prédiction rapide
            predicted_zone = self._fast_prediction()
            
            # 2.2 Détection selon mode performance
            if self.performance_mode == 'fast' or not self.use_multi_scale:
                detection_results = [self._enhanced_detect_at_scale(enhanced_frame, 1.0)]
            else:
                detection_results = self._optimized_multi_scale_detection(enhanced_frame, predicted_zone)
            
            # 2.3 Détection par zones (simplifiée en mode rapide)
            if self.performance_mode != 'fast':
                confidence_results = self._fast_confidence_zone_detection(enhanced_frame)
                detection_results.extend(confidence_results)
            
            # === PHASE 3: FUSION ET VALIDATION OPTIMISÉES ===
            
            # 3.1 Fusion rapide
            fused_detection = self._optimized_fusion(detection_results, predicted_zone)
            
            # 3.2 Validation adaptative
            if self.performance_mode == 'fast':
                validated_detection = self._fast_validation(fused_detection, enhanced_frame)
            else:
                validated_detection = self._cascade_validation(fused_detection, enhanced_frame)
            
            # 3.3 Filtrage Kalman conditionnel
            if self.kalman_filter and self.performance_mode != 'fast':
                final_detection = self._kalman_temporal_filtering(validated_detection)
            else:
                final_detection = validated_detection
            
            # === PHASE 4: MISE À JOUR OPTIMISÉE ===
            
            # 4.1 Tracking
            self._update_optimized_tracking(final_detection)
            
            # 4.2 Auto-calibration (réduite)
            if self.frame_count % 10 == 0:
                self._auto_color_calibration(enhanced_frame, final_detection)
            
            # 4.3 Zoom intelligent
            self._intelligent_zoom_adaptation(final_detection)
            
            # 4.4 Zones de confiance (simplifié)
            self._update_confidence_zones(final_detection)
            
            # === PHASE 5: RENDU ET OPTIMISATIONS DYNAMIQUES ===
            
            # 5.1 Rendu
            result_frame = self._render_enhanced_detection(original_frame, final_detection)
            
            # 5.2 Gestion performance dynamique
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.last_processing_time = processing_time
            
            # 5.3 Ajustement automatique performance
            if self.auto_performance_adjustment:
                self._adjust_performance_settings()
            
            # 5.4 Cache pour frames skippées
            if final_detection['detected']:
                self.detection_count += 1
                self.quality_scores.append(final_detection.get('quality_score', 0))
                self.last_successful_detection = final_detection
                self.error_recovery_counter = 0
            else:
                self.error_recovery_counter += 1
            
            return result_frame, final_detection['detected'], final_detection
            
        except Exception as e:
            logger.error(f"Enhanced detection error: {e}")
            return self._error_recovery(original_frame)

    def _adjust_performance_settings(self):
        """Ajustement automatique des paramètres de performance"""
        try:
            if len(self.processing_times) < 5:
                return
            
            avg_processing_time = np.mean(list(self.processing_times)[-5:]) * 1000
            current_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
            
            if current_fps < 10:
                if self.performance_mode != 'fast':
                    self.performance_mode = 'fast'
                    self.process_every_n_frames = 2
                    self.use_multi_scale = False
                    self.stabilization_enabled = False
                    logger.info("⚡ Mode FAST activé automatiquement (FPS trop bas)")
                    
            elif current_fps < self.target_fps and avg_processing_time > 50:
                if self.performance_mode == 'quality':
                    self.performance_mode = 'balanced'
                    self.process_every_n_frames = 1
                    logger.info("⚖️ Mode BALANCED activé automatiquement")
                elif self.performance_mode == 'balanced':
                    self.process_every_n_frames = 2
                    
            elif current_fps > self.target_fps * 1.5 and avg_processing_time < 30:
                if self.performance_mode == 'fast':
                    self.performance_mode = 'balanced'
                    self.process_every_n_frames = 1
                    self.use_multi_scale = True
                    self.stabilization_enabled = True
                    logger.info("⚖️ Mode BALANCED restauré (performance OK)")
                elif self.performance_mode == 'balanced' and avg_processing_time < 20:
                    self.performance_mode = 'quality'
                    logger.info("🎯 Mode QUALITY activé (performance excellente)")
                    
        except Exception as e:
            logger.debug(f"Performance adjustment error: {e}")

    def _optimized_stabilization(self, frame):
        """Stabilisation optimisée avec moins de points"""
        try:
            if len(self.stabilization_buffer) == 0:
                self.stabilization_buffer.append(frame)
                return frame
            
            prev_frame = self.stabilization_buffer[-1]
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.optical_flow_points is None or len(self.optical_flow_points) < 15:
                self.optical_flow_points = cv2.goodFeaturesToTrack(
                    prev_gray, 
                    maxCorners=50,
                    qualityLevel=0.02,
                    minDistance=15,
                    blockSize=7
                )
            
            if self.optical_flow_points is not None and len(self.optical_flow_points) > 8:
                lk_params = dict(
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02)
                )
                
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, self.optical_flow_points, None, **lk_params
                )
                
                good_new = new_points[status == 1]
                good_old = self.optical_flow_points[status == 1]
                
                if len(good_new) > 6:
                    transform, _ = cv2.estimateAffinePartial2D(
                        good_old, good_new, 
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0,
                        maxIters=1000,
                        confidence=0.95
                    )
                    
                    if transform is not None:
                        tx, ty = transform[0, 2], transform[1, 2]
                        tx = np.clip(tx, -20, 20)
                        ty = np.clip(ty, -20, 20)
                        
                        limited_transform = transform.copy()
                        limited_transform[0, 2] = tx
                        limited_transform[1, 2] = ty
                        
                        h, w = frame.shape[:2]
                        stabilized = cv2.warpAffine(frame, limited_transform, (w, h))
                        
                        self.optical_flow_points = good_new.reshape(-1, 1, 2)
                        self.stabilization_buffer.append(stabilized)
                        return stabilized
                
                self.optical_flow_points = None
            
            self.stabilization_buffer.append(frame)
            return frame
            
        except Exception as e:
            logger.debug(f"Optimized stabilization error: {e}")
            return frame

    def _fast_lighting_adaptation(self, frame):
        """Adaptation lumière rapide"""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            mean_brightness = np.mean(l_channel)
            
            if mean_brightness < 90:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
                l_enhanced = clahe.apply(l_channel)
                lab[:, :, 0] = l_enhanced
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif mean_brightness > 180:
                l_compressed = cv2.multiply(l_channel, 0.9)
                lab[:, :, 0] = l_compressed
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                result = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)
            
            if len(self.lighting_adaptation) >= 1:
                alpha = 0.7
                result = cv2.addWeighted(result, alpha, self.lighting_adaptation[-1], 1-alpha, 0)
            
            self.lighting_adaptation.append(result)
            return result
            
        except Exception as e:
            logger.debug(f"Fast lighting adaptation error: {e}")
            return frame

    def _fast_prediction(self):
        """Prédiction rapide optimisée"""
        try:
            if len(self.tracking_history) < 2:
                return None
            
            valid_tracks = [t for t in list(self.tracking_history)[-5:] if t is not None]
            
            if len(valid_tracks) < 2:
                return None
            
            recent_centers = [t['center'] for t in valid_tracks[-3:]]
            
            if len(recent_centers) >= 2:
                dx = recent_centers[-1][0] - recent_centers[-2][0]
                dy = recent_centers[-1][1] - recent_centers[-2][1]
                
                pred_x = recent_centers[-1][0] + dx * 2
                pred_y = recent_centers[-1][1] + dy * 2
                
                velocity_magnitude = np.sqrt(dx**2 + dy**2)
                search_radius = max(30, min(80, int(20 + velocity_magnitude * 1.5)))
                
                return {
                    'center': (int(pred_x), int(pred_y)),
                    'radius': search_radius,
                    'confidence': min(1.0, len(valid_tracks) / 3.0),
                    'type': 'fast_velocity'
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Fast prediction error: {e}")
            return None

    def _optimized_multi_scale_detection(self, frame, predicted_zone=None):
        """Détection multi-échelle optimisée"""
        try:
            results = []
            
            scales = self.multi_scale_levels
            if self.performance_mode == 'fast':
                scales = [1.0]
            elif self.performance_mode == 'balanced':
                scales = [1.0, 0.8]
            
            for i, scale in enumerate(scales):
                if predicted_zone and scale >= 0.8:
                    detection_frame = self._crop_prediction_zone(frame, predicted_zone, scale)
                    if detection_frame is None:
                        detection_frame = frame
                else:
                    detection_frame = frame
                
                if scale != 1.0:
                    h, w = detection_frame.shape[:2]
                    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                    scaled_frame = cv2.resize(detection_frame, (new_w, new_h), interpolation=interpolation)
                else:
                    scaled_frame = detection_frame
                
                detection_result = self._enhanced_detect_at_scale(scaled_frame, scale)
                
                if detection_result['detected']:
                    if scale != 1.0 or predicted_zone:
                        detection_result = self._remap_detection_to_original(
                            detection_result, scale, predicted_zone
                        )
                    
                    detection_result['scale_weight'] = self.scale_weights[i] if i < len(self.scale_weights) else 0.5
                    results.append(detection_result)
            
            return results
            
        except Exception as e:
            logger.debug(f"Optimized multi-scale detection error: {e}")
            return []

    def _enhanced_detect_at_scale(self, frame, scale):
        """Détection optimisée pour une échelle"""
        try:
            h, w = frame.shape[:2]
            
            if self.performance_mode == 'fast':
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            else:
                blur_size = 3 if scale < 0.7 else 5
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.GaussianBlur(hsv, (blur_size, blur_size), 0)
            
            # Ajustements adaptatifs
            sat_adjustment = max(0, int(15 * (2 - scale))) if self.performance_mode != 'fast' else 10
            val_adjustment = max(0, int(10 * (2 - scale))) if self.performance_mode != 'fast' else 5
            
            if len(self.brightness_history) > 0:
                avg_brightness = np.mean(self.brightness_history)
                if avg_brightness < 100:
                    val_adjustment -= 15
                elif avg_brightness > 180:
                    val_adjustment += 10
            
            # Masques optimisés
            mask_orange = self._create_optimized_color_mask(
                hsv, self.color_calibration['orange_ranges'], sat_adjustment, val_adjustment
            )
            
            mask_red = self._create_optimized_color_mask(
                hsv, self.color_calibration['red_ranges'], sat_adjustment, val_adjustment
            )
            
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # Exclusions rapides
            if self.performance_mode != 'fast':
                mask_skin = self._create_fast_skin_exclusion(hsv, scale)
                mask_skin_processed = cv2.dilate(mask_skin, np.ones((3,3), np.uint8))
                mask_glove = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_skin_processed))
            
            # Morphologie optimisée
            mask_final = self._optimized_morphology(mask_glove, scale)
            
            # Contours rapides
            contours, hierarchy = cv2.findContours(
                mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            best_contour, quality_score = self._fast_contour_selection(contours, scale, frame)
            
            result = {
                'detected': best_contour is not None,
                'contour': best_contour,
                'area': cv2.contourArea(best_contour) if best_contour is not None else 0,
                'scale': scale,
                'quality_score': quality_score,
                'mask': mask_final,
                'frame': frame,
                'color_confidence': self._fast_color_confidence(mask_orange, mask_red)
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Enhanced scale detection error: {e}")
            return {'detected': False, 'contour': None, 'area': 0, 'scale': scale, 'quality_score': 0}

    def _create_optimized_color_mask(self, hsv, color_ranges, sat_adj, val_adj):
        """Masque couleur optimisé"""
        try:
            h, w = hsv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for lower_base, upper_base in color_ranges:
                lower = np.array([
                    lower_base[0],
                    max(0, lower_base[1] - sat_adj),
                    max(0, lower_base[2] - val_adj)
                ])
                upper = np.array([
                    upper_base[0],
                    min(255, upper_base[1]),
                    min(255, upper_base[2])
                ])
                
                range_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, range_mask)
            
            return mask
            
        except Exception as e:
            logger.debug(f"Optimized color mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _create_fast_skin_exclusion(self, hsv, scale):
        """Exclusion peau rapide"""
        try:
            strictness = max(0.6, 1.0 - (scale - 1.0) * 0.2)
            
            skin_s_max = int(110 * strictness)
            skin_v_min = max(90, int(130 * strictness))
            
            skin_lower = np.array([6, 60, skin_v_min])
            skin_upper = np.array([14, skin_s_max, 210])
            
            return cv2.inRange(hsv, skin_lower, skin_upper)
            
        except Exception as e:
            logger.debug(f"Fast skin exclusion error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _optimized_morphology(self, mask, scale):
        """Morphologie optimisée"""
        try:
            if scale > 1.5:
                close_size, open_size = 6, 2
            else:
                close_size, open_size = max(3, int(4 * scale)), 2
            
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            
            # Bordures
            border_size = max(3, int(8 / scale))
            mask[:border_size, :] = 0
            mask[-border_size:, :] = 0
            mask[:, :border_size] = 0
            mask[:, -border_size:] = 0
            
            return mask
            
        except Exception as e:
            logger.debug(f"Optimized morphology error: {e}")
            return mask

    def _fast_contour_selection(self, contours, scale, frame):
        """Sélection contour rapide mais complète"""
        if not contours:
            return None, 0
            
        try:
            best_contour = None
            best_score = 0
            
            # Seuils adaptés
            min_area_scaled = self.min_area * (scale ** 1.2)
            max_area_scaled = self.max_area * (scale ** 1.2)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base
                if area < min_area_scaled or area > max_area_scaled:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Analyse géométrique optimisée
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Validation rapide
                if not (0.3 <= aspect_ratio <= 2.5):
                    continue
                
                # Score simplifié mais efficace
                score = 0
                
                # 1. Score géométrique (40%)
                geo_score = 1.0 - abs(aspect_ratio - 1.0) * 0.4 if 0.4 <= aspect_ratio <= 2.0 else 0.2
                
                # 2. Score de taille (30%)
                optimal_area = 3000 * (scale ** 1.3)
                size_score = max(0, 1.0 - abs(area - optimal_area) / optimal_area)
                
                # 3. Score de position (30%)
                center_x = x + w // 2
                center_y = y + h // 2
                frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
                
                dist_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                max_dist = np.sqrt(frame_center_x**2 + frame_center_y**2)
                pos_score = 1.0 - (dist_from_center / max_dist) * 0.5
                
                # Score final pondéré
                final_score = geo_score * 0.4 + size_score * 0.3 + pos_score * 0.3
                
                # Bonus échelle optimale
                if scale >= 0.8:
                    final_score *= 1.05
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
            
            return best_contour, best_score
            
        except Exception as e:
            logger.debug(f"Fast contour selection error: {e}")
            return None, 0

    def _fast_color_confidence(self, mask_orange, mask_red):
        """Confiance couleur rapide"""
        try:
            total_pixels = mask_orange.shape[0] * mask_orange.shape[1]
            orange_pixels = cv2.countNonZero(mask_orange)
            red_pixels = cv2.countNonZero(mask_red)
            
            orange_ratio = orange_pixels / total_pixels
            red_ratio = red_pixels / total_pixels
            
            if orange_ratio > 0.001 and red_ratio > 0.001:
                return min(1.0, (orange_ratio + red_ratio) * 100)
            elif orange_ratio > 0.002 or red_ratio > 0.002:
                return min(0.7, max(orange_ratio, red_ratio) * 80)
            else:
                return 0.1
                
        except Exception as e:
            return 0.0

    def _fast_confidence_zone_detection(self, frame):
        """Détection zones de confiance rapide"""
        try:
            h, w = frame.shape[:2]
            
            # Une seule zone pour vitesse
            zone = {'name': 'center', 'region': (w//4, h//4, w//2, h//2), 'weight': 1.1}
            
            x, y, zone_w, zone_h = zone['region']
            zone_frame = frame[y:y+zone_h, x:x+zone_w]
            
            zone_result = self._enhanced_detect_at_scale(zone_frame, 1.0)
            
            if zone_result['detected'] and zone_result['contour'] is not None:
                # Remapping
                zone_result['contour'][:, :, 0] += x
                zone_result['contour'][:, :, 1] += y
                zone_result['zone_weight'] = zone['weight']
                return [zone_result]
            
            return []
            
        except Exception as e:
            logger.debug(f"Fast confidence zone detection error: {e}")
            return []

    def _optimized_fusion(self, all_results, predicted_zone):
        """Fusion optimisée"""
        try:
            if not all_results:
                return {'detected': False, 'contour': None, 'area': 0, 'quality_score': 0}
            
            best_result = None
            best_fusion_score = 0
            
            for result in all_results:
                if not result['detected']:
                    continue
                
                # Score de fusion rapide
                fusion_score = result['quality_score']
                
                # Pondérations
                fusion_score *= result.get('scale_weight', 1.0)
                fusion_score *= result.get('zone_weight', 1.0)
                fusion_score *= (0.5 + result.get('color_confidence', 0.5) * 0.5)
                
                if predicted_zone and 'prediction_bonus' in result:
                    fusion_score *= 1.1
                
                if fusion_score > best_fusion_score:
                    best_fusion_score = fusion_score
                    best_result = result
            
            if best_result:
                best_result['fusion_score'] = best_fusion_score
                return best_result
            else:
                return {'detected': False, 'contour': None, 'area': 0, 'quality_score': 0}
                
        except Exception as e:
            logger.debug(f"Optimized fusion error: {e}")
            return {'detected': False, 'contour': None, 'area': 0, 'quality_score': 0}

    def _fast_validation(self, detection, frame):
        """Validation rapide simplifiée"""
        try:
            if not detection['detected']:
                return detection
            
            contour = detection['contour']
            if contour is None:
                detection['detected'] = False
                return detection
            
            # Validation basique
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                detection['detected'] = False
                detection['rejection_reason'] = 'area_bounds'
                self.false_positive_rejection += 1
                return detection
            
            # Validation couleur simple
            if detection.get('color_confidence', 0) < 0.2:
                detection['detected'] = False
                detection['rejection_reason'] = 'low_color_confidence'
                self.false_positive_rejection += 1
                return detection
            
            detection['validation_passed'] = True
            return detection
            
        except Exception as e:
            logger.debug(f"Fast validation error: {e}")
            detection['detected'] = False
            return detection

    def _cascade_validation(self, detection, frame):
        """Validation cascade complète (mode quality)"""
        try:
            if not detection['detected']:
                return detection
            
            contour = detection['contour']
            if contour is None:
                detection['detected'] = False
                return detection
            
            # === VALIDATION GÉOMÉTRIQUE ===
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                detection['detected'] = False
                detection['rejection_reason'] = 'area_out_of_bounds'
                self.false_positive_rejection += 1
                return detection
            
            # === VALIDATION COULEUR ===
            color_validation = self._validate_color_composition(frame, contour)
            if color_validation < self.validation_cascade['color_threshold']:
                detection['detected'] = False
                detection['rejection_reason'] = 'insufficient_color_match'
                self.false_positive_rejection += 1
                return detection
            
            # === VALIDATION MOUVEMENT ===
            if len(self.tracking_history) > 2:
                movement_validation = self._validate_movement_consistency(detection)
                if movement_validation < self.validation_cascade['motion_threshold']:
                    detection['detected'] = False
                    detection['rejection_reason'] = 'inconsistent_movement'
                    self.false_positive_rejection += 1
                    return detection
            
            # === VALIDATION TEMPORELLE ===
            temporal_validation = self._validate_temporal_consistency(detection)
            if temporal_validation < self.validation_cascade['temporal_threshold']:
                detection['detected'] = False
                detection['rejection_reason'] = 'temporal_inconsistency'
                self.false_positive_rejection += 1
                return detection
            
            # Ajout des scores
            detection['color_validation'] = color_validation
            detection['temporal_validation'] = temporal_validation
            detection['validation_passed'] = True
            
            return detection
            
        except Exception as e:
            logger.debug(f"Cascade validation error: {e}")
            detection['detected'] = False
            return detection

    def _validate_color_composition(self, frame, contour):
        """Validation composition couleur"""
        try:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            roi = cv2.bitwise_and(frame, frame, mask=mask)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            total_roi_pixels = cv2.countNonZero(mask)
            if total_roi_pixels == 0:
                return 0.0
            
            orange_pixels = 0
            red_pixels = 0
            
            for lower, upper in self.color_calibration['orange_ranges']:
                orange_mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
                orange_pixels += cv2.countNonZero(orange_mask)
            
            for lower, upper in self.color_calibration['red_ranges']:
                red_mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
                red_pixels += cv2.countNonZero(red_mask)
            
            target_pixels = orange_pixels + red_pixels
            color_ratio = target_pixels / total_roi_pixels
            
            both_colors_bonus = 1.3 if orange_pixels > 0 and red_pixels > 0 else 1.0
            
            return min(1.0, color_ratio * both_colors_bonus)
            
        except Exception as e:
            logger.debug(f"Color composition validation error: {e}")
            return 0.0

    def _validate_movement_consistency(self, detection):
        """Validation cohérence mouvement"""
        try:
            if detection['contour'] is None:
                return 0.0
            
            moments = cv2.moments(detection['contour'])
            if moments["m00"] == 0:
                return 0.0
                
            curr_x = int(moments["m10"] / moments["m00"])
            curr_y = int(moments["m01"] / moments["m00"])
            
            recent_positions = []
            for track in list(self.tracking_history)[-4:]:
                if track and 'center' in track:
                    recent_positions.append(track['center'])
            
            if len(recent_positions) < 2:
                return 1.0
            
            distances = []
            for pos in recent_positions:
                dist = np.sqrt((curr_x - pos[0])**2 + (curr_y - pos[1])**2)
                distances.append(dist)
            
            max_distance = max(distances)
            if max_distance > 120:
                return 0.1
            elif max_distance > 80:
                return 0.6
            else:
                return 1.0
                
        except Exception as e:
            logger.debug(f"Movement validation error: {e}")
            return 0.5

    def _validate_temporal_consistency(self, detection):
        """Validation cohérence temporelle"""
        try:
            recent_detections = list(self.detection_history)[-8:]
            if len(recent_detections) == 0:
                return 1.0
            
            detection_ratio = sum(recent_detections) / len(recent_detections)
            
            if detection_ratio > 0.6:
                return 1.0
            elif detection_ratio > 0.3:
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            logger.debug(f"Temporal validation error: {e}")
            return 0.5

    def _kalman_temporal_filtering(self, detection):
        """Filtrage Kalman optimisé"""
        try:
            if not detection['detected'] or self.kalman_filter is None:
                return detection
            
            contour = detection['contour']
            if contour is None:
                return detection
            
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                return detection
            
            curr_x = int(moments["m10"] / moments["m00"])
            curr_y = int(moments["m01"] / moments["m00"])
            
            measurement = np.array([[curr_x], [curr_y]], dtype=np.float32)
            
            prediction = self.kalman_filter.predict()
            corrected = self.kalman_filter.correct(measurement)
            
            filtered_x = int(corrected[0, 0])
            filtered_y = int(corrected[1, 0])
            
            offset_x = filtered_x - curr_x
            offset_y = filtered_y - curr_y
            
            if abs(offset_x) < 40 and abs(offset_y) < 40:
                filtered_contour = contour.copy()
                filtered_contour[:, :, 0] += offset_x
                filtered_contour[:, :, 1] += offset_y
                
                detection['contour'] = filtered_contour
                detection['kalman_filtered'] = True
                detection['filter_offset'] = (offset_x, offset_y)
            
            return detection
            
        except Exception as e:
            logger.debug(f"Kalman filtering error: {e}")
            return detection

    def _update_optimized_tracking(self, detection):
        """Tracking optimisé"""
        try:
            if detection['detected'] and detection['contour'] is not None:
                moments = cv2.moments(detection['contour'])
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    track_info = {
                        'center': (cx, cy),
                        'area': detection['area'],
                        'timestamp': time.time(),
                        'quality': detection.get('quality_score', 0),
                        'contour': detection['contour'],
                        'frame_id': self.frame_count
                    }
                    
                    self.tracking_history.append(track_info)
                else:
                    self.tracking_history.append(None)
            else:
                self.tracking_history.append(None)
                
        except Exception as e:
            logger.debug(f"Optimized tracking error: {e}")

    def _auto_color_calibration(self, frame, detection):
        """Auto-calibration optimisée"""
        try:
            if detection['detected'] and detection['contour'] is not None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [detection['contour']], 255)
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                roi_pixels = hsv[mask > 0]
                
                if len(roi_pixels) > 10:
                    # Échantillonnage réduit pour vitesse
                    sample_size = min(10, len(roi_pixels))
                    sample_indices = np.random.choice(len(roi_pixels), sample_size, replace=False)
                    for idx in sample_indices:
                        self.color_samples.append(roi_pixels[idx])
                    
                    # Recalibration moins fréquente
                    if len(self.color_samples) > 40 and self.frame_count % 200 == 0:
                        self._recalibrate_color_ranges()
                        
        except Exception as e:
            logger.debug(f"Auto color calibration error: {e}")

    def _recalibrate_color_ranges(self):
        """Recalibration optimisée"""
        try:
            if len(self.color_samples) < 20:
                return
            
            samples = np.array(list(self.color_samples))
            
            # Clustering optimisé
            kmeans = OptimizedKMeans(n_clusters=2)
            clusters = kmeans.fit(samples)
            
            centers = clusters.cluster_centers_
            labels = clusters.labels_
            
            for i, center in enumerate(centers):
                cluster_samples = samples[labels == i]
                
                if len(cluster_samples) < 5:
                    continue
                
                h_mean, s_mean, v_mean = center
                h_std = np.std(cluster_samples[:, 0])
                s_std = np.std(cluster_samples[:, 1])
                v_std = np.std(cluster_samples[:, 2])
                
                margin_h = max(4, h_std * 1.2)
                margin_s = max(15, s_std * 1.2)
                margin_v = max(15, v_std * 1.2)
                
                new_range = (
                    [max(0, h_mean - margin_h), max(0, s_mean - margin_s), max(0, v_mean - margin_v)],
                    [min(180, h_mean + margin_h), min(255, s_mean + margin_s), min(255, v_mean + margin_v)]
                )
                
                # Limitation du nombre de plages
                if h_mean < 25:
                    if h_mean < 15 and len(self.color_calibration['orange_ranges']) < 5:
                        self.color_calibration['orange_ranges'].append(new_range)
                    elif len(self.color_calibration['red_ranges']) < 5:
                        self.color_calibration['red_ranges'].append(new_range)
            
            logger.info(f"🎨 Recalibration: {len(self.color_calibration['orange_ranges'])} orange, {len(self.color_calibration['red_ranges'])} rouge")
            
        except Exception as e:
            logger.debug(f"Color recalibration error: {e}")

    def _intelligent_zoom_adaptation(self, detection):
        """Adaptation zoom intelligente"""
        try:
            if detection['detected'] and detection['area'] > 0:
                area = detection['area']
                predicted_zoom = self._predict_optimal_zoom(area)
                
                self.zoom_prediction.append(predicted_zoom)
                if len(self.zoom_prediction) >= 2:
                    weights = [0.6, 0.4]
                    weighted_zoom = sum(z * w for z, w in zip(list(self.zoom_prediction)[-2:], weights))
                    self.target_zoom = weighted_zoom
                else:
                    self.target_zoom = predicted_zoom
                
                if abs(self.target_zoom - self.zoom_factor) < 0.1:
                    self.zoom_stability_counter += 1
                else:
                    self.zoom_stability_counter = 0
                
                # Application zoom
                if self.zoom_stability_counter > 3:
                    self.zoom_smooth_factor = 0.04
                else:
                    self.zoom_smooth_factor = 0.08
                    
            else:
                if sum(list(self.stable_detections)[-2:]) == 0:
                    self.target_zoom = max(self.zoom_min, self.target_zoom * 0.98)
            
            # Application lissée
            self.zoom_factor += (self.target_zoom - self.zoom_factor) * self.zoom_smooth_factor
            self.zoom_factor = np.clip(self.zoom_factor, self.zoom_min, self.zoom_max)
                    
        except Exception as e:
            logger.debug(f"Zoom adaptation error: {e}")

    def _predict_optimal_zoom(self, area):
        """Prédiction zoom optimal"""
        try:
            if area < 600:
                return min(self.zoom_max, 4.0)
            elif area < 1200:
                return min(self.zoom_max, 3.0)
            elif area < 2500:
                return 2.2
            elif area < 5000:
                return 1.6
            elif area < 10000:
                return 1.2
            else:
                return 1.0
                
        except Exception as e:
            return 1.0

    def _update_confidence_zones(self, detection):
        """Mise à jour zones de confiance"""
        try:
            if detection['detected'] and detection['contour'] is not None:
                moments = cv2.moments(detection['contour'])
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    zone_info = self._determine_confidence_zone(cx, cy)
                    quality = detection.get('quality_score', 0)
                    
                    zone_entry = {
                        'position': (cx, cy),
                        'quality': quality,
                        'timestamp': time.time(),
                        'frame_id': self.frame_count
                    }
                    
                    self.confidence_zones[zone_info['level']].append(zone_entry)
                    
        except Exception as e:
            logger.debug(f"Confidence zones error: {e}")

    def _determine_confidence_zone(self, x, y):
        """Détermine zone de confiance"""
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        if distance_from_center < 80:
            return {'level': 'high', 'distance': distance_from_center}
        elif distance_from_center < 160:
            return {'level': 'medium', 'distance': distance_from_center}
        else:
            return {'level': 'low', 'distance': distance_from_center}

    def _render_cached_detection(self, frame):
        """Rendu rapide avec cache"""
        try:
            if self.last_successful_detection and 'contour' in self.last_successful_detection:
                contour = self.last_successful_detection['contour']
                if contour is not None:
                    return self._draw_enhanced_detection(frame, self.last_successful_detection)
            
            return self._add_performance_overlay(frame, False)
            
        except Exception as e:
            logger.debug(f"Cached render error: {e}")
            return frame

    def _render_enhanced_detection(self, frame, detection):
        """Rendu complet optimisé"""
        try:
            h, w = frame.shape[:2]
            result_frame = frame.copy()
            
            # Détection principale
            if detection['detected'] and detection['contour'] is not None:
                self._draw_enhanced_detection(result_frame, detection)
            
            # Overlay système
            self._add_system_overlay(result_frame, detection)
            
            # Zones de confiance (mode quality seulement)
            if self.performance_mode == 'quality':
                self._draw_confidence_zones(result_frame)
            
            # Tracking (mode balanced et quality)
            if self.performance_mode != 'fast':
                self._draw_tracking_info(result_frame, detection)
            
            # Performance et zoom
            self._add_performance_overlay(result_frame, detection['detected'])
            
            return result_frame
            
        except Exception as e:
            logger.debug(f"Enhanced render error: {e}")
            return frame

    def _draw_enhanced_detection(self, frame, detection):
        """Dessin détection optimisé"""
        try:
            contour = detection['contour']
            area = detection['area']
            quality = detection.get('quality_score', 0)
            
            # Couleur selon distance
            if area > 6000:
                color = (0, 255, 0)
                distance_text = "TRÈS PROCHE"
            elif area > 3000:
                color = (0, 255, 255)
                distance_text = "PROCHE"
            elif area > 1500:
                color = (0, 200, 255)
                distance_text = "MOYEN"
            elif area > 800:
                color = (0, 150, 255)
                distance_text = "LOIN"
            else:
                color = (0, 100, 255)
                distance_text = "TRÈS LOIN"
            
            quality_factor = max(0.4, quality)
            color = tuple(int(c * quality_factor) for c in color)
            
            # Contour
            thickness = max(2, int(3 * quality))
            cv2.drawContours(frame, [contour], -1, color, thickness)
            
            # Rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 2)
                
                # Textes
                info_y = max(y - 15, 25)
                cv2.putText(frame, f"GANT {distance_text}", (x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                           
                info_y += 20
                cv2.putText(frame, f"A:{int(area)} Q:{quality:.2f}", (x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Validation info (mode quality)
                if self.performance_mode == 'quality' and 'color_validation' in detection:
                    info_y += 15
                    cv2.putText(frame, f"C:{detection['color_validation']:.2f}", 
                               (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                
                # Kalman indicator
                if detection.get('kalman_filtered', False):
                    cv2.putText(frame, "K", (cx + 10, cy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        except Exception as e:
            logger.debug(f"Enhanced detection drawing error: {e}")

    def _add_system_overlay(self, frame, detection):
        """Overlay système optimisé"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal avec mode performance
            if detection['detected']:
                if detection.get('quality_score', 0) > 0.7:
                    status = "🎯 GANT DÉTECTÉ (HQ)"
                    color = (0, 255, 0)
                else:
                    status = "🎯 GANT DÉTECTÉ"
                    color = (0, 200, 255)
            else:
                status = "🔍 RECHERCHE"
                color = (0, 255, 255)
                
                if 'rejection_reason' in detection:
                    status += f" ({detection['rejection_reason'][:8]})"
            
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Mode performance
            mode_colors = {'fast': (0, 150, 255), 'balanced': (0, 255, 255), 'quality': (0, 255, 0)}
            mode_color = mode_colors.get(self.performance_mode, (255, 255, 255))
            cv2.putText(frame, f"Mode: {self.performance_mode.upper()}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
            
        except Exception as e:
            logger.debug(f"System overlay error: {e}")

    def _draw_confidence_zones(self, frame):
        """Zones de confiance optimisées"""
        try:
            h, w = frame.shape[:2]
            
            # Zone haute confiance
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 1)
            cv2.putText(frame, "HIGH", (w//4 + 5, h//4 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Stats zones
            cv2.putText(frame, f"H:{len(self.confidence_zones['high'])}", 
                       (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"M:{len(self.confidence_zones['medium'])}", 
                       (w - 70, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, f"L:{len(self.confidence_zones['low'])}", 
                       (w - 40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 255), 1)
            
        except Exception as e:
            logger.debug(f"Confidence zones drawing error: {e}")

    def _draw_tracking_info(self, frame, detection):
        """Tracking info optimisé"""
        try:
            h, w = frame.shape[:2]
            
            # Trajectoire simplifiée
            recent_tracks = [t for t in list(self.tracking_history)[-6:] if t is not None]
            if len(recent_tracks) > 1:
                points = [t['center'] for t in recent_tracks]
                
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    color = (int(200 * alpha), int(100 * alpha), int(100 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, 2)
            
            # Prédiction
            if hasattr(self, 'prediction_zone') and self.prediction_zone:
                pred_center = self.prediction_zone['center']
                pred_radius = min(self.prediction_zone['radius'], 60)
                cv2.circle(frame, pred_center, pred_radius, (255, 0, 255), 1)
                cv2.putText(frame, "P", (pred_center[0] + 8, pred_center[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Historique compact
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-8:]])
            cv2.putText(frame, f"H:{history}", (10, h - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
        except Exception as e:
            logger.debug(f"Tracking info drawing error: {e}")

    def _add_performance_overlay(self, frame, detected):
        """Overlay performance optimisé"""
        try:
            h, w = frame.shape[:2]
            
            # === BARRE ZOOM SIMPLIFIÉE ===
            zoom_bar_width = 200
            zoom_bar_height = 15
            zoom_x, zoom_y = 10, 85
            
            # Fond
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_bar_width, zoom_y + zoom_bar_height), 
                         (50, 50, 50), -1)
            
            # Barre zoom
            zoom_width = int(zoom_bar_width * (self.zoom_factor - 1.0) / (self.zoom_max - 1.0))
            zoom_color = (0, 255, 255) if self.zoom_factor > 2.0 else (100, 255, 100)
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_width, zoom_y + zoom_bar_height), 
                         zoom_color, -1)
            
            # Target zoom
            target_width = int(zoom_bar_width * (self.target_zoom - 1.0) / (self.zoom_max - 1.0))
            cv2.line(frame, (zoom_x + target_width, zoom_y - 3), 
                    (zoom_x + target_width, zoom_y + zoom_bar_height + 3), 
                    (255, 255, 255), 1)
            
            # Texte zoom
            cv2.putText(frame, f"Z:{self.zoom_factor:.1f}x", 
                       (zoom_x + zoom_bar_width + 10, zoom_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # === PERFORMANCE ===
            
            # FPS
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            # Couleur FPS selon performance
            fps_color = (100, 255, 100) if self.current_fps >= 15 else (0, 255, 255) if self.current_fps >= 10 else (0, 100, 255)
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 120, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            # Temps traitement
            if self.processing_times:
                avg_time = np.mean(list(self.processing_times)[-5:]) * 1000
                time_color = (100, 255, 100) if avg_time < 30 else (0, 255, 255) if avg_time < 50 else (0, 100, 255)
                cv2.putText(frame, f"T:{avg_time:.0f}ms", (w - 120, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, time_color, 1)
            
            # === STATISTIQUES COMPACTES ===
            
            # Taux détection
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            quality_avg = np.mean(self.quality_scores) if self.quality_scores else 0
            
            stats_y = h - 80
            cv2.putText(frame, f"Det: {detection_rate:.1f}% | Q: {quality_avg:.2f}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Rejets et skip
            stats_y += 20
            cv2.putText(frame, f"Skip:{self.process_every_n_frames} | FP:{self.false_positive_rejection}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # === INDICATEURS SYSTÈME ===
            
            indicators_x = w - 160
            indicator_y = 100
            
            # Détection
            if detected:
                cv2.circle(frame, (indicators_x, indicator_y), 6, (0, 255, 0), -1)
                cv2.putText(frame, "D", (indicators_x + 10, indicator_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Zoom actif
            if self.zoom_factor > 1.3:
                cv2.circle(frame, (indicators_x + 25, indicator_y), 6, (0, 255, 255), -1)
                cv2.putText(frame, "Z", (indicators_x + 35, indicator_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Tracking actif
            if len([t for t in self.tracking_history if t]) > 3:
                cv2.circle(frame, (indicators_x + 50, indicator_y), 6, (255, 0, 255), -1)
                cv2.putText(frame, "T", (indicators_x + 60, indicator_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Stabilisation
            if self.stabilization_enabled and self.performance_mode != 'fast':
                cv2.circle(frame, (indicators_x + 75, indicator_y), 6, (255, 255, 0), -1)
                cv2.putText(frame, "S", (indicators_x + 85, indicator_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        except Exception as e:
            logger.debug(f"Performance overlay error: {e}")

    def _error_recovery(self, frame):
        """Récupération d'erreur optimisée"""
        try:
            if self.last_successful_detection:
                logger.info(f"🔄 Récupération erreur #{self.error_recovery_counter}")
                
                self.target_zoom = max(1.0, self.target_zoom * 0.9)
                
                if self.error_recovery_counter > 5:
                    # Basculer en mode fast
                    self.performance_mode = 'fast'
                    self.process_every_n_frames = 2
                    logger.info("⚡ Mode FAST forcé (trop d'erreurs)")
                
                if self.error_recovery_counter > 15:
                    self.reset_detector()
                    logger.info("🔄 Reset complet (récupération)")
                
                return frame, False, {'detected': False, 'error_recovery': True}
            
            return frame, False, {'detected': False}
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            return frame, False, {'detected': False}

    def reset_detector(self):
        """Reset optimisé"""
        logger.info("🔄 Reset détecteur optimisé")
        # Garde les calibrations couleur
        old_calibration = self.color_calibration.copy()
        self.__init__()
        self.color_calibration = old_calibration

    def get_performance_stats(self):
        """Stats performance optimisées"""
        return {
            'detection_rate': (self.detection_count / max(self.frame_count, 1)) * 100,
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0,
            'false_positive_rejections': self.false_positive_rejection,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'tracking_history_length': len([t for t in self.tracking_history if t is not None]),
            'current_zoom': self.zoom_factor,
            'target_zoom': self.target_zoom,
            'color_samples_collected': len(self.color_samples),
            'performance_mode': self.performance_mode,
            'frame_skip_rate': self.process_every_n_frames,
            'current_fps': self.current_fps,
            'stabilization_enabled': self.stabilization_enabled,
            'multi_scale_enabled': self.use_multi_scale
        }

    # === MÉTHODES UTILITAIRES ===
    
    def _crop_prediction_zone(self, frame, predicted_zone, scale):
        """Découpage zone prédiction"""
        try:
            h, w = frame.shape[:2]
            center_x, center_y = predicted_zone['center']
            radius = predicted_zone['radius']
            
            expanded_radius = int(radius * (1.5 - scale * 0.3))
            
            x1 = max(0, center_x - expanded_radius)
            y1 = max(0, center_y - expanded_radius)
            x2 = min(w, center_x + expanded_radius)
            y2 = min(h, center_y + expanded_radius)
            
            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2]
            
            return None
            
        except Exception as e:
            logger.debug(f"Crop prediction zone error: {e}")
            return None

    def _remap_detection_to_original(self, detection_result, scale, predicted_zone):
        """Remapping vers coordonnées originales"""
        try:
            contour = detection_result.get('contour')
            if contour is None:
                return detection_result
            
            if scale != 1.0:
                scale_factor = 1.0 / scale
                contour = (contour * scale_factor).astype(np.int32)
            
            if predicted_zone:
                center_x, center_y = predicted_zone['center']
                radius = predicted_zone['radius']
                offset_x = max(0, center_x - radius)
                offset_y = max(0, center_y - radius)
                
                contour[:, :, 0] += offset_x
                contour[:, :, 1] += offset_y
            
            detection_result['contour'] = contour
            detection_result['area'] = cv2.contourArea(contour)
            
            return detection_result
            
        except Exception as e:
            logger.debug(f"Remap detection error: {e}")
            return detection_result

    def _adaptive_contrast_enhancement(self, frame):
        """Amélioration contraste (mode quality)"""
        try:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            # CLAHE léger
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
            y_enhanced = clahe.apply(y_channel)
            
            yuv[:, :, 0] = y_enhanced
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
        except Exception as e:
            logger.debug(f"Contrast enhancement error: {e}")
            return frame

# === CONTRÔLE DRONE AMÉLIORÉ ===
def optimized_drone_control(bebop):
    """Contrôle drone optimisé"""
    logger.info("🎮 Contrôle drone optimisé démarré")
    print("\n[Commandes drone optimisées]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f/b/g/d = mouvements | h/m = haut/bas | a/c = rotations\n"
          "  1/2/3 = vitesses | p = hover | x = urgence\n")
    
    speed_settings = {'1': 15, '2': 25, '3': 35}
    current_speed = 25
    
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
        elif key == 'x':
            bebop.emergency_land()
            print("🚨 Urgence")
        elif key == 'p':
            print("📍 Hover")
        elif key in speed_settings:
            current_speed = speed_settings[key]
            print(f"⚡ Vitesse: {current_speed}")
        elif key == 'f':
            bebop.fly_direct(roll=0, pitch=current_speed, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-current_speed, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'g':
            bebop.fly_direct(roll=-current_speed, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'd':
            bebop.fly_direct(roll=current_speed, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=current_speed//2, duration=0.3)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-current_speed//2, duration=0.3)
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-30, vertical_movement=0, duration=0.3)
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=30, vertical_movement=0, duration=0.3)

# === FONCTION PRINCIPALE OPTIMISÉE COMPLÈTE ===
def main():
    """Fonction principale complète optimisée"""
    logger.info("=" * 80)
    logger.info("🚀 BEBOP 2 - DÉTECTION COMPLÈTE OPTIMISÉE")
    logger.info("🎯 Toutes fonctionnalités + Performance maximale")
    logger.info("=" * 80)
    
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
        ctrl_thread = threading.Thread(target=optimized_drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG OPTIMISÉ ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg optimisé pour qualité et vitesse
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '500000',
            '-probesize', '500000',
            '-max_delay', '100000',
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-threads', '2',
            '-'
        ]
        
        logger.info("🚀 FFmpeg optimisé qualité/vitesse configuré")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=1024*1024)
            logger.info("✅ Pipeline optimisé initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR COMPLET OPTIMISÉ ===
        detector = OptimizedCompleteGloveDetector()
        
        # === INTERFACE COMPLÈTE ===
        window_name = "Bebop 2 - Détection Complète Optimisée"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 80)
        logger.info("🎮 COMMANDES COMPLÈTES:")
        logger.info("  'q'/'ESC' = Quitter | 's' = Screenshot | 'r' = Reset")
        logger.info("  'z' = Reset zoom | '+'/'-' = Zoom | 'c' = Calibration")
        logger.info("  'f' = Mode Fast | 'b' = Mode Balanced | 'Q' = Mode Quality")
        logger.info("  'd' = Debug | 'p' = Stats | 'h' = Aide")
        logger.info("=" * 80)
        logger.info("⚡ FONCTIONNALITÉS OPTIMISÉES:")
        logger.info("  ✓ Stabilisation adaptative")
        logger.info("  ✓ Adaptation lumière rapide")
        logger.info("  ✓ Multi-échelle dynamique")
        logger.info("  ✓ Tracking Kalman conditionnel")
        logger.info("  ✓ Validation cascade adaptative")
        logger.info("  ✓ Auto-calibration optimisée")
        logger.info("  ✓ Zones confiance intelligentes")
        logger.info("  ✓ Zoom adaptatif 5x")
        logger.info("  ✓ Ajustement performance automatique")
        logger.info("=" * 80)
        
        # === BOUCLE PRINCIPALE OPTIMISÉE ===
        logger.info("🎬 Démarrage détection complète optimisée...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        performance_log_interval = 150
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # === DÉTECTION COMPLÈTE OPTIMISÉE ===
                processed_frame, detected, detection_info = detector.enhanced_detect_glove(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # === LOGGING OPTIMISÉ ===
                fps_counter += 1
                if fps_counter % 60 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    
                    stats = detector.get_performance_stats()
                    
                    logger.info(f"📊 FPS: {display_fps:.1f} | "
                               f"Det: {detector.detection_count}/{detector.frame_count} "
                               f"({stats['detection_rate']:.1f}%) | "
                               f"Q: {stats['avg_quality']:.2f} | "
                               f"Mode: {stats['performance_mode']} | "
                               f"Zoom: {detector.zoom_factor:.1f}x→{detector.target_zoom:.1f}x | "
                               f"FP: {stats['false_positive_rejections']}")
                    
                    last_fps_log = current_time
                
                # === LOGGING DÉTAILLÉ RÉDUIT ===
                if fps_counter % performance_log_interval == 0:
                    stats = detector.get_performance_stats()
                    logger.info("🔍 PERFORMANCE:")
                    logger.info(f"   Proc: {stats['avg_processing_time']*1000:.1f}ms | "
                               f"Track: {stats['tracking_history_length']} | "
                               f"Samples: {stats['color_samples_collected']} | "
                               f"Skip: {stats['frame_skip_rate']}")
                
                # === GESTION TOUCHES COMPLÈTE ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    # Screenshot complet
                    timestamp = int(time.time())
                    screenshot_name = f"complete_capture_{timestamp}_{screenshot_count:03d}.png"
                    
                    info_frame = processed_frame.copy()
                    stats = detector.get_performance_stats()
                    info_text = (f"F:{detector.frame_count} Z:{detector.zoom_factor:.1f}x "
                               f"M:{stats['performance_mode']} Q:{detection_info.get('quality_score', 0):.2f}")
                    cv2.putText(info_frame, info_text, (10, HEIGHT - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imwrite(screenshot_name, info_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    # Reset complet
                    old_stats = detector.get_performance_stats()
                    detector.reset_detector()
                    logger.info(f"🔄 Reset (det: {old_stats['detection_rate']:.1f}%)")
                    
                elif key == ord('z'):
                    # Reset zoom
                    detector.zoom_factor = 1.0
                    detector.target_zoom = 1.0
                    detector.zoom_prediction.clear()
                    detector.zoom_stability_counter = 0
                    logger.info("🔍 Zoom reset")
                    
                elif key == ord('+') or key == ord('='):
                    # Zoom +
                    detector.target_zoom = min(detector.zoom_max, detector.target_zoom + 0.5)
                    detector.zoom_stability_counter = 0
                    logger.info(f"🔍 Zoom: {detector.target_zoom:.1f}x")
                    
                elif key == ord('-'):
                    # Zoom -
                    detector.target_zoom = max(detector.zoom_min, detector.target_zoom - 0.5)
                    detector.zoom_stability_counter = 0
                    logger.info(f"🔍 Zoom: {detector.target_zoom:.1f}x")
                    
                elif key == ord('c'):
                    # Calibration couleurs
                    if len(detector.color_samples) > 15:
                        detector._recalibrate_color_ranges()
                        logger.info(f"🎨 Calibration ({len(detector.color_samples)} échantillons)")
                    else:
                        logger.info("🎨 Pas assez d'échantillons")
                    
                elif key == ord('f'):
                    # Mode Fast
                    detector.performance_mode = 'fast'
                    detector.process_every_n_frames = 2
                    detector.use_multi_scale = False
                    detector.stabilization_enabled = False
                    logger.info("⚡ Mode FAST activé")
                    
                elif key == ord('b'):
                    # Mode Balanced
                    detector.performance_mode = 'balanced'
                    detector.process_every_n_frames = 1
                    detector.use_multi_scale = True
                    detector.stabilization_enabled = True
                    logger.info("⚖️ Mode BALANCED activé")
                    
                elif key == ord('Q'):
                    # Mode Quality
                    detector.performance_mode = 'quality'
                    detector.process_every_n_frames = 1
                    detector.use_multi_scale = True
                    detector.stabilization_enabled = True
                    logger.info("🎯 Mode QUALITY activé")
                    
                elif key == ord('p'):
                    # Stats complètes
                    stats = detector.get_performance_stats()
                    logger.info("📈 STATS COMPLÈTES:")
                    for key_stat, value in stats.items():
                        if isinstance(value, float):
                            logger.info(f"   {key_stat}: {value:.3f}")
                        else:
                            logger.info(f"   {key_stat}: {value}")
                    
                elif key == ord('d'):
                    # Debug complet
                    logger.info("🔍 DEBUG COMPLET:")
                    logger.info(f"   Frame: {detector.frame_count}")
                    logger.info(f"   Zoom: {detector.zoom_factor:.2f}x → {detector.target_zoom:.2f}x")
                    logger.info(f"   Mode: {detector.performance_mode}")
                    logger.info(f"   Skip: {detector.process_every_n_frames}")
                    logger.info(f"   Multi-scale: {detector.use_multi_scale}")
                    logger.info(f"   Stabilisation: {detector.stabilization_enabled}")
                    logger.info(f"   Détection: {detected}")
                    if detection_info.get('detected', False):
                        logger.info(f"   Aire: {detection_info.get('area', 0)}")
                        logger.info(f"   Qualité: {detection_info.get('quality_score', 0):.3f}")
                        if 'color_validation' in detection_info:
                            logger.info(f"   Validation couleur: {detection_info['color_validation']:.3f}")
                        if 'temporal_validation' in detection_info:
                            logger.info(f"   Validation temporelle: {detection_info['temporal_validation']:.3f}")
                    logger.info(f"   Tracking points: {len([t for t in detector.tracking_history if t])}")
                    logger.info(f"   Échantillons couleur: {len(detector.color_samples)}")
                    logger.info(f"   Erreurs: {detector.error_recovery_counter}")
                    
                elif key == ord('h'):
                    # Aide complète
                    print("\n" + "=" * 80)
                    print("🎮 AIDE COMPLÈTE - DÉTECTION OPTIMISÉE")
                    print("=" * 80)
                    print("CONTRÔLE PRINCIPAL:")
                    print("  s      = Screenshot avec métadonnées complètes")
                    print("  r      = Reset complet du détecteur")
                    print("  z      = Reset zoom uniquement")
                    print("  +/-    = Zoom manuel")
                    print("MODES PERFORMANCE:")
                    print("  f      = Mode FAST (vitesse max, fonctions réduites)")
                    print("  b      = Mode BALANCED (équilibre vitesse/qualité)")
                    print("  Q      = Mode QUALITY (qualité max, toutes fonctions)")
                    print("ANALYSE:")
                    print("  c      = Force calibration couleurs")
                    print("  p      = Afficher toutes les statistiques")
                    print("  d      = Debug informations complètes")
                    print("  h      = Cette aide")
                    print("FONCTIONNALITÉS PAR MODE:")
                    print("  FAST     : Détection simple, skip frames, pas de stabilisation")
                    print("  BALANCED : Multi-échelle partiel, stabilisation, tracking")
                    print("  QUALITY  : Toutes fonctions, validation cascade, Kalman")
                    print("OPTIMISATIONS AUTOMATIQUES:")
                    print("  ✓ Ajustement performance selon FPS")
                    print("  ✓ Skip frames adaptatif")
                    print("  ✓ Cache détections pour interpolation")
                    print("  ✓ Morphologie adaptée selon échelle")
                    print("=" * 80 + "\n")
                    
                elif key == ord('a'):
                    # Toggle auto performance
                    detector.auto_performance_adjustment = not detector.auto_performance_adjustment
                    status = "ACTIVÉ" if detector.auto_performance_adjustment else "DÉSACTIVÉ"
                    logger.info(f"🤖 Ajustement auto performance: {status}")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle principale: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # === NETTOYAGE COMPLET ===
        logger.info("🧹 Nettoyage complet...")
        
        if detector:
            total_runtime = time.time() - start_time
            final_stats = detector.get_performance_stats()
            
            logger.info("=" * 80)
            logger.info("📊 STATISTIQUES FINALES COMPLÈTES OPTIMISÉES")
            logger.info("=" * 80)
            logger.info(f"  ⏱️ Durée totale: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames traitées: {detector.frame_count}")
            logger.info(f"  ⚡ FPS moyen: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections: {detector.detection_count} ({final_stats['detection_rate']:.1f}%)")
            logger.info(f"  📈 Qualité moyenne: {final_stats['avg_quality']:.3f}")
            logger.info(f"  🔍 Zoom final: {detector.zoom_factor:.1f}x")
            logger.info(f"  🚫 Faux positifs rejetés: {final_stats['false_positive_rejections']}")
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            logger.info(f"  🎨 Échantillons couleur: {final_stats['color_samples_collected']}")
            logger.info(f"  📍 Points tracking: {final_stats['tracking_history_length']}")
            logger.info(f"  ⚙️ Temps traitement moy: {final_stats['avg_processing_time']*1000:.1f}ms")
            logger.info(f"  🎭 Mode final: {final_stats['performance_mode']}")
            logger.info(f"  ⏭️ Skip frames: {final_stats['frame_skip_rate']}")
            logger.info(f"  📊 FPS final: {final_stats['current_fps']:.1f}")
            logger.info(f"  🔧 Stabilisation: {'OUI' if final_stats['stabilization_enabled'] else 'NON'}")
            logger.info(f"  📐 Multi-échelle: {'OUI' if final_stats['multi_scale_enabled'] else 'NON'}")
            logger.info("=" * 80)
            
            # Recommandations finales
            if final_stats['avg_processing_time'] > 0.050:
                logger.info("💡 RECOMMANDATION: Utiliser mode FAST pour améliorer performance")
            elif final_stats['avg_processing_time'] < 0.020:
                logger.info("💡 RECOMMANDATION: Mode QUALITY disponible avec cette performance")
            
            if final_stats['detection_rate'] < 70:
                logger.info("💡 RECOMMANDATION: Vérifier éclairage ou recalibrer couleurs")
            
            # Sauvegarde statistiques complètes
            try:
                stats_filename = f"complete_detection_stats_{int(time.time())}.json"
                complete_stats = {
                    'runtime_seconds': total_runtime,
                    'total_frames': detector.frame_count,
                    'total_detections': detector.detection_count,
                    'performance_stats': final_stats,
                    'screenshots_taken': screenshot_count,
                    'system_info': {
                        'os': platform.system(),
                        'python': platform.python_version(),
                        'opencv': cv2.__version__
                    },
                    'configuration': {
                        'width': WIDTH,
                        'height': HEIGHT,
                        'zoom_max': detector.zoom_max,
                        'color_ranges': detector.color_calibration
                    },
                    'optimization_results': {
                        'avg_fps': detector.frame_count/max(total_runtime,1),
                        'processing_efficiency': final_stats['avg_processing_time'] * final_stats['current_fps'],
                        'detection_efficiency': final_stats['detection_rate'] / (final_stats['avg_processing_time']*1000)
                    }
                }
                
                with open(stats_filename, 'w') as f:
                    json.dump(complete_stats, f, indent=2)
                logger.info(f"📁 Statistiques complètes sauvées: {stats_filename}")
            except Exception as e:
                logger.debug(f"Stats save error: {e}")
        
        if pipe:
            try:
                pipe.terminate()
                pipe.wait(timeout=5)
                logger.info("✅ Pipeline FFmpeg fermé")
            except:
                try:
                    pipe.kill()
                except:
                    pass
        
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Interface fermée")
        except:
            pass
        
        if bebop:
            try:
                bebop.safe_land(10)
                bebop.disconnect()
                logger.info("✅ Drone atterri et déconnecté")
            except:
                logger.warning("⚠️ Déconnexion drone échouée")
        
        logger.info("🎉 Session détection complète optimisée terminée!")
    
    return True

# === FONCTIONS UTILITAIRES OPTIMISÉES ===

def get_system_info():
    """Informations système optimisées"""
    info = {
        'os': f"{platform.system()} {platform.release()}",
        'python': platform.python_version(),
        'processor': platform.processor() or 'Unknown',
        'architecture': platform.architecture()[0]
    }
    
    # Infos mémoire cross-platform
    try:
        if platform.system() == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        info['ram_gb'] = round(mem_kb / 1024 / 1024, 1)
                        break
        elif platform.system() == 'Darwin':
            import subprocess
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split()[-1])
                info['ram_gb'] = round(mem_bytes / 1024 / 1024 / 1024, 1)
        elif platform.system() == 'Windows':
            import subprocess
            result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    mem_bytes = int(lines[1].strip())
                    info['ram_gb'] = round(mem_bytes / 1024 / 1024 / 1024, 1)
    except:
        info['ram_gb'] = 'Unknown'
    
    try:
        info['cpu_cores'] = os.cpu_count() or 'Unknown'
    except:
        info['cpu_cores'] = 'Unknown'
    
    return info

def check_system_requirements():
    """Vérification des prérequis système"""
    logger.info("🔍 Vérification prérequis système...")
    
    # Vérification OpenCV
    try:
        cv_version = cv2.__version__
        major_version = int(cv_version.split('.')[0])
        if major_version < 4:
            logger.warning(f"⚠️ OpenCV {cv_version} détecté. Version 4.x recommandée.")
        else:
            logger.info(f"✅ OpenCV {cv_version} OK")
    except:
        logger.error("❌ OpenCV non détecté")
        return False
    
    # Vérification NumPy
    try:
        np_version = np.__version__
        logger.info(f"✅ NumPy {np_version} OK")
    except:
        logger.error("❌ NumPy non détecté")
        return False
    
    # Vérification pyparrot
    try:
        pp_version = pyparrot.__version__ if hasattr(pyparrot, '__version__') else 'Unknown'
        logger.info(f"✅ pyparrot {pp_version} OK")
    except:
        logger.error("❌ pyparrot non détecté")
        return False
    
    # Vérification mémoire
    system_info = get_system_info()
    if isinstance(system_info['ram_gb'], (int, float)) and system_info['ram_gb'] < 4:
        logger.warning(f"⚠️ RAM faible: {system_info['ram_gb']}GB. 8GB recommandé.")
    
    # Vérification CPU
    if isinstance(system_info['cpu_cores'], int) and system_info['cpu_cores'] < 4:
        logger.warning(f"⚠️ CPU limité: {system_info['cpu_cores']} cores. 4+ recommandé.")
    
    return True

def print_system_info():
    """Affichage informations système"""
    system_info = get_system_info()
    
    logger.info("💻 INFORMATIONS SYSTÈME:")
    logger.info(f"   OS: {system_info['os']}")
    logger.info(f"   Python: {system_info['python']}")
    logger.info(f"   RAM: {system_info['ram_gb']} GB")
    logger.info(f"   CPU: {system_info['cpu_cores']} cores")
    logger.info(f"   Architecture: {system_info['architecture']}")
    
    try:
        logger.info(f"   OpenCV: {cv2.__version__}")
    except:
        logger.warning("   OpenCV: Non détecté")
    
    logger.info("   Fonctionnalités: Complètes optimisées avec modes adaptatifs")

def test_camera_connection():
    """Test connexion caméra"""
    logger.info("🧪 Test connexion caméra...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.info("✅ Caméra locale détectée (pour tests)")
                # Test performance basique
                start_time = time.time()
                for _ in range(10):
                    ret, frame = cap.read()
                    if frame is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elapsed = time.time() - start_time
                test_fps = 10 / elapsed
                logger.info(f"   Performance test: {test_fps:.1f} FPS")
            cap.release()
        else:
            logger.info("ℹ️ Pas de caméra locale (normal pour Bebop)")
    except Exception as e:
        logger.debug(f"Camera test error: {e}")

if __name__ == "__main__":
    try:
        # === INITIALISATION COMPLÈTE ===
        print("\n" + "=" * 80)
        print("🚀 BEBOP 2 - SYSTÈME DÉTECTION GANT COMPLET OPTIMISÉ")
        print("🎯 Toutes fonctionnalités avancées + Performance maximale")
        print("=" * 80)
        
        # Vérifications système
        print_system_info()
        
        if not check_system_requirements():
            print("❌ Prérequis système non satisfaits!")
            sys.exit(1)
        
        test_camera_connection()
        
        # Lancement principal
        print("\n🎬 Lancement du système complet optimisé...")
        success = main()
        
        # Code de sortie
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        
        if success:
            print("✅ Session complète terminée avec succès!")
            print("📊 Consultez les logs pour les statistiques détaillées")
        else:
            print("❌ Session terminée avec erreurs")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("⌨️ Interruption utilisateur")
        print("\n🛑 Arrêt par l'utilisateur")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"💥 Exception critique: {e}")
        import traceback
        logger.error(f"Traceback complet: {traceback.format_exc()}")
        print(f"\n💥 Erreur critique: {e}")
        sys.exit(1)