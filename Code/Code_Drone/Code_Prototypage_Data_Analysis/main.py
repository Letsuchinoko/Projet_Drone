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
        logging.FileHandler('bebop_enhanced_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === UTILITAIRES DE CLUSTERING SIMPLE (REMPLACEMENT SCIKIT-LEARN) ===
class SimpleKMeans:
    """Implémentation simple de K-means pour éviter la dépendance scikit-learn"""
    
    def __init__(self, n_clusters=2, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit(self, data):
        """Ajustement du modèle K-means"""
        try:
            data = np.array(data)
            n_samples, n_features = data.shape
            
            # Initialisation aléatoire des centres
            np.random.seed(42)
            self.cluster_centers_ = data[np.random.choice(n_samples, self.n_clusters, replace=False)]
            
            for _ in range(self.max_iters):
                # Attribution des points aux clusters
                distances = np.sqrt(((data - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
                self.labels_ = np.argmin(distances, axis=0)
                
                # Mise à jour des centres
                new_centers = np.array([data[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
                
                # Vérification de convergence
                if np.allclose(self.cluster_centers_, new_centers):
                    break
                    
                self.cluster_centers_ = new_centers
            
            return self
            
        except Exception as e:
            logger.debug(f"Simple K-means error: {e}")
            # Fallback: retourner des centres par défaut
            self.cluster_centers_ = np.array([[10, 180, 180], [0, 200, 200]])  # Orange et rouge typiques
            self.labels_ = np.zeros(len(data), dtype=int)
            return self

# === UTILITAIRES SYSTÈME SANS PSUTIL ===
def get_system_info():
    """Informations système sans dépendance psutil"""
    info = {
        'os': f"{platform.system()} {platform.release()}",
        'python': platform.python_version(),
        'processor': platform.processor() or 'Unknown',
        'architecture': platform.architecture()[0]
    }
    
    # Tentative d'obtenir des infos mémoire sur Linux/Mac
    try:
        if platform.system() == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        info['ram_gb'] = round(mem_kb / 1024 / 1024, 1)
                        break
        elif platform.system() == 'Darwin':  # macOS
            import subprocess
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.split()[-1])
                info['ram_gb'] = round(mem_bytes / 1024 / 1024 / 1024, 1)
    except:
        info['ram_gb'] = 'Unknown'
    
    # Nombre de CPU
    try:
        info['cpu_cores'] = os.cpu_count() or 'Unknown'
    except:
        info['cpu_cores'] = 'Unknown'
    
    return info

# === PARAMÈTRES OPTIMISÉS AVEC AMÉLIORATIONS COMPLÈTES ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_enhanced_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT COMPLET AVEC TOUTES LES AMÉLIORATIONS ===
class CompleteEnhancedGloveDetector:
    def __init__(self):
        # Configuration de base améliorée
        self.detection_history = deque(maxlen=15)
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3
        
        # Paramètres de détection optimisés
        self.min_area = 80       # Plus petit pour distances extrêmes
        self.max_area = 150000   # Plus grand pour gros plans
        self.min_contour_points = 8
        
        # === STABILISATION IMAGE ===
        self.stabilization_buffer = deque(maxlen=3)
        self.optical_flow_points = None
        self.stabilization_transform = None
        
        # === TRACKING TEMPOREL AVANCÉ ===
        self.tracking_history = deque(maxlen=20)
        self.velocity_estimation = deque(maxlen=5)
        self.prediction_zone = None
        self.kalman_filter = self._init_kalman_filter()
        
        # === ADAPTATION LUMIÈRE ===
        self.lighting_adaptation = deque(maxlen=10)
        self.exposure_compensation = 0
        self.brightness_history = deque(maxlen=5)
        self.contrast_history = deque(maxlen=5)
        
        # === DÉTECTION MULTI-ÉCHELLE ===
        self.multi_scale_levels = [1.0, 0.8, 0.6, 0.4]
        self.scale_weights = [1.0, 0.8, 0.6, 0.4]
        self.scale_detection_cache = {}
        
        # === ZONES DE CONFIANCE ===
        self.confidence_zones = {
            'high': deque(maxlen=10),
            'medium': deque(maxlen=15), 
            'low': deque(maxlen=20)
        }
        
        # === FILTRAGE ADAPTATIF COULEURS ===
        self.color_calibration = {
            'orange_ranges': [
                ([10, 180, 180], [18, 255, 255]),  # Principal
                ([12, 200, 200], [20, 255, 255]),  # Vif
                ([14, 140, 160], [19, 220, 240])   # Ombré
            ],
            'red_ranges': [
                ([0, 180, 180], [8, 255, 255]),    # Rouge bas
                ([172, 180, 180], [180, 255, 255]), # Rouge haut
                ([0, 200, 200], [6, 255, 255])     # Rouge vif
            ]
        }
        
        # Auto-calibration des couleurs
        self.color_samples = deque(maxlen=100)
        self.background_model = None
        
        # === ZOOM ADAPTATIF AMÉLIORÉ ===
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.06  # Plus lent pour ultra-stabilité
        self.zoom_min = 1.0
        self.zoom_max = 6.0  # Zoom plus puissant
        
        # Prédiction de zoom intelligent
        self.zoom_prediction = deque(maxlen=5)
        self.zoom_stability_counter = 0
        
        # === VALIDATION AVANCÉE ===
        self.validation_cascade = {
            'geometry_threshold': 0.3,
            'color_threshold': 0.4,
            'motion_threshold': 0.2,
            'temporal_threshold': 0.3
        }
        
        # === PERFORMANCE ET STATS ===
        self.frame_count = 0
        self.detection_count = 0
        self.quality_scores = deque(maxlen=50)
        self.false_positive_rejection = 0
        self.processing_times = deque(maxlen=30)
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # === GESTION D'ERREURS ROBUSTE ===
        self.error_recovery_counter = 0
        self.last_successful_detection = None
        
        logger.info("🚀 Détecteur Enhanced Glove initialisé avec toutes les améliorations")

    def _init_kalman_filter(self):
        """Initialisation du filtre de Kalman pour tracking smooth"""
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
        """Détection complète avec toutes les améliorations"""
        if frame is None:
            return frame, False, {}
            
        start_time = time.time()
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            # === PHASE 1: PRÉTRAITEMENT AVANCÉ ===
            
            # 1.1 Stabilisation d'image
            stabilized_frame = self._advanced_stabilization(frame)
            
            # 1.2 Adaptation automatique de la lumière
            light_adapted_frame = self._intelligent_lighting_adaptation(stabilized_frame)
            
            # 1.3 Amélioration de contraste adaptatif
            enhanced_frame = self._adaptive_contrast_enhancement(light_adapted_frame)
            
            # === PHASE 2: DÉTECTION MULTI-STRATÉGIE ===
            
            # 2.1 Prédiction de zone de recherche
            predicted_zone = self._advanced_prediction()
            
            # 2.2 Détection multi-échelle avec cache
            multi_scale_results = self._optimized_multi_scale_detection(enhanced_frame, predicted_zone)
            
            # 2.3 Détection par zones de confiance
            confidence_results = self._confidence_zone_detection(enhanced_frame)
            
            # === PHASE 3: FUSION ET VALIDATION ===
            
            # 3.1 Fusion intelligente des résultats
            fused_detection = self._intelligent_fusion(
                multi_scale_results, confidence_results, predicted_zone
            )
            
            # 3.2 Validation en cascade
            validated_detection = self._cascade_validation(fused_detection, enhanced_frame)
            
            # 3.3 Filtrage temporel avec Kalman
            final_detection = self._kalman_temporal_filtering(validated_detection)
            
            # === PHASE 4: MISE À JOUR ET APPRENTISSAGE ===
            
            # 4.1 Mise à jour du tracking
            self._update_advanced_tracking(final_detection)
            
            # 4.2 Auto-calibration des couleurs
            self._auto_color_calibration(enhanced_frame, final_detection)
            
            # 4.3 Adaptation du zoom intelligent
            self._intelligent_zoom_adaptation(final_detection)
            
            # 4.4 Mise à jour des zones de confiance
            self._update_confidence_zones(final_detection)
            
            # === PHASE 5: RENDU ET DIAGNOSTICS ===
            
            # 5.1 Rendu avec informations complètes
            result_frame = self._comprehensive_rendering(original_frame, final_detection)
            
            # 5.2 Mise à jour des statistiques
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if final_detection['detected']:
                self.detection_count += 1
                self.quality_scores.append(final_detection.get('quality_score', 0))
                self.last_successful_detection = final_detection
                self.error_recovery_counter = 0
            else:
                self.error_recovery_counter += 1
            
            return result_frame, final_detection['detected'], final_detection
            
        except Exception as e:
            logger.error(f"Enhanced detection critical error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Récupération d'erreur
            return self._error_recovery(original_frame)

    def _advanced_stabilization(self, frame):
        """Stabilisation avancée avec flux optique et compensation"""
        try:
            if len(self.stabilization_buffer) == 0:
                self.stabilization_buffer.append(frame)
                return frame
            
            prev_frame = self.stabilization_buffer[-1]
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détection de points caractéristiques améliorée
            if self.optical_flow_points is None or len(self.optical_flow_points) < 20:
                self.optical_flow_points = cv2.goodFeaturesToTrack(
                    prev_gray, 
                    maxCorners=150,
                    qualityLevel=0.01,
                    minDistance=10,
                    blockSize=7,
                    useHarrisDetector=False,
                    k=0.04
                )
            
            if self.optical_flow_points is not None and len(self.optical_flow_points) > 10:
                # Paramètres Lucas-Kanade optimisés
                lk_params = dict(
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                    minEigThreshold=1e-4
                )
                
                # Calcul du flux optique
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, self.optical_flow_points, None, **lk_params
                )
                
                # Filtrage des points valides
                good_new = new_points[status == 1]
                good_old = self.optical_flow_points[status == 1]
                
                if len(good_new) > 8:
                    # Estimation robuste de la transformation (RANSAC)
                    transform, mask = cv2.estimateAffinePartial2D(
                        good_old, good_new, 
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.99
                    )
                    
                    if transform is not None:
                        # Limitation des transformations extrêmes
                        translation_limit = 30
                        rotation_limit = 0.1
                        scale_limit = 0.05
                        
                        # Extraction des paramètres
                        tx, ty = transform[0, 2], transform[1, 2]
                        scale_x = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
                        scale_y = np.sqrt(transform[1, 0]**2 + transform[1, 1]**2)
                        
                        # Limitation des translations
                        tx = np.clip(tx, -translation_limit, translation_limit)
                        ty = np.clip(ty, -translation_limit, translation_limit)
                        
                        # Limitation du scaling
                        scale_x = np.clip(scale_x, 1-scale_limit, 1+scale_limit)
                        scale_y = np.clip(scale_y, 1-scale_limit, 1+scale_limit)
                        
                        # Reconstruction de la transformation limitée
                        limited_transform = transform.copy()
                        limited_transform[0, 2] = tx
                        limited_transform[1, 2] = ty
                        
                        # Application de la stabilisation
                        h, w = frame.shape[:2]
                        stabilized = cv2.warpAffine(
                            frame, limited_transform, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101
                        )
                        
                        # Mise à jour des points pour la prochaine frame
                        self.optical_flow_points = good_new.reshape(-1, 1, 2)
                        
                        # Lissage temporel
                        if len(self.stabilization_buffer) >= 2:
                            alpha = 0.3
                            blended = cv2.addWeighted(
                                stabilized, alpha, 
                                self.stabilization_buffer[-1], 1-alpha, 0
                            )
                            self.stabilization_buffer.append(blended)
                            return blended
                        else:
                            self.stabilization_buffer.append(stabilized)
                            return stabilized
                    
                # Reset des points si transformation échouée
                self.optical_flow_points = None
            
            self.stabilization_buffer.append(frame)
            return frame
            
        except Exception as e:
            logger.debug(f"Advanced stabilization error: {e}")
            return frame

    def _intelligent_lighting_adaptation(self, frame):
        """Adaptation intelligente de l'éclairage avec analyse avancée"""
        try:
            # Conversion en différents espaces colorimétriques pour analyse
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Analyse de l'histogramme L*a*b*
            l_channel = lab[:, :, 0]
            hist_l = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
            
            # Calcul des métriques d'éclairage
            mean_brightness = np.mean(l_channel)
            std_brightness = np.std(l_channel)
            
            # Détection des conditions d'éclairage
            is_underexposed = mean_brightness < 80
            is_overexposed = mean_brightness > 200
            is_low_contrast = std_brightness < 20
            is_high_contrast = std_brightness > 60
            
            self.brightness_history.append(mean_brightness)
            self.contrast_history.append(std_brightness)
            
            # Adaptation selon les conditions
            if is_underexposed:
                # Amélioration pour sous-exposition
                l_enhanced = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(l_channel)
                
                # Boost sélectif des tons moyens
                lut = np.arange(256, dtype=np.uint8)
                lut[50:200] = np.clip(lut[50:200] * 1.3, 0, 255)
                l_enhanced = cv2.LUT(l_enhanced, lut)
                
                enhanced_lab = lab.copy()
                enhanced_lab[:, :, 0] = l_enhanced
                result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
            elif is_overexposed:
                # Correction pour surexposition
                l_compressed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l_channel)
                
                # Compression des hautes lumières
                lut = np.arange(256, dtype=np.uint8)
                lut[180:255] = 180 + (lut[180:255] - 180) * 0.7
                l_compressed = cv2.LUT(l_compressed, lut)
                
                enhanced_lab = lab.copy()
                enhanced_lab[:, :, 0] = l_compressed
                result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
            elif is_low_contrast:
                # Amélioration du contraste
                l_contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)).apply(l_channel)
                
                enhanced_lab = lab.copy()
                enhanced_lab[:, :, 0] = l_contrast
                result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Boost de la saturation
                result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                result_hsv[:, :, 1] = cv2.multiply(result_hsv[:, :, 1], 1.2)
                result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
                
            else:
                # Amélioration standard
                result = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)
            
            # Lissage temporel pour éviter le flickering
            if len(self.lighting_adaptation) >= 2:
                weights = [0.6, 0.3, 0.1] if len(self.lighting_adaptation) >= 3 else [0.7, 0.3]
                blended = np.zeros_like(result, dtype=np.float32)
                
                blended += result.astype(np.float32) * weights[0]
                for i in range(1, min(len(weights), len(self.lighting_adaptation))):
                    blended += self.lighting_adaptation[-(i)].astype(np.float32) * weights[i]
                
                result = np.clip(blended, 0, 255).astype(np.uint8)
            
            self.lighting_adaptation.append(result)
            return result
            
        except Exception as e:
            logger.debug(f"Intelligent lighting adaptation error: {e}")
            return frame

    def _adaptive_contrast_enhancement(self, frame):
        """Amélioration adaptative du contraste avec préservation des couleurs"""
        try:
            # Conversion YUV pour traiter la luminance séparément
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            # Analyse de la distribution des intensités
            hist_y = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
            
            # Détection des zones d'intérêt (potentiellement le gant)
            # Filtrage gaussien pour réduire le bruit
            y_smooth = cv2.GaussianBlur(y_channel, (5, 5), 0)
            
            # Égalisation adaptative avec préservation des détails
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            y_enhanced = clahe.apply(y_smooth)
            
            # Sharpening sélectif
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
            y_sharpened = cv2.filter2D(y_enhanced, -1, kernel_sharpen * 0.1)
            y_enhanced = cv2.addWeighted(y_enhanced, 0.8, y_sharpened, 0.2, 0)
            
            # Reconstruction
            enhanced_yuv = yuv.copy()
            enhanced_yuv[:, :, 0] = y_enhanced
            
            result = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)
            
            return result
            
        except Exception as e:
            logger.debug(f"Adaptive contrast enhancement error: {e}")
            return frame

    def _advanced_prediction(self):
        """Prédiction avancée avec Kalman et analyse de mouvement"""
        try:
            if len(self.tracking_history) < 3:
                return None
            
            # Extraction des positions valides récentes
            valid_tracks = [t for t in list(self.tracking_history)[-10:] if t is not None]
            
            if len(valid_tracks) < 2:
                return None
            
            # Mise à jour du filtre de Kalman
            if self.kalman_filter is not None and len(valid_tracks) >= 1:
                last_track = valid_tracks[-1]
                measurement = np.array([[last_track['center'][0]], 
                                      [last_track['center'][1]]], dtype=np.float32)
                
                # Prédiction
                prediction = self.kalman_filter.predict()
                
                # Correction
                self.kalman_filter.correct(measurement)
                
                # Position prédite
                pred_x, pred_y = int(prediction[0, 0]), int(prediction[1, 0])
                
                # Calcul de l'incertitude
                uncertainty = np.trace(self.kalman_filter.errorCovPre[:2, :2])
                search_radius = max(30, min(100, int(uncertainty * 50)))
                
                return {
                    'center': (pred_x, pred_y),
                    'radius': search_radius,
                    'confidence': min(1.0, len(valid_tracks) / 5.0),
                    'type': 'kalman'
                }
            
            # Fallback: prédiction basée sur la vélocité
            if len(valid_tracks) >= 2:
                recent_centers = [t['center'] for t in valid_tracks[-5:]]
                
                # Calcul de la vélocité moyenne
                velocities = []
                for i in range(1, len(recent_centers)):
                    dx = recent_centers[i][0] - recent_centers[i-1][0]
                    dy = recent_centers[i][1] - recent_centers[i-1][1]
                    velocities.append((dx, dy))
                
                if velocities:
                    avg_vx = np.mean([v[0] for v in velocities])
                    avg_vy = np.mean([v[1] for v in velocities])
                    
                    # Prédiction
                    last_pos = recent_centers[-1]
                    pred_x = last_pos[0] + avg_vx * 3
                    pred_y = last_pos[1] + avg_vy * 3
                    
                    # Calcul du rayon adaptatif
                    velocity_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
                    search_radius = max(40, min(120, int(30 + velocity_magnitude * 2)))
                    
                    return {
                        'center': (int(pred_x), int(pred_y)),
                        'radius': search_radius,
                        'confidence': min(1.0, len(valid_tracks) / 3.0),
                        'type': 'velocity'
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Advanced prediction error: {e}")
            return None

    def _optimized_multi_scale_detection(self, frame, predicted_zone=None):
        """Détection multi-échelle optimisée avec cache et zone prédictive"""
        try:
            results = []
            
            for i, scale in enumerate(self.multi_scale_levels):
                # Optimisation: focus sur la zone prédite pour échelles > 1.0
                if predicted_zone and scale >= 0.8:
                    detection_frame = self._crop_prediction_zone(frame, predicted_zone, scale)
                    if detection_frame is None:
                        detection_frame = frame
                else:
                    detection_frame = frame
                
                # Redimensionnement
                if scale != 1.0:
                    h, w = detection_frame.shape[:2]
                    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                    scaled_frame = cv2.resize(detection_frame, (new_w, new_h), 
                                            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                else:
                    scaled_frame = detection_frame
                
                # Cache des détections (pour optimiser les performances)
                cache_key = f"{scale}_{self.frame_count % 5}"
                
                # Détection sur cette échelle
                detection_result = self._enhanced_detect_at_scale(scaled_frame, scale)
                
                if detection_result['detected']:
                    # Remapping vers coordonnées originales
                    if scale != 1.0 or predicted_zone:
                        detection_result = self._remap_detection_to_original(
                            detection_result, scale, predicted_zone
                        )
                    
                    # Pondération par qualité d'échelle et prédiction
                    detection_result['scale_weight'] = self.scale_weights[i]
                    if predicted_zone:
                        detection_result['prediction_bonus'] = 1.1
                    
                    results.append(detection_result)
            
            return results
            
        except Exception as e:
            logger.debug(f"Optimized multi-scale detection error: {e}")
            return []

    def _crop_prediction_zone(self, frame, predicted_zone, scale):
        """Découpage de la zone de prédiction pour optimiser la détection"""
        try:
            h, w = frame.shape[:2]
            center_x, center_y = predicted_zone['center']
            radius = predicted_zone['radius']
            
            # Expansion de la zone selon l'échelle
            expanded_radius = int(radius * (2.0 - scale))
            
            # Calcul des limites
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

    def _enhanced_detect_at_scale(self, frame, scale):
        """Détection améliorée pour une échelle donnée"""
        try:
            h, w = frame.shape[:2]
            
            # Prétraitement adaptatif selon l'échelle
            if scale < 0.7:
                # Échelle réduite: préservation des détails
                blur_size = 3
                morph_size = 2
            else:
                # Échelle normale: lissage standard
                blur_size = 5
                morph_size = 3
            
            # Conversion HSV avec lissage adaptatif
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (blur_size, blur_size), 0)
            
            # === DÉTECTION COULEUR ADAPTATIVE ===
            
            # Ajustement des seuils selon l'échelle et l'historique
            sat_adjustment = max(0, int(20 * (2 - scale)))
            val_adjustment = max(0, int(15 * (2 - scale)))
            
            # Adaptation basée sur l'historique de luminosité
            if len(self.brightness_history) > 0:
                avg_brightness = np.mean(self.brightness_history)
                if avg_brightness < 100:  # Conditions sombres
                    val_adjustment -= 20
                elif avg_brightness > 180:  # Conditions claires
                    val_adjustment += 10
            
            # Masques couleur avec plages adaptatives
            mask_orange = self._create_adaptive_color_mask(
                hsv, self.color_calibration['orange_ranges'], sat_adjustment, val_adjustment
            )
            
            mask_red = self._create_adaptive_color_mask(
                hsv, self.color_calibration['red_ranges'], sat_adjustment, val_adjustment
            )
            
            # Combinaison intelligente des masques
            mask_glove = cv2.bitwise_or(mask_orange, mask_red)
            
            # === EXCLUSIONS AVANCÉES ===
            
            # Exclusion peau adaptée
            mask_skin = self._create_skin_exclusion_mask(hsv, scale)
            
            # Exclusion arrière-plan avec modèle adaptatif
            if self.background_model is not None:
                mask_bg = self._apply_background_model(hsv)
                mask_glove = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_bg))
            
            # Application exclusion peau
            mask_skin_processed = cv2.morphologyEx(
                mask_skin, cv2.MORPH_DILATE, 
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            )
            mask_final = cv2.bitwise_and(mask_glove, cv2.bitwise_not(mask_skin_processed))
            
            # === MORPHOLOGIE ADAPTATIVE ===
            
            mask_final = self._adaptive_morphological_processing(mask_final, scale)
            
            # === DÉTECTION DE CONTOURS AVANCÉE ===
            
            contours, hierarchy = cv2.findContours(
                mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            
            best_contour, quality_score = self._advanced_contour_selection(
                contours, hierarchy, scale, frame
            )
            
            result = {
                'detected': best_contour is not None,
                'contour': best_contour,
                'area': cv2.contourArea(best_contour) if best_contour is not None else 0,
                'scale': scale,
                'quality_score': quality_score,
                'mask': mask_final,
                'frame': frame,
                'color_confidence': self._calculate_color_confidence(mask_orange, mask_red)
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Enhanced scale detection error: {e}")
            return {'detected': False, 'contour': None, 'area': 0, 'scale': scale, 'quality_score': 0}

    def _create_adaptive_color_mask(self, hsv, color_ranges, sat_adj, val_adj):
        """Création de masque couleur adaptatif"""
        try:
            h, w = hsv.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for lower_base, upper_base in color_ranges:
                # Ajustement adaptatif des seuils
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
            logger.debug(f"Adaptive color mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _create_skin_exclusion_mask(self, hsv, scale):
        """Création du masque d'exclusion peau adaptatif"""
        try:
            # Adaptation des seuils de peau selon l'échelle
            strictness = max(0.5, 1.0 - (scale - 1.0) * 0.2)
            
            skin_h_range = 10  # Teinte peau
            skin_s_max = int(120 * strictness)
            skin_v_min = max(80, int(140 * strictness))
            
            skin_lower = np.array([5, 60, skin_v_min])
            skin_upper = np.array([15, skin_s_max, 220])
            
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)
            
            return mask_skin
            
        except Exception as e:
            logger.debug(f"Skin exclusion mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _adaptive_morphological_processing(self, mask, scale):
        """Traitement morphologique adaptatif selon l'échelle"""
        try:
            # Taille des kernels adaptatifs
            if scale > 1.5:
                close_size, open_size = 8, 3
                iterations = 2
            elif scale > 1.0:
                close_size, open_size = 6, 2
                iterations = 1
            else:
                close_size, open_size = max(3, int(5 * scale)), max(2, int(3 * scale))
                iterations = 1
            
            # Kernels elliptiques pour forme naturelle
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            
            # Traitement progressif
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            # Lissage final
            mask = cv2.medianBlur(mask, 3)
            
            return mask
            
        except Exception as e:
            logger.debug(f"Adaptive morphological processing error: {e}")
            return mask

    def _advanced_contour_selection(self, contours, hierarchy, scale, frame):
        """Sélection avancée de contour avec scoring multiple"""
        if not contours:
            return None, 0
            
        try:
            best_contour = None
            best_score = 0
            
            # Ajustement des seuils selon l'échelle
            min_area_scaled = self.min_area * (scale ** 1.3)
            max_area_scaled = self.max_area * (scale ** 1.3)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Filtres de base
                if area < min_area_scaled or area > max_area_scaled:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # === ANALYSE GÉOMÉTRIQUE COMPLÈTE ===
                
                # Rectangle englobant
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Analyse de convexité
                hull = cv2.convexHull(contour, returnPoints=False)
                if len(hull) > 3:
                    defects = cv2.convexityDefects(contour, hull)
                    convexity_ratio = area / cv2.contourArea(cv2.convexHull(contour))
                    
                    # Analyse des défauts (doigts potentiels)
                    finger_like_defects = 0
                    if defects is not None:
                        for defect in defects:
                            s, e, f, d = defect[0]
                            if d > 1000:  # Défaut significatif
                                finger_like_defects += 1
                else:
                    defects = None
                    convexity_ratio = 1.0
                    finger_like_defects = 0
                
                # Moments pour analyse de forme
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # Hu moments pour invariance
                    hu_moments = cv2.HuMoments(moments).flatten()
                    
                    # Compacité
                    perimeter = cv2.arcLength(contour, True)
                    compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 0
                else:
                    continue
                
                # === SCORING MULTI-CRITÈRES AVANCÉ ===
                
                score = 0
                
                # 1. Score géométrique (30%)
                if 0.3 <= aspect_ratio <= 2.2:  # Forme main raisonnable
                    geo_score = 1.0 - abs(aspect_ratio - 1.0) * 0.4
                else:
                    geo_score = 0.1
                
                # 2. Score de convexité (25%)
                if 0.65 <= convexity_ratio <= 0.92:  # Main avec doigts
                    conv_score = 1.0
                elif convexity_ratio > 0.92:  # Trop convexe
                    conv_score = 0.7
                else:  # Pas assez convexe
                    conv_score = max(0, convexity_ratio / 0.65)
                
                # 3. Score de position (20%)
                frame_center_x, frame_center_y = frame.shape[1] // 2, frame.shape[0] // 2
                dist_from_center = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
                max_dist = np.sqrt(frame_center_x**2 + frame_center_y**2)
                pos_score = 1.0 - (dist_from_center / max_dist) * 0.5
                
                # 4. Score de taille optimale (15%)
                optimal_area = 3000 * (scale ** 1.5)
                size_diff = abs(area - optimal_area) / optimal_area
                size_score = max(0, 1.0 - size_diff)
                
                # 5. Score de compacité (10%)
                # Compacité idéale pour une main: entre 2 et 4
                if 2 <= compactness <= 4:
                    compact_score = 1.0
                else:
                    compact_score = max(0, 1.0 - abs(compactness - 3) * 0.2)
                
                # Score final pondéré
                final_score = (geo_score * 0.3 + conv_score * 0.25 + 
                              pos_score * 0.2 + size_score * 0.15 + compact_score * 0.1)
                
                # === BONUS ET MALUS ===
                
                # Bonus pour défauts de convexité (doigts)
                if 1 <= finger_like_defects <= 4:
                    final_score *= 1.1
                elif finger_like_defects > 4:
                    final_score *= 0.9  # Trop de défauts
                
                # Bonus pour échelle optimale
                if scale >= 0.8:
                    final_score *= 1.05
                
                # Bonus historique (si proche des détections précédentes)
                if len(self.tracking_history) > 0:
                    last_track = self.tracking_history[-1]
                    if last_track and 'center' in last_track:
                        last_cx, last_cy = last_track['center']
                        tracking_dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                        if tracking_dist < 100:  # Proche de la dernière détection
                            final_score *= 1.15
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
            
            return best_contour, best_score
            
        except Exception as e:
            logger.debug(f"Advanced contour selection error: {e}")
            return None, 0

    def _calculate_color_confidence(self, mask_orange, mask_red):
        """Calcul de la confiance couleur"""
        try:
            total_pixels = mask_orange.shape[0] * mask_orange.shape[1]
            orange_pixels = cv2.countNonZero(mask_orange)
            red_pixels = cv2.countNonZero(mask_red)
            
            orange_ratio = orange_pixels / total_pixels
            red_ratio = red_pixels / total_pixels
            
            # Confiance basée sur la présence des deux couleurs
            if orange_ratio > 0.001 and red_ratio > 0.001:
                return min(1.0, (orange_ratio + red_ratio) * 100)
            elif orange_ratio > 0.002 or red_pixels > 0.002:
                return min(0.7, max(orange_ratio, red_ratio) * 80)
            else:
                return 0.1
                
        except Exception as e:
            logger.debug(f"Color confidence calculation error: {e}")
            return 0.0

    def _confidence_zone_detection(self, frame):
        """Détection par zones de confiance"""
        try:
            results = []
            h, w = frame.shape[:2]
            
            # Définition des zones de confiance
            zones = [
                # Zone haute confiance (centre élargi)
                {'name': 'high', 'region': (w//4, h//4, w//2, h//2), 'weight': 1.2},
                # Zone moyenne confiance (bords proches)
                {'name': 'medium', 'region': (w//8, h//8, 3*w//4, 3*h//4), 'weight': 1.0},
                # Zone faible confiance (bords extrêmes)
                {'name': 'low', 'region': (0, 0, w, h), 'weight': 0.8}
            ]
            
            for zone in zones:
                x, y, zone_w, zone_h = zone['region']
                zone_frame = frame[y:y+zone_h, x:x+zone_w]
                
                # Détection dans cette zone
                zone_result = self._enhanced_detect_at_scale(zone_frame, 1.0)
                
                if zone_result['detected']:
                    # Remapping vers coordonnées globales
                    if zone_result['contour'] is not None:
                        zone_result['contour'][:, :, 0] += x
                        zone_result['contour'][:, :, 1] += y
                    
                    zone_result['zone_weight'] = zone['weight']
                    zone_result['zone_name'] = zone['name']
                    results.append(zone_result)
            
            return results
            
        except Exception as e:
            logger.debug(f"Confidence zone detection error: {e}")
            return []

    def _intelligent_fusion(self, multi_scale_results, confidence_results, predicted_zone):
        """Fusion intelligente de tous les résultats de détection"""
        try:
            all_results = multi_scale_results + confidence_results
            
            if not all_results:
                return {'detected': False, 'contour': None, 'area': 0, 'quality_score': 0}
            
            # Scoring et fusion
            best_result = None
            best_fusion_score = 0
            
            for result in all_results:
                if not result['detected']:
                    continue
                
                # Score de base
                fusion_score = result['quality_score']
                
                # Pondération par échelle
                if 'scale_weight' in result:
                    fusion_score *= result['scale_weight']
                
                # Pondération par zone
                if 'zone_weight' in result:
                    fusion_score *= result['zone_weight']
                
                # Bonus prédiction
                if predicted_zone and 'prediction_bonus' in result:
                    fusion_score *= result['prediction_bonus']
                
                # Bonus cohérence couleur
                if 'color_confidence' in result:
                    fusion_score *= (0.5 + result['color_confidence'] * 0.5)
                
                if fusion_score > best_fusion_score:
                    best_fusion_score = fusion_score
                    best_result = result
            
            if best_result:
                best_result['fusion_score'] = best_fusion_score
                return best_result
            else:
                return {'detected': False, 'contour': None, 'area': 0, 'quality_score': 0}
                
        except Exception as e:
            logger.debug(f"Intelligent fusion error: {e}")
            return {'detected': False, 'contour': None, 'area': 0, 'quality_score': 0}

    def _cascade_validation(self, detection, frame):
        """Validation en cascade avec multiple critères"""
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
            
            # === VALIDATION COULEUR AVANCÉE ===
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
            
            # Ajout des scores de validation
            detection['color_validation'] = color_validation
            detection['temporal_validation'] = temporal_validation
            detection['validation_passed'] = True
            
            return detection
            
        except Exception as e:
            logger.debug(f"Cascade validation error: {e}")
            detection['detected'] = False
            return detection

    def _validate_color_composition(self, frame, contour):
        """Validation de la composition couleur à l'intérieur du contour"""
        try:
            # Création du masque du contour
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Extraction de la région d'intérêt
            roi = cv2.bitwise_and(frame, frame, mask=mask)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Comptage des pixels couleur cible
            total_roi_pixels = cv2.countNonZero(mask)
            if total_roi_pixels == 0:
                return 0.0
            
            # Test des couleurs orange et rouge
            orange_pixels = 0
            red_pixels = 0
            
            for lower, upper in self.color_calibration['orange_ranges']:
                orange_mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
                orange_pixels += cv2.countNonZero(orange_mask)
            
            for lower, upper in self.color_calibration['red_ranges']:
                red_mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
                red_pixels += cv2.countNonZero(red_mask)
            
            # Calcul du ratio de couleurs cibles
            target_pixels = orange_pixels + red_pixels
            color_ratio = target_pixels / total_roi_pixels
            
            # Bonus si les deux couleurs sont présentes
            both_colors_bonus = 1.0
            if orange_pixels > 0 and red_pixels > 0:
                both_colors_bonus = 1.3
            
            return min(1.0, color_ratio * both_colors_bonus)
            
        except Exception as e:
            logger.debug(f"Color composition validation error: {e}")
            return 0.0

    def _validate_movement_consistency(self, detection):
        """Validation de la cohérence du mouvement"""
        try:
            if not detection['contour'] is not None:
                return 0.0
            
            # Position actuelle
            moments = cv2.moments(detection['contour'])
            if moments["m00"] == 0:
                return 0.0
                
            curr_x = int(moments["m10"] / moments["m00"])
            curr_y = int(moments["m01"] / moments["m00"])
            
            # Positions récentes
            recent_positions = []
            for track in list(self.tracking_history)[-5:]:
                if track and 'center' in track:
                    recent_positions.append(track['center'])
            
            if len(recent_positions) < 2:
                return 1.0  # Pas assez d'historique
            
            # Calcul des distances
            distances = []
            for pos in recent_positions:
                dist = np.sqrt((curr_x - pos[0])**2 + (curr_y - pos[1])**2)
                distances.append(dist)
            
            # Validation: mouvement pas trop brusque
            max_distance = max(distances)
            if max_distance > 150:  # Mouvement trop brusque
                return 0.1
            elif max_distance > 100:
                return 0.5
            else:
                return 1.0
                
        except Exception as e:
            logger.debug(f"Movement consistency validation error: {e}")
            return 0.5

    def _validate_temporal_consistency(self, detection):
        """Validation de la cohérence temporelle"""
        try:
            # Cohérence avec l'historique récent
            recent_detections = list(self.detection_history)[-10:]
            if len(recent_detections) == 0:
                return 1.0
            
            # Ratio de détections récentes
            detection_ratio = sum(recent_detections) / len(recent_detections)
            
            # Score basé sur la cohérence
            if detection_ratio > 0.7:
                return 1.0  # Détections cohérentes
            elif detection_ratio > 0.4:
                return 0.8  # Moyennement cohérent
            else:
                return 0.3  # Peu cohérent
                
        except Exception as e:
            logger.debug(f"Temporal consistency validation error: {e}")
            return 0.5

    def _kalman_temporal_filtering(self, detection):
        """Filtrage temporel avec Kalman pour lissage"""
        try:
            if not detection['detected'] or self.kalman_filter is None:
                return detection
            
            contour = detection['contour']
            if contour is None:
                return detection
            
            # Position actuelle
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                return detection
            
            curr_x = int(moments["m10"] / moments["m00"])
            curr_y = int(moments["m01"] / moments["m00"])
            
            # Mise à jour Kalman
            measurement = np.array([[curr_x], [curr_y]], dtype=np.float32)
            
            # Prédiction et correction
            prediction = self.kalman_filter.predict()
            corrected = self.kalman_filter.correct(measurement)
            
            # Position filtrée
            filtered_x = int(corrected[0, 0])
            filtered_y = int(corrected[1, 0])
            
            # Ajustement du contour (translation)
            offset_x = filtered_x - curr_x
            offset_y = filtered_y - curr_y
            
            if abs(offset_x) < 50 and abs(offset_y) < 50:  # Ajustement raisonnable
                filtered_contour = contour.copy()
                filtered_contour[:, :, 0] += offset_x
                filtered_contour[:, :, 1] += offset_y
                
                detection['contour'] = filtered_contour
                detection['kalman_filtered'] = True
                detection['filter_offset'] = (offset_x, offset_y)
            
            return detection
            
        except Exception as e:
            logger.debug(f"Kalman temporal filtering error: {e}")
            return detection

    def _update_advanced_tracking(self, detection):
        """Mise à jour avancée du tracking"""
        try:
            if detection['detected'] and detection['contour'] is not None:
                # Calcul des propriétés
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
                        'validation_scores': {
                            'color': detection.get('color_validation', 0),
                            'temporal': detection.get('temporal_validation', 0)
                        },
                        'frame_id': self.frame_count
                    }
                    
                    self.tracking_history.append(track_info)
                    
                    # Mise à jour des zones de confiance
                    zone_info = self._determine_confidence_zone(cx, cy)
                    self.confidence_zones[zone_info['level']].append((cx, cy))
                else:
                    self.tracking_history.append(None)
            else:
                self.tracking_history.append(None)
                
        except Exception as e:
            logger.debug(f"Advanced tracking update error: {e}")

    def _determine_confidence_zone(self, x, y):
        """Détermine la zone de confiance pour une position"""
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        if distance_from_center < 100:
            return {'level': 'high', 'distance': distance_from_center}
        elif distance_from_center < 200:
            return {'level': 'medium', 'distance': distance_from_center}
        else:
            return {'level': 'low', 'distance': distance_from_center}

    def _auto_color_calibration(self, frame, detection):
        """Auto-calibration des couleurs basée sur les détections réussites"""
        try:
            if detection['detected'] and detection['contour'] is not None:
                # Échantillonnage des couleurs dans le contour détecté
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [detection['contour']], 255)
                
                # Extraction des pixels HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                roi_pixels = hsv[mask > 0]
                
                if len(roi_pixels) > 10:
                    # Ajout à l'échantillon
                    sample_indices = np.random.choice(len(roi_pixels), 
                                                    min(20, len(roi_pixels)), 
                                                    replace=False)
                    for idx in sample_indices:
                        self.color_samples.append(roi_pixels[idx])
                    
                    # Recalibration périodique
                    if len(self.color_samples) > 50 and self.frame_count % 100 == 0:
                        self._recalibrate_color_ranges()
                        
        except Exception as e:
            logger.debug(f"Auto color calibration error: {e}")

    def _recalibrate_color_ranges(self):
        """Recalibration des plages de couleurs"""
        try:
            if len(self.color_samples) < 30:
                return
            
            samples = np.array(list(self.color_samples))
            
            # Clustering simple pour séparer orange et rouge
            kmeans = SimpleKMeans(n_clusters=2)
            clusters = kmeans.fit(samples)
            
            # Identification des clusters
            centers = clusters.cluster_centers_
            labels = clusters.labels_
            
            # Mise à jour des plages (adaptation conservative)
            for i, center in enumerate(centers):
                cluster_samples = samples[labels == i]
                
                if len(cluster_samples) < 5:
                    continue
                
                # Calcul des plages
                h_mean, s_mean, v_mean = center
                h_std = np.std(cluster_samples[:, 0])
                s_std = np.std(cluster_samples[:, 1])
                v_std = np.std(cluster_samples[:, 2])
                
                # Extension conservative des plages
                margin_h = max(5, h_std * 1.5)
                margin_s = max(20, s_std * 1.5)
                margin_v = max(20, v_std * 1.5)
                
                new_range = (
                    [max(0, h_mean - margin_h), max(0, s_mean - margin_s), max(0, v_mean - margin_v)],
                    [min(180, h_mean + margin_h), min(255, s_mean + margin_s), min(255, v_mean + margin_v)]
                )
                
                # Classification orange vs rouge
                if h_mean < 25:  # Orange/Rouge
                    if h_mean < 15:
                        # Ajout prudent aux plages orange
                        if len(self.color_calibration['orange_ranges']) < 6:  # Limite
                            self.color_calibration['orange_ranges'].append(new_range)
                    else:
                        # Ajout aux plages rouge
                        if len(self.color_calibration['red_ranges']) < 6:  # Limite
                            self.color_calibration['red_ranges'].append(new_range)
            
            logger.info(f"🎨 Recalibration couleurs: {len(self.color_calibration['orange_ranges'])} orange, {len(self.color_calibration['red_ranges'])} rouge")
            
        except Exception as e:
            logger.debug(f"Color recalibration error: {e}")

    def _intelligent_zoom_adaptation(self, detection):
        """Adaptation intelligente du zoom avec prédiction"""
        try:
            if detection['detected'] and detection['area'] > 0:
                area = detection['area']
                
                # Prédiction du zoom optimal
                predicted_zoom = self._predict_optimal_zoom(area)
                
                # Lissage avec l'historique
                self.zoom_prediction.append(predicted_zoom)
                if len(self.zoom_prediction) >= 3:
                    # Moyenne pondérée
                    weights = [0.5, 0.3, 0.2]
                    weighted_zoom = sum(z * w for z, w in zip(list(self.zoom_prediction)[-3:], weights))
                    self.target_zoom = weighted_zoom
                else:
                    self.target_zoom = predicted_zoom
                
                # Stabilité du zoom
                if abs(self.target_zoom - self.zoom_factor) < 0.1:
                    self.zoom_stability_counter += 1
                else:
                    self.zoom_stability_counter = 0
                
                # Application progressive
                if self.zoom_stability_counter > 5:
                    self.zoom_smooth_factor = 0.03  # Plus lent quand stable
                else:
                    self.zoom_smooth_factor = 0.06  # Normal
                    
            else:
                # Zoom out graduel si pas de détection
                if sum(list(self.stable_detections)[-3:]) == 0:
                    self.target_zoom = max(self.zoom_min, self.target_zoom * 0.98)
                    
        except Exception as e:
            logger.debug(f"Intelligent zoom adaptation error: {e}")

    def _predict_optimal_zoom(self, area):
        """Prédiction du zoom optimal basée sur l'aire"""
        try:
            # Fonction de zoom adaptative améliorée
            if area < 500:           # Très petit (très loin)
                return min(self.zoom_max, 4.5)
            elif area < 1000:        # Petit (loin)
                return min(self.zoom_max, 3.5)
            elif area < 2000:        # Petit-moyen (moyennement loin)
                return min(self.zoom_max, 2.8)
            elif area < 4000:        # Moyen (distance normale)
                return 2.0
            elif area < 8000:        # Grand (proche)
                return 1.5
            elif area < 15000:       # Très grand (très proche)
                return 1.2
            else:                    # Énorme (trop proche)
                return 1.0
                
        except Exception as e:
            logger.debug(f"Optimal zoom prediction error: {e}")
            return 1.0

    def _update_confidence_zones(self, detection):
        """Mise à jour des zones de confiance"""
        try:
            if detection['detected'] and detection['contour'] is not None:
                moments = cv2.moments(detection['contour'])
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    zone_info = self._determine_confidence_zone(cx, cy)
                    quality = detection.get('quality_score', 0)
                    
                    # Ajout pondéré par qualité
                    zone_entry = {
                        'position': (cx, cy),
                        'quality': quality,
                        'timestamp': time.time(),
                        'frame_id': self.frame_count
                    }
                    
                    self.confidence_zones[zone_info['level']].append(zone_entry)
                    
        except Exception as e:
            logger.debug(f"Confidence zones update error: {e}")

    def _comprehensive_rendering(self, frame, detection):
        """Rendu complet avec toutes les informations"""
        try:
            h, w = frame.shape[:2]
            result_frame = frame.copy()
            
            # === AFFICHAGE DE LA DÉTECTION ===
            if detection['detected'] and detection['contour'] is not None:
                self._draw_enhanced_detection(result_frame, detection)
            
            # === OVERLAY INFORMATIONS SYSTÈME ===
            self._draw_system_overlay(result_frame, detection)
            
            # === AFFICHAGE ZONES DE CONFIANCE ===
            self._draw_confidence_zones(result_frame)
            
            # === AFFICHAGE TRACKING ET PRÉDICTION ===
            self._draw_tracking_info(result_frame, detection)
            
            # === AFFICHAGE ZOOM ET PERFORMANCE ===
            self._draw_zoom_and_performance(result_frame, detection)
            
            return result_frame
            
        except Exception as e:
            logger.debug(f"Comprehensive rendering error: {e}")
            return frame

    def _draw_enhanced_detection(self, frame, detection):
        """Dessin amélioré de la détection"""
        try:
            contour = detection['contour']
            area = detection['area']
            quality = detection.get('quality_score', 0)
            
            # Couleur selon la qualité et la distance
            if area > 8000:
                color = (0, 255, 0)      # Vert - très proche
                distance_text = "TRÈS PROCHE"
            elif area > 4000:
                color = (0, 255, 255)    # Jaune - proche
                distance_text = "PROCHE"
            elif area > 2000:
                color = (0, 200, 255)    # Orange - moyen
                distance_text = "MOYEN"
            elif area > 1000:
                color = (0, 150, 255)    # Orange foncé - loin
                distance_text = "LOIN"
            else:
                color = (0, 100, 255)    # Rouge - très loin
                distance_text = "TRÈS LOIN"
            
            # Ajustement couleur selon qualité
            quality_factor = max(0.3, quality)
            color = tuple(int(c * quality_factor) for c in color)
            
            # Contour principal avec épaisseur variable
            thickness = max(2, int(3 * quality))
            cv2.drawContours(frame, [contour], -1, color, thickness)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre avec croix de précision
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # Croix centrale
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
                
                # Informations détaillées
                info_y = max(y - 20, 30)
                cv2.putText(frame, f"GANT {distance_text}", (x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                           
                info_y += 25
                cv2.putText(frame, f"Aire: {int(area)} | Q: {quality:.2f}", (x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Informations de validation si disponibles
                if 'color_validation' in detection:
                    info_y += 20
                    cv2.putText(frame, f"Couleur: {detection['color_validation']:.2f} | "
                                     f"Temp: {detection.get('temporal_validation', 0):.2f}", 
                               (x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Indicateur Kalman si filtré
                if detection.get('kalman_filtered', False):
                    cv2.putText(frame, "K", (cx + 15, cy - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
        except Exception as e:
            logger.debug(f"Enhanced detection drawing error: {e}")

    def _draw_system_overlay(self, frame, detection):
        """Overlay système avec statut global"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal
            if detection['detected']:
                if detection.get('quality_score', 0) > 0.7:
                    status = "🎯 GANT DÉTECTÉ (HAUTE QUALITÉ)"
                    color = (0, 255, 0)
                else:
                    status = "🎯 GANT DÉTECTÉ"
                    color = (0, 200, 255)
            else:
                status = "🔍 RECHERCHE EN COURS"
                color = (0, 255, 255)
                
                # Raison de rejet si disponible
                if 'rejection_reason' in detection:
                    status += f" ({detection['rejection_reason']})"
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Mode de détection actif
            mode_text = "Mode: Multi-échelle + Tracking + Zoom adaptatif"
            cv2.putText(frame, mode_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            logger.debug(f"System overlay drawing error: {e}")

    def _draw_confidence_zones(self, frame):
        """Affichage des zones de confiance"""
        try:
            h, w = frame.shape[:2]
            
            # Zone haute confiance (centre)
            center_rect = (w//4, h//4, w//2, h//2)
            cv2.rectangle(frame, (center_rect[0], center_rect[1]), 
                         (center_rect[0] + center_rect[2], center_rect[1] + center_rect[3]), 
                         (0, 255, 0), 1)
            cv2.putText(frame, "HIGH", (center_rect[0] + 5, center_rect[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Zone moyenne confiance
            medium_rect = (w//8, h//8, 6*w//8, 6*h//8)
            cv2.rectangle(frame, (medium_rect[0], medium_rect[1]), 
                         (medium_rect[0] + medium_rect[2], medium_rect[1] + medium_rect[3]), 
                         (0, 255, 255), 1)
            cv2.putText(frame, "MED", (medium_rect[0] + 5, medium_rect[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Statistiques des zones
            if len(self.confidence_zones['high']) > 0:
                cv2.putText(frame, f"H:{len(self.confidence_zones['high'])}", 
                           (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(self.confidence_zones['medium']) > 0:
                cv2.putText(frame, f"M:{len(self.confidence_zones['medium'])}", 
                           (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if len(self.confidence_zones['low']) > 0:
                cv2.putText(frame, f"L:{len(self.confidence_zones['low'])}", 
                           (w - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
            
        except Exception as e:
            logger.debug(f"Confidence zones drawing error: {e}")

    def _draw_tracking_info(self, frame, detection):
        """Affichage des informations de tracking"""
        try:
            h, w = frame.shape[:2]
            
            # Trajectoire récente
            recent_tracks = [t for t in list(self.tracking_history)[-10:] if t is not None]
            if len(recent_tracks) > 1:
                points = [t['center'] for t in recent_tracks]
                
                # Dessin de la trajectoire
                for i in range(1, len(points)):
                    alpha = i / len(points)  # Transparence progressive
                    color = (int(255 * alpha), int(100 * alpha), int(100 * alpha))
                    cv2.line(frame, points[i-1], points[i], color, 2)
                
                # Point de prédiction si disponible
                if hasattr(self, 'prediction_zone') and self.prediction_zone:
                    pred_center = self.prediction_zone['center']
                    pred_radius = self.prediction_zone['radius']
                    cv2.circle(frame, pred_center, pred_radius, (255, 0, 255), 2)
                    cv2.putText(frame, "PRED", (pred_center[0] + 10, pred_center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Historique de détection visuel
            history_display = "".join(["●" if x else "○" for x in list(self.detection_history)[-15:]])
            cv2.putText(frame, f"Hist: {history_display}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        except Exception as e:
            logger.debug(f"Tracking info drawing error: {e}")

    def _draw_zoom_and_performance(self, frame, detection):
        """Affichage zoom et performance"""
        try:
            h, w = frame.shape[:2]
            
            # === BARRE DE ZOOM AVANCÉE ===
            zoom_bar_width = 250
            zoom_bar_height = 20
            zoom_x, zoom_y = 10, 100
            
            # Barre de fond avec graduations
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_bar_width, zoom_y + zoom_bar_height), 
                         (50, 50, 50), -1)
            
            # Graduations
            for i in range(1, int(self.zoom_max) + 1):
                grad_x = zoom_x + int(zoom_bar_width * (i - 1) / (self.zoom_max - 1))
                cv2.line(frame, (grad_x, zoom_y), (grad_x, zoom_y + zoom_bar_height), 
                        (100, 100, 100), 1)
                cv2.putText(frame, str(i), (grad_x - 5, zoom_y + zoom_bar_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            # Barre de zoom actuel
            zoom_width = int(zoom_bar_width * (self.zoom_factor - 1.0) / (self.zoom_max - 1.0))
            zoom_color = (0, 255, 255) if self.zoom_factor > 2.0 else (100, 255, 100)
            cv2.rectangle(frame, (zoom_x, zoom_y), 
                         (zoom_x + zoom_width, zoom_y + zoom_bar_height), 
                         zoom_color, -1)
            
            # Indicateur zoom cible
            target_width = int(zoom_bar_width * (self.target_zoom - 1.0) / (self.zoom_max - 1.0))
            cv2.line(frame, (zoom_x + target_width, zoom_y - 5), 
                    (zoom_x + target_width, zoom_y + zoom_bar_height + 5), 
                    (255, 255, 255), 2)
            
            # Textes zoom
            cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x", 
                       (zoom_x + zoom_bar_width + 10, zoom_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {self.target_zoom:.1f}x", 
                       (zoom_x + zoom_bar_width + 10, zoom_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # === STATISTIQUES DE PERFORMANCE ===
            
            # FPS et performance
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                self.current_fps = 30 / elapsed if elapsed > 0 else 0
                self.fps_start_time = now
            
            # Temps de traitement moyen
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            cv2.putText(frame, f"Proc: {avg_processing_time*1000:.1f}ms", (w - 150, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Statistiques de détection
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            quality_avg = np.mean(self.quality_scores) if self.quality_scores else 0
            
            stats_y = h - 120
            cv2.putText(frame, f"Détections: {self.detection_count}/{self.frame_count} ({detection_rate:.1f}%)", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            stats_y += 25
            cv2.putText(frame, f"Qualité moy: {quality_avg:.2f} | Rejets FP: {self.false_positive_rejection}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            stats_y += 20
            cv2.putText(frame, f"Ajust. zoom: {getattr(self, 'zoom_adjustments', 0)} | Stabilité: {self.zoom_stability_counter}", 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Indicateurs d'état système
            indicators_x = w - 200
            if detection['detected']:
                cv2.circle(frame, (indicators_x, 100), 8, (0, 255, 0), -1)
                cv2.putText(frame, "DET", (indicators_x + 15, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            if self.zoom_factor > 1.5:
                cv2.circle(frame, (indicators_x, 120), 8, (0, 255, 255), -1)
                cv2.putText(frame, "ZOOM", (indicators_x + 15, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            if len(self.tracking_history) > 5:
                cv2.circle(frame, (indicators_x, 140), 8, (255, 0, 255), -1)
                cv2.putText(frame, "TRACK", (indicators_x + 15, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
        except Exception as e:
            logger.debug(f"Zoom and performance drawing error: {e}")

    def _error_recovery(self, frame):
        """Récupération d'erreur avec fallback"""
        try:
            # Mode de récupération simplifié
            if self.last_successful_detection:
                logger.info(f"🔄 Mode récupération activé (erreur #{self.error_recovery_counter})")
                
                # Retour à un état stable
                self.target_zoom = max(1.0, self.target_zoom * 0.9)
                
                # Simplification temporaire des paramètres
                if self.error_recovery_counter > 10:
                    self.zoom_factor = 1.0
                    self.target_zoom = 1.0
                    self.tracking_history.clear()
                    self.error_recovery_counter = 0
                    logger.info("🔄 Reset complet du détecteur")
                
                return frame, False, {'detected': False, 'error_recovery': True}
            
            return frame, False, {'detected': False}
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            return frame, False, {'detected': False}

    def reset_detector(self):
        """Reset complet du détecteur"""
        logger.info("🔄 Reset complet du détecteur Enhanced")
        self.__init__()

    def get_performance_stats(self):
        """Récupération des statistiques de performance"""
        return {
            'detection_rate': (self.detection_count / max(self.frame_count, 1)) * 100,
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0,
            'false_positive_rejections': self.false_positive_rejection,
            'zoom_adjustments': getattr(self, 'zoom_adjustments', 0),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'tracking_history_length': len([t for t in self.tracking_history if t is not None]),
            'current_zoom': self.zoom_factor,
            'target_zoom': self.target_zoom,
            'color_samples_collected': len(self.color_samples)
        }

# === CONTRÔLE DRONE AMÉLIORÉ ===
def enhanced_drone_control(bebop):
    """Contrôle drone avec fonctionnalités avancées"""
    logger.info("🎮 Contrôle drone amélioré démarré")
    print("\n[Commandes drone avancées]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  f/b/g/d = mouvements | h/m = haut/bas | a/c = rotations\n"
          "  1/2/3 = vitesses (lent/moyen/rapide)\n"
          "  p = position hover | x = arrêt d'urgence\n")
    
    speed_settings = {'1': 15, '2': 25, '3': 40}
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
            print("🚨 Arrêt d'urgence")
        elif key == 'p':
            print("📍 Position hover")
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
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=current_speed, duration=0.3)
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-current_speed, duration=0.3)
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-35, vertical_movement=0, duration=0.3)
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=35, vertical_movement=0, duration=0.3)

# === FONCTION PRINCIPALE COMPLÈTE ===
def main():
    """Fonction principale avec détection complète améliorée"""
    logger.info("=" * 80)
    logger.info("🚀 BEBOP 2 - DÉTECTION GANT COMPLÈTE AMÉLIORÉE")
    logger.info("🎯 Multi-échelle + Tracking + Zoom adaptatif + Validation cascade")
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
        time.sleep(3)  # Plus de temps pour stabilisation
        
        # === CONTRÔLE DRONE ===
        ctrl_thread = threading.Thread(target=enhanced_drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG OPTIMISÉ ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # FFmpeg avec optimisations avancées
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer+fastseek',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '1000000',   # Plus d'analyse pour qualité
            '-probesize', '1000000',
            '-max_delay', '0',
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-threads', '0',  # Auto threads
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg optimisé: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, 
                                  bufsize=1024*1024*2)  # Buffer plus grand
            logger.info("✅ Pipeline optimisé initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR COMPLET AMÉLIORÉ ===
        detector = CompleteEnhancedGloveDetector()
        
        # === INTERFACE AVANCÉE ===
        window_name = "Bebop 2 - Détection Complète Améliorée"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 80)
        logger.info("🎮 COMMANDES AVANCÉES:")
        logger.info("  'q'/'ESC' = Quitter | 's' = Screenshot | 'r' = Reset complet")
        logger.info("  'z' = Reset zoom | '+'/'-' = Zoom manuel | 'c' = Calibration couleurs")
        logger.info("  'd' = Debug détaillé | 'p' = Stats performance | 'h' = Aide")
        logger.info("=" * 80)
        logger.info("🔍 FONCTIONNALITÉS:")
        logger.info("  ✓ Stabilisation d'image avec flux optique")
        logger.info("  ✓ Adaptation automatique de la lumière")
        logger.info("  ✓ Détection multi-échelle (4 niveaux)")
        logger.info("  ✓ Tracking prédictif avec Kalman")
        logger.info("  ✓ Validation en cascade")
        logger.info("  ✓ Auto-calibration des couleurs")
        logger.info("  ✓ Zones de confiance adaptatives")
        logger.info("  ✓ Zoom intelligent jusqu'à 6x")
        logger.info("=" * 80)
        
        # === BOUCLE PRINCIPALE AMÉLIORÉE ===
        logger.info("🎬 Démarrage détection complète améliorée...")
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        skip_counter = 0
        performance_log_interval = 100  # Log performance tous les 100 frames
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Skip frames adaptatif pour performance
                skip_counter += 1
                if skip_counter % 2 != 0 and detector.current_fps > 20:
                    continue
                
                # === DÉTECTION COMPLÈTE AMÉLIORÉE ===
                processed_frame, detected, detection_info = detector.enhanced_detect_glove(frame)
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # === LOGGING PERFORMANCE ===
                fps_counter += 1
                if fps_counter % 60 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 60 / elapsed if elapsed > 0 else 0
                    
                    # Log détaillé avec nouvelles métriques
                    stats = detector.get_performance_stats()
                    
                    logger.info(f"📊 FPS: {display_fps:.1f} | "
                               f"Détections: {detector.detection_count}/{detector.frame_count} "
                               f"({stats['detection_rate']:.1f}%) | "
                               f"Qualité: {stats['avg_quality']:.2f} | "
                               f"Zoom: {detector.zoom_factor:.1f}x→{detector.target_zoom:.1f}x | "
                               f"Rejets FP: {stats['false_positive_rejections']}")
                    
                    last_fps_log = current_time
                
                # === LOGGING PERFORMANCE DÉTAILLÉ ===
                if fps_counter % performance_log_interval == 0:
                    stats = detector.get_performance_stats()
                    logger.info("🔍 STATS DÉTAILLÉES:")
                    logger.info(f"   Temps traitement moy: {stats['avg_processing_time']*1000:.1f}ms")
                    logger.info(f"   Historique tracking: {stats['tracking_history_length']} points")
                    logger.info(f"   Échantillons couleur: {stats['color_samples_collected']}")
                    logger.info(f"   Stabilité zoom: {detector.zoom_stability_counter}")
                
                # === GESTION TOUCHES AVANCÉE ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q ou ESC
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    # Screenshot avec métadonnées
                    timestamp = int(time.time())
                    screenshot_name = f"enhanced_capture_{timestamp}_{screenshot_count:03d}.png"
                    
                    # Ajout informations dans le screenshot
                    info_frame = processed_frame.copy()
                    info_text = (f"Frame:{detector.frame_count} | Zoom:{detector.zoom_factor:.1f}x | "
                               f"Det:{detected} | Q:{detection_info.get('quality_score', 0):.2f}")
                    cv2.putText(info_frame, info_text, (10, HEIGHT - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imwrite(screenshot_name, info_frame)
                    logger.info(f"📸 Screenshot amélioré: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    # Reset complet
                    old_stats = detector.get_performance_stats()
                    detector.reset_detector()
                    logger.info(f"🔄 Reset complet (détections: {old_stats['detection_rate']:.1f}%)")
                    
                elif key == ord('z'):
                    # Reset zoom seulement
                    detector.zoom_factor = 1.0
                    detector.target_zoom = 1.0
                    detector.zoom_prediction.clear()
                    detector.zoom_stability_counter = 0
                    logger.info("🔍 Zoom reset à 1.0x")
                    
                elif key == ord('+') or key == ord('='):
                    # Zoom manuel +
                    detector.target_zoom = min(detector.zoom_max, detector.target_zoom + 0.5)
                    detector.zoom_stability_counter = 0
                    logger.info(f"🔍 Zoom manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('-'):
                    # Zoom manuel -
                    detector.target_zoom = max(detector.zoom_min, detector.target_zoom - 0.5)
                    detector.zoom_stability_counter = 0
                    logger.info(f"🔍 Zoom manuel: {detector.target_zoom:.1f}x")
                    
                elif key == ord('c'):
                    # Force calibration couleurs
                    if len(detector.color_samples) > 10:
                        detector._recalibrate_color_ranges()
                        logger.info(f"🎨 Calibration couleurs forcée ({len(detector.color_samples)} échantillons)")
                    else:
                        logger.info("🎨 Pas assez d'échantillons pour calibration")
                    
                elif key == ord('p'):
                    # Affichage stats performance
                    stats = detector.get_performance_stats()
                    logger.info("📈 PERFORMANCE ACTUELLE:")
                    for key_stat, value in stats.items():
                        logger.info(f"   {key_stat}: {value}")
                    
                elif key == ord('d'):
                    # Debug informations détaillées
                    logger.info("🔍 INFOS DEBUG COMPLÈTES:")
                    logger.info(f"   Frame actuelle: {detector.frame_count}")
                    logger.info(f"   Zoom: {detector.zoom_factor:.2f}x → {detector.target_zoom:.2f}x")
                    logger.info(f"   Détection actuelle: {detected}")
                    if detection_info.get('detected', False):
                        logger.info(f"   Aire: {detection_info.get('area', 0)}")
                        logger.info(f"   Qualité: {detection_info.get('quality_score', 0):.3f}")
                        logger.info(f"   Validation couleur: {detection_info.get('color_validation', 0):.3f}")
                        logger.info(f"   Validation temporelle: {detection_info.get('temporal_validation', 0):.3f}")
                    logger.info(f"   Historique tracking: {len([t for t in detector.tracking_history if t])}")
                    logger.info(f"   Échantillons couleur: {len(detector.color_samples)}")
                    logger.info(f"   Erreurs récupération: {detector.error_recovery_counter}")
                    
                elif key == ord('h'):
                    # Aide
                    print("\n" + "=" * 60)
                    print("🎮 AIDE - COMMANDES DÉTECTION AMÉLIORÉE")
                    print("=" * 60)
                    print("CONTRÔLE:")
                    print("  q/ESC  = Quitter")
                    print("  s      = Screenshot avec métadonnées")
                    print("  r      = Reset complet du détecteur")
                    print("  z      = Reset zoom uniquement")
                    print("  +/-    = Zoom manuel")
                    print("ANALYSE:")
                    print("  c      = Force calibration couleurs")
                    print("  p      = Afficher stats performance")
                    print("  d      = Debug informations complètes")
                    print("  h      = Cette aide")
                    print("FONCTIONNALITÉS ACTIVES:")
                    print("  ✓ Stabilisation optique")
                    print("  ✓ Adaptation lumière auto")
                    print("  ✓ Multi-échelle (4 niveaux)")
                    print("  ✓ Tracking Kalman")
                    print("  ✓ Validation cascade")
                    print("  ✓ Auto-calibration couleurs")
                    print("  ✓ Zones de confiance")
                    print("=" * 60 + "\n")

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier")
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle principale: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
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
            logger.info("📊 STATISTIQUES FINALES - DÉTECTION COMPLÈTE AMÉLIORÉE")
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
            logger.info("=" * 80)
            
            # Sauvegarde des statistiques finales
            try:
                stats_filename = f"detection_stats_{int(time.time())}.json"
                with open(stats_filename, 'w') as f:
                    json.dump({
                        'runtime_seconds': total_runtime,
                        'total_frames': detector.frame_count,
                        'total_detections': detector.detection_count,
                        'performance_stats': final_stats,
                        'screenshots_taken': screenshot_count,
                        'system_info': get_system_info()
                    }, f, indent=2)
                logger.info(f"📁 Statistiques sauvées: {stats_filename}")
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
        
        logger.info("🎉 Session détection complète améliorée terminée!")
    
    return True

# === FONCTIONS UTILITAIRES ===

def check_dependencies():
    """Vérification des dépendances requises"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'pyparrot': 'pyparrot'
    }
    
    missing_required = []
    
    # Vérification des packages requis
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(package)
    
    if missing_required:
        logger.error(f"❌ Packages requis manquants: {missing_required}")
        logger.info("📦 Installation requise:")
        logger.info(f"   pip install {' '.join(missing_required)}")
        return False
    
    logger.info("✅ Toutes les dépendances requises sont disponibles")
    return True

def test_camera_connection():
    """Test de connexion caméra pour debug"""
    logger.info("🧪 Test connexion caméra...")
    
    try:
        # Test avec une caméra locale pour debug
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.info("✅ Caméra locale détectée (pour tests)")
            cap.release()
        else:
            logger.info("ℹ️ Pas de caméra locale (normal pour Bebop)")
    except Exception as e:
        logger.debug(f"Camera test error: {e}")

def print_system_info():
    """Affichage des informations système"""
    system_info = get_system_info()
    
    logger.info("💻 INFORMATIONS SYSTÈME:")
    logger.info(f"   OS: {system_info['os']}")
    logger.info(f"   Python: {system_info['python']}")
    logger.info(f"   RAM: {system_info['ram_gb']} GB")
    logger.info(f"   CPU: {system_info['cpu_cores']} cores")
    logger.info(f"   Architecture: {system_info['architecture']}")
    
    try:
        import cv2
        logger.info(f"   OpenCV: {cv2.__version__}")
    except:
        logger.warning("   OpenCV: Non détecté")
    
    logger.info("   Fonctionnalités: Clustering intégré + Toutes optimisations")

if __name__ == "__main__":
    try:
        # === INITIALISATION ===
        print("\n" + "=" * 80)
        print("🚀 BEBOP 2 - SYSTÈME DE DÉTECTION GANT COMPLET AMÉLIORÉ")
        print("=" * 80)
        
        # Vérifications préalables
        print_system_info()
        
        if not check_dependencies():
            print("❌ Dépendances manquantes!")
            sys.exit(1)
        
        test_camera_connection()
        
        # Lancement principal
        print("\n🎬 Lancement du système de détection...")
        success = main()
        
        # Code de sortie
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        
        if success:
            print("✅ Session terminée avec succès!")
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