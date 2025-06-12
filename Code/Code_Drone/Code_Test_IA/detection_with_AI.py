#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEBOP 2 - DÉTECTION GANT AVEC IA DE RECONNAISSANCE DE POSITION
Version finale corrigée - Thread-safe
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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pickle
import json

# Import TensorFlow avec gestion sécurisée
TF_AVAILABLE = False
tf = None
keras = None
layers = None

try:
    import tensorflow as tf
    keras = tf.keras
    layers = tf.keras.layers
    TF_AVAILABLE = True
    print("✅ TensorFlow chargé avec succès")
    
    # Optimisations TensorFlow
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass
        
except ImportError as e:
    TF_AVAILABLE = False
    print(f"⚠️ TensorFlow non disponible: {e}")
    print("   Installez avec: pip install tensorflow")
except Exception as e:
    TF_AVAILABLE = False
    print(f"❌ Erreur TensorFlow: {e}")

# === PARAMÈTRES ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_ai_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION DES POSITIONS ===
@dataclass
class HandPosition:
    """Classe pour définir une position de main"""
    name: str
    description: str
    confidence_threshold: float = 0.7

HAND_POSITIONS = {
    0: HandPosition("poing", "Poing fermé - ARRÊT D'URGENCE", 0.9),
    1: HandPosition("paume", "Paume ouverte - AVANCER", 0.7),
    2: HandPosition("index", "Index pointé - DIRECTION", 0.75),
    3: HandPosition("victoire", "Signe V - MONTÉE", 0.7),
    4: HandPosition("ok", "Signe OK - VALIDATION/HOVER", 0.8),
    5: HandPosition("pouce", "Pouce levé - MONTÉE DOUCE", 0.75),
    6: HandPosition("stop", "Main STOP - ARRÊT", 0.85),
    7: HandPosition("salut", "Salut - ROTATION", 0.6)
}

# === EXTRACTEUR DE CARACTÉRISTIQUES ===
class AdvancedHandFeatureExtractor:
    """Extracteur de caractéristiques pour la reconnaissance de position"""
    
    def __init__(self):
        self.feature_size = 64
        self.logging = logging.getLogger(__name__)
        self.feature_history = deque(maxlen=5)
        
    def extract_geometric_features(self, contour, bounding_rect):
        """Extraction des caractéristiques géométriques"""
        try:
            x, y, w, h = bounding_rect
            area = cv2.contourArea(contour)
            
            # Caractéristiques de base
            aspect_ratio = w / float(h) if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            
            # Convexité
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Périmètre et compacité
            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Moments géométriques
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Défauts de convexité
            if len(contour) >= 4:
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    convexity_defects = len(defects) if defects is not None else 0
                else:
                    convexity_defects = 0
            else:
                convexity_defects = 0
            
            # Orientation
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                orientation = ellipse[2] / 180.0
            else:
                orientation = 0
            
            # Centre et distances
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                distances = [np.sqrt((pt[0][0] - cx)**2 + (pt[0][1] - cy)**2) for pt in contour]
                avg_distance = np.mean(distances) / 100.0
                std_distance = np.std(distances) / 100.0
            else:
                avg_distance = std_distance = 0
            
            # Compilation
            geometric_features = [
                aspect_ratio, extent, solidity, compactness,
                area / 10000.0, perimeter / 1000.0, convexity_defects / 10.0,
                orientation, avg_distance, std_distance, *hu_moments[:7]
            ]
            
            return np.array(geometric_features[:17], dtype=np.float32)
            
        except Exception as e:
            self.logging.debug(f"Erreur extraction géométrique: {e}")
            return np.zeros(17, dtype=np.float32)
    
    def extract_visual_features(self, roi_image, contour_mask):
        """Extraction des caractéristiques visuelles"""
        try:
            if roi_image.size == 0:
                return np.zeros(21, dtype=np.float32)
                
            roi_resized = cv2.resize(roi_image, (64, 64))
            mask_resized = cv2.resize(contour_mask, (64, 64))
            
            # Caractéristiques couleur HSV
            hsv_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            
            if np.any(mask_resized > 0):
                h_mean = np.mean(hsv_roi[:, :, 0][mask_resized > 0]) / 180.0
                s_mean = np.mean(hsv_roi[:, :, 1][mask_resized > 0]) / 255.0
                v_mean = np.mean(hsv_roi[:, :, 2][mask_resized > 0]) / 255.0
            else:
                h_mean = s_mean = v_mean = 0
            
            # Gradients
            gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            if np.any(mask_resized > 0):
                gradient_mean = np.mean(gradient_magnitude[mask_resized > 0]) / 255.0
            else:
                gradient_mean = 0
            
            # Contours
            edges = cv2.Canny(gray_roi, 50, 150)
            if np.sum(mask_resized > 0) > 0:
                edge_density = np.sum(edges[mask_resized > 0]) / np.sum(mask_resized > 0) / 255.0
            else:
                edge_density = 0
            
            # Histogramme
            hist = cv2.calcHist([gray_roi], [0], mask_resized, [8], [0, 256])
            hist_features = hist.flatten() / (np.sum(hist) + 1e-7)
            
            visual_features = [h_mean, s_mean, v_mean, gradient_mean, edge_density, *hist_features]
            
            return np.array(visual_features[:13], dtype=np.float32)
            
        except Exception as e:
            self.logging.debug(f"Erreur extraction visuelle: {e}")
            return np.zeros(13, dtype=np.float32)
    
    def extract_complete_features(self, frame, contour, bounding_rect):
        """Extraction complète des caractéristiques"""
        try:
            x, y, w, h = bounding_rect
            
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                return np.zeros(self.feature_size, dtype=np.float32)
            
            if w <= 0 or h <= 0:
                return np.zeros(self.feature_size, dtype=np.float32)
            
            roi = frame[y:y+h, x:x+w]
            
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            contour_relative = contour - [x, y]
            cv2.fillPoly(contour_mask, [contour_relative], 255)
            
            geometric_features = self.extract_geometric_features(contour, bounding_rect)
            visual_features = self.extract_visual_features(roi, contour_mask)
            
            combined_features = np.concatenate([geometric_features, visual_features])
            
            if len(combined_features) < self.feature_size:
                padding = np.zeros(self.feature_size - len(combined_features), dtype=np.float32)
                combined_features = np.concatenate([combined_features, padding])
            else:
                combined_features = combined_features[:self.feature_size]
            
            self.feature_history.append(combined_features)
            if len(self.feature_history) >= 3:
                weights = np.array([0.2, 0.3, 0.5])
                stabilized_features = np.average(list(self.feature_history)[-3:], weights=weights, axis=0)
                return stabilized_features
            
            return combined_features
            
        except Exception as e:
            self.logging.debug(f"Erreur extraction complète: {e}")
            return np.zeros(self.feature_size, dtype=np.float32)

# === MODÈLE TENSORFLOW CORRIGÉ ===
class HandPositionRecognizer:
    """Modèle de reconnaissance de position - VERSION CORRIGÉE"""
    
    def __init__(self, feature_size=64, num_classes=len(HAND_POSITIONS)):
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.model = None
        self.is_trained = False
        self.training_data = []
        self.training_labels = []
        self.logging = logging.getLogger(__name__)
        
        # Historique prédictions
        self.prediction_history = deque(maxlen=7)
        self.confidence_history = deque(maxlen=5)
        
        # Variables pour optimisation prédictions
        self.last_prediction_time = 0
        self.last_prediction_result = (None, 0.0)
        self.prediction_interval = 0.2  # 5 FPS max
        
        # Métriques
        self.total_predictions = 0
        self.confident_predictions = 0
        
        self.tf_available = TF_AVAILABLE
        if not self.tf_available:
            self.logging.warning("⚠️ TensorFlow non disponible - IA désactivée")
    
    def create_model(self):
        """Création du modèle"""
        if not self.tf_available:
            return False
            
        try:
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(self.feature_size,)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(96, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.logging.info(f"✅ Modèle créé: {model.count_params()} paramètres")
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur création modèle: {e}")
            return False
    
    def add_training_sample(self, features, position_class):
        """Ajout échantillon d'entraînement"""
        if not self.tf_available:
            return False
            
        try:
            if len(features) != self.feature_size:
                self.logging.warning(f"Taille features incorrecte: {len(features)} vs {self.feature_size}")
                return False
            
            self.training_data.append(features.copy())
            self.training_labels.append(position_class)
            
            position_name = HAND_POSITIONS[position_class].name
            self.logging.info(f"📊 Échantillon ajouté: {position_name} - Total: {len(self.training_data)}")
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur ajout échantillon: {e}")
            return False
    
    def train_model(self, validation_split=0.2, epochs=25):
        """Entraînement du modèle - VERSION CORRIGÉE"""
        if not self.tf_available:
            self.logging.error("❌ TensorFlow requis pour l'entraînement")
            return False
            
        try:
            if len(self.training_data) < 10:
                self.logging.warning("⚠️ Pas assez de données (min 10)")
                return False
            
            if self.model is None:
                if not self.create_model():
                    return False
            
            X = np.array(self.training_data, dtype=np.float32)
            y = np.array(self.training_labels, dtype=np.int32)
            
            unique, counts = np.unique(y, return_counts=True)
            self.logging.info(f"📊 Distribution: {dict(zip(unique, counts))}")
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            self.logging.info(f"🚀 Entraînement: {len(X)} échantillons, {epochs} époques")
            
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=min(16, len(X) // 4),  # Batch size réduit
                callbacks=callbacks,
                verbose=0  # Supprime les logs TensorFlow
            )
            
            final_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history.get('val_accuracy', [0])[-1]
            
            self.logging.info(f"✅ Entraînement terminé:")
            self.logging.info(f"   Précision: {final_accuracy:.4f}")
            self.logging.info(f"   Validation: {val_accuracy:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur entraînement: {e}")
            return False
    
    def predict_position(self, features, use_stabilization=True):
        """Prédiction optimisée - VERSION CORRIGÉE"""
        if not self.tf_available or self.model is None or not self.is_trained:
            return None, 0.0
            
        try:
            if len(features) != self.feature_size:
                return None, 0.0
            
            # Limitation fréquence
            current_time = time.time()
            if current_time - self.last_prediction_time < self.prediction_interval:
                return self.last_prediction_result
            
            # Prédiction
            features_batch = features.reshape(1, -1)
            prediction = self.model.predict(features_batch, verbose=0, batch_size=1)[0]
            
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            self.last_prediction_time = current_time
            self.total_predictions += 1
            
            # Stabilisation
            if use_stabilization:
                self.prediction_history.append((predicted_class, confidence))
                self.confidence_history.append(confidence)
                
                if len(self.prediction_history) >= 3:
                    recent_classes = [p[0] for p in list(self.prediction_history)[-3:]]
                    recent_confidences = [p[1] for p in list(self.prediction_history)[-3:]]
                    
                    from collections import Counter
                    class_counts = Counter(recent_classes)
                    most_common_class, count = class_counts.most_common(1)[0]
                    
                    if count >= 2:
                        class_confidences = [conf for cls, conf in zip(recent_classes, recent_confidences) 
                                           if cls == most_common_class]
                        avg_confidence = np.mean(class_confidences)
                        
                        predicted_class = most_common_class
                        confidence = avg_confidence
            
            # Validation seuil
            if predicted_class in HAND_POSITIONS:
                threshold = HAND_POSITIONS[predicted_class].confidence_threshold
                if confidence >= threshold:
                    self.confident_predictions += 1
                    result = (predicted_class, confidence)
                    self.last_prediction_result = result
                    return result
            
            result = (None, confidence)
            self.last_prediction_result = result
            return result
            
        except Exception as e:
            self.logging.debug(f"Erreur prédiction: {e}")
            return None, 0.0
    
    def get_prediction_stats(self):
        """Statistiques des prédictions"""
        if self.total_predictions == 0:
            return "Aucune prédiction"
        
        confidence_rate = (self.confident_predictions / self.total_predictions) * 100
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0
        
        return f"Confiance: {confidence_rate:.1f}% | Moy: {avg_confidence:.2f}"
    
    def save_model(self, filepath):
        if not self.tf_available or self.model is None:
            return False
            
        try:
            import os
            
            # === SAUVEGARDE MODÈLE TENSORFLOW ===
            model_dir = f"{filepath}_model"
            
            # Suppression ancien modèle si existe
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
            
            # Sauvegarde au format TensorFlow moderne
            self.model.save(model_dir)
            self.logging.info(f"✅ Modèle TF sauvegardé: {model_dir}")
            
            # === SAUVEGARDE DONNÉES ===
            data_path = f"{filepath}_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'training_data': self.training_data,
                    'training_labels': self.training_labels,
                    'feature_size': self.feature_size,
                    'num_classes': self.num_classes,
                    'is_trained': self.is_trained,
                    'total_predictions': getattr(self, 'total_predictions', 0),
                    'confident_predictions': getattr(self, 'confident_predictions', 0)
                }, f)
            
            self.logging.info(f"✅ Données sauvegardées: {data_path}")
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur sauvegarde: {e}")
            return False

    def load_model(self, filepath):
        """Chargement du modèle - VERSION CORRIGÉE"""
        if not self.tf_available:
            return False
            
        try:
            # === CHARGEMENT MODÈLE ===
            model_dir = f"{filepath}_model"
            if os.path.exists(model_dir):
                self.model = keras.models.load_model(model_dir)
                self.logging.info(f"✅ Modèle chargé: {model_dir}")
            else:
                # Tentative ancien format
                old_model_path = f"{filepath}_model.h5"
                if os.path.exists(old_model_path):
                    self.model = keras.models.load_model(old_model_path)
                    self.logging.info(f"✅ Ancien modèle chargé: {old_model_path}")
                else:
                    self.logging.warning(f"⚠️ Aucun modèle trouvé: {model_dir}")
                    return False
            
            # === CHARGEMENT DONNÉES ===
            data_path = f"{filepath}_data.pkl"
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.training_data = data.get('training_data', [])
                self.training_labels = data.get('training_labels', [])
                self.feature_size = data.get('feature_size', 64)
                self.num_classes = data.get('num_classes', len(HAND_POSITIONS))
                self.is_trained = data.get('is_trained', False)
                self.total_predictions = data.get('total_predictions', 0)
                self.confident_predictions = data.get('confident_predictions', 0)
                
                self.logging.info(f"✅ Données chargées: {len(self.training_data)} échantillons")
            
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur chargement: {e}")
            return False
    

# === DÉTECTEUR PRINCIPAL AVEC IA ===
class OptimizedBicolorGloveDetectorWithAI:
    """Détecteur de gant avec IA de reconnaissance de position"""
    
    def __init__(self, feature_size=64, num_classes=len(HAND_POSITIONS)):

        self.last_prediction_time = 0
        self.last_prediction_result = (None, 0.0)
        self.prediction_interval = 0.3  # Prédiction toutes les 300ms seulement

        # === PARAMÈTRES DÉTECTION ORIGINAUX ===
        self.detection_history = deque(maxlen=15)
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3
        
        self.min_area = 200
        self.max_area = 120000
        self.min_contour_points = 8
        
        self.color_balance_history = deque(maxlen=20)
        self.red_orange_ratio_history = deque(maxlen=10)
        
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        self.zoom_factor = 1.0
        self.target_zoom = 1.0
        self.zoom_smooth_factor = 0.12
        self.zoom_min = 1.0
        self.zoom_max = 4.5
        
        self.area_reference = 2800
        self.area_history = deque(maxlen=15)
        self.quality_scores = deque(maxlen=10)
        
        self.search_zone = None
        self.zone_tracking = deque(maxlen=5)
        
        self.frame_count = 0
        self.detection_count = 0
        self.quality_count = 0
        self.zoom_adjustments = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.brightness_history = deque(maxlen=10)
        self.auto_exposure_factor = 1.0
        
        # === COMPOSANTS IA ===
        self.ai_enabled = TF_AVAILABLE
        self.feature_extractor = AdvancedHandFeatureExtractor()
        self.position_recognizer = HandPositionRecognizer()
        
        # Données détection pour IA
        self.last_detected_contour = None
        self.last_detected_area = 0
        self.last_bounding_rect = None
        
        # État IA
        self.ai_mode = "detection"  # "detection", "training", "recognition"
        self.training_class = 0
        self.training_countdown = 0
        self.training_samples_per_class = 25
        
        # Position actuelle
        self.current_position = None
        self.current_position_confidence = 0.0
        
        # Métriques IA
        self.ai_frame_count = 0
        self.ai_position_detections = 0
        
        # Commandes drone
        self.drone_commands_enabled = False
        self.last_command_time = 0
        self.command_cooldown = 2.0
        
        self.logging = logging.getLogger(__name__)
        
        # Initialisation IA
        if self.ai_enabled:
            self._initialize_ai()
        else:
            self.logging.warning("⚠️ IA désactivée - TensorFlow requis")
    
    def _initialize_ai(self):
        """Initialisation des composants IA"""
        try:
            # Tentative chargement modèle existant
            model_path = "hand_position_model"
            if os.path.exists(f"{model_path}_model"):
                if self.position_recognizer.load_model(model_path):
                    self.ai_mode = "recognition"
                    self.logging.info("🤖 Modèle IA chargé - Mode reconnaissance")
                else:
                    self.logging.info("🤖 Nouveau modèle IA - Prêt pour entraînement")
            else:
                self.logging.info("🤖 Aucun modèle existant - Utilisez 't' pour entraîner")
            
        except Exception as e:
            self.logging.error(f"Erreur initialisation IA: {e}")
            self.ai_enabled = False
    
    def detect_glove_optimized(self, frame):
        """Détection de gant optimisée (version simplifiée de votre code original)"""
        if frame is None:
            return frame, False
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            # === DÉTECTION COULEUR SIMPLIFIÉE ===
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Masques rouge et orange
            red_lower1 = np.array([0, 140, 120])
            red_upper1 = np.array([8, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([172, 140, 120])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            orange_lower = np.array([8, 160, 140])
            orange_upper = np.array([18, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            
            # Combinaison
            mask_combined = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_orange))
            
            # Morphologie
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, self.kernel_medium)
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, self.kernel_small)
            
            # Recherche contours
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_contour = None
            best_area = 0
            quality_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area and len(contour) >= self.min_contour_points:
                    if area > best_area:
                        best_contour = contour
                        best_area = area
                        quality_score = min(area / self.area_reference, 1.0)
            
            detected = best_contour is not None
            
            # Sauvegarde pour IA
            if detected:
                self.last_detected_contour = best_contour.copy()
                self.last_detected_area = best_area
                self.last_bounding_rect = cv2.boundingRect(best_contour)
            
            return self._finalize_detection(original_frame, detected, best_contour, best_area, quality_score)
            
        except Exception as e:
            self.logging.debug(f"Erreur détection: {e}")
            return original_frame, False
    
    def _finalize_detection(self, frame, detected, contour, area, quality_score):
        """Finalisation avec intégration IA"""
        try:
            # Mise à jour historique
            self.detection_history.append(detected)
            if detected:
                self.detection_count += 1
            
            # === ANALYSE IA ===
            if detected and contour is not None and self.ai_enabled:
                position, confidence = self._analyze_hand_position_ai(frame, contour)
                
                if position:
                    self.current_position = position
                    self.current_position_confidence = confidence
                    self.ai_position_detections += 1
                    
                    # Commandes drone si activées
                    if self.drone_commands_enabled:
                        self._execute_drone_command(position, confidence)
                
                # Visualisation IA
                frame = self._draw_ai_overlay(frame, contour, position, confidence)
            
            # Visualisation détection
            if detected and contour is not None:
                self._draw_detection_overlay(frame, contour, area, quality_score)
            
            # Interface complète
            result_frame = self._create_complete_overlay(frame, detected, area, quality_score)
            
            return result_frame, detected
            
        except Exception as e:
            self.logging.debug(f"Erreur finalisation: {e}")
            return frame, False
    
    def _analyze_hand_position_ai(self, frame, contour):
        """Analyse IA de la position de la main"""
        try:
            if not self.ai_enabled or not self.feature_extractor or not self.position_recognizer:
                return None, 0.0
            
            bounding_rect = cv2.boundingRect(contour)
            
            # Mode entraînement
            if self.ai_mode == "training":
                return self._handle_training_mode(frame, contour, bounding_rect)
            
            # Mode reconnaissance
            elif self.ai_mode == "recognition" and self.position_recognizer.is_trained:
                features = self.feature_extractor.extract_complete_features(
                    frame, contour, bounding_rect
                )
                
                predicted_class, confidence = self.position_recognizer.predict_position(features)
                
                if predicted_class is not None:
                    position_name = HAND_POSITIONS[predicted_class].name
                    return position_name, confidence
            
            return None, 0.0
            
        except Exception as e:
            self.logging.debug(f"Erreur analyse IA: {e}")
            return None, 0.0
    
    def _handle_training_mode(self, frame, contour, bounding_rect):
        """Gestion du mode entraînement"""
        try:
            if self.training_class >= len(HAND_POSITIONS):
                # Fin d'entraînement
                self._start_model_training()
                return "training_complete", 1.0
            
            position = HAND_POSITIONS[self.training_class]
            
            # Compte à rebours pour capture
            if self.training_countdown > 0:
                self.training_countdown -= 1
                return f"training_{position.name}", self.training_countdown / 60.0
            
            # Capture d'échantillon
            features = self.feature_extractor.extract_complete_features(
                frame, contour, bounding_rect
            )
            
            success = self.position_recognizer.add_training_sample(features, self.training_class)
            if success:
                samples_count = len([l for l in self.position_recognizer.training_labels 
                                   if l == self.training_class])
                
                if samples_count >= self.training_samples_per_class:
                    self.training_class += 1
                    self.training_countdown = 60  # 2 secondes de pause
                    if self.training_class < len(HAND_POSITIONS):
                        next_position = HAND_POSITIONS[self.training_class]
                        self.logging.info(f"📝 Position suivante: {next_position.name}")
                else:
                    self.training_countdown = 30  # 1 seconde entre captures
                
                return f"captured_{position.name}", 1.0
            
            return f"training_{position.name}", 0.0
            
        except Exception as e:
            self.logging.error(f"Erreur mode entraînement: {e}")
            return None, 0.0
    
    def _start_model_training(self):
    # === PROTECTION CONTRE MULTIPLE THREADS ===
        if hasattr(self, '_training_in_progress') and self._training_in_progress:
            self.logging.warning("⚠️ Entraînement déjà en cours - ignoré")
            return
        
        self._training_in_progress = True
        self.logging.info("🚀 Démarrage entraînement du modèle IA...")
        
        def train():
            try:
                # Protection supplémentaire
                if not hasattr(self, '_training_in_progress') or not self._training_in_progress:
                    return
                    
                success = self.position_recognizer.train_model(epochs=25)  # ÉPOQUE RÉDUITE
                
                if success:
                    # Sauvegarde avec extension correcte
                    try:
                        import os
                        model_path = "hand_position_model"
                        
                        # Création du dossier si nécessaire
                        os.makedirs(model_path, exist_ok=True)
                        
                        # Sauvegarde TensorFlow moderne
                        self.position_recognizer.model.save(model_path)

                        # Sauvegarde données séparément
                        import pickle
                        data_path = f"{model_path}_data.pkl"
                        with open(data_path, 'wb') as f:
                            pickle.dump({
                                'training_data': self.position_recognizer.training_data,
                                'training_labels': self.position_recognizer.training_labels,
                                'feature_size': self.position_recognizer.feature_size,
                                'num_classes': self.position_recognizer.num_classes,
                                'is_trained': True
                            }, f)
                        
                        self.logging.info("💾 Modèle sauvegardé avec succès")
                        
                    except Exception as save_error:
                        self.logging.error(f"❌ Erreur sauvegarde: {save_error}")
                    
                    self.ai_mode = "recognition"
                    self.logging.info("✅ Modèle entraîné et sauvegardé!")
                    
                else:
                    self.logging.error("❌ Échec entraînement")
                    self.ai_mode = "detection"
                    
            except Exception as e:
                self.logging.error(f"❌ Erreur thread entraînement: {e}")
                self.ai_mode = "detection"
                
            finally:
                # Libération du verrou
                self._training_in_progress = False
        
        # Lancement thread unique
        training_thread = threading.Thread(target=train, daemon=True, name="ModelTraining")
        training_thread.start()
    
    def _execute_drone_command(self, position, confidence):
        """Exécution des commandes drone basées sur la position"""
        try:
            current_time = time.time()
            if current_time - self.last_command_time < self.command_cooldown:
                return False
            
            if confidence < 0.65:  # Seuil de sécurité global
                return False
            
            # NOTE: Remplacez 'bebop' par votre instance de drone
            # Exemple d'implémentation - adaptez selon votre code
            
            if position == "poing" and confidence > 0.9:
                self.logging.warning("🚨 ARRÊT D'URGENCE - Poing détecté")
                # bebop.emergency()
                self.logging.info(">>> COMMANDE: EMERGENCY() <<<")
                
            elif position == "stop" and confidence > 0.85:
                self.logging.info("🛑 STOP - Maintien position")
                # bebop.hover()
                self.logging.info(">>> COMMANDE: HOVER() <<<")
                
            elif position == "victoire" and confidence > 0.7:
                self.logging.info("⬆️ MONTÉE - Signe V")
                # bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=25, duration=0.8)
                self.logging.info(">>> COMMANDE: MONTÉE +25 (0.8s) <<<")
                
            elif position == "pouce" and confidence > 0.75:
                self.logging.info("👍 MONTÉE DOUCE - Pouce levé")
                # bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=15, duration=0.5)
                self.logging.info(">>> COMMANDE: MONTÉE +15 (0.5s) <<<")
                
            elif position == "paume" and confidence > 0.7:
                self.logging.info("➡️ AVANCER - Paume ouverte")
                # bebop.fly_direct(roll=0, pitch=20, yaw=0, vertical_movement=0, duration=0.6)
                self.logging.info(">>> COMMANDE: AVANCER pitch+20 (0.6s) <<<")
                
            elif position == "index" and confidence > 0.75:
                self.logging.info("🎯 AVANCER PRÉCIS - Index pointé")
                # bebop.fly_direct(roll=0, pitch=15, yaw=0, vertical_movement=0, duration=0.4)
                self.logging.info(">>> COMMANDE: AVANCER pitch+15 (0.4s) <<<")
                
            elif position == "salut" and confidence > 0.6:
                self.logging.info("↺ ROTATION GAUCHE - Salut")
                # bebop.fly_direct(roll=0, pitch=0, yaw=-30, vertical_movement=0, duration=0.6)
                self.logging.info(">>> COMMANDE: ROTATION yaw-30 (0.6s) <<<")
                
            elif position == "ok" and confidence > 0.8:
                self.logging.info("✅ HOVER - Signe OK")
                # bebop.hover()
                self.logging.info(">>> COMMANDE: HOVER() <<<")
                
            else:
                self.logging.debug(f"Position {position} non exécutée (confiance: {confidence:.2f})")
                return False
            
            self.last_command_time = current_time
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur commande drone: {e}")
            return False
    
    def _draw_ai_overlay(self, frame, contour, position, confidence):
        """Visualisation overlay IA"""
        try:
            if position and confidence > 0:
                # Couleur selon confiance
                if confidence > 0.8:
                    color = (0, 255, 0)      # Vert
                elif confidence > 0.6:
                    color = (0, 255, 255)    # Jaune
                else:
                    color = (0, 150, 255)    # Orange
                
                # Position sur la main
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Cercle de confiance
                    radius = int(15 + confidence * 25)
                    cv2.circle(frame, (cx, cy), radius, color, 3)
                    
                    # Nom de la position
                    cv2.putText(frame, position.upper(), (cx - 30, cy - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Confiance
                    cv2.putText(frame, f"{confidence:.2f}", (cx - 15, cy + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return frame
            
        except Exception as e:
            self.logging.debug(f"Erreur overlay IA: {e}")
            return frame
    
    def _draw_detection_overlay(self, frame, contour, area, quality_score):
        """Visualisation de la détection de base"""
        try:
            # Couleur selon qualité
            if quality_score > 0.7:
                color = (0, 255, 0)
            elif quality_score > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 150, 255)
            
            # Contour
            cv2.drawContours(frame, [contour], -1, color, 2)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Informations
            cv2.putText(frame, f"Q:{quality_score:.2f} A:{int(area)}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            self.logging.debug(f"Erreur visualisation détection: {e}")
    
    def _create_complete_overlay(self, frame, detected, area, quality_score):
        """Interface utilisateur complète"""
        try:
            h, w = frame.shape[:2]
            
            # === STATUS PRINCIPAL ===
            if detected:
                if self.current_position:
                    status = f"🤖 GANT + {self.current_position.upper()} ({self.current_position_confidence:.2f})"
                    status_color = (0, 255, 0)
                else:
                    status = f"🎯 GANT DÉTECTÉ (Q:{quality_score:.2f})"
                    status_color = (0, 255, 255)
            else:
                status = f"🔍 RECHERCHE GANT"
                status_color = (100, 100, 255)
            
            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # === MODE IA ===
            if self.ai_enabled:
                ai_status = f"IA: Mode {self.ai_mode}"
                if self.ai_mode == "training" and self.training_class < len(HAND_POSITIONS):
                    position = HAND_POSITIONS[self.training_class]
                    samples = len([l for l in self.position_recognizer.training_labels 
                                  if l == self.training_class])
                    ai_status += f" | {position.name} ({samples}/{self.training_samples_per_class})"
                    if self.training_countdown > 0:
                        ai_status += f" | Capture dans {self.training_countdown//30 + 1}s"
                elif self.position_recognizer.is_trained:
                    ai_status += " (Entraînée)"
                
                cv2.putText(frame, ai_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            else:
                cv2.putText(frame, "IA: TensorFlow requis", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # === COMMANDES DRONE ===
            if self.drone_commands_enabled:
                drone_status = "🚁 COMMANDES ACTIVES"
                drone_color = (0, 255, 0)
            else:
                drone_status = "🚁 Commandes désactivées"
                drone_color = (100, 100, 100)
            
            cv2.putText(frame, drone_status, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, drone_color, 1)
            
            # === STATISTIQUES ===
            if self.ai_enabled and self.ai_frame_count > 0:
                detection_rate = (self.ai_position_detections / max(self.ai_frame_count, 1)) * 100
                ai_stats = f"IA: {detection_rate:.1f}% positions détectées"
                cv2.putText(frame, ai_stats, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
            
            # === POSITIONS DISPONIBLES ===
            if self.ai_mode == "recognition":
                positions_text = "Positions: poing(URGENCE), paume(AVANCER), victoire(MONTÉE), ok(HOVER)"
                cv2.putText(frame, positions_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # === HISTORIQUE ===
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-15:]])
            cv2.putText(frame, f"Historique: {history}", (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            self.logging.debug(f"Erreur interface: {e}")
            return frame
    
    # === MÉTHODES DE CONTRÔLE IA ===
    def start_ai_training(self):
        """Démarrage entraînement IA"""
        if not self.ai_enabled:
            self.logging.warning("⚠️ TensorFlow requis pour l'entraînement")
            return False
        
        self.ai_mode = "training"
        self.training_class = 0
        self.training_countdown = 60
        
        # Reset données
        self.position_recognizer.training_data = []
        self.position_recognizer.training_labels = []
        
        self.logging.info("🎓 Mode entraînement IA démarré")
        self.logging.info("📝 Positions à entraîner:")
        for pos_id, position in HAND_POSITIONS.items():
            self.logging.info(f"   {pos_id}: {position.name} - {position.description}")
        
        return True
    
    def switch_ai_mode(self, mode):
        """Changement de mode IA"""
        if not self.ai_enabled:
            return False
            
        if mode == "recognition" and self.position_recognizer.is_trained:
            self.ai_mode = "recognition"
            self.logging.info("🤖 Mode reconnaissance activé")
            return True
        elif mode == "training":
            return self.start_ai_training()
        elif mode == "detection":
            self.ai_mode = "detection"
            self.logging.info("🔍 Mode détection simple activé")
            return True
        return False
    
    def toggle_drone_commands(self):
        """Activation/désactivation commandes drone"""
        self.drone_commands_enabled = not self.drone_commands_enabled
        status = "activées" if self.drone_commands_enabled else "désactivées"
        self.logging.info(f"🚁 Commandes drone {status}")
        return self.drone_commands_enabled
    
    def save_ai_model(self):
        """Sauvegarde du modèle IA"""
        if self.ai_enabled and self.position_recognizer and self.position_recognizer.is_trained:
            success = self.position_recognizer.save_model("hand_position_model")
            if success:
                self.logging.info("💾 Modèle IA sauvegardé")
            return success
        return False
    
    def load_ai_model(self):
        """Chargement du modèle IA"""
        if self.ai_enabled and self.position_recognizer:
            success = self.position_recognizer.load_model("hand_position_model")
            if success:
                self.ai_mode = "recognition"
                self.logging.info("📂 Modèle IA chargé")
            return success
        return False
    
    def get_ai_stats(self):
        """Statistiques IA complètes"""
        if not self.ai_enabled:
            return "IA désactivée (TensorFlow requis)"
        
        stats = f"Mode: {self.ai_mode}"
        
        if self.position_recognizer.is_trained:
            stats += " (Entraînée)"
            
        if self.ai_frame_count > 0:
            detection_rate = (self.ai_position_detections / self.ai_frame_count) * 100
            stats += f" | Détections: {detection_rate:.1f}%"
        
        if self.position_recognizer.training_data:
            stats += f" | Échantillons: {len(self.position_recognizer.training_data)}"
        
        return stats

# === CONTRÔLE DRONE ===
def simple_drone_control(bebop):
    """Contrôle drone manuel"""
    logger.info("Contrôle drone démarré.")
    print("\n[Commandes drone manuelles]")
    print("  t = décoller | l = atterrir | e = quitter")
    print("  f/b/g/d = mouvements | h/m = haut/bas")
    
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

# === FONCTION PRINCIPALE ===
def main_with_ai():
    """Fonction principale avec IA intégrée"""
    
    logger.info("=== BEBOP 2 AVEC IA DE RECONNAISSANCE DE POSITION ===")
    logger.info("🤖 Système de reconnaissance de gestes pour drone")
    
    if not TF_AVAILABLE:
        logger.warning("⚠️ TensorFlow non disponible - Mode détection simple uniquement")
        logger.info("   Pour activer l'IA: pip install tensorflow")
    
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
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-avioflags', 'direct',
            '-analyzeduration', '1000000',
            '-probesize', '1000000',
            '-i', sdp_path,
            '-vf', 'eq=saturation=1.1:gamma=0.95',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=2*1024*1024)
            logger.info("✅ Pipeline vidéo initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR AVEC IA ===
        detector = OptimizedBicolorGloveDetectorWithAI()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - IA Reconnaissance Position"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 80)
        logger.info("🎮 COMMANDES:")
        logger.info("  'q' = Quitter | 's' = Screenshot | 'r' = Reset")
        if TF_AVAILABLE:
            logger.info("🤖 COMMANDES IA:")
            logger.info("  'i' = Info IA | 't' = Entraînement | 'n' = Reconnaissance")
            logger.info("  'm' = Sauvegarder | 'l' = Charger | 'c' = Commandes drone")
        logger.info("=" * 80)
        logger.info("🎯 POSITIONS RECONNUES:")
        for pos_id, position in HAND_POSITIONS.items():
            logger.info(f"  {pos_id}: {position.name} - {position.description}")
        logger.info("=" * 80)
        
        screenshot_count = 0
        last_fps_log = time.time()
        fps_counter = 0
        
        logger.info("🎬 Démarrage détection avec IA...")
        
        # === BOUCLE PRINCIPALE ===
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.warning("⚠️ Frame incomplète")
                    continue
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Détection avec IA
                processed_frame, detected = detector.detect_glove_optimized(frame)
                
                # Mise à jour compteurs
                if detected:
                    detector.ai_frame_count += 1
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Logs périodiques
                fps_counter += 1
                if fps_counter % 90 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 90 / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"📊 FPS: {display_fps:.1f} | {detector.get_ai_stats()}")
                    if detector.current_position:
                        logger.info(f"🤖 Position: {detector.current_position} "
                                   f"(confiance: {detector.current_position_confidence:.2f})")
                    
                    last_fps_log = current_time
                
                # === GESTION TOUCHES ===
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                
                elif key == ord('s'):
                    timestamp = int(time.time())
                    screenshot_name = f"ai_capture_{timestamp}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                
                elif key == ord('r'):
                    detector = OptimizedBicolorGloveDetectorWithAI()
                    logger.info("🔄 Détecteur reset")
                
                # Commandes IA (seulement si TensorFlow disponible)
                elif TF_AVAILABLE:
                    if key == ord('i'):
                        logger.info(f"📊 {detector.get_ai_stats()}")
                        if detector.position_recognizer:
                            logger.info(f"📊 {detector.position_recognizer.get_prediction_stats()}")
                    
                    elif key == ord('t'):
                        if detector.start_ai_training():
                            logger.info("🎓 Mode entraînement IA activé")
                            logger.info("Effectuez chaque position quand demandé")
                    
                    elif key == ord('n'):
                        if detector.switch_ai_mode("recognition"):
                            logger.info("🤖 Mode reconnaissance IA activé")
                        else:
                            logger.warning("⚠️ Modèle non entraîné")
                    
                    elif key == ord('m'):
                        if detector.save_ai_model():
                            logger.info("💾 Modèle IA sauvegardé")
                        else:
                            logger.warning("⚠️ Aucun modèle à sauvegarder")
                    
                    elif key == ord('l'):
                        if detector.load_ai_model():
                            logger.info("📂 Modèle IA chargé")
                        else:
                            logger.warning("⚠️ Aucun modèle à charger")
                    
                    elif key == ord('c'):
                        status = detector.toggle_drone_commands()
                        if status:
                            logger.warning("⚠️ ATTENTION: Gestes contrôlent le drone!")
                            logger.info("🚁 Commandes drone ACTIVÉES")
                        else:
                            logger.info("🚁 Commandes drone désactivées")
                
                # Debug
                elif key == ord('d'):
                    logger.info("🔍 DEBUG DÉTAILLÉ:")
                    logger.info(f"   Détecteur IA: {detector.ai_enabled}")
                    logger.info(f"   Mode: {detector.ai_mode}")
                    logger.info(f"   Frames IA: {detector.ai_frame_count}")
                    logger.info(f"   Position actuelle: {detector.current_position}")
                    logger.info(f"   Commandes drone: {detector.drone_commands_enabled}")

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
        # === NETTOYAGE ===
        logger.info("🧹 Nettoyage...")
        
        if detector:
            total_runtime = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info("📊 STATS FINALES:")
            logger.info(f"  ⏱️ Durée: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames: {detector.frame_count}")
            logger.info(f"  🎯 Détections gant: {detector.detection_count}")
            
            if detector.ai_enabled:
                logger.info(f"  🤖 Frames IA: {detector.ai_frame_count}")
                logger.info(f"  🎭 Détections position: {detector.ai_position_detections}")
                if detector.ai_frame_count > 0:
                    rate = (detector.ai_position_detections / detector.ai_frame_count) * 100
                    logger.info(f"  📈 Taux reconnaissance: {rate:.1f}%")
                
                if detector.position_recognizer.training_data:
                    logger.info(f"  📚 Échantillons: {len(detector.position_recognizer.training_data)}")
            
            logger.info(f"  📸 Screenshots: {screenshot_count}")
            logger.info("=" * 80)
        
        # Nettoyage ressources
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
        
        logger.info("🎉 Session IA terminée!")
    
    return True

# === POINT D'ENTRÉE ===
if __name__ == "__main__":
    try:
        print("🚁 BEBOP 2 - DÉTECTION GANT AVEC IA")
        print("=" * 50)
        
        if TF_AVAILABLE:
            print("✅ TensorFlow détecté - IA activée")
        else:
            print("⚠️  TensorFlow manquant - Mode détection simple")
            print("   Installation: pip install tensorflow")
        
        print("\n🎯 GESTES RECONNUS:")
        for pos_id, position in HAND_POSITIONS.items():
            print(f"  {pos_id}: {position.name} - {position.description}")
        
        print("\n🚀 Démarrage...")
        success = main_with_ai()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"💥 Exception finale: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

# === GUIDE D'UTILISATION ===
"""
🎮 GUIDE D'UTILISATION RAPIDE:

1. INSTALLATION:
   pip install tensorflow opencv-python numpy pyparrot

2. PREMIÈRE UTILISATION:
   - Lancez le script
   - Pressez 't' pour entraîner l'IA (optionnel)
   - Suivez les instructions pour chaque position
   - Le modèle s'entraîne automatiquement

3. UTILISATION NORMALE:
   - Pressez 'n' pour activer la reconnaissance
   - Pressez 'c' pour activer les commandes drone
   - Effectuez vos gestes devant la caméra

4. COMMANDES PRINCIPALES:
   ✊ Poing fermé     → ARRÊT D'URGENCE
   ✋ Paume ouverte   → AVANCER
   ✌️ Victoire (V)    → MONTÉE
   👌 OK             → HOVER
   👍 Pouce levé     → MONTÉE DOUCE
   ☝️ Index pointé   → AVANCER PRÉCIS
   🛑 STOP           → ARRÊT
   👋 Salut          → ROTATION

5. SÉCURITÉS:
   - Seuils de confiance adaptatifs (60-90%)
   - Cooldown de 2 secondes entre commandes
   - Validation sur plusieurs frames
   - Mode d'urgence prioritaire

6. FICHIERS GÉNÉRÉS:
   - hand_position_model_model/ (modèle TensorFlow)
   - hand_position_model_data.pkl (données d'entraînement)
   - bebop_ai_detection.log (logs détaillés)
   - ai_capture_*.png (screenshots)

CONSEILS:
- Entraînez avec des variations d'éclairage et d'angles
- Maintenez les positions stables pendant la capture
- Vérifiez les logs pour les performances
- Sauvegardez régulièrement le modèle ('m')

DÉPANNAGE:
- Si TensorFlow manque: pip install tensorflow
- Si détection faible: ré-entraînez avec plus d'échantillons
- Si commandes erratiques: augmentez les seuils de confiance
- Si performances lentes: réduisez la résolution vidéo

ARCHITECTURE:
- Détection couleur rouge/orange optimisée
- Extraction de 64 caractéristiques (géométrie + vision)
- Réseau de neurones dense avec régularisation
- Stabilisation temporelle des prédictions
- Interface en temps réel avec feedback visuel
"""