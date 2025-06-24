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

import matplotlib.pyplot as plt

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
    confidence_threshold: float = 0.10

HAND_POSITIONS = {
    0: HandPosition("poing", "Poing fermé - ARRÊT D'URGENCE"),
    1: HandPosition("paume_avant", "Paume ouverte vers la caméra - AVANCER"),
    2: HandPosition("dos_main", "Dos de la main vers la caméra (ongles visibles) - RECULER"),
    3: HandPosition("index_droite", "Index tendu à droite (autres doigts repliés) - DROITE"),
    4: HandPosition("index_gauche", "Index tendu à gauche (autres doigts repliés) - GAUCHE"),
    5: HandPosition("doigts_haut", "Main ouverte, doigts pointés vers le haut - HAUT"),
    6: HandPosition("doigts_bas", "Main ouverte, doigts pointés vers le bas - BAS"),
}

DISTANCE_LABELS = [
    "PROCHE de la caméra",
    "À MI-DISTANCE",
    "ÉLOIGNÉ du drone"
]

# === EXTRACTEUR DE CARACTÉRISTIQUES ===
class AdvancedHandFeatureExtractor:
    """Extracteur de caractéristiques pour la reconnaissance de position"""
    
    def __init__(self):
        self.feature_size = 64
        self.logging = logging.getLogger(__name__)
        self.feature_history = deque(maxlen=5)
        self.position_history = deque(maxlen=3)
        
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

class HandPositionRecognizer:
    """Modèle de reconnaissance de position - VERSION COURBES & FEEDBACK"""
    
    def __init__(self, feature_size=64, num_classes=8):
        from collections import deque
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.model = None
        self.is_trained = False
        self.training_data = []
        self.training_labels = []
        self.logging = logging.getLogger(__name__)

        self.prediction_history = deque(maxlen=7)
        self.confidence_history = deque(maxlen=5)
        self.last_prediction_time = 0
        self.last_prediction_result = (None, 0.0)
        self.prediction_interval = 0.2

        self.total_predictions = 0
        self.confident_predictions = 0

        try:
            import tensorflow as tf
            self.tf_available = True
            self.keras = tf.keras
            self.layers = tf.keras.layers
        except ImportError:
            self.tf_available = False
            self.keras = None
            self.layers = None
            self.logging.warning("TensorFlow non disponible.")

    def create_model(self):
        if not self.tf_available:
            return False
        try:
            model = self.keras.Sequential([
                self.layers.Dense(128, activation='relu', input_shape=(self.feature_size,)),
                self.layers.BatchNormalization(),
                self.layers.Dropout(0.3),
                self.layers.Dense(96, activation='relu'),
                self.layers.BatchNormalization(),
                self.layers.Dropout(0.4),
                self.layers.Dense(64, activation='relu'),
                self.layers.BatchNormalization(),
                self.layers.Dropout(0.3),
                self.layers.Dense(32, activation='relu'),
                self.layers.Dropout(0.2),
                self.layers.Dense(self.num_classes, activation='softmax')
            ])
            model.compile(
                optimizer=self.keras.optimizers.Adam(learning_rate=0.001),
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
        if not self.tf_available:
            return False
        try:
            if len(features) != self.feature_size:
                self.logging.warning(f"Taille features incorrecte: {len(features)} vs {self.feature_size}")
                return False
            self.training_data.append(features.copy())
            self.training_labels.append(position_class)
            self.logging.info(f"📊 Échantillon ajouté: {position_class} - Total: {len(self.training_data)}")
            return True
        except Exception as e:
            self.logging.error(f"❌ Erreur ajout échantillon: {e}")
            return False

    def train_model(self, validation_split=0.2, epochs=25, plot_curves=True, save_fig=True, show_confusion=True):
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
            self.logging.info(
                f"[DEBUG TRAIN] First train features: {X[0][:8]}... sum={np.sum(X[0]):.2f}, min={np.min(X[0]):.2f}, max={np.max(X[0]):.2f}"
            )
            self.logging.info(f"[DEBUG TRAIN] Labels (premiers): {y[:10]}")
            unique, counts = np.unique(y, return_counts=True)
            self.logging.info(f"[DEBUG TRAIN] Distribution des labels: {dict(zip(unique, counts))}")
            self.logging.info(f"[DEBUG TRAIN] X shape: {X.shape}, y shape: {y.shape}")

            # Split train/validation manuellement pour garder l'équilibre par classe
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
            for train_idx, val_idx in sss.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

            self.logging.info(f"Train y distrib: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            self.logging.info(f"Val y distrib: {dict(zip(*np.unique(y_val, return_counts=True)))}")
            self.logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

            callbacks = [
                self.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                self.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6
                )
            ]
            self.logging.info(f"🚀 Entraînement: {len(X)} échantillons, {epochs} époques")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=min(16, len(X) // 4),
                callbacks=callbacks,
                verbose=0
            )
            final_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history.get('val_accuracy', [0])[-1]
            self.logging.info(f"✅ Entraînement terminé:")
            self.logging.info(f"   Précision: {final_accuracy:.4f}")
            self.logging.info(f"   Validation: {val_accuracy:.4f}")

            # === Courbes d’apprentissage (toujours en sauvegarde PNG, jamais de plt.show()) ===
            if plot_curves:
                try:
                    import time
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['accuracy'], label='Train Acc')
                    if 'val_accuracy' in history.history:
                        plt.plot(history.history['val_accuracy'], label='Val Acc')
                    plt.title('Accuracy')
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['loss'], label='Train Loss')
                    if 'val_loss' in history.history:
                        plt.plot(history.history['val_loss'], label='Val Loss')
                    plt.title('Loss')
                    plt.legend()
                    plt.tight_layout()
                    if save_fig:
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        figname = f"training_curves_{stamp}.png"
                        plt.savefig(figname)
                        self.logging.info(f"📈 Courbes sauvegardées : {figname}")
                    plt.close()  # <-- FERMETURE explicite, pas de plt.show()
                except Exception as e:
                    self.logging.warning(f"Erreur affichage/sauvegarde courbes : {e}")

            # === MATRICE DE CONFUSION (train/val) —> uniquement en sauvegarde ===
            from sklearn.metrics import confusion_matrix, classification_report
            import seaborn as sns
            import time
            class_names = [HAND_POSITIONS[i].name for i in range(self.num_classes)]
            stamp = time.strftime("%Y%m%d_%H%M%S")

            # Train
            train_preds = np.argmax(self.model.predict(X_train, verbose=0), axis=1)
            cm = confusion_matrix(y_train, train_preds)
            plt.figure(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Prédit')
            plt.ylabel('Vrai')
            plt.title("Matrice de confusion (train)")
            plt.tight_layout()
            figname_cm = f"confusion_matrix_train_{stamp}.png"
            plt.savefig(figname_cm)
            self.logging.info(f"🟦 Matrice de confusion (train) sauvegardée : {figname_cm}")
            plt.close()
            rep = classification_report(y_train, train_preds, target_names=class_names, digits=3)
            self.logging.info("\n" + rep)

            # Validation
            if len(X_val) > 0:
                val_preds = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
                cm_val = confusion_matrix(y_val, val_preds)
                plt.figure(figsize=(8, 7))
                sns.heatmap(cm_val, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
                plt.xlabel('Prédit')
                plt.ylabel('Vrai')
                plt.title("Matrice de confusion (validation)")
                plt.tight_layout()
                figname_cmval = f"confusion_matrix_val_{stamp}.png"
                plt.savefig(figname_cmval)
                self.logging.info(f"🟧 Matrice de confusion (val) sauvegardée : {figname_cmval}")
                plt.close()
                rep_val = classification_report(y_val, val_preds, target_names=class_names, digits=3)
                self.logging.info("\n" + rep_val)

            # === ACCURACY PAR ÉCHANTILLON —> nouveau graphique ===
            train_preds_full = np.argmax(self.model.predict(X, verbose=0), axis=1)
            accuracies = (train_preds_full == y).astype(int)
            plt.figure(figsize=(8,4))
            plt.plot(accuracies, label="Correct/Incorrect (1/0)")
            plt.title("Performance sur tous les échantillons d'entraînement")
            plt.xlabel("Échantillon")
            plt.ylabel("Réussite")
            plt.legend()
            figname_full = f"full_sample_accuracy_{stamp}.png"
            plt.savefig(figname_full)
            self.logging.info(f"📉 Courbe échantillon par échantillon sauvegardée: {figname_full}")
            plt.close()

            # Score sur tout le train (pour logs)
            train_acc_full = np.mean(train_preds_full == y)
            self.logging.info(f"[DEBUG TRAIN] Accuracy (full train set): {train_acc_full:.4f}")
            self.logging.info(f"[DEBUG TRAIN] Sample preds (train set): {train_preds_full[:10]} vs {y[:10]}")
            test_pred = self.model.predict(X[0].reshape(1, -1), verbose=0)[0]
            self.logging.info(f"[DEBUG TRAIN] Pred 1st sample: class={np.argmax(test_pred)}, raw={test_pred}")

            self.is_trained = True
            return True
        except Exception as e:
            self.logging.error(f"❌ Erreur entraînement: {e}")
            return False

    def predict_position(self, features, use_stabilization=True):
        if not self.tf_available or self.model is None or not self.is_trained:
            return None, 0.0
        try:
            if len(features) != self.feature_size:
                return None, 0.0
            current_time = time.time()
            if current_time - self.last_prediction_time < self.prediction_interval:
                return self.last_prediction_result
            features_batch = features.reshape(1, -1)    
            prediction = self.model.predict(features_batch, verbose=0, batch_size=1)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            self.last_prediction_time = current_time
            self.total_predictions += 1
            if use_stabilization:
                self.prediction_history.append((predicted_class, confidence))
                self.confidence_history.append(confidence)
                if len(self.prediction_history) >= 3:
                    from collections import Counter
                    recent_classes = [p[0] for p in list(self.prediction_history)[-3:]]
                    recent_confidences = [p[1] for p in list(self.prediction_history)[-3:]]
                    class_counts = Counter(recent_classes)
                    most_common_class, count = class_counts.most_common(1)[0]
                    if count >= 2:
                        class_confidences = [conf for cls, conf in zip(recent_classes, recent_confidences)
                                            if cls == most_common_class]
                        avg_confidence = np.mean(class_confidences)
                        predicted_class = most_common_class
                        confidence = avg_confidence
            # ---- UTILISER LE DICT GLOBAL ----
            if predicted_class in HAND_POSITIONS:
                threshold = 0.3
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
        if self.total_predictions == 0:
            return "Aucune prédiction"
        confidence_rate = (self.confident_predictions / self.total_predictions) * 100
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0
        return f"Confiance: {confidence_rate:.1f}% | Moy: {avg_confidence:.2f}"

    def save_model(self, filepath="hand_position_model"):
        if not self.tf_available or self.model is None:
            return False
        try:
            model_file = f"{filepath}_model.keras"
            self.model.save(model_file)
            self.logging.info(f"✅ Modèle sauvegardé dans {model_file}")
            data_path = f"{filepath}_data.pkl"
            with open(data_path, 'wb') as f:
                import pickle
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

    def load_model(self, filepath="hand_position_model"):
        if not self.tf_available:
            return False
        try:
            model_file = f"{filepath}_model.keras"
            if os.path.exists(model_file):
                self.model = self.keras.models.load_model(model_file)
                self.logging.info(f"✅ Modèle chargé: {model_file}")
            else:
                old_model_path = f"{filepath}_model.h5"
                if os.path.exists(old_model_path):
                    self.model = self.keras.models.load_model(old_model_path)
                    self.logging.info(f"✅ Ancien modèle chargé: {old_model_path}")
                else:
                    self.logging.warning(f"⚠️ Aucun modèle trouvé: {model_file} ou {old_model_path}")
                    return False
            data_path = f"{filepath}_data.pkl"
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    import pickle
                    data = pickle.load(f)
                self.training_data = data.get('training_data', [])
                self.training_labels = data.get('training_labels', [])
                self.feature_size = data.get('feature_size', 64)
                self.num_classes = data.get('num_classes', 8)
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
        self.training_samples_per_class = 50
        
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

        # === AJOUT ICI (stabilisation commandes drone) ===
        self.position_history = deque(maxlen=3)   # <---- AJOUT OBLIGATOIRE

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
        """Détection de gant améliorée avec nettoyage du masque"""
        if frame is None:
            return frame, False

        original_frame = frame.copy()
        self.frame_count += 1
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Masques couleur - élargir un peu les bornes si besoin
            red_lower1 = np.array([0, 120, 90])
            red_upper1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)

            red_lower2 = np.array([170, 120, 90])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)

            orange_lower = np.array([10, 120, 120])
            orange_upper = np.array([22, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)

            # Fusion masques
            mask_combined = cv2.bitwise_or(mask_red1, mask_red2)
            mask_combined = cv2.bitwise_or(mask_combined, mask_orange)

            # --- FILTRAGE AVANCÉ ---
            mask_combined = cv2.medianBlur(mask_combined, 5)
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
            mask_combined = cv2.dilate(mask_combined, self.kernel_medium, iterations=2)
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, self.kernel_large, iterations=2)

            # Recherche de contours
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour, best_area, best_quality = None, 0, 0

            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                if (self.min_area < area < self.max_area) and (len(contour) > self.min_contour_points):
                    # **Prioriser les contours compacts et larges**
                    if compactness > 0.25 and area > best_area:
                        best_contour, best_area, best_quality = contour, area, min(area / self.area_reference, 1.0)

            detected = best_contour is not None

            # Vérifie la "fermeture" du contour (si besoin, ajouter un check sur cv2.isContourConvex)
            if detected and not cv2.isContourConvex(best_contour):
                # Tu peux logguer ici si tu veux
                self.logging.info("Contour détecté mais pas convexe (main trop ouverte ou contour brisé)")
            
            # Tu peux ici refuser le contour s'il est trop allongé (erreur classique)
            if detected and best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                if aspect_ratio < 0.35 or aspect_ratio > 3.0:
                    # Probablement du bruit ou une main mal détectée
                    self.logging.info("Contour refusé car aspect ratio douteux.")
                    detected = False

            # Sauvegarde et return classique
            if detected:
                self.last_detected_contour = best_contour.copy()
                self.last_detected_area = best_area
                self.last_bounding_rect = cv2.boundingRect(best_contour)
            
            return self._finalize_detection(original_frame, detected, best_contour, best_area, best_quality)

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

                # --- Correction: n'afficher que les vraies positions en mode reconnaissance ---
                if (
                    self.ai_mode == "recognition"
                    and position
                    and position != "training_complete"
                    and position in [p.name for p in HAND_POSITIONS.values()]
                ):
                    self.current_position = position
                    self.current_position_confidence = confidence
                    self.ai_position_detections += 1

                    # Commandes drone si activées
                    if self.drone_commands_enabled:
                        self._execute_drone_command(position, confidence)
                else:
                    self.current_position = None
                    self.current_position_confidence = 0.0

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
                self.logging.info("[DEBUG] IA non activée ou modules manquants")
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
                self.logging.info(f"[DEBUG] Features: {features[:8]}... sum={np.sum(features):.2f}")

                predicted_class, confidence = self.position_recognizer.predict_position(features)
                self.logging.info(f"[DEBUG] Prédiction brute: class={predicted_class}, confiance={confidence:.3f}")

                if predicted_class is not None:
                    position_name = HAND_POSITIONS[predicted_class].name
                    self.logging.info(f"[DEBUG] Position IA: {position_name}, confiance: {confidence:.3f}")
                    return position_name, confidence

            self.logging.info("[DEBUG] Aucun résultat IA ou modèle non entraîné")
            return None, 0.0

        except Exception as e:
            self.logging.debug(f"Erreur analyse IA: {e}")
            return None, 0.0
    
    def _handle_training_mode(self, frame, contour, bounding_rect):
        try:
            if self.training_class >= len(HAND_POSITIONS):
                # Fin d'entraînement, on lance le training du modèle
                self._start_model_training()
                self.current_training_distance_msg = ""
                return "training_complete", 1.0

            position = HAND_POSITIONS[self.training_class]

            # --- QUALITÉ DU CONTOUR ---
            min_area = 3500        # Aire minimale pour être sûr que le gant est bien détecté
            min_extent = 0.25      # Proportion du contour par rapport au bounding rect
            min_solidity = 0.85    # Evite les contours “troués”
            min_ratio = 0.3        # Ratio largeur/hauteur raisonnable (évite les barres fines)

            x, y, w, h = bounding_rect
            area = cv2.contourArea(contour)
            extent = area / (w * h) if w * h > 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            aspect_ratio = w / float(h) if h > 0 else 0

            # Critère "aspect ratio" dans les deux sens pour accepter mains orientées
            aspect_ok = min_ratio < aspect_ratio < 1.0/min_ratio

            # Si la main n'est pas bien vue : contour trop petit, pas assez “plein”, etc.
            if (area < min_area or extent < min_extent or solidity < min_solidity or not aspect_ok):
                self.current_training_distance_msg = "Gant mal détecté : repositionnez bien la main"
                self.training_countdown = 15   # Petite pause, le temps de replacer la main
                return f"training_{position.name}", 0.0

            # --- Vérifie la distance à la caméra pour feedback visuel ---
            if area > 11000:
                distance_msg = "Reculez un peu la main"
            elif area < 2500:
                distance_msg = "Avancez la main"
            else:
                distance_msg = "Distance OK"
            self.current_training_distance_msg = distance_msg

            # Attend le countdown avant de capturer
            if self.training_countdown > 0:
                self.training_countdown -= 1
                return f"training_{position.name}", self.training_countdown / 60.0

            # --- Capture d'échantillon si tout est ok ---
            features = self.feature_extractor.extract_complete_features(
                frame, contour, bounding_rect
            )
            success = self.position_recognizer.add_training_sample(features, self.training_class)
            if success:
                samples_count = len([l for l in self.position_recognizer.training_labels if l == self.training_class])
                self.current_training_distance_msg = "✅ Échantillon capturé!"

                # Passe à la position suivante si assez d'échantillons pour cette position
                if samples_count >= self.training_samples_per_class:
                    self.training_class += 1
                    self.training_countdown = 60  # 2 secondes de pause pour changer de position
                    if self.training_class < len(HAND_POSITIONS):
                        next_position = HAND_POSITIONS[self.training_class]
                        self.logging.info(f"📝 Prochain geste attendu : {next_position.description}")
                        self.current_training_distance_msg = ""
                    else:
                        self.training_countdown = 30  # petite pause avant la fin
                    return f"captured_{position.name}", 1.0

                # Sinon continue la capture pour cette même position (petite pause)
                self.training_countdown = 18   # ~0.6s pour laisser le temps à l'utilisateur de rebouger la main
                return f"captured_{position.name}", 1.0

            # S'il n'y a pas eu de capture, on continue l'affichage et l'attente
            return f"training_{position.name}", 0.0

        except Exception as e:
            self.logging.error(f"Erreur mode entraînement: {e}")
            self.current_training_distance_msg = ""
            return None, 0.0
    
    def _start_model_training(self):
        # On NE fait plus de thread : tout dans le thread principal pour éviter les bugs et que matplotlib/keras fonctionne bien.
        if hasattr(self, '_training_in_progress') and self._training_in_progress:
            self.logging.warning("⚠️ Entraînement déjà en cours - ignoré")
            return

        self._training_in_progress = True
        self.logging.info("🚀 Démarrage entraînement du modèle IA...")

        try:
            success = self.position_recognizer.train_model(epochs=25)
            if success:
                model_path = "hand_position_model"
                self.position_recognizer.save_model(model_path)
                n_samples = len(self.position_recognizer.training_data)
                self.logging.info(f"💾 Modèle sauvegardé avec succès ({n_samples} échantillons)")
                self.ai_mode = "recognition"
                self.logging.info("✅ Modèle entraîné et sauvegardé!")
            else:
                self.logging.error("❌ Échec entraînement")
                self.ai_mode = "detection"
        except Exception as e:
            self.logging.error(f"❌ Erreur entraînement: {e}")
            self.ai_mode = "detection"
        finally:
            self._training_in_progress = False
    
    def _execute_drone_command(self, position, confidence):
        try:
            import time
            current_time = time.time()
            self.logging.info(f"[DRONE CMD] DEMANDE: position={position}, confidence={confidence:.3f}, t={current_time:.3f}")

            # Vérifie le cooldown
            if current_time - self.last_command_time < self.command_cooldown:
                self.logging.info("[DRONE CMD] Cooldown actif, commande ignorée.")
                return False
            
            self.position_history.append((position, confidence))
            if list(self.position_history).count((position, confidence)) < 2:  # besoin 2x d'affilée
                self.logging.info(f"[DRONE CMD] Position '{position}' pas assez stable, commande ignorée.")
                return False

            # Seuils de confiance (modifiable)
            thresholds = {
                "paume_avant": 0.75,      # Avancer
                "dos_main": 0.75,         # Reculer
                "index_droite": 0.75,     # Droite
                "index_gauche": 0.75,     # Gauche
                "doigts_haut": 0.75,      # Haut
                "doigts_bas": 0.75,       # Bas
                "poing": 0.75,            # Arrêt d'urgence
            }
            threshold = thresholds.get(position, 0.85)
            self.logging.info(f"[DRONE CMD] Seuil pour {position}: {threshold}")

            # Limite de distance : évite les commandes si le gant est trop loin
            MIN_GLOVE_AREA = 3500  # Ajuste si besoin (taille en pixels)
            if position in ["paume", "index", "ok", "salut"]:
                area = getattr(self, "last_detected_area", None)
                if area is not None and area < MIN_GLOVE_AREA:
                    self.logging.warning(f"[DRONE CMD] Gant trop loin (area={area:.0f}), commande ignorée.")
                    return False

            if confidence < threshold:
                self.logging.info(f"[DRONE CMD] Confiance trop basse ({confidence:.3f} < {threshold}), commande ignorée.")
                return False

            # Accès au drone
            bebop = getattr(self, "bebop", None)
            if bebop is None:
                self.logging.warning("[DRONE CMD] Aucune instance drone !")
                return False

            # Sensors complet pour debug
            sensors_dict = vars(bebop.sensors) if hasattr(bebop, 'sensors') else {}
            self.logging.info(f"[DRONE CMD] bebop OK: {bebop}")
            self.logging.info(f"[DRONE CMD] sensors: {sensors_dict}")

            # Statut drone
            flying_state = getattr(bebop.sensors, "flying_state", "unknown")
            self.logging.info(f"[DRONE CMD] flying_state = '{flying_state}'")

            # ========== Commandes ==========
            if position == "poing":
                # Arrêt d'urgence : Atterrir tout de suite
                self.logging.warning("[DRONE CMD] 🚨 ARRÊT D'URGENCE - Poing détecté")
                if flying_state == "landed":
                    self.logging.info("[DRONE CMD] Déjà posé. Aucun mouvement.")
                else:
                    self.logging.info("[DRONE CMD] safe_land appelé !")
                    bebop.safe_land(10)
                self.last_command_time = current_time
                return True

            elif position == "paume_avant" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ➡️ Avancer (paume avant)")
                bebop.fly_direct(roll=0, pitch=70, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "dos_main" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬅️ Reculer (dos de la main)")
                bebop.fly_direct(roll=0, pitch=-70, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "index_droite" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ➡️ Droite (index droit)")
                bebop.fly_direct(roll=30, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "index_gauche" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬅️ Gauche (index gauche)")
                bebop.fly_direct(roll=-30, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "doigts_haut" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬆️ Monter (doigts haut)")
                bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=25, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "doigts_bas" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬇️ Descendre (doigts bas)")
                bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-25, duration=0.3)
                self.last_command_time = current_time
                return True

            else:
                self.logging.info(f"[DRONE CMD] Position {position} non exécutée ou condition non remplie (flying_state={flying_state}, confiance={confidence:.2f})")
                return False

        except Exception as e:
            self.logging.error(f"[DRONE CMD] ❌ Erreur commande drone: {e}")
            import traceback
            self.logging.error(f"[DRONE CMD] Traceback: {traceback.format_exc()}")
            return False

    
    def _draw_ai_overlay(self, frame, contour, position, confidence):
        """Visualisation overlay IA"""
        try:
            if (
                position
                and confidence > 0
                and position != "training_complete"
                and position in [p.name for p in HAND_POSITIONS.values()]
            ):
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
        """
        Visualisation de la détection de base avec retour couleur qualité et affichage des problèmes.
        """
        try:
            # Couleur selon la qualité (en général)
            color = (0, 255, 0)  # Vert = bonne qualité

            # En mode entraînement : si le message de distance signale une mauvaise détection, on force le rouge
            if hasattr(self, "current_training_distance_msg"):
                msg = self.current_training_distance_msg.lower()
                if "mal détecté" in msg or "repositionnez" in msg or "gant" in msg and "mal" in msg:
                    color = (0, 0, 255)  # Rouge

            # Sinon, code couleur selon score
            if color == (0, 255, 0):
                if quality_score > 0.7:
                    color = (0, 255, 0)    # Vert
                elif quality_score > 0.5:
                    color = (0, 255, 255)  # Orange
                else:
                    color = (0, 0, 255)    # Rouge

            # Dessin du contour
            cv2.drawContours(frame, [contour], -1, color, 3)

            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Affichage d'infos techniques
            cv2.putText(frame, f"Q:{quality_score:.2f} A:{int(area)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Affichage du message problème si mauvais
            if hasattr(self, "current_training_distance_msg"):
                msg = self.current_training_distance_msg
                if "mal détecté" in msg or "repositionnez" in msg:
                    cv2.putText(frame, msg, (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
                    samples = len([l for l in self.position_recognizer.training_labels if l == self.training_class])
                    ai_status += f" | {position.name} ({samples}/{self.training_samples_per_class})"
                    if self.training_countdown > 0:
                        ai_status += f" | Capture dans {self.training_countdown//30 + 1}s"
                    # Affichage de la description du geste attendu
                    instruction = position.description
                    cv2.putText(frame, f"Geste attendu : {instruction}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
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

            # === AFFICHAGE DU MESSAGE DE DISTANCE EN TRAINING ===
            if self.ai_mode == "training" and hasattr(self, 'current_training_distance_msg'):
                msg = self.current_training_distance_msg
                if msg:
                    cv2.putText(frame, f"Distance : {msg}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            # === STATISTIQUES ===
            if self.ai_enabled and self.ai_frame_count > 0:
                detection_rate = (self.ai_position_detections / max(self.ai_frame_count, 1)) * 100
                ai_stats = f"IA: {detection_rate:.1f}% positions détectées"
                cv2.putText(frame, ai_stats, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

            # === POSITIONS DISPONIBLES ===
            if self.ai_mode == "recognition":
                positions_text = ("Positions : "
                    "avancer(paume), reculer(dos main), droite(index droit), "
                    "gauche(index gauche), haut(doigts haut), bas(doigts bas), urgence(poing)")
                cv2.putText(frame, positions_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # === HISTORIQUE ===
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-15:]])
            cv2.putText(frame, f"Historique: {history}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # === ÉCHANTILLONS PAR CLASSE EN TRAINING ===
            if self.ai_mode == "training" and hasattr(self, 'position_recognizer'):
                ylabels = self.position_recognizer.training_labels
                txt = "Échantillons : "
                for i, pos in HAND_POSITIONS.items():
                    n = len([y for y in ylabels if y == i])
                    txt += f"{pos.name}({n}) "
                cv2.putText(frame, txt, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

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
        detector.bebop = bebop  # Ajoute cette ligne juste après la création du detector
        
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
                    detector.bebop = bebop  # ← AJOUTE BIEN cette ligne ICI aussi !
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
                        logger.info("[DEBUG] Touche 'c' pressée, toggle_drone_commands va être appelé")
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
        
        logger.info("=" * 80)
        logger.info("🎯 GESTES RECONNUS :")
        for pos_id, position in HAND_POSITIONS.items():
            logger.info(f"  {pos_id}: {position.name} - {position.description}")
        logger.info("=" * 80)

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