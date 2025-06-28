#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================
BEBOP 2 - DÉTECTION GANT AVEC IA DE RECONNAISSANCE DE POSITION
Projet Universitaire - Paris-Saclay - Licence Pro MECSE
Étudiant: RAYAN DJOUDI
Version finale corrigée - Thread-safe
=================================================================================

DESCRIPTION:
Ce programme contrôle un drone Bebop 2 par reconnaissance de gestes de la main.
Il utilise OpenCV pour la détection d'un gant rouge/orange et TensorFlow pour
reconnaitre 9 positions différentes qui correspondent à des commandes de vol.

FONCTIONNALITÉS:
- Détection en temps réel d'un gant coloré (rouge/orange)
- Intelligence artificielle pour reconnaître 9 gestes différents
- Contrôle sécurisé du drone avec multiples vérifications
- Interface utilisateur complète avec feedback visuel
- Mode d'entraînement interactif pour l'IA
- Sauvegarde/chargement des modèles entraînés

SÉCURITÉS IMPLÉMENTÉES:
- Arrêt d'urgence avec le poing fermé
- Cooldown entre commandes (5 secondes)
- Seuils de confiance adaptatifs (75-85%)
- Vérification de la distance du gant
- Stabilisation sur plusieurs frames
=================================================================================
"""

# =============================================================================
# IMPORTS ET CONFIGURATION GLOBALE
# =============================================================================

# Imports système et utilitaires de base
import cv2              # OpenCV pour la vision par ordinateur
import numpy as np      # NumPy pour les calculs matriciels
import time            # Gestion du temps et des delays
import subprocess      # Pour lancer FFmpeg et gérer le flux vidéo
import threading       # Threading pour le contrôle drone parallèle
import sys            # Interface système (arguments, exit, etc.)
import logging        # Système de logs pour le debugging
import os             # Opérations sur les fichiers et dossiers
import pyparrot       # Bibliothèque pour contrôler le drone Bebop
from pyparrot.Bebop import Bebop

# Imports pour structures de données avancées
from collections import deque      # Files FIFO pour l'historique
from dataclasses import dataclass  # Classes de données simplifiées
from typing import List, Tuple, Optional, Dict  # Type hints pour clarté

# Imports pour la persistance des données
import pickle  # Sérialisation d'objets Python
import json   # Format JSON pour configuration

# Import pour les graphiques et visualisations
import matplotlib.pyplot as plt

# =============================================================================
# GESTION DE TENSORFLOW (INTELLIGENCE ARTIFICIELLE)
# =============================================================================

# Variables globales pour TensorFlow (IA optionnelle)
TF_AVAILABLE = False  # Flag pour savoir si TensorFlow est disponible
tf = None            # Module TensorFlow principal
keras = None         # Interface Keras pour les réseaux de neurones
layers = None        # Couches de réseaux de neurones

try:
    # Tentative d'import de TensorFlow
    import tensorflow as tf
    keras = tf.keras                    # Interface haut niveau pour ML
    layers = tf.keras.layers           # Couches du réseau de neurones
    TF_AVAILABLE = True
    print("✅ TensorFlow chargé avec succès")
    
    # Optimisations TensorFlow pour les performances
    tf.config.threading.set_inter_op_parallelism_threads(1)  # Parallélisme entre opérations
    tf.config.threading.set_intra_op_parallelism_threads(1)  # Parallélisme dans opérations
    
    try:
        # Configuration GPU si disponible (optionnel)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Croissance progressive de la mémoire GPU (évite l'allocation complète)
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass  # GPU non disponible, on continue avec CPU
        
except ImportError as e:
    # TensorFlow pas installé
    TF_AVAILABLE = False
    print(f"⚠️ TensorFlow non disponible: {e}")
    print("   Installez avec: pip install tensorflow")
except Exception as e:
    # Autre erreur TensorFlow
    TF_AVAILABLE = False
    print(f"❌ Erreur TensorFlow: {e}")

# =============================================================================
# PARAMÈTRES DE CONFIGURATION
# =============================================================================

# Configuration réseau drone
BEBOP_IP = "192.168.42.1"  # Adresse IP fixe du drone Bebop 2

# Résolution vidéo (optimisée pour performance/qualité)
WIDTH, HEIGHT = 856, 480

# =============================================================================
# CONFIGURATION DU SYSTÈME DE LOGS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,  # Niveau de détail des logs
    format='%(asctime)s - %(levelname)s - %(message)s',  # Format d'affichage
    handlers=[
        logging.StreamHandler(sys.stdout),  # Affichage console
        logging.FileHandler('bebop_ai_detection.log', mode='w', encoding='utf-8')  # Fichier de logs
    ]
)
logger = logging.getLogger(__name__)  # Logger pour ce module

# =============================================================================
# CONFIGURATION DES POSITIONS DE MAIN RECONNUES
# =============================================================================

@dataclass
class HandPosition:
    """
    Classe pour définir une position de main reconnue par l'IA
    
    Attributes:
        name (str): Nom court de la position
        description (str): Description détaillée du geste
        confidence_threshold (float): Seuil minimal de confiance pour validation
    """
    name: str
    description: str
    confidence_threshold: float = 0.10  # Seuil par défaut très bas (ajusté dynamiquement)

# Dictionnaire des 9 positions reconnues par l'IA
HAND_POSITIONS = {
    0: HandPosition("poing", "Poing fermé - ARRÊT D'URGENCE"),
    1: HandPosition("avancer", "Main vers l'avant, doigts vers le bas (style karaté vers caméra)"),
    2: HandPosition("reculer", "Paume vers caméra, doigts écartés (main à plat stop classique)"),
    3: HandPosition("monter", "Pouce vers le haut (pouce en l'air 👍)"),
    4: HandPosition("descendre", "Pouce vers le bas (👎)"),
    5: HandPosition("droite", "Index pointé vers la gauche de l'image (pour faire aller le drone à droite)"),
    6: HandPosition("gauche", "Index pointé vers la droite de l'image (pour faire aller le drone à gauche)"),
    7: HandPosition("rotation_gauche", "Paume main penché vers droite visible (rotation à gauche)"),
    8: HandPosition("rotation_droite", "Dos main penché vers gauche visible (rotation à droite)"),
}

# Labels pour la classification de distance (non utilisé actuellement)
DISTANCE_LABELS = [
    "PROCHE de la caméra",
    "À MI-DISTANCE", 
    "ÉLOIGNÉ du drone"
]

# =============================================================================
# EXTRACTEUR DE CARACTÉRISTIQUES POUR L'IA
# =============================================================================

class AdvancedHandFeatureExtractor:
    """
    Classe pour extraire les caractéristiques d'une main détectée.
    
    Cette classe analyse un contour de main et en extrait 64 caractéristiques
    numériques qui serviront d'entrée au réseau de neurones :
    - 17 caractéristiques géométriques (forme, orientation, etc.)
    - 13 caractéristiques visuelles (couleur, gradients, etc.)
    - 34 caractéristiques de padding pour atteindre 64
    
    La stabilisation temporelle lisse les variations entre frames.
    """
    
    def __init__(self):
        """Initialise l'extracteur avec ses paramètres."""
        self.feature_size = 64  # Taille fixe du vecteur de caractéristiques
        self.logging = logging.getLogger(__name__)
        
        # Historiques pour la stabilisation temporelle
        self.feature_history = deque(maxlen=5)    # 5 derniers vecteurs de features
        self.position_history = deque(maxlen=3)   # 3 dernières positions détectées
        
    def extract_geometric_features(self, contour, bounding_rect):
        """
        Extrait les caractéristiques géométriques d'un contour de main.
        
        Args:
            contour: Contour OpenCV de la main détectée
            bounding_rect: Rectangle englobant (x, y, width, height)
            
        Returns:
            np.array: Vecteur de 17 caractéristiques géométriques normalisées
        """
        try:
            x, y, w, h = bounding_rect
            area = cv2.contourArea(contour)  # Aire du contour
            
            # === CARACTÉRISTIQUES DE FORME BASIQUES ===
            
            # Ratio largeur/hauteur (forme allongée vs carrée)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Proportion du contour dans son rectangle englobant
            extent = area / (w * h) if w * h > 0 else 0
            
            # === CONVEXITÉ ET SOLIDITÉ ===
            
            # Enveloppe convexe (forme "gonflée" sans creux)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            # Solidité = rapport entre aire réelle et aire convexe
            # (proche de 1 = forme pleine, proche de 0 = forme avec beaucoup de creux)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # === COMPACITÉ ===
            
            # Périmètre du contour
            perimeter = cv2.arcLength(contour, True)
            
            # Compacité = 4π × aire / périmètre²
            # (cercle parfait = 1, forme très allongée = proche de 0)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # === MOMENTS GÉOMÉTRIQUES (INVARIANTS) ===
            
            # Moments statistiques de la forme (invariants aux transformations)
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()  # 7 moments de Hu
            
            # === DÉFAUTS DE CONVEXITÉ ===
            
            # Nombre de "creux" dans la forme (doigts écartés créent des défauts)
            if len(contour) >= 4:
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    convexity_defects = len(defects) if defects is not None else 0
                else:
                    convexity_defects = 0
            else:
                convexity_defects = 0
            
            # === ORIENTATION ===
            
            # Angle principal de la forme (ellipse ajustée)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                orientation = ellipse[2] / 180.0  # Normalisation [0,1]
            else:
                orientation = 0
            
            # === ANALYSE DU CENTRE ET DISTANCES ===
            
            # Centre de masse du contour
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])  # Centre x
                cy = int(moments["m01"] / moments["m00"])  # Centre y
                
                # Distances de tous les points au centre
                distances = [np.sqrt((pt[0][0] - cx)**2 + (pt[0][1] - cy)**2) for pt in contour]
                avg_distance = np.mean(distances) / 100.0      # Distance moyenne normalisée
                std_distance = np.std(distances) / 100.0       # Écart-type des distances
            else:
                avg_distance = std_distance = 0
            
            # === COMPILATION DES CARACTÉRISTIQUES ===
            
            geometric_features = [
                aspect_ratio,           # Ratio largeur/hauteur
                extent,                # Proportion dans rectangle
                solidity,              # Solidité (convexité)
                compactness,           # Compacité (circularité)
                area / 10000.0,        # Aire normalisée
                perimeter / 1000.0,    # Périmètre normalisé
                convexity_defects / 10.0,  # Nombre de défauts normalisé
                orientation,           # Orientation normalisée
                avg_distance,          # Distance moyenne normalisée
                std_distance,          # Écart-type distance normalisé
                *hu_moments[:7]        # 7 moments de Hu (invariants géométriques)
            ]
            
            # Retourne exactement 17 caractéristiques
            return np.array(geometric_features[:17], dtype=np.float32)
            
        except Exception as e:
            self.logging.debug(f"Erreur extraction géométrique: {e}")
            return np.zeros(17, dtype=np.float32)  # Vecteur zéro en cas d'erreur
    
    def extract_visual_features(self, roi_image, contour_mask):
        """
        Extrait les caractéristiques visuelles (couleur, texture) d'une région d'intérêt.
        
        Args:
            roi_image: Image de la région d'intérêt (main détectée)
            contour_mask: Masque binaire de la forme de la main
            
        Returns:
            np.array: Vecteur de 13 caractéristiques visuelles normalisées
        """
        try:
            # Vérification de validité
            if roi_image.size == 0:
                return np.zeros(21, dtype=np.float32)
                
            # Redimensionnement pour uniformiser (64x64 pixels)
            roi_resized = cv2.resize(roi_image, (64, 64))
            mask_resized = cv2.resize(contour_mask, (64, 64))
            
            # === ANALYSE COULEUR (ESPACE HSV) ===
            
            # Conversion en HSV (Hue-Saturation-Value)
            hsv_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            
            # Moyennes des composantes HSV dans la zone de la main
            if np.any(mask_resized > 0):
                h_mean = np.mean(hsv_roi[:, :, 0][mask_resized > 0]) / 180.0  # Teinte [0,1]
                s_mean = np.mean(hsv_roi[:, :, 1][mask_resized > 0]) / 255.0  # Saturation [0,1]
                v_mean = np.mean(hsv_roi[:, :, 2][mask_resized > 0]) / 255.0  # Valeur [0,1]
            else:
                h_mean = s_mean = v_mean = 0
            
            # === ANALYSE DES GRADIENTS (TEXTURE) ===
            
            # Conversion en niveaux de gris
            gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            
            # Gradients de Sobel (variations d'intensité)
            sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)  # Gradient horizontal
            sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)  # Gradient vertical
            
            # Magnitude du gradient (force des variations)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Moyenne des gradients dans la zone de la main
            if np.any(mask_resized > 0):
                gradient_mean = np.mean(gradient_magnitude[mask_resized > 0]) / 255.0
            else:
                gradient_mean = 0
            
            # === DÉTECTION DE CONTOURS (CANNY) ===
            
            # Détecteur de bords de Canny
            edges = cv2.Canny(gray_roi, 50, 150)
            
            # Densité de bords dans la zone de la main
            if np.sum(mask_resized > 0) > 0:
                edge_density = np.sum(edges[mask_resized > 0]) / np.sum(mask_resized > 0) / 255.0
            else:
                edge_density = 0
            
            # === HISTOGRAMME DES NIVEAUX DE GRIS ===
            
            # Histogramme simplifié (8 bins pour réduire la dimensionnalité)
            hist = cv2.calcHist([gray_roi], [0], mask_resized, [8], [0, 256])
            hist_features = hist.flatten() / (np.sum(hist) + 1e-7)  # Normalisation
            
            # === COMPILATION DES CARACTÉRISTIQUES VISUELLES ===
            
            visual_features = [
                h_mean,          # Teinte moyenne
                s_mean,          # Saturation moyenne
                v_mean,          # Valeur moyenne
                gradient_mean,   # Gradient moyen
                edge_density,    # Densité de contours
                *hist_features   # 8 bins d'histogramme
            ]
            
            # Retourne exactement 13 caractéristiques
            return np.array(visual_features[:13], dtype=np.float32)
            
        except Exception as e:
            self.logging.debug(f"Erreur extraction visuelle: {e}")
            return np.zeros(13, dtype=np.float32)  # Vecteur zéro en cas d'erreur
    
    def extract_complete_features(self, frame, contour, bounding_rect):
        """
        Extraction complète des caractéristiques avec stabilisation temporelle.
        
        Args:
            frame: Image complète de la caméra
            contour: Contour de la main détectée
            bounding_rect: Rectangle englobant
            
        Returns:
            np.array: Vecteur final de 64 caractéristiques stabilisées
        """
        try:
            x, y, w, h = bounding_rect
            
            # === VÉRIFICATIONS DE VALIDITÉ ===
            
            # Vérification des limites de l'image
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                return np.zeros(self.feature_size, dtype=np.float32)
            
            # Vérification de la taille minimale
            if w <= 0 or h <= 0:
                return np.zeros(self.feature_size, dtype=np.float32)
            
            # === EXTRACTION DE LA RÉGION D'INTÉRÊT ===
            
            # Découpage de la zone de la main
            roi = frame[y:y+h, x:x+w]
            
            # Création du masque de contour dans la ROI
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            contour_relative = contour - [x, y]  # Contour relatif à la ROI
            cv2.fillPoly(contour_mask, [contour_relative], 255)
            
            # === EXTRACTION DES CARACTÉRISTIQUES ===
            
            # Caractéristiques géométriques (17 features)
            geometric_features = self.extract_geometric_features(contour, bounding_rect)
            
            # Caractéristiques visuelles (13 features)
            visual_features = self.extract_visual_features(roi, contour_mask)
            
            # === COMBINAISON ET PADDING ===
            
            # Concaténation des deux types de features (17 + 13 = 30)
            combined_features = np.concatenate([geometric_features, visual_features])
            
            # Padding avec des zéros pour atteindre 64 features
            if len(combined_features) < self.feature_size:
                padding = np.zeros(self.feature_size - len(combined_features), dtype=np.float32)
                combined_features = np.concatenate([combined_features, padding])
            else:
                combined_features = combined_features[:self.feature_size]
            
            # === STABILISATION TEMPORELLE ===
            
            # Ajout à l'historique
            self.feature_history.append(combined_features)
            
            # Lissage avec pondération des 3 dernières frames
            if len(self.feature_history) >= 3:
                weights = np.array([0.2, 0.3, 0.5])  # Plus de poids sur la frame récente
                stabilized_features = np.average(list(self.feature_history)[-3:], weights=weights, axis=0)
                return stabilized_features
            
            # Pas assez d'historique, retourne les features directes
            return combined_features
            
        except Exception as e:
            self.logging.debug(f"Erreur extraction complète: {e}")
            return np.zeros(self.feature_size, dtype=np.float32)

# =============================================================================
# RÉSEAU DE NEURONES POUR LA RECONNAISSANCE DE POSITION
# =============================================================================

class HandPositionRecognizer:
    """
    Modèle de reconnaissance de position utilisant un réseau de neurones dense.
    
    Ce modèle prend en entrée un vecteur de 64 caractéristiques et prédit
    l'une des 9 positions de main définies. Il inclut :
    - Architecture de réseau dense avec régularisation
    - Système d'entraînement avec validation
    - Stabilisation des prédictions sur plusieurs frames
    - Sauvegarde/chargement des modèles
    - Métriques de performance et visualisations
    """
    
    def __init__(self, feature_size=64, num_classes=9):
        """
        Initialise le modèle de reconnaissance.
        
        Args:
            feature_size (int): Taille du vecteur d'entrée (64 caractéristiques)
            num_classes (int): Nombre de classes à reconnaître (9 positions)
        """
        # === PARAMÈTRES DU MODÈLE ===
        self.feature_size = feature_size    # 64 caractéristiques en entrée
        self.num_classes = num_classes      # 9 positions possibles
        self.model = None                   # Modèle TensorFlow (créé plus tard)
        self.is_trained = False             # Flag d'entraînement
        
        # === DONNÉES D'ENTRAÎNEMENT ===
        self.training_data = []             # Liste des vecteurs de features
        self.training_labels = []           # Liste des labels correspondants
        
        # === HISTORIQUES POUR STABILISATION ===
        from collections import deque
        self.prediction_history = deque(maxlen=7)    # 7 dernières prédictions
        self.confidence_history = deque(maxlen=5)    # 5 dernières confidences
        
        # === CONTRÔLE DE FRÉQUENCE ===
        self.last_prediction_time = 0           # Timestamp de la dernière prédiction
        self.last_prediction_result = (None, 0.0)  # Cache du dernier résultat
        self.prediction_interval = 0.2          # Prédiction toutes les 200ms seulement
        
        # === MÉTRIQUES DE PERFORMANCE ===
        self.total_predictions = 0              # Nombre total de prédictions
        self.confident_predictions = 0          # Nombre de prédictions confiantes
        
        # === GESTION DE TENSORFLOW ===
        self.logging = logging.getLogger(__name__)
        self.force_stop = False  # Flag d'arrêt d'urgence
        
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
        """
        Crée l'architecture du réseau de neurones.
        
        Architecture choisie :
        - Réseau dense (fully connected) à 5 couches
        - Décroissance progressive : 128 → 96 → 64 → 32 → 9
        - BatchNormalization pour stabiliser l'entraînement
        - Dropout pour éviter le surapprentissage
        - Activation ReLU pour les couches cachées
        - Softmax final pour la classification multiclasse
        
        Returns:
            bool: True si le modèle a été créé avec succès
        """
        if not self.tf_available:
            return False
            
        try:
            # === ARCHITECTURE DU RÉSEAU ===
            model = self.keras.Sequential([
                # Couche d'entrée + première couche cachée
                self.layers.Dense(128, activation='relu', input_shape=(self.feature_size,)),
                self.layers.BatchNormalization(),  # Normalisation pour stabilité
                self.layers.Dropout(0.3),          # 30% de dropout contre surapprentissage
                
                # Deuxième couche cachée
                self.layers.Dense(96, activation='relu'),
                self.layers.BatchNormalization(),
                self.layers.Dropout(0.4),          # 40% de dropout (plus agressif)
                
                # Troisième couche cachée  
                self.layers.Dense(64, activation='relu'),
                self.layers.BatchNormalization(),
                self.layers.Dropout(0.3),
                
                # Quatrième couche cachée (plus petite)
                self.layers.Dense(32, activation='relu'),
                self.layers.Dropout(0.2),          # Moins de dropout proche de la sortie
                
                # Couche de sortie (classification)
                self.layers.Dense(self.num_classes, activation='softmax')  # 9 classes
            ])
            
            # === COMPILATION DU MODÈLE ===
            model.compile(
                optimizer=self.keras.optimizers.Adam(learning_rate=0.001),  # Optimiseur Adam
                loss='sparse_categorical_crossentropy',  # Perte pour classification
                metrics=['accuracy']  # Métrique de suivi
            )
            
            self.model = model
            self.logging.info(f"✅ Modèle créé: {model.count_params()} paramètres")
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur création modèle: {e}")
            return False

    def add_training_sample(self, features, position_class):
        """
        Ajoute un échantillon d'entraînement.
        
        Args:
            features (np.array): Vecteur de 64 caractéristiques
            position_class (int): Classe de la position (0-8)
            
        Returns:
            bool: True si l'ajout a réussi
        """
        if not self.tf_available:
            return False
            
        try:
            # Vérification de la taille des features
            if len(features) != self.feature_size:
                self.logging.warning(f"Taille features incorrecte: {len(features)} vs {self.feature_size}")
                return False
                
            # Ajout aux données d'entraînement
            self.training_data.append(features.copy())
            self.training_labels.append(position_class)
            
            self.logging.info(f"📊 Échantillon ajouté: {position_class} - Total: {len(self.training_data)}")
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur ajout échantillon: {e}")
            return False

    def train_model(self, validation_split=0.2, epochs=50, plot_curves=True, save_fig=True, show_confusion=True):
        """
        Entraîne le modèle avec les données collectées.
        
        Args:
            validation_split (float): Proportion des données pour validation (20%)
            epochs (int): Nombre d'époques d'entraînement (50)
            plot_curves (bool): Génère les courbes d'apprentissage
            save_fig (bool): Sauvegarde les figures
            show_confusion (bool): Affiche la matrice de confusion
            
        Returns:
            bool: True si l'entraînement a réussi
        """
        if not self.tf_available:
            self.logging.error("❌ TensorFlow requis pour l'entraînement")
            return False
            
        try:
            # === VÉRIFICATIONS PRÉLIMINAIRES ===
            
            # Vérification du nombre d'échantillons
            if len(self.training_data) < 10:
                self.logging.warning("⚠️ Pas assez de données (min 10)")
                return False
                
            # Création du modèle si nécessaire
            if self.model is None:
                if not self.create_model():
                    return False
            
            # === PRÉPARATION DES DONNÉES ===
            
            # Conversion en arrays NumPy
            X = np.array(self.training_data, dtype=np.float32)
            y = np.array(self.training_labels, dtype=np.int32)
            
            # Logs de debugging pour vérifier les données
            self.logging.info(
                f"[DEBUG TRAIN] First train features: {X[0][:8]}... sum={np.sum(X[0]):.2f}, "
                f"min={np.min(X[0]):.2f}, max={np.max(X[0]):.2f}"
            )
            self.logging.info(f"[DEBUG TRAIN] Labels (premiers): {y[:10]}")
            
            # Vérification de la distribution des classes
            unique, counts = np.unique(y, return_counts=True)
            self.logging.info(f"[DEBUG TRAIN] Distribution des labels: {dict(zip(unique, counts))}")
            self.logging.info(f"[DEBUG TRAIN] X shape: {X.shape}, y shape: {y.shape}")

            # === SPLIT TRAIN/VALIDATION ÉQUILIBRÉ ===
            
            # Utilisation de StratifiedShuffleSplit pour garder l'équilibre des classes
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=42)
            
            for train_idx, val_idx in sss.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

            # Vérification des distributions après split
            self.logging.info(f"Train y distrib: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            self.logging.info(f"Val y distrib: {dict(zip(*np.unique(y_val, return_counts=True)))}")
            self.logging.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, "
                            f"X_val: {X_val.shape}, y_val: {y_val.shape}")

            # === CALLBACKS D'ENTRAÎNEMENT ===
            
            callbacks = [
                # Arrêt anticipé si pas d'amélioration sur validation
                self.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                ),
                # Réduction du learning rate si plateau
                self.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.7, 
                    patience=5, 
                    min_lr=1e-6
                )
            ]
            
            # === ENTRAÎNEMENT ===
            
            self.logging.info(f"🚀 Entraînement: {len(X)} échantillons, {epochs} époques")
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=min(16, len(X) // 4),  # Batch size adaptatif
                callbacks=callbacks,
                verbose=0  # Pas d'affichage pendant l'entraînement
            )
            
            # === RÉSULTATS D'ENTRAÎNEMENT ===
            
            final_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history.get('val_accuracy', [0])[-1]
            
            self.logging.info(f"✅ Entraînement terminé:")
            self.logging.info(f"   Précision: {final_accuracy:.4f}")
            self.logging.info(f"   Validation: {val_accuracy:.4f}")

            # === GÉNÉRATION DES COURBES D'APPRENTISSAGE ===
            
            if plot_curves:
                try:
                    import time
                    
                    # Création de la figure avec 2 sous-graphiques
                    plt.figure(figsize=(10, 4))
                    
                    # Graphique de précision
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['accuracy'], label='Train Acc')
                    if 'val_accuracy' in history.history:
                        plt.plot(history.history['val_accuracy'], label='Val Acc')
                    plt.title('Accuracy')
                    plt.legend()
                    
                    # Graphique de perte
                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['loss'], label='Train Loss')
                    if 'val_loss' in history.history:
                        plt.plot(history.history['val_loss'], label='Val Loss')
                    plt.title('Loss')
                    plt.legend()
                    
                    plt.tight_layout()
                    
                    # Sauvegarde (jamais d'affichage plt.show() pour éviter blocage)
                    if save_fig:
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        figname = f"training_curves_{stamp}.png"
                        plt.savefig(figname)
                        self.logging.info(f"📈 Courbes sauvegardées : {figname}")
                    
                    plt.close()  # Fermeture explicite
                    
                except Exception as e:
                    self.logging.warning(f"Erreur affichage/sauvegarde courbes : {e}")

            # === MATRICES DE CONFUSION ===
            
            from sklearn.metrics import confusion_matrix, classification_report
            import seaborn as sns
            import time
            
            # Noms des classes pour l'affichage
            class_names = [HAND_POSITIONS[i].name for i in range(self.num_classes)]
            stamp = time.strftime("%Y%m%d_%H%M%S")

            # Matrice de confusion sur les données d'entraînement
            train_preds = np.argmax(self.model.predict(X_train, verbose=0), axis=1)
            cm = confusion_matrix(y_train, train_preds)
            
            plt.figure(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Prédit')
            plt.ylabel('Vrai')
            plt.title("Matrice de confusion (train)")
            plt.tight_layout()
            
            figname_cm = f"confusion_matrix_train_{stamp}.png"
            plt.savefig(figname_cm)
            self.logging.info(f"🟦 Matrice de confusion (train) sauvegardée : {figname_cm}")
            plt.close()
            
            # Rapport de classification détaillé
            rep = classification_report(y_train, train_preds, target_names=class_names, digits=3)
            self.logging.info("\n" + rep)

            # Matrice de confusion sur les données de validation
            if len(X_val) > 0:
                val_preds = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
                cm_val = confusion_matrix(y_val, val_preds)
                
                plt.figure(figsize=(8, 7))
                sns.heatmap(cm_val, annot=True, fmt="d", cmap="Oranges", 
                           xticklabels=class_names, yticklabels=class_names)
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

            # === ANALYSE ÉCHANTILLON PAR ÉCHANTILLON ===
            
            # Performance sur chaque échantillon d'entraînement
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

            # === LOGS DE DEBUGGING FINAL ===
            
            train_acc_full = np.mean(train_preds_full == y)
            self.logging.info(f"[DEBUG TRAIN] Accuracy (full train set): {train_acc_full:.4f}")
            self.logging.info(f"[DEBUG TRAIN] Sample preds (train set): {train_preds_full[:10]} vs {y[:10]}")
            
            # Test de prédiction sur le premier échantillon
            test_pred = self.model.predict(X[0].reshape(1, -1), verbose=0)[0]
            self.logging.info(f"[DEBUG TRAIN] Pred 1st sample: class={np.argmax(test_pred)}, raw={test_pred}")

            # === FINALISATION ===
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur entraînement: {e}")
            return False

    def predict_position(self, features, use_stabilization=True):
        """
        Prédit la position de la main à partir des caractéristiques.
        
        Args:
            features (np.array): Vecteur de 64 caractéristiques
            use_stabilization (bool): Active la stabilisation temporelle
            
        Returns:
            tuple: (classe_prédite, confiance) ou (None, confiance) si pas assez confiant
        """
        # Vérifications préliminaires
        if not self.tf_available or self.model is None or not self.is_trained:
            return None, 0.0
            
        try:
            # Vérification de la taille des features
            if len(features) != self.feature_size:
                return None, 0.0
            
            # === CONTRÔLE DE FRÉQUENCE ===
            
            current_time = time.time()
            
            # Limite la fréquence des prédictions (performance)
            if current_time - self.last_prediction_time < self.prediction_interval:
                return self.last_prediction_result
            
            # === PRÉDICTION ===
            
            # Reshape pour le modèle (batch de 1)
            features_batch = features.reshape(1, -1)    
            
            # Prédiction du modèle
            prediction = self.model.predict(features_batch, verbose=0, batch_size=1)[0]
            
            # Classe avec la plus haute probabilité
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Mise à jour des timestamps
            self.last_prediction_time = current_time
            self.total_predictions += 1
            
            # === STABILISATION TEMPORELLE ===
            
            if use_stabilization:
                # Ajout à l'historique
                self.prediction_history.append((predicted_class, confidence))
                self.confidence_history.append(confidence)
                
                # Stabilisation sur les 3 dernières prédictions
                if len(self.prediction_history) >= 3:
                    from collections import Counter
                    
                    # Classes et confidences récentes
                    recent_classes = [p[0] for p in list(self.prediction_history)[-3:]]
                    recent_confidences = [p[1] for p in list(self.prediction_history)[-3:]]
                    
                    # Classe la plus fréquente
                    class_counts = Counter(recent_classes)
                    most_common_class, count = class_counts.most_common(1)[0]
                    
                    # Si une classe domine (2/3 ou plus), on l'utilise
                    if count >= 2:
                        class_confidences = [conf for cls, conf in zip(recent_classes, recent_confidences)
                                           if cls == most_common_class]
                        avg_confidence = np.mean(class_confidences)
                        predicted_class = most_common_class
                        confidence = avg_confidence
            
            # === VALIDATION AVEC SEUIL ===
            
            # Vérification que la classe prédite existe dans notre dictionnaire
            if predicted_class in HAND_POSITIONS:
                threshold = 0.3  # Seuil de confiance minimal
                
                if confidence >= threshold:
                    self.confident_predictions += 1
                    result = (predicted_class, confidence)
                    self.last_prediction_result = result
                    return result
            
            # Prédiction pas assez confiante
            result = (None, confidence)
            self.last_prediction_result = result
            return result
            
        except Exception as e:
            self.logging.debug(f"Erreur prédiction: {e}")
            return None, 0.0

    def get_prediction_stats(self):
        """
        Retourne les statistiques de performance des prédictions.
        
        Returns:
            str: Chaîne formatée avec les stats
        """
        if self.total_predictions == 0:
            return "Aucune prédiction"
            
        # Taux de prédictions confiantes
        confidence_rate = (self.confident_predictions / self.total_predictions) * 100
        
        # Confiance moyenne récente
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0
        
        return f"Confiance: {confidence_rate:.1f}% | Moy: {avg_confidence:.2f}"

    def save_model(self, filepath="hand_position_model"):
        """
        Sauvegarde le modèle et les données d'entraînement.
        
        Args:
            filepath (str): Chemin de base pour les fichiers
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        if not self.tf_available or self.model is None:
            return False
            
        try:
            # === SAUVEGARDE DU MODÈLE TENSORFLOW ===
            
            model_file = f"{filepath}_model.keras"
            self.model.save(model_file)
            self.logging.info(f"✅ Modèle sauvegardé dans {model_file}")
            
            # === SAUVEGARDE DES DONNÉES ASSOCIÉES ===
            
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
        """
        Charge un modèle et ses données depuis les fichiers.
        
        Args:
            filepath (str): Chemin de base des fichiers
            
        Returns:
            bool: True si le chargement a réussi
        """
        if not self.tf_available:
            return False
            
        try:
            # === CHARGEMENT DU MODÈLE TENSORFLOW ===
            
            # Priorité au format .keras (nouveau)
            model_file = f"{filepath}_model.keras"
            if os.path.exists(model_file):
                self.model = self.keras.models.load_model(model_file)
                self.logging.info(f"✅ Modèle chargé: {model_file}")
            else:
                # Fallback sur l'ancien format .h5
                old_model_path = f"{filepath}_model.h5"
                if os.path.exists(old_model_path):
                    self.model = self.keras.models.load_model(old_model_path)
                    self.logging.info(f"✅ Ancien modèle chargé: {old_model_path}")
                else:
                    self.logging.warning(f"⚠️ Aucun modèle trouvé: {model_file} ou {old_model_path}")
                    return False
            
            # === CHARGEMENT DES DONNÉES ASSOCIÉES ===
            
            data_path = f"{filepath}_data.pkl"
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    import pickle
                    data = pickle.load(f)
                    
                # Restauration des attributs
                self.training_data = data.get('training_data', [])
                self.training_labels = data.get('training_labels', [])
                self.feature_size = data.get('feature_size', 64)
                self.num_classes = data.get('num_classes', 9)
                self.is_trained = data.get('is_trained', False)
                self.total_predictions = data.get('total_predictions', 0)
                self.confident_predictions = data.get('confident_predictions', 0)
                
                self.logging.info(f"✅ Données chargées: {len(self.training_data)} échantillons")
                
            return True
            
        except Exception as e:
            self.logging.error(f"❌ Erreur chargement: {e}")
            return False

# =============================================================================
# DÉTECTEUR PRINCIPAL AVEC INTELLIGENCE ARTIFICIELLE
# =============================================================================

class OptimizedBicolorGloveDetectorWithAI:
    """
    Détecteur principal combinant vision par ordinateur et intelligence artificielle.
    
    Cette classe intègre :
    1. Détection de gant coloré (rouge/orange) avec OpenCV
    2. Reconnaissance de position avec IA (TensorFlow)
    3. Génération de commandes pour le drone
    4. Interface utilisateur temps réel
    5. Système de sécurité multicouche
    
    Le détecteur fonctionne en plusieurs modes :
    - "detection" : Détection simple du gant
    - "training" : Collecte d'échantillons pour entraîner l'IA
    - "recognition" : Reconnaissance active des gestes avec contrôle drone
    """
    
    def __init__(self, feature_size=64, num_classes=len(HAND_POSITIONS)):
        """
        Initialise le détecteur avec tous ses composants.
        
        Args:
            feature_size (int): Taille du vecteur de caractéristiques (64)
            num_classes (int): Nombre de positions à reconnaître (9)
        """
        
        # === CONTRÔLE DE FRÉQUENCE DES PRÉDICTIONS ===
        self.last_prediction_time = 0                    # Timestamp dernière prédiction
        self.last_prediction_result = (None, 0.0)       # Cache du dernier résultat
        self.prediction_interval = 0.3                   # Prédiction toutes les 300ms seulement

        # === HISTORIQUES POUR STABILISATION ===
        self.detection_history = deque(maxlen=15)        # 15 dernières détections gant
        self.stable_detections = deque(maxlen=5)         # 5 détections stables
        self.confidence_threshold = 3                    # Seuil pour validation stabilité
        
        # === PARAMÈTRES DE DÉTECTION COULEUR ===
        self.min_area = 200                              # Aire minimale du contour
        self.max_area = 120000                           # Aire maximale du contour
        self.min_contour_points = 8                      # Points minimum du contour
        
        # === HISTORIQUES POUR ADAPTATION AUTOMATIQUE ===
        self.color_balance_history = deque(maxlen=20)    # Balance couleur
        self.red_orange_ratio_history = deque(maxlen=10) # Ratio rouge/orange
        
        # === ÉLÉMENTS STRUCTURANTS POUR MORPHOLOGIE ===
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # === SYSTÈME DE ZOOM ADAPTATIF ===
        self.zoom_factor = 1.0                           # Facteur de zoom actuel
        self.target_zoom = 1.0                           # Zoom cible
        self.zoom_smooth_factor = 0.12                   # Lissage du zoom
        self.zoom_min = 1.0                              # Zoom minimum
        self.zoom_max = 4.5                              # Zoom maximum
        
        # === HISTORIQUES POUR MÉTRIQUES ===
        self.area_reference = 2800                       # Aire de référence pour zoom
        self.area_history = deque(maxlen=15)             # Historique des aires
        self.quality_scores = deque(maxlen=10)           # Scores de qualité
        
        # === ZONE DE RECHERCHE INTELLIGENTE ===
        self.search_zone = None                          # Zone de recherche préférentielle
        self.zone_tracking = deque(maxlen=5)             # Suivi de zone
        
        # === COMPTEURS ET MÉTRIQUES ===
        self.frame_count = 0                             # Nombre de frames traitées
        self.detection_count = 0                         # Nombre de détections réussies
        self.quality_count = 0                           # Nombre de détections de qualité
        self.zoom_adjustments = 0                        # Nombre d'ajustements de zoom
        self.fps_start_time = time.time()                # Timestamp de début pour FPS
        self.current_fps = 0                             # FPS actuel
        
        # === ADAPTATION AUTOMATIQUE DE L'EXPOSITION ===
        self.brightness_history = deque(maxlen=10)       # Historique luminosité
        self.auto_exposure_factor = 1.0                  # Facteur d'exposition automatique
        
        # === COMPOSANTS INTELLIGENCE ARTIFICIELLE ===
        self.ai_enabled = TF_AVAILABLE                   # IA activée si TensorFlow dispo
        self.feature_extractor = AdvancedHandFeatureExtractor()  # Extracteur de features
        self.position_recognizer = HandPositionRecognizer()      # Réseau de neurones
        
        # === DONNÉES DE DÉTECTION POUR IA ===
        self.last_detected_contour = None               # Dernier contour détecté
        self.last_detected_area = 0                     # Dernière aire détectée
        self.last_bounding_rect = None                   # Dernier rectangle englobant
        
        # === ÉTATS ET MODES IA ===
        self.ai_mode = "detection"                       # Mode: "detection", "training", "recognition"
        self.training_class = 0                          # Classe en cours d'entraînement
        self.training_countdown = 0                      # Compteur avant capture
        self.training_samples_per_class = 150            # Échantillons par position
        
        # === POSITION ACTUELLE RECONNUE ===
        self.current_position = None                     # Position actuellement reconnue
        self.current_position_confidence = 0.0          # Confiance de la position
        
        # === MÉTRIQUES IA ===
        self.ai_frame_count = 0                          # Frames traitées par l'IA
        self.ai_position_detections = 0                  # Positions détectées par l'IA
        
        # === CONTRÔLE DRONE ===
        self.drone_commands_enabled = False              # Commandes drone activées/désactivées
        self.last_command_time = 0                       # Timestamp dernière commande
        self.command_cooldown = 5.0                      # Cooldown entre commandes (5 sec)

        # === STABILISATION COMMANDES DRONE ===
        self.position_history = deque(maxlen=3)          # Historique positions pour stabilité

        # === LOGGING ===
        self.logging = logging.getLogger(__name__)
        
        # === INITIALISATION IA ===
        if self.ai_enabled:
            self._initialize_ai()
        else:
            self.logging.warning("⚠️ IA désactivée - TensorFlow requis")
    
    def _initialize_ai(self):
        """
        Initialise les composants d'intelligence artificielle.
        
        Tente de charger un modèle existant, sinon prépare pour un nouvel entraînement.
        """
        try:
            # Tentative de chargement d'un modèle existant
            model_path = "hand_position_model"
            if os.path.exists(f"{model_path}_model"):
                if self.position_recognizer.load_model(model_path):
                    self.ai_mode = "recognition"  # Passe en mode reconnaissance
                    self.logging.info("🤖 Modèle IA chargé - Mode reconnaissance")
                else:
                    self.logging.info("🤖 Nouveau modèle IA - Prêt pour entraînement")
            else:
                self.logging.info("🤖 Aucun modèle existant - Utilisez 't' pour entraîner")
            
        except Exception as e:
            self.logging.error(f"Erreur initialisation IA: {e}")
            self.ai_enabled = False
    
    def detect_glove_optimized(self, frame):
        """
        Détection optimisée du gant rouge/orange dans l'image.
        
        Cette méthode implémente un pipeline de vision par ordinateur pour détecter
        un gant coloré en temps réel :
        
        1. Conversion dans l'espace colorimétrique HSV
        2. Création de masques pour rouge et orange
        3. Opérations morphologiques pour nettoyer
        4. Détection et filtrage des contours
        5. Sélection du meilleur candidat
        
        Args:
            frame (np.array): Image d'entrée de la caméra
            
        Returns:
            tuple: (image_traitée, détection_réussie)
        """
        # Vérification de validité de l'image
        if frame is None:
            return frame, False

        # Sauvegarde de l'image originale
        original_frame = frame.copy()
        self.frame_count += 1

        try:
            # === CONVERSION D'ESPACE COLORIMÉTRIQUE ===
            
            # Conversion BGR (Blue-Green-Red) vers HSV (Hue-Saturation-Value)
            # HSV est plus robuste pour la détection de couleur que RGB
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # === CRÉATION DES MASQUES DE COULEUR ===
            
            # Masque rouge (2 intervalles car le rouge wrappe autour de 0/180 en HSV)
            # Premier intervalle : rouge pur (0-8°)
            mask_red1 = cv2.inRange(hsv, np.array([0, 140, 120]), np.array([8, 255, 255]))
            # Deuxième intervalle : rouge profond (172-180°)
            mask_red2 = cv2.inRange(hsv, np.array([172, 140, 120]), np.array([180, 255, 255]))

            # Masque orange (entre rouge et jaune, 8-18°)
            mask_orange = cv2.inRange(hsv, np.array([8, 160, 140]), np.array([18, 255, 255]))

            # === COMBINAISON DES MASQUES ===
            
            # Union des deux masques rouges
            mask_combined = cv2.bitwise_or(mask_red1, mask_red2)
            # Ajout de l'orange
            mask_combined = cv2.bitwise_or(mask_combined, mask_orange)

            # === NETTOYAGE MORPHOLOGIQUE ===
            
            # Fermeture morphologique pour combler les petits trous
            # (connecte les régions proches de même couleur)
            mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, self.kernel_small)

            # === DÉTECTION DES CONTOURS ===
            
            # Recherche des contours dans le masque binaire
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrage par aire minimale (élimine le bruit)
            contours = [c for c in contours if cv2.contourArea(c) > 500]

            # === SÉLECTION DU MEILLEUR CONTOUR ===
            
            best_contour = None
            best_area = 0
            quality_score = 0

            # Recherche du contour avec la plus grande aire
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > best_area:
                    best_contour = contour
                    best_area = area
                    # Score de qualité basé sur l'aire (normalisé par rapport à référence)
                    quality_score = min(area / 2800, 1.0)

            # === VALIDATION DE LA DÉTECTION ===
            
            detected = best_contour is not None

            if detected:
                # Sauvegarde des informations de détection pour l'IA
                self.last_detected_contour = best_contour.copy()
                self.last_detected_area = best_area
                self.last_bounding_rect = cv2.boundingRect(best_contour)

            # === FINALISATION ===
            
            # Pas de lissage du contour pour garder les détails des doigts
            return self._finalize_detection(original_frame, detected, best_contour, best_area, quality_score)

        except Exception as e:
            self.logging.debug(f"Erreur détection: {e}")
            return original_frame, False

    def _finalize_detection(self, frame, detected, contour, area, quality_score):
        """
        Finalise la détection en intégrant l'analyse IA et les visualisations.
        
        Args:
            frame: Image à traiter
            detected: Booléen de détection
            contour: Contour détecté (peut être None)
            area: Aire du contour
            quality_score: Score de qualité de la détection
            
        Returns:
            tuple: (image_finale, détection_réussie)
        """
        try:
            # === MISE À JOUR DES HISTORIQUES ===
            
            # Ajout de la détection à l'historique
            self.detection_history.append(detected)
            if detected:
                self.detection_count += 1
            
            # === ANALYSE INTELLIGENCE ARTIFICIELLE ===
            
            if detected and contour is not None and self.ai_enabled:
                # Analyse de la position de la main par l'IA
                position, confidence = self._analyze_hand_position_ai(frame, contour)

                # Mise à jour de la position actuelle seulement en mode reconnaissance
                if (
                    self.ai_mode == "recognition"
                    and position
                    and position != "training_complete"
                    and position in [p.name for p in HAND_POSITIONS.values()]
                ):
                    self.current_position = position
                    self.current_position_confidence = confidence
                    self.ai_position_detections += 1

                    # Exécution des commandes drone si activées
                    if self.drone_commands_enabled:
                        self._execute_drone_command(position, confidence)
                else:
                    self.current_position = None
                    self.current_position_confidence = 0.0

                # Ajout de la visualisation IA sur l'image
                frame = self._draw_ai_overlay(frame, contour, position, confidence)

            # === VISUALISATION DE LA DÉTECTION ===
            
            if detected and contour is not None:
                self._draw_detection_overlay(frame, contour, area, quality_score)
            
            # === INTERFACE UTILISATEUR COMPLÈTE ===
            
            result_frame = self._create_complete_overlay(frame, detected, area, quality_score)
            
            return result_frame, detected
            
        except Exception as e:
            self.logging.debug(f"Erreur finalisation: {e}")
            return frame, False
    
    def _analyze_hand_position_ai(self, frame, contour):
        """
        Analyse la position de la main en utilisant l'intelligence artificielle.
        
        Args:
            frame: Image complète
            contour: Contour de la main détectée
            
        Returns:
            tuple: (position_name, confidence) ou (None, 0.0)
        """
        try:
            # Vérifications préliminaires
            if not self.ai_enabled or not self.feature_extractor or not self.position_recognizer:
                self.logging.info("[DEBUG] IA non activée ou modules manquants")
                return None, 0.0

            # Calcul du rectangle englobant
            bounding_rect = cv2.boundingRect(contour)

            # === MODE ENTRAÎNEMENT ===
            if self.ai_mode == "training":
                return self._handle_training_mode(frame, contour, bounding_rect)

            # === MODE RECONNAISSANCE ===
            elif self.ai_mode == "recognition" and self.position_recognizer.is_trained:
                # Extraction des caractéristiques de la main
                features = self.feature_extractor.extract_complete_features(
                    frame, contour, bounding_rect
                )
                self.logging.info(f"[DEBUG] Features: {features[:8]}... sum={np.sum(features):.2f}")

                # Prédiction par le réseau de neurones
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
        """
        Gère le mode d'entraînement de l'IA.
        
        Ce mode collecte automatiquement des échantillons pour chaque position :
        1. Vérifie la qualité de la détection
        2. Donne des indications à l'utilisateur
        3. Capture les échantillons quand les conditions sont bonnes
        4. Passe automatiquement à la position suivante
        
        Args:
            frame: Image complète
            contour: Contour de la main
            bounding_rect: Rectangle englobant
            
        Returns:
            tuple: (status_message, progress_value)
        """
        try:
            # === VÉRIFICATION DE FIN D'ENTRAÎNEMENT ===
            
            if self.training_class >= len(HAND_POSITIONS):
                # Toutes les positions ont été entraînées, lancement du training
                self._start_model_training()
                self.current_training_distance_msg = ""
                return "training_complete", 1.0

            # Position actuelle à entraîner
            position = HAND_POSITIONS[self.training_class]

            # === ANALYSE DE QUALITÉ DU CONTOUR ===
            
            # Seuils de qualité pour une bonne détection
            min_area = 2000        # Aire minimale pour être sûr que le gant est bien visible
            min_extent = 0.18      # Proportion du contour par rapport au bounding rect
            min_solidity = 0.75    # Évite les contours "troués" ou mal formés
            min_ratio = 0.1        # Ratio largeur/hauteur raisonnable

            # Calcul des métriques de qualité
            x, y, w, h = bounding_rect
            area = cv2.contourArea(contour)
            extent = area / (w * h) if w * h > 0 else 0
            
            # Calcul de la solidité (compacité du contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Ratio d'aspect (doit être raisonnable pour une main)
            aspect_ratio = w / float(h) if h > 0 else 0
            aspect_ok = min_ratio < aspect_ratio < 1.0/min_ratio

            # === VALIDATION DE LA QUALITÉ ===
            
            # Si la main n'est pas bien détectée
            if (area < min_area or extent < min_extent or solidity < min_solidity or not aspect_ok):
                self.current_training_distance_msg = "Gant mal détecté : repositionnez bien la main"
                self.training_countdown = 15   # Pause pour repositionner
                return f"training_{position.name}", 0.0

            # === FEEDBACK DISTANCE À LA CAMÉRA ===
            
            # Guidance pour la distance optimale
            if area > 11000:
                distance_msg = "Reculez un peu la main"
            elif area < 2500:
                distance_msg = "Avancez la main"
            else:
                distance_msg = "Distance OK"
            self.current_training_distance_msg = distance_msg

            # === GESTION DU COUNTDOWN ===
            
            # Attente avant capture (laisse le temps de se positionner)
            if self.training_countdown > 0:
                self.training_countdown -= 1
                return f"training_{position.name}", self.training_countdown / 60.0

            # === CAPTURE D'ÉCHANTILLON ===
            
            # Extraction des caractéristiques
            features = self.feature_extractor.extract_complete_features(
                frame, contour, bounding_rect
            )
            
            # Ajout aux données d'entraînement
            success = self.position_recognizer.add_training_sample(features, self.training_class)
            
            if success:
                # Comptage des échantillons pour cette position
                samples_count = len([l for l in self.position_recognizer.training_labels 
                                   if l == self.training_class])
                self.current_training_distance_msg = "✅ Échantillon capturé!"

                # === PASSAGE À LA POSITION SUIVANTE ===
                
                # Si assez d'échantillons pour cette position
                if samples_count >= self.training_samples_per_class:
                    self.training_class += 1
                    self.training_countdown = 60  # Pause pour changer de position
                    
                    if self.training_class < len(HAND_POSITIONS):
                        next_position = HAND_POSITIONS[self.training_class]
                        self.logging.info(f"📝 Prochain geste attendu : {next_position.description}")
                        self.current_training_distance_msg = ""
                    else:
                        self.training_countdown = 30  # Pause avant fin
                        
                    return f"captured_{position.name}", 1.0

                # Sinon continue la capture pour cette position
                self.training_countdown = 18   # Pause entre captures
                return f"captured_{position.name}", 1.0

            # Pas de capture, continue l'affichage
            return f"training_{position.name}", 0.0

        except Exception as e:
            self.logging.error(f"Erreur mode entraînement: {e}")
            self.current_training_distance_msg = ""
            return None, 0.0
    
    def _execute_drone_command(self, position, confidence):
        """
        Exécute une commande de drone basée sur la position détectée.
        
        Cette méthode implémente un système de sécurité multicouche :
        1. Vérification du cooldown temporel
        2. Validation de la stabilité de la position
        3. Vérification des seuils de confiance
        4. Contrôle de la distance du gant
        5. Vérification de l'état de vol du drone
        
        Args:
            position (str): Nom de la position détectée
            confidence (float): Confiance de la prédiction
            
        Returns:
            bool: True si la commande a été exécutée
        """
        # Vérification de l'arrêt manuel
        if getattr(self, "force_stop", False):
            self.logging.info("[DRONE CMD] Commandes bloquées (arrêt manuel activé)")
            return False
            
        try:
            import time
            current_time = time.time()
            
            self.logging.info(f"[DRONE CMD] DEMANDE: position={position}, confidence={confidence:.3f}, t={current_time:.3f}")

            # === VÉRIFICATION DU COOLDOWN ===
            
            # Évite les commandes trop rapprochées (sécurité)
            if current_time - self.last_command_time < self.command_cooldown:
                self.logging.info("[DRONE CMD] Cooldown actif, commande ignorée.")
                return False

            # === VÉRIFICATION DE LA STABILITÉ ===
            
            # Ajout à l'historique des positions
            self.position_history.append((position, confidence))
            
            # Nécessite au moins 2 détections identiques récentes
            if list(self.position_history).count((position, confidence)) < 2:
                self.logging.info(f"[DRONE CMD] Position '{position}' pas assez stable, commande ignorée.")
                return False

            # === SEUILS DE CONFIANCE ADAPTATIFS ===
            
            # Seuils différents selon la criticité de la commande
            thresholds = {
                "avancer": 0.75,          # Mouvements de base
                "reculer": 0.75,
                "monter": 0.75,
                "descendre": 0.75,
                "droite": 0.75,
                "gauche": 0.75,
                "rotation_gauche": 0.75,  # Rotations
                "rotation_droite": 0.75,
                "poing": 0.75,            # Arrêt d'urgence (plus permissif)
            }
            
            threshold = thresholds.get(position, 0.85)  # Seuil par défaut élevé
            self.logging.info(f"[DRONE CMD] Seuil pour {position}: {threshold}")

            # === VÉRIFICATION DE LA DISTANCE ===
            
            # Pour les mouvements directionnels, le gant doit être assez proche
            MIN_GLOVE_AREA = 3500
            if position in ["avancer", "reculer", "droite", "gauche"]:
                area = getattr(self, "last_detected_area", None)
                if area is not None and area < MIN_GLOVE_AREA:
                    self.logging.warning(f"[DRONE CMD] Gant trop loin (area={area:.0f}), commande ignorée.")
                    return False

            # === VALIDATION DE LA CONFIANCE ===
            
            if confidence < threshold:
                self.logging.info(f"[DRONE CMD] Confiance trop basse ({confidence:.3f} < {threshold}), commande ignorée.")
                return False

            # === VÉRIFICATION DE L'INSTANCE DRONE ===
            
            bebop = getattr(self, "bebop", None)
            if bebop is None:
                self.logging.warning("[DRONE CMD] Aucune instance drone !")
                return False

            # Debug de l'état du drone
            sensors_dict = vars(bebop.sensors) if hasattr(bebop, 'sensors') else {}
            self.logging.info(f"[DRONE CMD] bebop OK: {bebop}")
            self.logging.info(f"[DRONE CMD] sensors: {sensors_dict}")

            # État de vol actuel
            flying_state = getattr(bebop.sensors, "flying_state", "unknown")
            self.logging.info(f"[DRONE CMD] flying_state = '{flying_state}'")

            # === EXÉCUTION DES COMMANDES ===

            # ARRÊT D'URGENCE (priorité absolue)
            if position == "poing":
                self.logging.warning("[DRONE CMD] 🚨 ARRÊT D'URGENCE - Poing détecté")
                if flying_state == "landed":
                    self.logging.info("[DRONE CMD] Déjà posé. Aucun mouvement.")
                else:
                    self.logging.info("[DRONE CMD] safe_land appelé !")
                    bebop.safe_land(10)
                self.last_command_time = current_time
                return True

            # COMMANDES DE MOUVEMENT (seulement si en vol)
            elif position == "avancer" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ➡️ Avancer")
                bebop.fly_direct(roll=0, pitch=30, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "reculer" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬅️ Reculer")
                bebop.fly_direct(roll=0, pitch=-30, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "monter" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬆️ Monter")
                bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "descendre" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬇️ Descendre")
                bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "droite" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ➡️ Droite")
                bebop.fly_direct(roll=20, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "gauche" and flying_state != "landed":
                self.logging.info("[DRONE CMD] ⬅️ Gauche")
                bebop.fly_direct(roll=-20, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
                self.last_command_time = current_time
                return True

            elif position == "rotation_gauche" and flying_state != "landed":
                self.logging.info("[DRONE CMD] 🔄 Rotation GAUCHE")
                bebop.fly_direct(roll=0, pitch=0, yaw=-25, vertical_movement=0, duration=0.4)
                self.last_command_time = current_time
                return True

            elif position == "rotation_droite" and flying_state != "landed":
                self.logging.info("[DRONE CMD] 🔄 Rotation DROITE")
                bebop.fly_direct(roll=0, pitch=0, yaw=25, vertical_movement=0, duration=0.4)
                self.last_command_time = current_time
                return True

            else:
                self.logging.info(f"[DRONE CMD] Position {position} non exécutée ou condition non remplie "
                                f"(flying_state={flying_state}, confiance={confidence:.2f})")
                return False

        except Exception as e:
            self.logging.error(f"[DRONE CMD] ❌ Erreur commande drone: {e}")
            import traceback
            self.logging.error(f"[DRONE CMD] Traceback: {traceback.format_exc()}")
            return False

    def _draw_ai_overlay(self, frame, contour, position, confidence):
        """
        Dessine la visualisation de l'analyse IA sur l'image.
        
        Args:
            frame: Image à modifier
            contour: Contour de la main
            position: Position reconnue
            confidence: Confiance de la prédiction
            
        Returns:
            np.array: Image avec overlay IA
        """
        try:
            # Vérification que c'est une vraie position reconnue
            if (
                position
                and confidence > 0
                and position != "training_complete"
                and position in [p.name for p in HAND_POSITIONS.values()]
            ):
                # === COULEUR SELON LA CONFIANCE ===
                
                if confidence > 0.8:
                    color = (0, 255, 0)      # Vert : très confiant
                elif confidence > 0.6:
                    color = (0, 255, 255)    # Jaune : moyennement confiant
                else:
                    color = (0, 150, 255)    # Orange : peu confiant

                # === CALCUL DU CENTRE DE LA MAIN ===
                
                # Utilisation des moments pour trouver le centroïde
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # Centre x
                    cy = int(M["m01"] / M["m00"])  # Centre y

                    # === CERCLE DE CONFIANCE ===
                    
                    # Rayon proportionnel à la confiance
                    radius = int(15 + confidence * 25)
                    cv2.circle(frame, (cx, cy), radius, color, 3)

                    # === TEXTE POSITION ===
                    
                    # Nom de la position au-dessus du centre
                    cv2.putText(frame, position.upper(), (cx - 30, cy - 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # === VALEUR DE CONFIANCE ===
                    
                    # Confiance en dessous du centre
                    cv2.putText(frame, f"{confidence:.2f}", (cx - 15, cy + 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                              
            return frame

        except Exception as e:
            self.logging.debug(f"Erreur overlay IA: {e}")
            return frame
    
    def _draw_detection_overlay(self, frame, contour, area, quality_score):
        """
        Dessine la visualisation de la détection de base avec feedback qualité.
        
        Args:
            frame: Image à modifier
            contour: Contour détecté
            area: Aire du contour
            quality_score: Score de qualité
        """
        try:
            # === DÉTERMINATION DE LA COULEUR ===
            
            # Couleur par défaut (vert = bonne qualité)
            color = (0, 255, 0)

            # En mode entraînement : si message d'erreur, forcer rouge
            if hasattr(self, "current_training_distance_msg"):
                msg = self.current_training_distance_msg.lower()
                if "mal détecté" in msg or "repositionnez" in msg or ("gant" in msg and "mal" in msg):
                    color = (0, 0, 255)  # Rouge pour erreur

            # Sinon, code couleur selon score de qualité
            if color == (0, 255, 0):
                if quality_score > 0.7:
                    color = (0, 255, 0)    # Vert : excellente qualité
                elif quality_score > 0.5:
                    color = (0, 255, 255)  # Jaune : qualité moyenne
                else:
                    color = (0, 0, 255)    # Rouge : mauvaise qualité

            # === DESSIN DU CONTOUR ===
            
            # Contour principal
            cv2.drawContours(frame, [contour], -1, color, 3)

            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # === INFORMATIONS TECHNIQUES ===
            
            # Affichage des métriques
            cv2.putText(frame, f"Q:{quality_score:.2f} A:{int(area)}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # === MESSAGE D'ERREUR SI MAUVAISE DÉTECTION ===
            
            # Affichage du message de problème en mode entraînement
            if hasattr(self, "current_training_distance_msg"):
                msg = self.current_training_distance_msg
                if "mal détecté" in msg or "repositionnez" in msg:
                    cv2.putText(frame, msg, (x, y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            self.logging.debug(f"Erreur visualisation détection: {e}")
    
    def _create_complete_overlay(self, frame, detected, area, quality_score):
        """
        Crée l'interface utilisateur complète avec toutes les informations.
        
        Args:
            frame: Image de base
            detected: Statut de détection
            area: Aire détectée
            quality_score: Score de qualité
            
        Returns:
            np.array: Image avec interface complète
        """
        try:
            h, w = frame.shape[:2]

            # === STATUS PRINCIPAL ===
            
            if detected:
                if self.current_position:
                    status = f"🤖 GANT + {self.current_position.upper()} ({self.current_position_confidence:.2f})"
                    status_color = (0, 255, 0)  # Vert : détection + reconnaissance
                else:
                    status = f"🎯 GANT DÉTECTÉ (Q:{quality_score:.2f})"
                    status_color = (0, 255, 255)  # Jaune : détection seulement
            else:
                status = f"🔍 RECHERCHE GANT"
                status_color = (100, 100, 255)  # Bleu : recherche

            cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # === INFORMATIONS MODE IA ===
            
            if self.ai_enabled:
                ai_status = f"IA: Mode {self.ai_mode}"
                
                # Informations spécifiques au mode entraînement
                if self.ai_mode == "training" and self.training_class < len(HAND_POSITIONS):
                    position = HAND_POSITIONS[self.training_class]
                    samples = len([l for l in self.position_recognizer.training_labels 
                                 if l == self.training_class])
                    ai_status += f" | {position.name} ({samples}/{self.training_samples_per_class})"
                    
                    if self.training_countdown > 0:
                        ai_status += f" | Capture dans {self.training_countdown//30 + 1}s"
                        
                    # Description du geste attendu
                    instruction = position.description
                    cv2.putText(frame, f"Geste attendu : {instruction}", (10, 95), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
                               
                elif self.position_recognizer.is_trained:
                    ai_status += " (Entraînée)"

                cv2.putText(frame, ai_status, (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            else:
                cv2.putText(frame, "IA: TensorFlow requis", (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

            # === STATUS COMMANDES DRONE ===
            
            if self.drone_commands_enabled:
                drone_status = "🚁 COMMANDES ACTIVES"
                drone_color = (0, 255, 0)  # Vert : danger actif
            else:
                drone_status = "🚁 Commandes désactivées"
                drone_color = (100, 100, 100)  # Gris : sécurisé

            cv2.putText(frame, drone_status, (10, 95 if self.ai_mode != "training" else 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, drone_color, 1)

            # === MESSAGE DE DISTANCE EN MODE ENTRAÎNEMENT ===
            
            if self.ai_mode == "training" and hasattr(self, 'current_training_distance_msg'):
                msg = self.current_training_distance_msg
                if msg:
                    cv2.putText(frame, f"Distance : {msg}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            # === STATISTIQUES PERFORMANCE IA ===
            
            if self.ai_enabled and self.ai_frame_count > 0:
                detection_rate = (self.ai_position_detections / max(self.ai_frame_count, 1)) * 100
                ai_stats = f"IA: {detection_rate:.1f}% positions détectées"
                cv2.putText(frame, ai_stats, (10, h - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

            # === AIDE POSITIONS DISPONIBLES ===
            
            if self.ai_mode == "recognition":
                positions_text = (
                    "Positions : avancer(main doigts bas), reculer(paume doigts écartés), "
                    "monter(pouce haut), descendre(pouce bas), droite(index gauche), "
                    "gauche(index droite), rot.gauche(dos main droite), rot.droite(dos main gauche), urgence(poing)"
                )
                cv2.putText(frame, positions_text, (10, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # === HISTORIQUE DES DÉTECTIONS ===
            
            # Visualisation des 15 dernières détections
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-15:]])
            cv2.putText(frame, f"Historique: {history}", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # === STATISTIQUES D'ENTRAÎNEMENT ===
            
            # Affichage du nombre d'échantillons par classe en mode training
            if self.ai_mode == "training" and hasattr(self, 'position_recognizer'):
                ylabels = self.position_recognizer.training_labels
                txt = "Échantillons : "
                for i, pos in HAND_POSITIONS.items():
                    n = len([y for y in ylabels if y == i])
                    txt += f"{pos.name}({n}) "
                cv2.putText(frame, txt, (10, h - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
                
            return frame

        except Exception as e:
            self.logging.debug(f"Erreur interface: {e}")
            return frame
    
    # =========================================================================
    # MÉTHODES DE CONTRÔLE ET GESTION DE L'IA
    # =========================================================================
    
    def start_ai_training(self):
        """
        Démarre le mode d'entraînement de l'intelligence artificielle.
        
        Cette méthode :
        1. Vérifie que TensorFlow est disponible
        2. Reset toutes les données d'entraînement existantes
        3. Initialise les paramètres de capture
        4. Affiche la liste des positions à entraîner
        
        Returns:
            bool: True si le mode entraînement a été activé avec succès
        """
        if not self.ai_enabled:
            self.logging.warning("⚠️ TensorFlow requis pour l'entraînement")
            return False
        
        # === INITIALISATION DU MODE ENTRAÎNEMENT ===
        self.ai_mode = "training"
        self.training_class = 0              # Commence par la première position
        self.training_countdown = 60         # 2 secondes de préparation
        
        # === RESET DES DONNÉES PRÉCÉDENTES ===
        self.position_recognizer.training_data = []
        self.position_recognizer.training_labels = []
        
        # === AFFICHAGE DES INFORMATIONS ===
        self.logging.info("🎓 Mode entraînement IA démarré")
        self.logging.info("📝 Positions à entraîner:")
        for pos_id, position in HAND_POSITIONS.items():
            self.logging.info(f"   {pos_id}: {position.name} - {position.description}")
        
        return True
    
    def switch_ai_mode(self, mode):
        """
        Change le mode de fonctionnement de l'IA.
        
        Args:
            mode (str): Mode souhaité ("recognition", "training", "detection")
            
        Returns:
            bool: True si le changement a réussi
        """
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
        """
        Active ou désactive les commandes de drone.
        
        ATTENTION: Cette fonction contrôle un système critique de sécurité.
        Quand les commandes sont activées, les gestes contrôlent directement le drone.
        
        Returns:
            bool: État final des commandes (True = activées, False = désactivées)
        """
        self.drone_commands_enabled = not self.drone_commands_enabled
        status = "activées" if self.drone_commands_enabled else "désactivées"
        self.logging.info(f"🚁 Commandes drone {status}")
        return self.drone_commands_enabled
    
    def save_ai_model(self):
        """
        Sauvegarde le modèle IA entraîné sur disque.
        
        Returns:
            bool: True si la sauvegarde a réussi
        """
        if self.ai_enabled and self.position_recognizer and self.position_recognizer.is_trained:
            success = self.position_recognizer.save_model("hand_position_model")
            if success:
                self.logging.info("💾 Modèle IA sauvegardé")
            return success
        return False
    
    def load_ai_model(self):
        """
        Charge un modèle IA existant depuis le disque.
        
        Returns:
            bool: True si le chargement a réussi
        """
        if self.ai_enabled and self.position_recognizer:
            success = self.position_recognizer.load_model("hand_position_model")
            if success:
                self.ai_mode = "recognition"
                self.logging.info("📂 Modèle IA chargé")
            return success
        return False
    
    def get_ai_stats(self):
        """
        Retourne les statistiques complètes de l'IA.
        
        Returns:
            str: Chaîne formatée avec toutes les stats
        """
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

    def _start_model_training(self):
        """
        Lance l'entraînement du modèle après la collecte des échantillons.
        
        Cette méthode est appelée automatiquement quand tous les échantillons
        ont été collectés pour toutes les positions.
        """
        if self.position_recognizer and len(self.position_recognizer.training_data) > 0:
            self.logging.info("🚀 Démarrage entraînement du modèle...")
            success = self.position_recognizer.train_model(
                validation_split=0.2,
                epochs=50,
                plot_curves=True,
                save_fig=True,
                show_confusion=True
            )
            
            if success:
                self.ai_mode = "recognition"  # Passe en mode reconnaissance
                self.logging.info("✅ Entraînement terminé, modèle prêt à l'emploi!")
                # Sauvegarde automatique
                self.save_ai_model()
            else:
                self.logging.error("❌ Échec de l'entraînement")

# =============================================================================
# CONTRÔLE MANUEL DU DRONE (THREAD SÉPARÉ)
# =============================================================================

def simple_drone_control(bebop):
    """
    Interface de contrôle manuel du drone dans un thread séparé.
    
    Cette fonction permet de contrôler le drone avec le clavier pendant
    que la reconnaissance de gestes fonctionne en parallèle.
    
    Commandes disponibles :
    - t : décollage
    - l : atterrissage
    - f/b/g/d : mouvements avant/arrière/gauche/droite
    - h/m : montée/descente
    - e : quitter et déconnecter
    
    Args:
        bebop: Instance du drone Bebop connecté
    """
    logger.info("Contrôle drone démarré.")
    print("\n[Commandes drone manuelles]")
    print("  t = décoller | l = atterrir | e = quitter")
    print("  f/b/g/d = mouvements | h/m = haut/bas")
    
    while True:
        try:
            # Lecture d'une commande clavier
            key = input("> ").strip().lower()
        except EOFError:
            break
            
        # === COMMANDES DE BASE ===
        if key == 't':
            # Décollage sécurisé
            bebop.safe_takeoff(10)
            time.sleep(1)
            # Stabilisation après décollage
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=3)
            print("✈️ Décollage")
            
        elif key == 'l':
            # Atterrissage sécurisé
            bebop.safe_land(10)
            print("🛬 Atterrissage")
            
        elif key == 'e':
            # Arrêt complet et déconnexion
            bebop.safe_land(10)
            bebop.disconnect()
            print("🔚 Arrêt")
            break
            
        # === COMMANDES DE MOUVEMENT ===
        elif key == 'f':
            # Avancer (pitch positif)
            bebop.fly_direct(roll=0, pitch=25, yaw=0, vertical_movement=0, duration=0.3)
            
        elif key == 'b':
            # Reculer (pitch négatif)
            bebop.fly_direct(roll=0, pitch=-25, yaw=0, vertical_movement=0, duration=0.3)
            
        elif key == 'g':
            # Gauche (roll négatif)
            bebop.fly_direct(roll=-25, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
            
        elif key == 'd':
            # Droite (roll positif)
            bebop.fly_direct(roll=25, pitch=0, yaw=0, vertical_movement=0, duration=0.3)
            
        elif key == 'h':
            # Monter (vertical_movement positif)
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=0.3)
            
        elif key == 'm':
            # Descendre (vertical_movement négatif)
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=0.3)

# =============================================================================
# FONCTION PRINCIPALE AVEC INTÉGRATION IA
# =============================================================================

def main_with_ai():
    """
    Fonction principale du programme intégrant toutes les fonctionnalités.
    
    Cette fonction orchestre :
    1. Connexion au drone Bebop 2
    2. Initialisation du flux vidéo via FFmpeg
    3. Création du détecteur avec IA
    4. Lancement des threads de contrôle
    5. Boucle principale de traitement des images
    6. Gestion des interactions clavier
    7. Nettoyage et fermeture propre
    
    Returns:
        bool: True si l'exécution s'est bien déroulée
    """
    
    logger.info("=== BEBOP 2 AVEC IA DE RECONNAISSANCE DE POSITION ===")
    logger.info("🤖 Système de reconnaissance de gestes pour drone")
    
    # Vérification de la disponibilité de TensorFlow
    if not TF_AVAILABLE:
        logger.warning("⚠️ TensorFlow non disponible - Mode détection simple uniquement")
        logger.info("   Pour activer l'IA: pip install tensorflow")
    
    # === INITIALISATION DES VARIABLES ===
    bebop = None                    # Instance du drone
    pipe = None                     # Pipeline FFmpeg pour vidéo
    detector = None                 # Détecteur avec IA
    start_time = time.time()        # Timestamp de début pour stats
    
    try:
        # === CONNEXION AU DRONE ===
        logger.info("📡 Connexion au drone...")
        bebop = Bebop()
        
        if not bebop.connect(10):  # Timeout de 10 secondes
            logger.error("❌ Échec connexion drone")
            return False

        logger.info("✅ Drone connecté!")
        
        # === DÉMARRAGE DU FLUX VIDÉO ===
        logger.info("📹 Démarrage flux vidéo...")
        bebop.start_video_stream()
        time.sleep(2)  # Attente stabilisation du flux
        
        # === LANCEMENT DU CONTRÔLE DRONE MANUEL ===
        # Thread séparé pour les commandes clavier du drone
        ctrl_thread = threading.Thread(target=simple_drone_control, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === CONFIGURATION DU PIPELINE FFMPEG ===
        # Localisation du fichier SDP (Session Description Protocol)
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # Commande FFmpeg optimisée pour faible latence
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',  # Protocoles autorisés
            '-fflags', 'nobuffer',                   # Pas de mise en buffer
            '-flags', 'low_delay',                   # Priorité faible latence
            '-avioflags', 'direct',                  # I/O direct
            '-analyzeduration', '1000000',           # Temps d'analyse réduit
            '-probesize', '1000000',                 # Taille de sonde réduite
            '-i', sdp_path,                          # Fichier source SDP
            '-vf', 'eq=saturation=1.1:gamma=0.95',  # Amélioration visuelle
            '-f', 'rawvideo',                        # Format de sortie brut
            '-pix_fmt', 'bgr24',                     # Format pixel OpenCV
            '-'                                      # Sortie vers stdout
        ]
        
        try:
            # Lancement du processus FFmpeg
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=2*1024*1024)
            logger.info("✅ Pipeline vidéo initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === CRÉATION DU DÉTECTEUR AVEC IA ===
        detector = OptimizedBicolorGloveDetectorWithAI()
        detector.bebop = bebop  # Liaison avec l'instance du drone
        
        # === CONFIGURATION DE L'INTERFACE ===
        window_name = "Bebop 2 - IA Reconnaissance Position"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # === AFFICHAGE DES INSTRUCTIONS ===
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
        
        # === VARIABLES DE BOUCLE PRINCIPALE ===
        screenshot_count = 0           # Compteur de captures d'écran
        last_fps_log = time.time()     # Dernier affichage FPS
        fps_counter = 0                # Compteur de frames pour FPS
        
        logger.info("🎬 Démarrage détection avec IA...")
        
        # === BOUCLE PRINCIPALE DE TRAITEMENT ===
        while True:
            try:
                # === LECTURE D'UNE FRAME VIDÉO ===
                
                # Lecture des données brutes depuis FFmpeg
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)  # 3 bytes par pixel (BGR)
                
                # Vérification de l'intégrité de la frame
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.warning("⚠️ Frame incomplète")
                    continue
                
                # Conversion des bytes en image NumPy
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # === TRAITEMENT AVEC DÉTECTION + IA ===
                
                # Application du pipeline de détection et reconnaissance
                processed_frame, detected = detector.detect_glove_optimized(frame)
                
                # === MISE À JOUR DES COMPTEURS IA ===
                
                if detected:
                    detector.ai_frame_count += 1
                
                # === AFFICHAGE DE L'IMAGE TRAITÉE ===
                
                cv2.imshow(window_name, processed_frame)
                
                # === LOGS PÉRIODIQUES DE PERFORMANCE ===
                
                fps_counter += 1
                if fps_counter % 90 == 0:  # Tous les 90 frames (environ 3 secondes)
                    current_time = time.time()
                    elapsed = current_time - last_fps_log
                    display_fps = 90 / elapsed if elapsed > 0 else 0
                    
                    # Affichage des statistiques
                    logger.info(f"📊 FPS: {display_fps:.1f} | {detector.get_ai_stats()}")
                    if detector.current_position:
                        logger.info(f"🤖 Position: {detector.current_position} "
                                   f"(confiance: {detector.current_position_confidence:.2f})")
                    
                    last_fps_log = current_time
                
                # === GESTION DES INTERACTIONS CLAVIER ===
                
                # Lecture non-bloquante du clavier (timeout 1ms)
                key = cv2.waitKey(1) & 0xFF
                
                # === COMMANDES GÉNÉRALES ===
                
                if key == ord('q') or key == 27:  # 'q' ou Échap
                    logger.info("🛑 Arrêt demandé")
                    break
                
                elif key == ord('s'):  # Screenshot
                    timestamp = int(time.time())
                    screenshot_name = f"ai_capture_{timestamp}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    logger.info(f"📸 Screenshot: {screenshot_name}")
                    screenshot_count += 1
                
                elif key == ord('r'):  # Reset du détecteur
                    detector = OptimizedBicolorGloveDetectorWithAI()
                    detector.bebop = bebop  # Re-liaison avec le drone
                    logger.info("🔄 Détecteur reset")
                
                # === COMMANDES IA (SEULEMENT SI TENSORFLOW DISPONIBLE) ===
                
                elif TF_AVAILABLE:
                    
                    if key == ord('i'):  # Informations IA
                        logger.info(f"📊 {detector.get_ai_stats()}")
                        if detector.position_recognizer:
                            logger.info(f"📊 {detector.position_recognizer.get_prediction_stats()}")
                    
                    elif key == ord('t'):  # Mode entraînement
                        if detector.start_ai_training():
                            logger.info("🎓 Mode entraînement IA activé")
                            logger.info("Effectuez chaque position quand demandé")
                    
                    elif key == ord('n'):  # Mode reconnaissance
                        if detector.switch_ai_mode("recognition"):
                            logger.info("🤖 Mode reconnaissance IA activé")
                        else:
                            logger.warning("⚠️ Modèle non entraîné")
                    
                    elif key == ord('m'):  # Sauvegarde modèle
                        if detector.save_ai_model():
                            logger.info("💾 Modèle IA sauvegardé")
                        else:
                            logger.warning("⚠️ Aucun modèle à sauvegarder")
                    
                    elif key == ord('l'):  # Chargement modèle
                        if detector.load_ai_model():
                            logger.info("📂 Modèle IA chargé")
                        else:
                            logger.warning("⚠️ Aucun modèle à charger")
                    
                    elif key == ord('a'):  # Arrêt d'urgence manuel
                        logger.warning("🛑 Touche A détectée : arrêt manuel demandé")
                        detector.drone_commands_enabled = False

                        if hasattr(detector, "bebop") and detector.bebop is not None:
                            try:
                                logger.info("[DRONE CMD] Stabilisation (hover)...")
                                detector.bebop.fly_direct(roll=0, pitch=0, yaw=0, 
                                                        vertical_movement=0, duration=0.5)
                                time.sleep(0.5)
                                logger.info("[DRONE CMD] Atterrissage...")
                                detector.bebop.safe_land(10)
                                logger.info("[DRONE CMD] ✅ Drone posé manuellement via touche A")
                            except Exception as e:
                                logger.error(f"[DRONE CMD] ❌ Erreur arrêt manuel (A): {e}")
                        else:
                            logger.warning("[DRONE CMD] Aucun drone actif pour atterrissage manuel.")

                    elif key == ord('c'):  # Toggle commandes drone
                        logger.info("[DEBUG] Touche 'c' pressée, toggle_drone_commands va être appelé")
                        status = detector.toggle_drone_commands()
                        if status:
                            logger.warning("⚠️ ATTENTION: Gestes contrôlent le drone!")
                            logger.info("🚁 Commandes drone ACTIVÉES")
                        else:
                            logger.info("🚁 Commandes drone désactivées")
                
                # === COMMANDE DEBUG ===
                elif key == ord('d'):  # Debug détaillé
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
        # === NETTOYAGE ET STATISTIQUES FINALES ===
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
        
        # === FERMETURE DES RESSOURCES ===
        
        # Fermeture du pipeline FFmpeg
        if pipe:
            try:
                pipe.terminate()
                logger.info("✅ Pipeline fermé")
            except:
                pass
        
        # Fermeture des fenêtres OpenCV
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Interface fermée")
        except:
            pass
        
        # Déconnexion du drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("✅ Drone déconnecté")
            except:
                pass
        
        logger.info("🎉 Session IA terminée!")
    
    return True

# =============================================================================
# POINT D'ENTRÉE DU PROGRAMME
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée principal du programme.
    
    Cette section est exécutée uniquement quand le script est lancé directement
    (pas quand il est importé comme module).
    """
    try:
        print("🚁 BEBOP 2 - DÉTECTION GANT AVEC IA")
        print("=" * 50)
        
        # Vérification de TensorFlow et affichage du statut
        if TF_AVAILABLE:
            print("✅ TensorFlow détecté - IA activée")
        else:
            print("⚠️  TensorFlow manquant - Mode détection simple")
            print("   Installation: pip install tensorflow")
        
        # Affichage des gestes reconnus
        logger.info("=" * 80)
        logger.info("🎯 GESTES RECONNUS :")
        for pos_id, position in HAND_POSITIONS.items():
            logger.info(f"  {pos_id}: {position.name} - {position.description}")
        logger.info("=" * 80)

        print("\n🚀 Démarrage...")
        
        # Lancement du programme principal
        success = main_with_ai()
        
        # Code de sortie selon le succès
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"💥 Exception finale: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

# =============================================================================
# GUIDE D'UTILISATION DÉTAILLÉ
# =============================================================================
"""
🎮 GUIDE D'UTILISATION COMPLET - RAYAN DJOUDI - PARIS-SACLAY MECSE

=== INSTALLATION ET CONFIGURATION ===

1. PRÉREQUIS SYSTÈME:
   - Python 3.7+ recommandé
   - FFmpeg installé et accessible dans PATH
   - Drone Bebop 2 avec batterie chargée
   - Gant rouge ou orange pour la détection

2. INSTALLATION DES DÉPENDANCES:
   pip install tensorflow opencv-python numpy pyparrot matplotlib scikit-learn seaborn

3. VÉRIFICATION DE L'INSTALLATION:
   - python -c "import tensorflow as tf; print(tf.__version__)"
   - ffmpeg -version

=== PREMIÈRE UTILISATION ===

4. CONFIGURATION DU DRONE:
   - Allumer le Bebop 2
   - Se connecter au WiFi du drone (Bebop2-XXXXXX)
   - Vérifier que l'adresse IP est bien 192.168.42.1

5. LANCEMENT DU PROGRAMME:
   python bebop_ai_detection.py

6. ENTRAÎNEMENT DE L'IA (optionnel mais recommandé):
   - Presser 't' pour démarrer l'entraînement
   - Suivre les instructions pour chaque position (9 gestes)
   - Maintenir chaque position 3-5 secondes quand demandé
   - L'entraînement se lance automatiquement après collecte

=== UTILISATION NORMALE ===

7. MODE RECONNAISSANCE:
   - Presser 'n' pour activer la reconnaissance (après entraînement)
   - Presser 'c' pour activer les commandes drone (ATTENTION!)
   - Effectuer les gestes devant la caméra
   - Observer les réactions du drone

8. COMMANDES CLAVIER PRINCIPALES:
   ✋ GESTION IA:
   - 'i' : Afficher informations et statistiques IA
   - 't' : Démarrer entraînement des 9 positions
   - 'n' : Activer mode reconnaissance
   - 'm' : Sauvegarder le modèle entraîné
   - 'l' : Charger un modèle existant
   
   🚁 CONTRÔLE DRONE:
   - 'c' : Activer/désactiver commandes par gestes
   - 'a' : Arrêt d'urgence manuel (atterrissage immédiat)
   
   📸 INTERFACE:
   - 's' : Capture d'écran
   - 'r' : Reset du détecteur
   - 'd' : Informations de debug
   - 'q' : Quitter le programme

=== GESTES ET COMMANDES DRONE ===

9. POSITIONS RECONNUES PAR L'IA:
   ✊ Poing fermé         → ARRÊT D'URGENCE (priorité absolue)
   ✋ Main karaté          → AVANCER (doigts vers le bas)
   🛑 Paume ouverte       → RECULER (doigts écartés)
   👍 Pouce levé         → MONTER (pouce vers le haut)
   👎 Pouce baissé       → DESCENDRE (pouce vers le bas)
   ☝️ Index gauche       → DRONE À DROITE
   ☝️ Index droite       → DRONE À GAUCHE
   🔄 Dos main droite    → ROTATION GAUCHE
   🔄 Dos main gauche    → ROTATION DROITE

10. SYSTÈME DE SÉCURITÉ:
    - Cooldown de 5 secondes entre commandes
    - Seuils de confiance adaptatifs (75-85%)
    - Validation sur 3 frames consécutives
    - Vérification de distance minimale
    - Arrêt d'urgence prioritaire (poing)

=== CONTRÔLE MANUEL DU DRONE (THREAD PARALLÈLE) ===

11. COMMANDES CLAVIER DRONE (dans terminal):
    - 't' : Décollage automatique
    - 'l' : Atterrissage automatique
    - 'f' : Avancer
    - 'b' : Reculer
    - 'g' : Gauche
    - 'd' : Droite
    - 'h' : Monter
    - 'm' : Descendre
    - 'e' : Quitter et déconnecter

=== FICHIERS GÉNÉRÉS ===

12. SAUVEGARDE ET LOGS:
    - hand_position_model_model.keras : Modèle TensorFlow entraîné
    - hand_position_model_data.pkl : Données d'entraînement
    - bebop_ai_detection.log : Logs détaillés de la session
    - ai_capture_*.png : Captures d'écran
    - training_curves_*.png : Courbes d'apprentissage
    - confusion_matrix_*.png : Matrices de confusion

=== CONSEILS D'OPTIMISATION ===

13. AMÉLIORER LA DÉTECTION:
    - Porter un gant rouge/orange uni
    - Éviter les arrière-plans colorés similaires
    - Assurer un éclairage uniforme
    - Maintenir une distance de 1-2 mètres de la caméra

14. OPTIMISER L'IA:
    - Entraîner avec différents angles et distances
    - Varier l'éclairage pendant l'entraînement
    - Maintenir les positions stables lors de la capture
    - Refaire l'entraînement si performance insuffisante

15. SÉCURITÉ EN VOL:
    - Toujours avoir le poing prêt pour l'arrêt d'urgence
    - Tester d'abord sans les commandes activées
    - Voler dans un espace dégagé
    - Surveiller le niveau de batterie du drone

=== DÉPANNAGE ===

16. PROBLÈMES COURANTS:
    
    ❌ "TensorFlow non disponible":
    → pip install tensorflow
    
    ❌ "FFmpeg non trouvé":
    → Installer FFmpeg et l'ajouter au PATH
    
    ❌ "Échec connexion drone":
    → Vérifier WiFi, redémarrer le drone
    
    ❌ "Détection faible":
    → Améliorer l'éclairage, changer de gant
    
    ❌ "IA pas assez précise":
    → Refaire l'entraînement avec plus d'échantillons
    
    ❌ "Commandes erratiques":
    → Augmenter les seuils de confiance dans le code

=== ARCHITECTURE TECHNIQUE ===

17. COMPOSANTS PRINCIPAUX:
    - 🎥 Détection couleur HSV (OpenCV)
    - 🧠 Réseau de neurones dense (TensorFlow)
    - 📊 Extraction de 64 caractéristiques (géométrie + vision)
    - 🚁 Interface drone (PyParrot)
    - 🖥️ Interface utilisateur temps réel (OpenCV)

18. PIPELINE DE TRAITEMENT:
    Image → Détection gant → Extraction features → IA → Commande drone

=== EXTENSIONS POSSIBLES ===

19. AMÉLIORATIONS ENVISAGEABLES:
    - Reconnaissance à deux mains
    - Gestes plus complexes (séquences)
    - Mode autonome avec planification
    - Interface mobile/tablette
    - Contrôle de multiple drones
    - Enregistrement/replay de vols

20. OPTIMISATIONS PERFORMANCE:
    - Quantification du modèle IA
    - Optimisation GPU pour temps réel
    - Réduction de la latence réseau
    - Parallélisation des traitements

=== PROJET UNIVERSITAIRE ===

21. ASPECTS PÉDAGOGIQUES:
    ✅ Vision par ordinateur (OpenCV, HSV, morphologie)
    ✅ Intelligence artificielle (TensorFlow, réseaux de neurones)
    ✅ Systèmes embarqués (temps réel, threading)
    ✅ Robotique (contrôle de drone, sécurité)
    ✅ Génie logiciel (architecture, documentation)

22. COMPÉTENCES DÉMONTRÉES:
    - Intégration de technologies complexes
    - Gestion de contraintes temps réel
    - Implémentation de systèmes de sécurité
    - Documentation et logging professionnel
    - Interface utilisateur complète

Projet réalisé par RAYAN DJOUDI
Licence Professionnelle MECSE - Université Paris-Saclay
Année universitaire 2024-2025

🎉 Bon vol avec votre drone intelligent! 🚁
"""