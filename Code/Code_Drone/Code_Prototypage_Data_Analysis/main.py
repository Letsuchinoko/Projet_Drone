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

# === PARAMÈTRES OPTIMISÉS POUR CONTRÔLE GESTUEL ===
BEBOP_IP = "192.168.42.1"
WIDTH, HEIGHT = 856, 480
DISTANCE_RANGE = (5, 10)  # Mètres de distance cible

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_gesture_control.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === DÉTECTEUR GANT LONGUE DISTANCE ===
class LongDistanceGestureDetector:
    def __init__(self):
        # Historique pour stabilité à distance
        self.detection_history = deque(maxlen=15)
        self.stable_detections = deque(maxlen=5)
        self.confidence_threshold = 3  # 3 sur 5 pour robustesse
        
        # Paramètres adaptés pour 5-10m de distance
        self.min_area = 100   # Plus petit pour distance
        self.max_area = 50000 # Adaptable selon zoom
        self.min_contour_points = 8
        
        # Kernels morphologiques pour longue distance
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Stats pour contrôle gestuel
        self.frame_count = 0
        self.detection_count = 0
        self.gesture_ready = False
        self.last_gesture_position = None
        self.gesture_history = deque(maxlen=30)
        
        # Paramètres avancés pour masque IA
        self.mask_quality_threshold = 0.6
        self.current_mask_quality = 0.0
        self.best_mask = None
        self.best_mask_score = 0.0
        
        # Zone de détection adaptative
        self.detection_zone = None
        self.auto_zoom_factor = 1.0
        
        # Cache pour performance
        self.frame_cache = {}
        self.mask_cache = {}

    def detect_gesture_glove(self, frame):
        """Détection optimisée pour contrôle gestuel à distance"""
        if frame is None:
            return frame, False, None
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            h, w = frame.shape[:2]
            
            # Prétraitement adaptatif selon la qualité
            work_frame = self._adaptive_preprocessing(frame)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Création du masque ultra-robuste
            mask = self._create_robust_distance_mask(hsv)
            
            # Post-traitement morphologique adaptatif
            mask = self._adaptive_morphology(mask, self.auto_zoom_factor)
            
            # Détection avec scoring de qualité
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour, quality_score = self._select_gesture_contour(contours, (h, w))
            
            # Validation pour contrôle gestuel
            detected, gesture_data = self._validate_gesture_detection(best_contour, quality_score)
            
            # Stabilisation avec historique gestuel
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            # Mise à jour données gestuelles
            if stable_detection and gesture_data:
                self.gesture_ready = True
                self.gesture_history.append(gesture_data)
                self.detection_count += 1
                
                # Sauvegarde du meilleur masque pour IA
                if quality_score > self.best_mask_score:
                    self.best_mask = mask.copy()
                    self.best_mask_score = quality_score
            else:
                self.gesture_ready = False
            
            # Mise à jour historique
            self.detection_history.append(stable_detection)
            self.current_mask_quality = quality_score
            
            # Visualisation pour contrôle gestuel
            if stable_detection and best_contour is not None:
                result_frame = self._draw_gesture_detection(original_frame, best_contour, gesture_data, quality_score)
            else:
                result_frame = original_frame.copy()
            
            # Overlay spécialisé contrôle gestuel
            result_frame = self._add_gesture_overlay(result_frame, stable_detection, mask, gesture_data)
            
            return result_frame, stable_detection, gesture_data
            
        except Exception as e:
            logger.debug(f"Gesture detection error: {e}")
            return original_frame, False, None

    def _adaptive_preprocessing(self, frame):
        """Prétraitement adaptatif pour longue distance"""
        try:
            # Amélioration du contraste pour distance
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE adaptatif
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Réduction du bruit pour distance
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
            
        except Exception as e:
            logger.debug(f"Preprocessing error: {e}")
            return frame

    def _create_robust_distance_mask(self, hsv):
        """Masque ultra-robuste pour détection à distance"""
        try:
            h, w = hsv.shape[:2]
            
            # Masques orange étendus pour toutes conditions
            orange_masks = []
            
            # Orange vif (conditions normales)
            orange_masks.append(cv2.inRange(hsv, np.array([8, 120, 120]), np.array([25, 255, 255])))
            
            # Orange foncé (ombre/distance)
            orange_masks.append(cv2.inRange(hsv, np.array([10, 80, 80]), np.array([22, 200, 200])))
            
            # Orange clair (surexposition)
            orange_masks.append(cv2.inRange(hsv, np.array([12, 60, 150]), np.array([20, 180, 255])))
            
            # Orange désaturé (distance)
            orange_masks.append(cv2.inRange(hsv, np.array([15, 40, 100]), np.array([25, 120, 220])))
            
            # Masques rouge étendus
            red_masks = []
            
            # Rouge vif
            red_masks.append(cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255])))
            red_masks.append(cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255])))
            
            # Rouge foncé
            red_masks.append(cv2.inRange(hsv, np.array([0, 80, 80]), np.array([8, 200, 200])))
            red_masks.append(cv2.inRange(hsv, np.array([172, 80, 80]), np.array([180, 200, 200])))
            
            # Rouge désaturé
            red_masks.append(cv2.inRange(hsv, np.array([0, 40, 100]), np.array([12, 120, 220])))
            red_masks.append(cv2.inRange(hsv, np.array([168, 40, 100]), np.array([180, 120, 220])))
            
            # Combinaison optimale
            orange_combined = orange_masks[0]
            for mask in orange_masks[1:]:
                orange_combined = cv2.bitwise_or(orange_combined, mask)
            
            red_combined = red_masks[0]
            for mask in red_masks[1:]:
                red_combined = cv2.bitwise_or(red_combined, mask)
            
            # Masque final gant
            glove_mask = cv2.bitwise_or(orange_combined, red_combined)
            
            # Exclusion peau très ciblée (pour ne pas perdre le gant)
            skin_lower = np.array([0, 50, 120])
            skin_upper = np.array([15, 120, 200])
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            
            # Érosion de la peau pour garder plus de gant
            skin_mask = cv2.erode(skin_mask, self.kernel_small, iterations=2)
            
            # Masque final
            final_mask = cv2.bitwise_and(glove_mask, cv2.bitwise_not(skin_mask))
            
            # Nettoyage des petits artefacts
            final_mask = cv2.medianBlur(final_mask, 5)
            
            return final_mask
            
        except Exception as e:
            logger.debug(f"Robust mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _adaptive_morphology(self, mask, zoom_factor):
        """Morphologie adaptative selon la distance estimée"""
        try:
            # Adapter les kernels selon le zoom/distance
            if zoom_factor < 0.5:  # Très loin
                kernel = self.kernel_large
                iterations = 3
            elif zoom_factor < 1.0:  # Loin
                kernel = self.kernel_medium
                iterations = 2
            else:  # Proche
                kernel = self.kernel_small
                iterations = 1
            
            # Fermeture pour connecter les parties
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Dilatation pour épaissir
            mask = cv2.dilate(mask, kernel, iterations=iterations)
            
            # Ouverture pour nettoyer
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            
            return mask
            
        except Exception as e:
            logger.debug(f"Adaptive morphology error: {e}")
            return mask

    def _select_gesture_contour(self, contours, frame_shape):
        """Sélection de contour optimisée pour gestes"""
        if not contours:
            return None, 0.0
            
        try:
            h, w = frame_shape
            best_contour = None
            best_score = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Calculs géométriques
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                aspect_ratio = w_rect / float(h_rect)
                
                # Rectangle englobant pas trop aux bords
                margin = 30
                if (x < margin or y < margin or 
                    (x + w_rect) > (w - margin) or 
                    (y + h_rect) > (h - margin)):
                    continue
                
                # Analyse de forme
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Moments pour centroïde
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Scores pour gestes
                
                # 1. Score d'aire (optimal pour geste à distance)
                ideal_area = 2000  # Aire optimale pour geste
                area_score = 1.0 - abs(area - ideal_area) / ideal_area
                area_score = max(0.0, min(1.0, area_score))
                
                # 2. Score de forme (main/gant)
                if 0.3 <= aspect_ratio <= 2.5 and solidity >= 0.4:
                    shape_score = 1.0
                else:
                    shape_score = 0.5
                
                # 3. Score de position (éviter les bords)
                center_x_norm = cx / w
                center_y_norm = cy / h
                if 0.2 <= center_x_norm <= 0.8 and 0.1 <= center_y_norm <= 0.9:
                    position_score = 1.0
                else:
                    position_score = 0.6
                
                # 4. Score de complexité (doigts séparés = bon)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter * perimeter)
                    # Moins compact = plus de détails = meilleur pour gestes
                    complexity_score = 1.0 - compactness
                    complexity_score = max(0.0, min(1.0, complexity_score))
                else:
                    complexity_score = 0.0
                
                # Score final pondéré pour contrôle gestuel
                final_score = (area_score * 0.3 + 
                              shape_score * 0.25 + 
                              position_score * 0.25 + 
                              complexity_score * 0.2)
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
            
            return best_contour, best_score
            
        except Exception as e:
            logger.debug(f"Gesture contour selection error: {e}")
            return None, 0.0

    def _validate_gesture_detection(self, contour, quality_score):
        """Validation spécialisée pour gestes"""
        if contour is None or quality_score < 0.3:
            return False, None
        
        try:
            # Extraction des données gestuelles
            gesture_data = self._extract_gesture_data(contour)
            
            # Validation cohérence temporelle
            if self.last_gesture_position is not None:
                current_pos = gesture_data['center']
                last_pos = self.last_gesture_position
                
                # Distance de mouvement
                movement = np.sqrt((current_pos[0] - last_pos[0])**2 + 
                                 (current_pos[1] - last_pos[1])**2)
                
                # Mouvement trop rapide = probablement faux positif
                if movement > 150:  # pixels
                    return False, None
            
            # Mise à jour position
            self.last_gesture_position = gesture_data['center']
            
            return True, gesture_data
            
        except Exception as e:
            logger.debug(f"Gesture validation error: {e}")
            return False, None

    def _extract_gesture_data(self, contour):
        """Extraction des données gestuelles pour IA"""
        try:
            # Données de base
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Centre de masse
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w//2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h//2
            
            # Analyse de forme avancée
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Points extrêmes (pour orientation)
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])
            topmost = tuple(contour[contour[:,:,1].argmin()][0])
            bottommost = tuple(contour[contour[:,:,1].argmax()][0])
            
            # Défauts de convexité (doigts séparés)
            defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
            num_defects = len(defects) if defects is not None else 0
            
            return {
                'center': (cx, cy),
                'area': area,
                'bounding_box': (x, y, w, h),
                'solidity': solidity,
                'aspect_ratio': w / h,
                'extremes': {
                    'left': leftmost,
                    'right': rightmost,
                    'top': topmost,
                    'bottom': bottommost
                },
                'finger_defects': num_defects,
                'contour_points': len(contour),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.debug(f"Gesture data extraction error: {e}")
            return None

    def _draw_gesture_detection(self, frame, contour, gesture_data, quality_score):
        """Dessin spécialisé pour contrôle gestuel"""
        try:
            # Couleur basée sur la qualité
            if quality_score > 0.8:
                color = (0, 255, 0)  # Vert - excellent
                status = "EXCELLENT"
            elif quality_score > 0.6:
                color = (0, 255, 255)  # Jaune - bon
                status = "BON"
            else:
                color = (0, 150, 255)  # Orange - acceptable
                status = "ACCEPTABLE"
            
            # Contour principal
            cv2.drawContours(frame, [contour], -1, color, 3)
            
            # Rectangle englobant
            x, y, w, h = gesture_data['bounding_box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Centre avec croix
            cx, cy = gesture_data['center']
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 15, 3)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 12, color, 2)
            
            # Points extrêmes
            extremes = gesture_data['extremes']
            cv2.circle(frame, extremes['left'], 4, (255, 0, 0), -1)    # Bleu
            cv2.circle(frame, extremes['right'], 4, (0, 0, 255), -1)   # Rouge
            cv2.circle(frame, extremes['top'], 4, (255, 255, 0), -1)   # Cyan
            cv2.circle(frame, extremes['bottom'], 4, (255, 0, 255), -1) # Magenta
            
            # Informations détaillées
            cv2.putText(frame, f"GESTE {status}", (x, max(y - 15, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            info_text = f"Q:{quality_score:.2f} A:{int(gesture_data['area'])} D:{gesture_data['finger_defects']}"
            cv2.putText(frame, info_text, (x, max(y - 40, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Gesture drawing error: {e}")
            return frame

    def _add_gesture_overlay(self, frame, detected, mask, gesture_data):
        """Overlay spécialisé pour contrôle gestuel"""
        try:
            h, w = frame.shape[:2]
            
            # Status gestuel principal
            if detected and self.gesture_ready:
                status = "🎯 GESTE DÉTECTÉ - CONTRÔLE ACTIF"
                color = (0, 255, 0)
            elif detected:
                status = "⚠️ GESTE EN COURS DE VALIDATION"
                color = (0, 255, 255)
            else:
                status = "🔍 RECHERCHE GESTE DE CONTRÔLE"
                color = (100, 100, 255)
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Qualité de détection
            quality_bar_width = 200
            quality_bar_height = 20
            quality_x, quality_y = 10, 60
            
            # Barre de fond
            cv2.rectangle(frame, (quality_x, quality_y), 
                         (quality_x + quality_bar_width, quality_y + quality_bar_height), 
                         (50, 50, 50), -1)
            
            # Barre de qualité
            quality_width = int(quality_bar_width * self.current_mask_quality)
            quality_color = (0, 255, 0) if self.current_mask_quality > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (quality_x, quality_y), 
                         (quality_x + quality_width, quality_y + quality_bar_height), 
                         quality_color, -1)
            
            cv2.putText(frame, f"Qualité: {self.current_mask_quality:.1%}", 
                       (quality_x + quality_bar_width + 10, quality_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stats de performance
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            stats_text = f"Frames: {self.frame_count} | Détections: {self.detection_count} ({detection_rate:.1f}%)"
            cv2.putText(frame, stats_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Données gestuelles actuelles
            if gesture_data:
                gesture_info = (f"Centre: ({gesture_data['center'][0]}, {gesture_data['center'][1]}) | "
                               f"Aire: {int(gesture_data['area'])} | "
                               f"Doigts: {gesture_data['finger_defects']}")
                cv2.putText(frame, gesture_info, (10, h - 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            
            # Zone de contrôle recommandée
            zone_x1, zone_y1 = int(w * 0.25), int(h * 0.15)
            zone_x2, zone_y2 = int(w * 0.75), int(h * 0.85)
            cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (100, 100, 100), 2)
            cv2.putText(frame, "ZONE DE CONTRÔLE", (zone_x1, zone_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Historique gestuel
            history = "".join(["●" if x else "○" for x in list(self.detection_history)[-15:]])
            cv2.putText(frame, f"Hist: {history}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Distance estimée (basée sur la taille du gant)
            if gesture_data:
                estimated_distance = self._estimate_distance(gesture_data['area'])
                distance_color = (0, 255, 0) if 5 <= estimated_distance <= 10 else (0, 255, 255)
                cv2.putText(frame, f"Distance: ~{estimated_distance:.1f}m", (w - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
            
            # Timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Masque pour IA (coin supérieur droit)
            if mask is not None and mask.size > 0:
                try:
                    mask_display_size = (160, 120)
                    mask_small = cv2.resize(mask, mask_display_size)
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    mask_x, mask_y = w - 170, 90
                    frame[mask_y:mask_y+120, mask_x:mask_x+160] = mask_colored
                    
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+160, mask_y+120), (255, 255, 255), 2)
                    cv2.putText(frame, "Masque IA", (mask_x, mask_y + 135), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Indicateur qualité masque
                    mask_pixels = np.count_nonzero(mask)
                    mask_coverage = (mask_pixels / mask.size) * 100
                    cv2.putText(frame, f"{mask_coverage:.1f}%", (mask_x, mask_y + 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                except Exception:
                    pass
            
            return frame
            
        except Exception as e:
            logger.debug(f"Gesture overlay error: {e}")
            return frame

    def _estimate_distance(self, area):
        """Estimation de distance basée sur la taille du gant"""
        try:
            # Calibration approximative (à ajuster selon votre gant)
            # Surface du gant à 1m ≈ 15000 pixels
            # Distance ∝ 1/√(aire)
            reference_area = 15000  # Aire de référence à 1m
            if area > 0:
                estimated_distance = np.sqrt(reference_area / area)
                return max(1.0, min(20.0, estimated_distance))  # Limiter entre 1 et 20m
            return 10.0  # Distance par défaut
        except:
            return 10.0

    def save_training_data(self, frame, mask, gesture_data, base_path="training_data"):
        """Sauvegarde des données pour entraînement IA"""
        try:
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            
            timestamp = int(time.time() * 1000)  # Millisecondes pour unicité
            
            # Sauvegarde frame originale
            frame_path = f"{base_path}/frame_{timestamp}.png"
            cv2.imwrite(frame_path, frame)
            
            # Sauvegarde masque
            mask_path = f"{base_path}/mask_{timestamp}.png"
            cv2.imwrite(mask_path, mask)
            
            # Sauvegarde métadonnées JSON
            import json
            metadata = {
                'timestamp': timestamp,
                'gesture_data': gesture_data,
                'quality_score': self.current_mask_quality,
                'frame_path': frame_path,
                'mask_path': mask_path
            }
            
            json_path = f"{base_path}/metadata_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Training data save error: {e}")
            return False

# === CONTRÔLE DRONE SIMPLIFIÉ ===
def drone_control_thread(bebop):
    """Thread de contrôle drone"""
    logger.info("Contrôle drone démarré.")
    print("\n[Commandes drone]\n"
          "  t = décoller | l = atterrir | e = quitter\n"
          "  Mouvements: f/b/g/d (avant/arrière/gauche/droite)\n"
          "  Altitude: h/m (haut/bas) | Rotation: a/c (gauche/droite)\n")
    
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
            bebop.fly_direct(roll=0, pitch=25, yaw=0, vertical_movement=0, duration=0.5)
            print("⬆️ Avant")
        elif key == 'b':
            bebop.fly_direct(roll=0, pitch=-25, yaw=0, vertical_movement=0, duration=0.5)
            print("⬇️ Arrière")
        elif key == 'g':
            bebop.fly_direct(roll=-25, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
            print("⬅️ Gauche")
        elif key == 'd':
            bebop.fly_direct(roll=25, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
            print("➡️ Droite")
        elif key == 'h':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=20, duration=0.5)
            print("⬆️ Haut")
        elif key == 'm':
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-20, duration=0.5)
            print("⬇️ Bas")
        elif key == 'a':
            bebop.fly_direct(roll=0, pitch=0, yaw=-35, vertical_movement=0, duration=0.5)
            print("🔄 Rotation gauche")
        elif key == 'c':
            bebop.fly_direct(roll=0, pitch=0, yaw=35, vertical_movement=0, duration=0.5)
            print("🔄 Rotation droite")

def main():
    """Fonction principale pour contrôle gestuel longue distance"""
    logger.info("=== BEBOP 2 GESTURE CONTROL SYSTEM ===")
    logger.info("🎯 Système de contrôle gestuel optimisé 5-10m")
    
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
            logger.error("Vérifiez: drone allumé, WiFi connecté, IP accessible")
            return False

        logger.info("✅ Drone connecté!")
        
        # === FLUX VIDÉO ===
        logger.info("📹 Démarrage flux vidéo...")
        bebop.start_video_stream()
        time.sleep(3)  # Stabilisation
        
        # === CONTRÔLE DRONE ===
        ctrl_thread = threading.Thread(target=drone_control_thread, args=(bebop,), daemon=True)
        ctrl_thread.start()
        
        # === PIPELINE FFMPEG OPTIMISÉ ===
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ SDP introuvable: {sdp_path}")
            return False
        
        # Commande FFmpeg pour qualité maximale (longue distance)
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-analyzeduration', '3000000',   # 3s d'analyse pour stabilité
            '-probesize', '3000000',         # 3MB probe pour qualité
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 FFmpeg qualité: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=2*1024*1024)
            logger.info("✅ Pipeline qualité initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé!")
            return False

        # === DÉTECTEUR GESTUEL ===
        detector = LongDistanceGestureDetector()
        
        # === INTERFACE ===
        window_name = "Bebop 2 - Contrôle Gestuel (5-10m)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        logger.info("=" * 60)
        logger.info("🎮 COMMANDES INTERFACE:")
        logger.info("  'q' = Quitter")
        logger.info("  's' = Screenshot + Sauvegarde données IA")
        logger.info("  'r' = Reset détecteur")
        logger.info("  'd' = Stats détaillées")
        logger.info("  'c' = Calibrage distance")
        logger.info("=" * 60)
        logger.info("🎯 CONTRÔLE GESTUEL:")
        logger.info("  Position optimale: 5-10m du drone")
        logger.info("  Zone de détection: centre de l'image")
        logger.info("  Qualité minimale: 60% pour contrôle fiable")
        logger.info("=" * 60)
        
        # === BOUCLE PRINCIPALE GESTUELLE ===
        logger.info("🎬 Démarrage détection gestuelle...")
        
        screenshot_count = 0
        training_data_count = 0
        last_stats_time = time.time()
        gesture_stable_count = 0
        
        # Calibrage initial
        calibration_frames = 0
        calibration_areas = []
        
        while True:
            try:
                # Lecture frame
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Erreur lecture frame")
                    break
                
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Détection gestuelle
                processed_frame, detected, gesture_data = detector.detect_gesture_glove(frame)
                
                # Suivi de stabilité gestuelle
                if detected and detector.gesture_ready:
                    gesture_stable_count += 1
                else:
                    gesture_stable_count = 0
                
                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Stats périodiques
                current_time = time.time()
                if current_time - last_stats_time >= 5:  # Toutes les 5 secondes
                    detection_rate = (detector.detection_count / max(detector.frame_count, 1)) * 100
                    stability = gesture_stable_count / max(detector.frame_count, 1) * 100
                    
                    logger.info(f"📊 Stats: Frames={detector.frame_count}, "
                               f"Détections={detector.detection_count} ({detection_rate:.1f}%), "
                               f"Qualité={detector.current_mask_quality:.1%}, "
                               f"Stabilité={stability:.1f}%")
                    
                    if gesture_data:
                        estimated_dist = detector._estimate_distance(gesture_data['area'])
                        logger.info(f"🎯 Geste: Centre=({gesture_data['center'][0]}, {gesture_data['center'][1]}), "
                                   f"Aire={int(gesture_data['area'])}, "
                                   f"Distance≈{estimated_dist:.1f}m")
                    
                    last_stats_time = current_time
                
                # Gestion touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    logger.info("🛑 Arrêt demandé")
                    break
                    
                elif key == ord('s'):
                    # Screenshot + données d'entraînement
                    timestamp = int(time.time())
                    screenshot_name = f"gesture_capture_{timestamp}_{screenshot_count:03d}.png"
                    cv2.imwrite(screenshot_name, processed_frame)
                    
                    # Sauvegarde données IA si geste détecté
                    if detected and gesture_data and detector.current_mask_quality > 0.6:
                        mask = detector.best_mask if detector.best_mask is not None else np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                        if detector.save_training_data(frame, mask, gesture_data):
                            training_data_count += 1
                            logger.info(f"📸 Screenshot + données IA sauvegardées: {screenshot_name}")
                            logger.info(f"💾 Total données IA: {training_data_count}")
                        else:
                            logger.info(f"📸 Screenshot sauvegardé: {screenshot_name}")
                    else:
                        logger.info(f"📸 Screenshot sauvegardé: {screenshot_name} (qualité insuffisante pour IA)")
                    
                    screenshot_count += 1
                    
                elif key == ord('r'):
                    # Reset complet du détecteur
                    old_count = detector.detection_count
                    detector.__init__()
                    logger.info(f"🔄 Détecteur reset (anciennes détections: {old_count})")
                    
                elif key == ord('d'):
                    # Stats détaillées
                    stats_data = {
                        'frame_count': detector.frame_count,
                        'detection_count': detector.detection_count,
                        'detection_rate': (detector.detection_count / max(detector.frame_count, 1)) * 100,
                        'current_quality': detector.current_mask_quality,
                        'best_quality': detector.best_mask_score,
                        'gesture_ready': detector.gesture_ready,
                        'training_data_saved': training_data_count
                    }
                    
                    logger.info("📊 STATISTIQUES DÉTAILLÉES:")
                    for key, value in stats_data.items():
                        logger.info(f"   {key}: {value}")
                    
                    if gesture_data:
                        logger.info("🎯 DONNÉES GESTUELLE ACTUELLES:")
                        logger.info(f"   Centre: {gesture_data['center']}")
                        logger.info(f"   Aire: {gesture_data['area']}")
                        logger.info(f"   Ratio aspect: {gesture_data['aspect_ratio']:.2f}")
                        logger.info(f"   Solidité: {gesture_data['solidity']:.2f}")
                        logger.info(f"   Défauts doigts: {gesture_data['finger_defects']}")
                        logger.info(f"   Distance estimée: {detector._estimate_distance(gesture_data['area']):.1f}m")
                    
                elif key == ord('c'):
                    # Calibrage de distance
                    if detected and gesture_data:
                        print("\n🎯 CALIBRAGE DISTANCE")
                        try:
                            real_distance = float(input("Entrez la distance réelle en mètres: "))
                            current_area = gesture_data['area']
                            
                            # Mise à jour de la référence de calibrage
                            # reference_area = area * distance²
                            new_reference = current_area * (real_distance ** 2)
                            
                            logger.info(f"📏 Calibrage: {real_distance}m → aire {current_area}")
                            logger.info(f"🔧 Nouvelle référence de calcul: {new_reference:.0f}")
                            
                            # Vous pouvez ici modifier la constante de calibrage dans le détecteur
                            # detector.distance_calibration_factor = new_reference
                            
                        except ValueError:
                            logger.info("❌ Distance invalide")
                    else:
                        logger.info("⚠️ Aucun geste détecté pour calibrage")

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
        # === NETTOYAGE FINAL ===
        logger.info("🧹 Nettoyage système...")
        
        if detector:
            total_runtime = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("📊 RAPPORT FINAL CONTRÔLE GESTUEL:")
            logger.info(f"  ⏱️ Durée session: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames traitées: {detector.frame_count}")
            logger.info(f"  ⚡ FPS moyen: {detector.frame_count/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Gestes détectés: {detector.detection_count}")
            logger.info(f"  📈 Taux détection: {(detector.detection_count/max(detector.frame_count,1))*100:.1f}%")
            logger.info(f"  🏆 Meilleure qualité: {detector.best_mask_score:.1%}")
            logger.info(f"  💾 Données IA sauvegardées: {training_data_count}")
            logger.info(f"  📸 Screenshots totaux: {screenshot_count}")
            logger.info("=" * 60)
            
            # Sauvegarde du meilleur masque
            if detector.best_mask is not None:
                best_mask_path = f"best_mask_{int(time.time())}.png"
                cv2.imwrite(best_mask_path, detector.best_mask)
                logger.info(f"💎 Meilleur masque sauvegardé: {best_mask_path}")
        
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
        
        logger.info("🎉 Session de contrôle gestuel terminée!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Code de sortie: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception non gérée: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)