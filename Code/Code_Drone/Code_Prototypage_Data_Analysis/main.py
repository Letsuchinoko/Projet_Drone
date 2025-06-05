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

# --- PARAMÈTRES VIDÉO ---
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

# --- DÉTECTEUR GANT BICOLORE AMÉLIORÉ ---
class EnhancedGloveDetector:
    def __init__(self):
        # Historique et stabilisation
        self.detection_history = deque(maxlen=50)  # Historique plus long
        self.stable_detections = deque(maxlen=8)   # Fenêtre de stabilisation
        self.confidence_threshold = 5              # Sur 8 détections
        
        # Paramètres de détection affinés
        self.min_area = 300
        self.max_area = 80000
        self.min_contour_points = 8
        
        # Kernels morphologiques optimisés
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Statistiques détaillées
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        self.fps_start_time = time.time()
        self.total_detections = 0
        self.confirmed_detections = 0
        
        # Tracking amélioré
        self.last_detection_center = None
        self.detection_positions = deque(maxlen=10)
        self.max_movement = 120
        self.detection_cooldown = 0
        
        # Cache pour optimisation
        self.last_frame_id = None
        self.last_result = None
        
        # Qualité de détection
        self.detection_scores = deque(maxlen=20)
        self.min_detection_score = 0.3

    def detect_glove(self, frame):
        """Détection améliorée avec tracking et validation"""
        if frame is None:
            return frame, False
            
        # Cache par ID de frame
        frame_id = id(frame)
        if frame_id == self.last_frame_id and self.last_result is not None:
            return self.last_result
            
        original_frame = frame.copy()
        self.frame_count += 1
        
        try:
            h, w = frame.shape[:2]
            
            # Redimensionnement adaptatif pour performance
            scale_factor = 1.0
            if w > 680:
                scale_factor = 680.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            
            # Prétraitement amélioré
            work_frame = cv2.medianBlur(work_frame, 5)  # Réduction bruit
            work_frame = cv2.GaussianBlur(work_frame, (3, 3), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)
            
            # Création du masque couleur amélioré
            mask = self._create_enhanced_color_mask(hsv)
            
            # Morphologie progressive
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.erode(mask, self.kernel_erode, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
            
            # Détection de contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour, detection_score = self._select_best_contour_with_score(contours, work_frame.shape)
            
            # Validation avancée
            raw_detected = best_contour is not None and detection_score > self.min_detection_score
            validated_detected = self._validate_detection_advanced(best_contour, scale_factor, detection_score)
            
            # Gestion du cooldown
            if self.detection_cooldown > 0:
                self.detection_cooldown -= 1
                validated_detected = False
            
            # Stabilisation avec confiance
            self.stable_detections.append(validated_detected)
            stable_detection = sum(self.stable_detections) >= self.confidence_threshold
            
            # Mise à jour historique et stats
            self.detection_history.append(stable_detection)
            if raw_detected:
                self.total_detections += 1
            if stable_detection:
                self.confirmed_detections += 1
                self.detection_count += 1
            
            # Enregistrement du score
            if detection_score > 0:
                self.detection_scores.append(detection_score)
            
            # Visualisation
            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_enhanced_detection(original_frame, best_contour, detection_score)
            
            # Overlay d'informations enrichi
            result_frame = self._add_enhanced_overlay(original_frame, stable_detection, mask, detection_score)
            
            # Cache du résultat
            self.last_frame_id = frame_id
            self.last_result = (result_frame, stable_detection)
            
            return result_frame, stable_detection
            
        except Exception as e:
            logger.debug(f"Detection error: {e}")
            self.error_count += 1
            return original_frame, False

    def _create_enhanced_color_mask(self, hsv):
        """Masque couleur amélioré avec exclusions intelligentes"""
        try:
            h, w = hsv.shape[:2]
            
            # Masques de couleurs multiples pour plus de robustesse
            
            # 1. Masque peau étendu (à exclure)
            skin_lower1 = np.array([0, 25, 60])
            skin_upper1 = np.array([30, 140, 255])
            mask_skin1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
            
            skin_lower2 = np.array([5, 40, 80])
            skin_upper2 = np.array([25, 120, 200])
            mask_skin2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
            
            mask_skin = cv2.bitwise_or(mask_skin1, mask_skin2)
            
            # 2. Masque orange optimisé (gant principal)
            orange_lower1 = np.array([8, 100, 100])
            orange_upper1 = np.array([25, 255, 255])
            mask_orange1 = cv2.inRange(hsv, orange_lower1, orange_upper1)
            
            orange_lower2 = np.array([12, 120, 120])
            orange_upper2 = np.array([22, 240, 240])
            mask_orange2 = cv2.inRange(hsv, orange_lower2, orange_upper2)
            
            mask_orange = cv2.bitwise_or(mask_orange1, mask_orange2)
            
            # 3. Masque rouge étendu (gant secondaire)
            red_lower1 = np.array([0, 120, 100])
            red_upper1 = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            
            red_lower2 = np.array([165, 120, 100])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            
            red_lower3 = np.array([170, 140, 120])
            red_upper3 = np.array([180, 240, 240])
            mask_red3 = cv2.inRange(hsv, red_lower3, red_upper3)
            
            mask_red = cv2.bitwise_or(mask_red1, cv2.bitwise_or(mask_red2, mask_red3))
            
            # 4. Combinaison des couleurs du gant
            mask_gant = cv2.bitwise_or(mask_orange, mask_red)
            
            # 5. Exclusion de la peau dilatée
            mask_skin_dilated = cv2.dilate(mask_skin, self.kernel_close, iterations=2)
            mask_final = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin_dilated))
            
            # 6. Exclusion des bords avec marge variable
            border_mask = np.ones((h, w), dtype=np.uint8) * 255
            border_size = max(15, min(w, h) // 30)  # Bordure adaptative
            border_mask[:border_size, :] = 0
            border_mask[-border_size:, :] = 0
            border_mask[:, :border_size] = 0
            border_mask[:, -border_size:] = 0
            
            mask_final = cv2.bitwise_and(mask_final, border_mask)
            
            # 7. Filtre supplémentaire pour éliminer le bruit
            mask_final = cv2.medianBlur(mask_final, 3)
            
            return mask_final
            
        except Exception as e:
            logger.debug(f"Enhanced mask error: {e}")
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

    def _select_best_contour_with_score(self, contours, frame_shape):
        """Sélection du meilleur contour avec score de qualité"""
        if not contours:
            return None, 0.0
            
        try:
            h, w = frame_shape[:2]
            best_contour = None
            best_score = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtres de base
                if area < self.min_area or area > self.max_area:
                    continue
                if len(contour) < self.min_contour_points:
                    continue
                
                # Rectangle englobant
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Ratio d'aspect
                aspect_ratio = w_rect / float(h_rect)
                if not (0.15 <= aspect_ratio <= 5.0):
                    continue
                
                # Éviter les bords avec marge
                margin = max(8, min(w, h) // 40)
                if (x < margin or y < margin or 
                    (x + w_rect) > (w - margin) or 
                    (y + h_rect) > (h - margin)):
                    continue
                
                # Analyse de forme avancée
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area <= 0:
                    continue
                
                solidity = area / hull_area
                if solidity < 0.25:  # Plus permissif
                    continue
                
                # Calcul du score composite
                
                # 1. Score d'aire (optimal autour de 5000)
                area_score = min(area / 5000.0, 1.0) if area < 5000 else max(0.5, 1.0 - (area - 5000) / 15000)
                
                # 2. Score de solidité (optimal autour de 0.7)
                solidity_score = min(solidity * 2, 1.0) if solidity < 0.5 else min((1.0 - solidity) * 2, 1.0)
                
                # 3. Score de position (éviter le haut de l'image)
                position_score = 1.0 if y > h * 0.15 else 0.6
                
                # 4. Score de ratio d'aspect (optimal autour de 1.0)
                aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.3
                aspect_score = max(0.3, min(1.0, aspect_score))
                
                # 5. Score de compacité
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter * perimeter)
                    compactness_score = min(compactness * 2, 1.0)
                else:
                    compactness_score = 0.0
                
                # Score final pondéré
                final_score = (area_score * 0.3 + 
                              solidity_score * 0.25 + 
                              position_score * 0.2 + 
                              aspect_score * 0.15 + 
                              compactness_score * 0.1)
                
                if final_score > best_score:
                    best_score = final_score
                    best_contour = contour
            
            return best_contour, best_score
            
        except Exception as e:
            logger.debug(f"Enhanced contour selection error: {e}")
            return None, 0.0

    def _validate_detection_advanced(self, contour, scale_factor, detection_score):
        """Validation avancée avec tracking spatial"""
        if contour is None or detection_score < self.min_detection_score:
            self.last_detection_center = None
            return False
        
        try:
            # Calculer le centre
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return False
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            current_center = (cx, cy)
            
            # Validation du mouvement si détection précédente
            if self.last_detection_center is not None:
                distance = np.sqrt((cx - self.last_detection_center[0])**2 + 
                                 (cy - self.last_detection_center[1])**2)
                
                # Seuil adaptatif basé sur la taille de l'objet
                area = cv2.contourArea(contour)
                adaptive_max_movement = self.max_movement * (1 + area / 10000.0)
                
                if distance > adaptive_max_movement:
                    logger.debug(f"Movement too large: {distance:.1f} > {adaptive_max_movement:.1f}")
                    self.detection_cooldown = 5
                    return False
            
            # Enregistrer la position
            self.last_detection_center = current_center
            self.detection_positions.append(current_center)
            
            # Validation de la trajectoire (si suffisamment de points)
            if len(self.detection_positions) >= 3:
                # Vérifier que la trajectoire n'est pas trop erratique
                positions = list(self.detection_positions)
                total_distance = 0
                for i in range(1, len(positions)):
                    dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                  (positions[i][1] - positions[i-1][1])**2)
                    total_distance += dist
                
                avg_movement = total_distance / (len(positions) - 1)
                if avg_movement > 50:  # Mouvement trop erratique
                    logger.debug(f"Trajectory too erratic: {avg_movement:.1f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Advanced validation error: {e}")
            return False

    def _draw_enhanced_detection(self, frame, contour, detection_score):
        """Dessin amélioré avec informations détaillées"""
        try:
            # Couleur basée sur la confiance
            confidence_color = (0, int(255 * detection_score), int(255 * (1 - detection_score)))
            
            # Contour principal avec épaisseur variable
            thickness = max(2, int(4 * detection_score))
            cv2.drawContours(frame, [contour], -1, confidence_color, thickness)
            
            # Rectangle englobant
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 0), 2)
            
            # Centre avec cercles concentriques
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 16, confidence_color, 1)
            
            # Informations détaillées
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Texte principal
            cv2.putText(frame, f"GANT DETECTE", (x, max(y - 15, 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, confidence_color, 2)
            
            # Informations techniques
            info_text = f"Score: {detection_score:.2f} | Aire: {int(area)} | Sol: {solidity:.2f}"
            cv2.putText(frame, info_text, (x, max(y - 40, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Trajectoire récente
            if len(self.detection_positions) > 1:
                positions = list(self.detection_positions)
                for i in range(1, len(positions)):
                    alpha = i / len(positions)
                    color = (int(100 * alpha), int(150 * alpha), int(200 * alpha))
                    cv2.line(frame, positions[i-1], positions[i], color, 2)
                    
        except Exception as e:
            logger.debug(f"Enhanced drawing error: {e}")

    def _add_enhanced_overlay(self, frame, detected, mask=None, detection_score=0.0):
        """Overlay d'informations enrichi"""
        try:
            h, w = frame.shape[:2]
            
            # Status principal avec couleur dynamique
            if detected:
                status = "🎯 GANT DETECTE"
                color = (0, 255, 0)
            else:
                status = "🔍 RECHERCHE GANT..."
                color = (0, 255, 255)
            
            cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Statistiques principales
            detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
            accuracy = (self.confirmed_detections / max(self.total_detections, 1)) * 100
            
            stats_text = f"Frames: {self.frame_count} | Detections: {self.detection_count} ({detection_rate:.1f}%)"
            cv2.putText(frame, stats_text, (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Statistiques avancées
            avg_score = np.mean(self.detection_scores) if self.detection_scores else 0.0
            stability = sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0.0
            
            advanced_text = f"Precision: {accuracy:.1f}% | Score moy: {avg_score:.2f} | Stabilite: {stability:.1%}"
            cv2.putText(frame, advanced_text, (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            
            # Score actuel si détection
            if detected and detection_score > 0:
                score_text = f"Score actuel: {detection_score:.3f}"
                cv2.putText(frame, score_text, (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Historique visuel étendu
            history_colors = []
            for det in list(self.detection_history)[-50:]:
                if det:
                    history_colors.append("●")
                else:
                    history_colors.append("○")
            
            # Diviser l'historique en lignes
            history_line1 = "".join(history_colors[-25:]) if len(history_colors) >= 25 else "".join(history_colors)
            history_line2 = "".join(history_colors[-50:-25]) if len(history_colors) > 25 else ""
            
            cv2.putText(frame, f"Hist: {history_line1}", (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if history_line2:
                cv2.putText(frame, f"     {history_line2}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Informations temporelles
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # FPS en temps réel
            if self.frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self.fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                self.current_fps = fps
                self.fps_start_time = now
            
            if hasattr(self, 'current_fps'):
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (w - 120, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            
            # Masque amélioré avec informations
            if mask is not None and mask.size > 0:
                try:
                    mask_small = cv2.resize(mask, (180, 135))
                    mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
                    
                    # Position adaptative du masque
                    mask_x, mask_y = w - 190, 90
                    frame[mask_y:mask_y+135, mask_x:mask_x+180] = mask_colored
                    
                    # Cadre et informations
                    cv2.rectangle(frame, (mask_x, mask_y), (mask_x+180, mask_y+135), (255, 255, 255), 2)
                    cv2.putText(frame, "Masque Couleur", (mask_x, mask_y + 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Statistiques du masque
                    mask_pixels = np.count_nonzero(mask)
                    mask_percentage = (mask_pixels / mask.size) * 100
                    cv2.putText(frame, f"Pixels: {mask_percentage:.1f}%", (mask_x, mask_y + 165), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                except Exception:
                    pass
            
            # Indicateur de qualité global
            quality_indicators = []
            if self.frame_count > 50:
                recent_detections = sum(list(self.detection_history)[-50:])
                if recent_detections > 25:
                    quality_indicators.append("🟢 Haute")
                elif recent_detections > 10:
                    quality_indicators.append("🟡 Moyenne")
                else:
                    quality_indicators.append("🔴 Faible")
                
                quality_text = f"Qualite: {' '.join(quality_indicators)}"
                cv2.putText(frame, quality_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            
            return frame
            
        except Exception as e:
            logger.debug(f"Enhanced overlay error: {e}")
            return frame

    def get_statistics(self):
        """Retourne des statistiques détaillées"""
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'total_detections': self.total_detections,
            'confirmed_detections': self.confirmed_detections,
            'error_count': self.error_count,
            'detection_rate': (self.detection_count / max(self.frame_count, 1)) * 100,
            'accuracy': (self.confirmed_detections / max(self.total_detections, 1)) * 100,
            'average_score': np.mean(self.detection_scores) if self.detection_scores else 0.0,
            'stability': sum(self.stable_detections) / len(self.stable_detections) if self.stable_detections else 0.0
        }

# --- THREAD DE PILOTAGE DRONE AMÉLIORÉ ---
def enhanced_drone_control_thread(bebop):
    """Thread de contrôle drone avec commandes étendues"""
    logger.info("Contrôle du drone (pyparrot) démarré.")
    print("\n[Commandes clavier étendues]\n"
          "  t = décoller\n"
          "  l = atterrir\n"
          "  e = quitter\n"
          "  f = avancer\n"
          "  b = reculer\n"
          "  g = gauche\n"
          "  d = droite\n"
          "  h = haut\n"
          "  m = bas\n"
          "  a = rotation gauche\n"
          "  c = rotation droite\n"
          "  s = stationnaire (hover)\n"
          "  1 = mouvement lent\n"
          "  2 = mouvement normal\n"
          "  3 = mouvement rapide\n")
    
    # Paramètres de vitesse
    speed_levels = {
        1: {'roll': 20, 'pitch': 20, 'yaw': 30, 'vertical': 20},
        2: {'roll': 40, 'pitch': 40, 'yaw': 50, 'vertical': 30},
        3: {'roll': 60, 'pitch': 60, 'yaw': 70, 'vertical': 40}
    }
    current_speed = 2
    
    while True:
        try:
            key = input(f"[Vitesse {current_speed}] > ").strip().lower()
        except EOFError:
            print("Arrêt du thread contrôle drone (entrée clavier coupée).")
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
            print("🔚 Fin du vol, arrêt du script.")
            break
        elif key == 's':
            bebop.hover()
            print("⏸️ Mode stationnaire")
        elif key in ['1', '2', '3']:
            current_speed = int(key)
            print(f"⚡ Vitesse réglée sur niveau {current_speed}")
        elif key == 'f':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=0, pitch=params['pitch'], yaw=0, vertical_movement=0, duration=0.5)
            print("⬆️ Avancer")
        elif key == 'b':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=0, pitch=-params['pitch'], yaw=0, vertical_movement=0, duration=0.5)
            print("⬇️ Reculer")
        elif key == 'g':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=-params['roll'], pitch=0, yaw=0, vertical_movement=0, duration=0.5)
            print("⬅️ Gauche")
        elif key == 'd':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=params['roll'], pitch=0, yaw=0, vertical_movement=0, duration=0.5)
            print("➡️ Droite")
        elif key == 'h':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=params['vertical'], duration=0.5)
            print("⬆️ Haut")
        elif key == 'm':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-params['vertical'], duration=0.5)
            print("⬇️ Bas")
        elif key == 'a':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=0, pitch=0, yaw=-params['yaw'], vertical_movement=0, duration=0.5)
            print("🔄 Rotation gauche")
        elif key == 'c':
            params = speed_levels[current_speed]
            bebop.fly_direct(roll=0, pitch=0, yaw=params['yaw'], vertical_movement=0, duration=0.5)
            print("🔄 Rotation droite")
        else:
            print("❌ Commande inconnue.")

def main():
    """Fonction principale améliorée basée sur le prototype qui marche"""
    logger.info("=== BEBOP 2 ENHANCED DETECTION SYSTEM ===")
    logger.info("Démarrage du système de détection amélioré...")
    
    bebop = None
    pipe = None
    detector = None
    start_time = time.time()
    
    try:
        # === PHASE 1: CONNEXION DRONE ===
        logger.info("📡 Connexion au drone...")
        bebop = Bebop()
        if not bebop.connect(10):
            logger.error("❌ Echec connexion drone")
            logger.error("Vérifiez:")
            logger.error("  - Drone allumé et prêt")
            logger.error("  - Connexion WiFi au drone")
            logger.error("  - Adresse IP accessible (192.168.42.1)")
            return False

        logger.info("✅ Drone connecté avec succès!")
        
        # === PHASE 2: DÉMARRAGE FLUX VIDÉO ===
        logger.info("📹 Démarrage du flux vidéo...")
        bebop.start_video_stream()
        logger.info("✅ Flux vidéo demandé au drone (start_video_stream).")
        time.sleep(3)  # Attendre stabilisation

        # === PHASE 3: CONTRÔLE DRONE EN ARRIÈRE-PLAN ===
        ctrl_thread = threading.Thread(
            target=enhanced_drone_control_thread, 
            args=(bebop,), 
            daemon=True
        )
        ctrl_thread.start()
        logger.info("🎮 Thread de contrôle drone démarré")

        # === PHASE 4: CONFIGURATION FFMPEG ===
        logger.info("🔧 Configuration du pipeline vidéo...")
        
        # Chemin SDP (dans site-packages pyparrot)
        sdp_path = os.path.join(os.path.dirname(pyparrot.__file__), "utils", "bebop.sdp")
        if not os.path.exists(sdp_path):
            logger.error(f"❌ Fichier SDP introuvable: {sdp_path}")
            return False
        
        logger.info(f"✅ Fichier SDP trouvé: {sdp_path}")

        # Commande FFmpeg optimisée
        ffmpeg_cmd = [
            'ffmpeg',
            '-protocol_whitelist', 'file,rtp,udp',
            '-analyzeduration', '2000000',  # 2 secondes d'analyse
            '-probesize', '2000000',        # 2MB de probe
            '-i', sdp_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        logger.info(f"🚀 Lancement FFmpeg: {' '.join(ffmpeg_cmd)}")
        
        try:
            pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)
            logger.info("✅ Pipeline FFmpeg initialisé")
        except FileNotFoundError:
            logger.error("❌ FFmpeg non trouvé ! Installez FFmpeg et ajoutez-le au PATH.")
            return False

        # === PHASE 5: INITIALISATION DÉTECTEUR ===
        logger.info("🎯 Initialisation du détecteur amélioré...")
        detector = EnhancedGloveDetector()
        
        # === PHASE 6: INTERFACE UTILISATEUR ===
        window_name = "Bebop 2 - Détection Gant Améliorée"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        screenshot_count = 0
        
        logger.info("=" * 60)
        logger.info("🎮 COMMANDES CLAVIER (Fenêtre vidéo):")
        logger.info("  'q' ou ESC    = Quitter")
        logger.info("  'r'           = Reset statistiques")
        logger.info("  's'           = Screenshot")
        logger.info("  'd'           = Afficher stats détaillées")
        logger.info("  'p'           = Pause/Reprendre")
        logger.info("=" * 60)
        logger.info("🚁 COMMANDES DRONE (Terminal):")
        logger.info("  Voir le terminal de contrôle pour les commandes de vol")
        logger.info("=" * 60)
        
        # === PHASE 7: BOUCLE PRINCIPALE ===
        logger.info("🎬 Démarrage de la boucle principale...")
        
        paused = False
        frame_skip_count = 0
        last_stats_time = time.time()
        
        while True:
            try:
                # Lecture de la frame brute
                raw_frame = pipe.stdout.read(WIDTH * HEIGHT * 3)
                
                if len(raw_frame) != WIDTH * HEIGHT * 3:
                    logger.error("❌ Problème lecture frame vidéo, arrêt.")
                    logger.error(f"Reçu {len(raw_frame)} bytes, attendu {WIDTH * HEIGHT * 3}")
                    break
                
                # Conversion en image OpenCV
                frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))
                
                # Traitement de détection (si pas en pause)
                if not paused:
                    processed_frame, detected = detector.detect_glove(frame)
                else:
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, "⏸️ PAUSE", (WIDTH//2 - 60, HEIGHT//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    detected = False

                # Affichage
                cv2.imshow(window_name, processed_frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # ESC
                    logger.info("🛑 Arrêt demandé par l'utilisateur.")
                    break
                    
                elif key == ord('r'):
                    # Reset complet des statistiques
                    old_stats = detector.get_statistics()
                    detector.__init__()  # Réinitialisation complète
                    logger.info("🔄 Statistiques réinitialisées.")
                    logger.info(f"   Anciennes stats: {old_stats['frame_count']} frames, "
                               f"{old_stats['detection_count']} détections")
                    
                elif key == ord('s'):
                    # Screenshot avec métadonnées
                    timestamp = int(time.time())
                    stats = detector.get_statistics()
                    screenshot_name = f"screenshot_bebop_{timestamp}_{screenshot_count:03d}.png"
                    
                    # Ajout d'informations dans l'image
                    info_frame = processed_frame.copy()
                    info_text = f"Frame #{stats['frame_count']} | Det: {stats['detection_rate']:.1f}% | {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    cv2.putText(info_frame, info_text, (10, HEIGHT - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imwrite(screenshot_name, info_frame)
                    logger.info(f"📸 Screenshot sauvegardé: {screenshot_name}")
                    screenshot_count += 1
                    
                elif key == ord('d'):
                    # Affichage des statistiques détaillées
                    stats = detector.get_statistics()
                    logger.info("📊 STATISTIQUES DÉTAILLÉES:")
                    logger.info(f"   Frames traitées: {stats['frame_count']}")
                    logger.info(f"   Détections confirmées: {stats['detection_count']}")
                    logger.info(f"   Taux de détection: {stats['detection_rate']:.2f}%")
                    logger.info(f"   Précision: {stats['accuracy']:.2f}%")
                    logger.info(f"   Score moyen: {stats['average_score']:.3f}")
                    logger.info(f"   Stabilité: {stats['stability']:.1%}")
                    logger.info(f"   Erreurs: {stats['error_count']}")
                    
                elif key == ord('p'):
                    # Pause/Reprendre
                    paused = not paused
                    status = "⏸️ PAUSE" if paused else "▶️ REPRENDRE"
                    logger.info(f"{status} - Traitement {'suspendu' if paused else 'repris'}")

                # Affichage périodique des FPS et stats
                if detector.frame_count % 150 == 0 and detector.frame_count > 0:  # Toutes les 5 secondes environ
                    current_time = time.time()
                    if current_time - last_stats_time >= 5:
                        stats = detector.get_statistics()
                        elapsed = current_time - start_time
                        overall_fps = stats['frame_count'] / elapsed
                        
                        logger.info(f"📈 Stats: {stats['frame_count']} frames | "
                                   f"FPS: {overall_fps:.1f} | "
                                   f"Détections: {stats['detection_count']} ({stats['detection_rate']:.1f}%) | "
                                   f"Stabilité: {stats['stability']:.1%}")
                        last_stats_time = current_time

            except KeyboardInterrupt:
                logger.info("⌨️ Interruption clavier détectée.")
                break
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle principale: {e}")
                frame_skip_count += 1
                if frame_skip_count > 10:
                    logger.error("Trop d'erreurs consécutives, arrêt.")
                    break
                continue

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        # === PHASE 8: NETTOYAGE FINAL ===
        logger.info("🧹 Nettoyage et arrêt du système...")
        
        # Statistiques finales
        if detector:
            final_stats = detector.get_statistics()
            total_runtime = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("📊 STATISTIQUES FINALES:")
            logger.info(f"  ⏱️ Durée totale: {total_runtime:.1f}s")
            logger.info(f"  🎞️ Frames traitées: {final_stats['frame_count']}")
            logger.info(f"  ⚡ FPS moyen: {final_stats['frame_count']/max(total_runtime,1):.1f}")
            logger.info(f"  🎯 Détections confirmées: {final_stats['detection_count']}")
            logger.info(f"  📈 Taux de détection: {final_stats['detection_rate']:.2f}%")
            logger.info(f"  🎯 Précision: {final_stats['accuracy']:.2f}%")
            logger.info(f"  📊 Score moyen: {final_stats['average_score']:.3f}")
            logger.info(f"  ⚖️ Stabilité: {final_stats['stability']:.1%}")
            logger.info(f"  ❌ Erreurs: {final_stats['error_count']}")
            logger.info("=" * 60)
        
        # Arrêt du pipeline vidéo
        if pipe:
            try:
                pipe.terminate()
                pipe.wait(timeout=5)
                logger.info("✅ Pipeline FFmpeg fermé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture pipeline: {e}")
        
        # Fermeture OpenCV
        try:
            cv2.destroyAllWindows()
            logger.info("✅ Fenêtres OpenCV fermées")
        except Exception as e:
            logger.warning(f"⚠️ Erreur fermeture OpenCV: {e}")
        
        # Déconnexion drone
        if bebop:
            try:
                bebop.disconnect()
                logger.info("✅ Drone déconnecté")
            except Exception as e:
                logger.warning(f"⚠️ Erreur déconnexion drone: {e}")
        
        logger.info("🎉 Script terminé avec succès.")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Programme terminé avec le code {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 Exception non gérée: {e}")
        import traceback
        logger.error(f"Traceback complet: {traceback.format_exc()}")
        sys.exit(1)