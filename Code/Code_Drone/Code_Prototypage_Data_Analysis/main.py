import cv2
import time
import numpy as np
import threading
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from queue import Queue, Empty
import logging
import os
import signal
import sys
from collections import deque
import glob

DISPLAY_FPS = 20
MAX_QUEUE_SIZE = 3
IMAGES_DIR = "C:/Users/Baptiste/anaconda3/Lib/site-packages/pyparrot/images"
CONNECTION_TIMEOUT = 15
MAX_IMAGE_FILES = 80    # plus large pour ne pas effacer des images actives
IMAGE_KEEP_COUNT = 40   # idem
WATCHDOG_TIMEOUT = 8

frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
processing_active = threading.Event()
processing_active.set()
frame_stats = {
    'frame_count': 0,
    'detection_count': 0,
    'error_count': 0,
    'last_frame_time': time.time(),
    'last_processed_file': None
}
stats_lock = threading.Lock()
image_dir_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bebop_detection.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedGloveDetector:
    def __init__(self):
        self.detection_history = deque(maxlen=25)
        self.min_area = 400
        self.max_area = 70000
        self.min_contour_points = 10
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((7, 7), np.uint8)
        self.stable_detections = deque(maxlen=5)

    def detect_glove(self, frame):
        if frame is None:
            return frame, False
        original_frame = frame.copy()
        h, w = frame.shape[:2]
        try:
            scale_factor = 1.0
            if w > 640:
                scale_factor = 640.0 / w
                work_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))
            else:
                work_frame = frame.copy()
            work_frame = cv2.GaussianBlur(work_frame, (5, 5), 0)
            hsv = cv2.cvtColor(work_frame, cv2.COLOR_BGR2HSV)

            # --- Masque couleur peau (à exclure) ---
            skin_lower = np.array([0, 30, 80])
            skin_upper = np.array([25, 130, 255])
            mask_skin = cv2.inRange(hsv, skin_lower, skin_upper)

            # --- Masque orange du gant ---
            orange_lower = np.array([10, 120, 120])
            orange_upper = np.array([23, 255, 255])
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)

            # --- Masque rouge du gant (deux plages à cause du Hue circulaire) ---
            red_lower1 = np.array([0, 140, 120])
            red_upper1 = np.array([8, 255, 255])
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_lower2 = np.array([170, 140, 120])
            red_upper2 = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            # --- Fusionne orange + rouge, puis enlève la peau ---
            mask_gant = cv2.bitwise_or(mask_orange, mask_red)
            mask = cv2.bitwise_and(mask_gant, cv2.bitwise_not(mask_skin))

            # --- Nettoyage morpho ---
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_contour = self._select_best_contour(contours, work_frame.shape)
            detected = best_contour is not None
            self.stable_detections.append(detected)
            stable_detection = sum(self.stable_detections) >= 2
            self.detection_history.append(stable_detection)

            if stable_detection and best_contour is not None:
                if scale_factor != 1.0:
                    best_contour = (best_contour / scale_factor).astype(np.int32)
                self._draw_detection(original_frame, best_contour)
                with stats_lock:
                    frame_stats['detection_count'] += 1
            result_frame = self._add_overlay(original_frame, stable_detection, mask)
            return result_frame, stable_detection

        except Exception as e:
            logger.error(f"Detection error: {e}")
            with stats_lock:
                frame_stats['error_count'] += 1
            return original_frame, False

    def _select_best_contour(self, contours, frame_shape):
        if not contours:
            return None
        h, w = frame_shape[:2]
        best_contour = None
        best_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            if len(contour) < self.min_contour_points:
                continue
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            aspect_ratio = w_rect / float(h_rect)
            if not (0.2 <= aspect_ratio <= 4.0):
                continue
            if x < 5 or y < 5 or (x + w_rect) > (w - 5) or (y + h_rect) > (h - 5):
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.35:
                    continue
            position_score = 1.0 if y > h * 0.1 else 0.5
            area_score = min(area / 4000.0, 1.0)
            score = area_score * position_score
            if score > best_score:
                best_score = score
                best_contour = contour
        return best_contour

    def _draw_detection(self, frame, contour):
        try:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"GANT DETECTE", (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Aire: {int(area)}", (x, max(y - 35, 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            logger.debug(f"Drawing error: {e}")

    def _add_overlay(self, frame, detected, mask=None):
        try:
            h, w = frame.shape[:2]
            status = "GANT DETECTE" if detected else "RECHERCHE..."
            color = (0, 255, 0) if detected else (0, 255, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            with stats_lock:
                detection_rate = (frame_stats['detection_count'] / max(frame_stats['frame_count'], 1)) * 100
                stats_text = f"Frames: {frame_stats['frame_count']} | Detections: {frame_stats['detection_count']} ({detection_rate:.1f}%)"
            cv2.putText(frame, stats_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            history_text = "Historique: " + "".join(["●" if x else "○" for x in list(self.detection_history)[-25:]])
            cv2.putText(frame, history_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if mask is not None and mask.size > 0:
                mask_small = cv2.resize(mask, (160, 120))
                mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_HOT)
                frame[10:130, w-170:w-10] = mask_colored
                cv2.rectangle(frame, (w-170, 10), (w-10, 130), (255, 255, 255), 1)
                cv2.putText(frame, "Masque", (w-160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return frame
        except Exception as e:
            logger.debug(f"Overlay error: {e}")
            return frame

def vision_callback(args):
    try:
        with image_dir_lock:
            pattern = os.path.join(IMAGES_DIR, "image_*.png")
            files = glob.glob(pattern)
            if not files:
                return
            latest_file = max(files, key=os.path.getmtime)
            if not os.path.exists(latest_file) or os.path.getsize(latest_file) < 1000:
                return
            time.sleep(0.01)
            frame = cv2.imread(latest_file)
            if frame is None:
                return
            h, w = frame.shape[:2]
            if h < 150 or w < 150:
                return
            # Si jamais on prend du retard, flush le buffer pour éviter le lag
            while frame_queue.qsize() > 1:
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put_nowait(frame)
        logger.debug(f"Vision callback: new frame {os.path.basename(latest_file)}")
        with stats_lock:
            frame_stats['frame_count'] += 1
            frame_stats['last_frame_time'] = time.time()
    except Exception as e:
        logger.debug(f"Vision callback error: {e}")

def cleanup_thread():
    logger.info("Cleanup thread started")
    while processing_active.is_set():
        try:
            with image_dir_lock:
                files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
                if len(files) > MAX_IMAGE_FILES:
                    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
                    files_to_remove = files_sorted[IMAGE_KEEP_COUNT:]
                    removed_count = 0
                    for file_path in files_to_remove:
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except OSError:
                            pass
                    if removed_count > 0:
                        logger.info(f"Cleaned up {removed_count} old image files")
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")
        for _ in range(20):
            if not processing_active.is_set():
                break
            time.sleep(0.5)
    logger.info("Cleanup thread terminated")

def display_thread():
    detector = ImprovedGloveDetector()
    logger.info("Display thread started")
    window_name = "Bebop 2 - Detection Gant Amelioree"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    fps_counter = 0
    fps_start_time = time.time()
    while processing_active.is_set():
        try:
            try:
                frame = frame_queue.get(timeout=2.0)
            except Empty:
                logger.warning("No frames received for 2 seconds")
                continue
            if frame is None:
                continue
            processed_frame, detected = detector.detect_glove(frame)
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_elapsed = time.time() - fps_start_time
                current_fps = 30 / fps_elapsed if fps_elapsed > 0 else 0
                logger.info(f"FPS: {current_fps:.1f}")
                fps_start_time = time.time()
            cv2.imshow(window_name, processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("User requested quit")
                processing_active.clear()
                break
            elif key == ord('r'):
                with stats_lock:
                    frame_stats['frame_count'] = 0
                    frame_stats['detection_count'] = 0
                    frame_stats['error_count'] = 0
                logger.info("Statistics reset")
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, processed_frame)
                logger.info(f"Screenshot saved: {screenshot_name}")
        except Exception as e:
            logger.error(f"Display thread error: {e}")
            time.sleep(0.1)
    cv2.destroyAllWindows()
    logger.info("Display thread terminated")

def connection_monitor_thread():
    logger.info("Connection monitor started")
    last_frame_count = 0
    check_interval = 4
    while processing_active.is_set():
        time.sleep(check_interval)
        with stats_lock:
            current_frames = frame_stats['frame_count']
            errors = frame_stats['error_count']
            detections = frame_stats['detection_count']
            last_received_time = frame_stats['last_frame_time']
        frame_diff = current_frames - last_frame_count
        last_frame_count = current_frames
        time_since_last_frame = time.time() - last_received_time
        with image_dir_lock:
            files = glob.glob(os.path.join(IMAGES_DIR, "image_*.png"))
        if frame_diff == 0 or time_since_last_frame > WATCHDOG_TIMEOUT:
            logger.warning("No new frames received for monitoring (stream may be frozen). Manual restart required.")
        else:
            avg_fps = frame_diff / check_interval
            detection_rate = (detections / max(current_frames, 1)) * 100
            logger.info(f"STATS - Frames: {current_frames}, FPS: {avg_fps:.1f}, Detections: {detection_rate:.1f}%, Errors: {errors}")
    logger.info("Connection monitor terminated")

def signal_handler(sig, frame):
    logger.info("Stop signal received")
    processing_active.clear()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Starting Bebop 2 glove detection system")
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        return False
    bebop = None
    vision = None
    threads = []
    try:
        bebop = Bebop()
        logger.info("Connecting to Bebop 2...")
        success = bebop.connect(CONNECTION_TIMEOUT)
        if not success:
            logger.error("Failed to connect to drone")
            return False
        logger.info("Drone connected successfully")
        vision = DroneVision(bebop, is_bebop=True)
        vision.set_user_callback_function(vision_callback)
        display_thread_obj = threading.Thread(target=display_thread, daemon=True)
        cleanup_thread_obj = threading.Thread(target=cleanup_thread, daemon=True)
        monitor_thread_obj = threading.Thread(target=connection_monitor_thread, daemon=True)
        threads = [display_thread_obj, cleanup_thread_obj, monitor_thread_obj]
        for thread in threads:
            thread.start()
        logger.info("All threads started (display, cleanup, monitor)")
        logger.info("Opening video stream...")
        if not vision.open_video():
            logger.error("Failed to open video stream")
            return False
        logger.info("Video stream opened successfully")
        logger.info("Detection system active")
        logger.info("Controls: 'q'/ESC=Quit, 'r'=Reset stats, 's'=Screenshot")
        try:
            while processing_active.is_set():
                time.sleep(1)
                try:
                    if cv2.getWindowProperty("Bebop 2 - Detection Gant Amelioree", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed")
                        break
                except:
                    pass
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return False
    finally:
        logger.info("Starting cleanup...")
        processing_active.clear()
        time.sleep(2)
        if vision:
            try:
                vision.close_video()
                logger.info("Video stream closed")
            except:
                pass
        if bebop:
            try:
                bebop.disconnect()
                logger.info("Drone disconnected")
            except:
                pass
        cv2.destroyAllWindows()
        for thread in threads:
            try:
                thread.join(timeout=3)
            except:
                pass
        logger.info("Cleanup completed")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
