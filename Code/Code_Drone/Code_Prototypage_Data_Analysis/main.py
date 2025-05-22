import os
import glob
import cv2
import time
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

DISPLAY_INTERVAL = 1 / 10  # 10 FPS
last_display_time = 0
last_image_name = None

def vision_callback(_):
    global last_display_time, last_image_name

    now = time.time()
    if now - last_display_time < DISPLAY_INTERVAL:
        return

    images_dir = os.path.join(os.path.dirname(__file__), "images")
    files = glob.glob(os.path.join(images_dir, "image_*.png"))

    try:
        files = sorted(files, key=os.path.getmtime)
        if not files:
            return
        last_file = files[-1]
        if last_file == last_image_name:
            return

        last_image_name = last_file
        last_display_time = now

        img = cv2.imread(last_file)
        if img is not None:
            img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))  # x2 pour lisibilité
            cv2.imshow("🎥 Flux Bebop2", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    except Exception as e:
        print(f"⚠️ Callback error: {e}")

# === Drone connection ===
bebop = Bebop()
bebop.drone_ip = "192.168.42.1"

print("Connexion au drone...")
if bebop.connect(10):
    print("✅ Connecté.")
    vision = DroneVision(bebop, is_bebop=True)
    vision.set_user_callback_function(vision_callback)

    if vision.open_video():
        print("🎥 Flux actif. Ctrl+C pour quitter.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("⏹ Arrêt manuel.")
        finally:
            vision.close_video()
            bebop.disconnect()
            cv2.destroyAllWindows()
    else:
        print("❌ Erreur ouverture flux.")
        bebop.disconnect()
else:
    print("❌ Erreur connexion.")
