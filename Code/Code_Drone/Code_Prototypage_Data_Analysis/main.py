import cv2
import os

# 🧠 Résout dynamiquement le chemin du fichier .sdp à partir de ce script
current_dir = os.path.dirname(os.path.abspath(__file__))
sdp_path = os.path.join(current_dir, "bebop.sdp")

# 👇 Assure-toi que ffmpeg le trouve
cap = cv2.VideoCapture(sdp_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir le flux vidéo à {sdp_path}")
else:
    print("✅ Flux vidéo ouvert.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Trame incomplète")
            continue

        cv2.imshow("🎥 Flux direct Bebop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
