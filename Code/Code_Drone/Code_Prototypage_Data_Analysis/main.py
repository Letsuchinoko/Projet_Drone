from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

bebop = Bebop()
if bebop.connect(10):
    vision = DroneVision(bebop, is_bebop=True)
    vision.open_video()  # cela crée bebop.sdp dans le répertoire pyparrot
    print("✅ SDP généré")
    time.sleep(2)
    vision.close_video()
    bebop.disconnect()
