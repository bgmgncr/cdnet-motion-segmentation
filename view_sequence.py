import cv2
import glob

paths = sorted(glob.glob("outputs/dynamicBackground/fountain01/overlays/*.png"))

for p in paths:
    img = cv2.imread(p)
    cv2.imshow("Overlay sequence", img)
    key = cv2.waitKey(30)  # 30 ms per frame
    if key == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
