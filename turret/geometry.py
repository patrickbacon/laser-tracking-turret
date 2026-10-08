import math
import numpy as np
import cv2


class Camera:
    def __init__(self, cfg):
        i = cfg["camera"]["intrinsics"]
        if i["fx"] is None:
            raise ValueError("Camera not calibrated.")
        self.K = np.array([[i["fx"], 0, i["cx"]],
                           [0, i["fy"], i["cy"]],
                           [0, 0, 1]], dtype=np.float64)
        self.dist = np.array(i["dist"], dtype=np.float64)
        self.fy = i["fy"]

    def pixel_to_angles(self, x, y):
        pts = np.array([[[float(x), float(y)]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K, self.dist)
        xn, yn = und[0, 0]
        az = math.degrees(math.atan(xn))
        el = math.degrees(math.atan(-yn))
        return az, el

    def angular_height_to_range(self, box_h_px, real_h_m):
        if box_h_px <= 1:
            return float("inf")
        return (real_h_m * self.fy) / box_h_px
