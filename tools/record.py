import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2, yaml

from turret.geometry import Camera

cfg = yaml.safe_load(open("config.yaml"))
geo = Camera(cfg)
PERSON_H = cfg["safety"]["person_height_assumed_m"]

name = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%m%d_%H%M%S")
out = f"sessions/{name}"
os.makedirs(out, exist_ok=True)
log = open(f"{out}/detections.jsonl", "w")
print(f"recording to {out}  --  q to stop")

model = YOLO("yolov8n_ncnn_model")
face = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx", "", (320, 320),
    score_threshold=0.6)

cam = Picamera2()
cam.configure(cam.create_video_configuration(
    main={"size": (1332, 990), "format": "RGB888"}))
cam.start()
time.sleep(1)

frame_i = 0
chin_cache = {}
t_start = time.time()

while True:
    frame = cam.capture_array()
    t = time.time()

    res = model(frame, classes=[0], conf=0.60, imgsz=640, verbose=False)

    dets = []
    for bi, b in enumerate(res[0].boxes):
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cx = (x1 + x2) / 2
        cy = y1 + (y2 - y1) * 0.25
        az, el = geo.pixel_to_angles(cx, cy)
        rng = geo.angular_height_to_range(y2 - y1, PERSON_H)

        if frame_i % 3 == 0:
            fy2 = y1 + int((y2 - y1) * 0.4)
            crop = frame[max(0, y1):fy2, max(0, x1):x2]
            if crop.size and crop.shape[0] >= 30 and crop.shape[1] >= 30:
                face.setInputSize((crop.shape[1], crop.shape[0]))
                _, faces = face.detect(crop)
                if faces is not None and len(faces):
                    f = faces[0]
                    eye_mid = (y1 + f[5] + y1 + f[7]) / 2
                    mth_mid = (y1 + f[11] + y1 + f[13]) / 2
                    chin_y = mth_mid + (mth_mid - eye_mid)
                    chin_cache[bi] = float((chin_y - y1) / max(1, (y2 - y1)))

        chin_el = None
        chin_py = None
        if bi in chin_cache:
            chin_py = float(y1 + chin_cache[bi] * (y2 - y1))
            _, chin_el = geo.pixel_to_angles(cx, chin_py)
            chin_el = float(chin_el)

        dets.append({"az": float(az), "el": float(el), "chin_el": chin_el,
                     "range_m": None if math.isinf(rng) else float(rng),
                     "box": [x1, y1, x2, y2],
                     "chin_py": chin_py})

    cv2.imwrite(f"{out}/{frame_i:06d}.jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    log.write(json.dumps({"i": frame_i, "t": t, "dets": dets}) + "\n")

    disp = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(disp, f"REC {frame_i}  {t-t_start:.1f}s", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow("recording", disp)

    frame_i += 1
    if cv2.waitKey(1) == ord('q'):
        break

log.close()
cam.stop()
cv2.destroyAllWindows()
print(f"saved {frame_i} frames to {out}")
