import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2, time, yaml, math
import pygame

from turret.geometry import Camera
from turret.tracking import Tracker
from turret.safety import AimGate

cfg = yaml.safe_load(open("config.yaml"))
geo = Camera(cfg)
tracker = Tracker(cfg)
gate = AimGate(cfg)
LEAD_S = cfg["tracking"]["predict_ahead_ms"] / 1000.0
PERSON_H = cfg["safety"]["person_height_assumed_m"]

pygame.mixer.init()
SND = pygame.mixer.Sound("sounds/detect.wav")
last_played = {}
COOLDOWN = 8.0

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

while True:
    frame = cam.capture_array()
    t = time.time()
    t0 = time.perf_counter()

    res = model(frame, classes=[0], conf=0.60, imgsz=640, verbose=False)

    detections = []
    for bi, b in enumerate(res[0].boxes):
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        cx = (x1 + x2) / 2
        cy = y1 + (y2 - y1) * 0.25
        az, el = geo.pixel_to_angles(cx, cy)
        rng = geo.angular_height_to_range(y2 - y1, PERSON_H)

        chin_el = None

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
                    chin_cache[bi] = (chin_y - y1) / max(1, (y2 - y1))

        if bi in chin_cache:
            chin_y = y1 + chin_cache[bi] * (y2 - y1)
            _, chin_el = geo.pixel_to_angles(cx, chin_y)
            cv2.line(frame, (x1, int(chin_y)), (x2, int(chin_y)),
                     (0, 0, 255), 2)

        detections.append({"az": az, "el": el,
                           "chin_el": chin_el, "range_m": rng,
                           "px": (cx, cy), "box": (x1, y1, x2, y2)})

    tracks, newly = tracker.step(detections, t)

    for tr in newly:
        print(f"*** NEW PERSON  track {tr.id} ***")
        if t - last_played.get(tr.id, -999) > COOLDOWN:
            SND.play()
            last_played[tr.id] = t

    for d in detections:
        best, bd = None, 6.0
        for tr in tracks:
            dist = ((d["az"]-tr.x[0])**2 + (d["el"]-tr.x[1])**2) ** 0.5
            if dist < bd:
                best, bd = tr, dist
        if best:
            x1, y1, _, _ = d["box"]
            cv2.putText(frame, f"ID {best.id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

    primary = tracker.primary()
    if primary:
        lead_az, lead_el = primary.lead(LEAD_S)
    else:
        lead_az = lead_el = 0.0
    cmd = gate.resolve(primary, lead_az, lead_el, t)

    if primary:
        K = geo.K
        px = int(K[0,0] * math.tan(math.radians(cmd.az_deg)) + K[0,2])
        py = int(K[1,1] * math.tan(math.radians(-cmd.el_deg)) + K[1,2])
        color = (0,0,255) if cmd.emitter_on else (128,128,128)
        cv2.line(frame, (px-25, py), (px+25, py), color, 2)
        cv2.line(frame, (px, py-25), (px, py+25), color, 2)
        cv2.circle(frame, (px, py), 12, color, 2)

    dt_ms = (time.perf_counter() - t0) * 1000
    hud = [
        f"{1000/dt_ms:.1f} fps   lead {LEAD_S*1000:.0f} ms",
        f"tracks {len(tracks)}",
        f"EMITTER {'ON' if cmd.emitter_on else 'off'}   {cmd.reason}",
        f"clamped {cmd.clamped}   az{cmd.az_deg:+.1f} el{cmd.el_deg:+.1f}",
    ]
    for i, line in enumerate(hud):
        c = (0,0,255) if (i == 2 and cmd.emitter_on) else (0,255,255)
        cv2.putText(frame, line, (10, 35 + i*32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)

    cv2.imshow("phase1", frame)
    frame_i += 1
    if cv2.waitKey(1) == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
