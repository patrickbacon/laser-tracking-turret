import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2, yaml
from turret.geometry import Camera
from turret.tracking import Tracker, Track
from turret.safety import AimGate

if len(sys.argv) < 2:
    print("usage: python3 tools/replay.py <session_name>")
    print("sessions:", os.listdir("sessions") if os.path.isdir("sessions") else "none")
    sys.exit(1)

sess = f"sessions/{sys.argv[1]}"
cfg = yaml.safe_load(open("config.yaml"))
geo = Camera(cfg)
LEAD_S = cfg["tracking"]["predict_ahead_ms"] / 1000.0

rows = [json.loads(l) for l in open(f"{sess}/detections.jsonl")]
print(f"{len(rows)} frames.  space=pause  n=step  r=restart  q=quit")


def run():
    Track._next_id = 0
    return Tracker(cfg), AimGate(cfg)


tracker, gate = run()
idx = 0
paused = False

while idx < len(rows):
    row = rows[idx]
    frame = cv2.imread(f"{sess}/{row['i']:06d}.jpg")
    if frame is None:
        idx += 1
        continue

    dets = []
    for d in row["dets"]:
        dets.append({"az": d["az"], "el": d["el"],
                     "chin_el": d["chin_el"],
                     "range_m": d["range_m"] if d["range_m"] else float("inf")})
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        if d["chin_py"]:
            cy = int(d["chin_py"])
            cv2.line(frame, (x1, cy), (x2, cy), (0, 0, 255), 2)

    tracks, newly = tracker.step(dets, row["t"])
    for tr in newly:
        print(f"frame {row['i']}: NEW track {tr.id}")

    primary = tracker.primary()
    lead_az, lead_el = primary.lead(LEAD_S) if primary else (0.0, 0.0)
    cmd = gate.resolve(primary, lead_az, lead_el, row["t"])

    K = geo.K
    if primary:
        for (a, e, col, r) in [(primary.x[0], primary.x[1], (255, 180, 0), 8),
                               (cmd.az_deg, cmd.el_deg,
                                (0, 0, 255) if cmd.emitter_on else (150, 150, 150), 14)]:
            px = int(K[0, 0]*math.tan(math.radians(a)) + K[0, 2])
            py = int(K[1, 1]*math.tan(math.radians(-e)) + K[1, 2])
            cv2.circle(frame, (px, py), r, col, 2)
            cv2.line(frame, (px-20, py), (px+20, py), col, 1)
            cv2.line(frame, (px, py-20), (px, py+20), col, 1)

    hud = [
        f"frame {idx}/{len(rows)}   lead {LEAD_S*1000:.0f} ms",
        f"qvel {cfg['tracking']['process_noise_vel']}  mnoise {cfg['tracking']['measurement_noise']}",
        f"tracks {len(tracks)}   EMITTER {'ON' if cmd.emitter_on else 'off'}  {cmd.reason}",
        f"clamped {cmd.clamped}",
    ]
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (10, 35 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "blue=now  red=predicted", (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("replay", frame)

    k = cv2.waitKey(0 if paused else 40) & 0xFF
    if k == ord('q'):
        break
    elif k == ord(' '):
        paused = not paused
    elif k == ord('n'):
        paused = True
        idx += 1
        continue
    elif k == ord('r'):
        tracker, gate = run()
        idx = 0
        continue
    idx += 1

cv2.destroyAllWindows()
