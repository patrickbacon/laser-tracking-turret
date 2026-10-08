# Automated Laser Tracking Turret



![Demo](docs/demo.gif)



A Raspberry Pi 5 pan-tilt turret that detects people in real time, locates facial landmarks, and predicts target motion to keep a laser aimed ahead of a moving subject.

## Status
- **Phase 1 – Vision & Tracking:** Complete
- **Phase 2 – Mechanical Integration:** In progress
- **Phase 3 – Emitter & Audio Integration:** Planned

## Features
- Person detection with YOLOv8n (NCNN) at 7–8 FPS on the Pi 5
- Facial landmark detection with YuNet
- Kalman filter tracking in angle space
- Predictive aiming with 300 ms lead to offset vision and servo latency
- Safety gate that clamps all aim commands below a computed chin elevation
- Calibrated camera model (OpenCV chessboard intrinsics and distortion correction)
- Session recording and replay tools for offline tuning

## Hardware
| Component | Purpose |
|---|---|
| Raspberry Pi 5 | Vision processing and control |
| Arducam IMX477 | Camera |
| PCA9685 | 16-channel PWM servo driver (I2C) |
| 2x MG996R metal-gear servos | Pan and tilt |
| Aluminum pan-tilt bracket | Turret frame |
| ALITOVE 5V 5A supply | Servo power |
| Green laser pointer | Manually operated emitter |
| Custom CAD camera plate | Camera and laser mount |

## How It Works
1. The camera frame is undistorted with calibrated intrinsics.
2. YOLOv8n detects people; YuNet finds facial landmarks within each detection.
3. Pixel positions are converted to azimuth and elevation angles.
4. A Kalman filter tracks the target and predicts its position 300 ms ahead.
5. The safety gate validates each aim command before it reaches the servos.
6. Angles are sent to the PCA9685, which drives the pan and tilt servos.

## Repository Structure
- `turret/` – core modules (geometry, tracking, safety, aiming, main loop)
- `tools/` – calibration, recording, replay, and servo test utilities
- `config.yaml` – camera calibration and tuning parameters
- `docs/` – photos and demo media

## Future Improvements
- 90° sweep scanning with dwell-weighted search patterns
- Wider sensor mode for increased field of view
- Reduced pipeline latency for faster targets
