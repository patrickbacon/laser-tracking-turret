from adafruit_servokit import ServoKit
import time

kit = ServoKit(channels=16)

print("centering both")
kit.servo[0].angle = 90
kit.servo[1].angle = 90
time.sleep(2)

for ch in (0, 1):
    print(f"sweeping channel {ch}")
    for a in (60, 120, 90):
        kit.servo[ch].angle = a
        time.sleep(0.8)

print("done")
