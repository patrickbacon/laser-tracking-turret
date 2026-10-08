import numpy as np


class Track:
    _next_id = 0

    def __init__(self, az, el, t, meas_noise, q_pos, q_vel):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = np.array([az, el, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 100.0, 100.0])
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.R = np.eye(2) * (meas_noise ** 2)
        self.last_update = t
        self.frames_seen = 1
        self.confirmed = False
        self.chin_el = None
        self.range_m = float("inf")

    def _F(self, dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt):
        return np.diag([(self.q_pos*dt)**2, (self.q_pos*dt)**2,
                        (self.q_vel*dt)**2, (self.q_vel*dt)**2])

    def predict(self, dt):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, az, el, t):
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = np.array([az, el]) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.last_update = t
        self.frames_seen += 1

    def lead(self, ahead_s):
        return (self.x[0] + self.x[2]*ahead_s,
                self.x[1] + self.x[3]*ahead_s)

    @property
    def angles(self):
        return self.x[0], self.x[1]

    def staleness(self, t):
        return t - self.last_update


class Tracker:
    def __init__(self, cfg):
        c = cfg["tracking"]
        self.gate = c["gate_deg"]
        self.max_coast = c["max_coast_s"]
        self.confirm_frames = c["confirm_frames"]
        self.meas_noise = c["measurement_noise"]
        self.q_pos = c["process_noise_pos"]
        self.q_vel = c["process_noise_vel"]
        self.tracks = []
        self._last_t = None

    def step(self, detections, t):
        dt = 0.0 if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t

        for tr in self.tracks:
            tr.predict(dt)
            tr.chin_el = None

        unmatched = list(range(len(detections)))
        for tr in self.tracks:
            best, best_d = None, self.gate
            for di in unmatched:
                d = detections[di]
                dist = np.hypot(d["az"] - tr.x[0], d["el"] - tr.x[1])
                if dist < best_d:
                    best, best_d = di, dist
            if best is not None:
                d = detections[best]
                tr.update(d["az"], d["el"], t)
                tr.chin_el = d.get("chin_el")
                tr.range_m = d.get("range_m", float("inf"))
                unmatched.remove(best)

        for di in unmatched:
            d = detections[di]
            tr = Track(d["az"], d["el"], t,
                       self.meas_noise, self.q_pos, self.q_vel)
            tr.chin_el = d.get("chin_el")
            tr.range_m = d.get("range_m", float("inf"))
            self.tracks.append(tr)

        self.tracks = [tr for tr in self.tracks
                       if tr.staleness(t) < self.max_coast]

        newly_confirmed = []
        for tr in self.tracks:
            if not tr.confirmed and tr.frames_seen >= self.confirm_frames:
                tr.confirmed = True
                newly_confirmed.append(tr)

        return self.tracks, newly_confirmed

    def primary(self):
        c = [t for t in self.tracks if t.confirmed]
        if not c:
            return None
        return min(c, key=lambda t: np.hypot(t.x[0], t.x[1]))
