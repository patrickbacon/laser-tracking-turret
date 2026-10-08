from dataclasses import dataclass


@dataclass
class AimCommand:
    az_deg: float
    el_deg: float
    emitter_on: bool
    clamped: bool
    reason: str


class AimGate:
    def __init__(self, cfg):
        s = cfg["safety"]
        a = cfg["aim"]
        self.chin_margin = s["chin_margin_deg"]
        self.fallback_offset = s["fallback_el_offset_deg"]
        self.min_range = s["min_range_m"]
        self.min_age = s["min_track_age_frames"]
        self.max_stale = s["max_track_staleness_s"]
        self.bs_az = a["boresight_az_deg"]
        self.bs_el = a["boresight_el_deg"]
        self.az_min, self.az_max = a["az_min_deg"], a["az_max_deg"]
        self.el_min, self.el_max = a["el_min_deg"], a["el_max_deg"]

    def ceiling_for(self, track):
        if track.chin_el is not None:
            return track.chin_el - self.chin_margin
        return track.angles[1] - self.fallback_offset

    def resolve(self, track, lead_az, lead_el, now):
        if track is None:
            return AimCommand(0.0, 0.0, False, False, "no track")

        az = lead_az + self.bs_az
        el = lead_el + self.bs_el

        ceiling = self.ceiling_for(track) + self.bs_el
        clamped = el > ceiling
        if clamped:
            el = ceiling

        az = max(self.az_min, min(self.az_max, az))
        el = max(self.el_min, min(self.el_max, el))

        reason = ""
        if track.frames_seen < self.min_age:
            reason = "track too young"
        elif track.staleness(now) > self.max_stale:
            reason = "track stale"
        elif track.range_m < self.min_range:
            reason = f"too close ({track.range_m:.1f}m)"

        return AimCommand(az, el, reason == "", clamped, reason or "ok")
