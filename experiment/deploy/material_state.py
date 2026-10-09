"""MaterialStateDecider — production decision layer on top of Ultralytics YOLO26-seg.

Design (research-backed):
  - Event-based, NOT per-frame emission (FEQS Appl. Sci. 2019; Lu et al. RCR 2022).
  - Presence gate with k-of-n debounce/hysteresis (industrial debounce practice).
  - Votes = conf * sieved mask area per class per frame (confidence-weighted,
    cf. Frame-Importance-Voting / M2CAI temporal averaging).
  - Emit ONE class per event (truck visit) + margin + uncertainty flag.

Usage:
    python material_state.py --video in.mp4 --model ../runs/final_best/weights/best.pt
    # or streaming:
    decider = MaterialStateDecider("best.pt")
    decider.feed(frame_bgr)            # -> DecisionEvent | None
    decider.close()                    # flush open event
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = {0: "gravel", 1: "sand"}


class GateState(Enum):
    EMPTY = auto()        # no truck/material
    ENTERING = auto()     # candidate frames accumulating
    PRESENT = auto()      # collecting votes
    LEAVING = auto()      # candidate empty frames accumulating


@dataclass
class DecisionEvent:
    decision: str          # "gravel" | "sand" | "no-material"
    sand_frac: float       # weighted fraction of sand mass (0..1)
    margin: float          # |sand_frac-0.5|*2  (1.0 = decisive)
    n_frames: int          # frames that contributed votes
    coverage_mean: float   # mean material coverage while present
    uncertain: bool        # margin < min_margin -> human review
    t_start: float
    t_end: float


@dataclass
class FrameObs:
    masses: dict           # {cls: conf*area summed}
    coverage: float        # total sieved mask fraction
    n_inst: int


class MaterialStateDecider:
    def __init__(self, model: str, imgsz: int = 640, conf: float = 0.25,
                 area_min: float = 0.005, coverage_gate: float = 0.02,
                 k_enter: int = 3, k_leave: int = 5,
                 min_margin: float = 0.30, stride: int = 1,
                 device: int | str = 0, temporal_boost: float = 0.0):
        self.model = YOLO(str(model))
        self.imgsz, self.conf, self.area_min = imgsz, conf, area_min
        self.coverage_gate = coverage_gate
        self.k_enter, self.k_leave = k_enter, k_leave
        self.min_margin = min_margin
        self.stride = max(1, stride)
        self.device = device
        self.temporal_boost = temporal_boost     # 0 disables attention-area trick

        self.state = GateState.EMPTY
        self._enter_run = self._leave_run = 0
        self._votes: list[dict] = []
        self._cover: list[float] = []
        self._t0 = 0.0
        self._frame_i = 0
        self._prev_hi: list[dict] = []           # temporal memory
        self.events: list[DecisionEvent] = []

    # ------------------------------------------------ perception
    def _observe(self, frame) -> FrameObs:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=0.001,
                               device=self.device, retina_masks=True,
                               verbose=False)[0]
        masses, cover, n = {0: 0.0, 1: 0.0}, 0.0, 0
        if r.masks is not None:
            H, W = frame.shape[:2]
            preds = []
            for c, cf, m in zip(r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy(),
                                r.masks.data.cpu().numpy()):
                mk = cv2.resize(m, (W, H)) > 0.5
                a = float(mk.mean())
                preds.append((int(c), float(cf), a, mk))
            # optional temporal boost: overlap with previous hi-conf boxes
            if self.temporal_boost > 0:
                hi_new = []
                for i, (c, cf, a, mk) in enumerate(preds):
                    bb = self._bbox(mk)
                    for qc, qb in self._prev_hi:
                        if qc == c and bb is not None and qb is not None \
                                and self._box_iou(bb, qb) > 0.3:
                            preds[i] = (c, min(1.0, cf + self.temporal_boost), a, mk)
                            break
                    if cf >= 0.35:
                        hi_new.append((c, bb))
                self._prev_hi = hi_new
            for c, cf, a, mk in preds:
                if cf >= self.conf and a >= self.area_min:
                    masses[c] += cf * a
                    cover += a
                    n += 1
        return FrameObs(masses, cover, n)

    @staticmethod
    def _bbox(m):
        ys, xs = np.where(m)
        return None if not len(xs) else (xs.min(), ys.min(), xs.max(), ys.max())

    @staticmethod
    def _box_iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        i = max(0, x1 - x0) * max(0, y1 - y0)
        den = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
        return i / den if den and i else 0.0

    # ------------------------------------------------ state machine
    def feed(self, frame, timestamp: float | None = None) -> DecisionEvent | None:
        """Feed one BGR frame. Returns a DecisionEvent when an event closes."""
        self._frame_i += 1
        if self._frame_i % self.stride:
            return None
        t = timestamp if timestamp is not None else time.time()
        obs = self._observe(frame)
        occupied = obs.coverage > self.coverage_gate
        ev = None

        if self.state is GateState.EMPTY:
            self._enter_run = self._enter_run + 1 if occupied else 0
            if self._enter_run >= self.k_enter:
                self.state, self._votes, self._cover = GateState.PRESENT, [], []
                self._t0, self._leave_run = t, 0

        elif self.state is GateState.PRESENT:
            self._votes.append(obs.masses)
            self._cover.append(obs.coverage)
            self._leave_run = self._leave_run + 1 if not occupied else 0
            if self._leave_run >= self.k_leave:
                ev = self._emit(t)
                self.state = GateState.EMPTY
                self._enter_run = 0
                self._prev_hi = []
        return ev

    def _emit(self, t_end) -> DecisionEvent:
        g = sum(v[0] for v in self._votes)
        s = sum(v[1] for v in self._votes)
        tot = g + s
        frac = s / tot if tot else float("nan")
        dec = "no-material" if tot <= 0 else ("sand" if frac > 0.5 else "gravel")
        margin = abs(frac - 0.5) * 2 if tot else 0.0
        ev = DecisionEvent(dec, round(float(frac), 4) if tot else 0.0,
                           round(float(margin), 4), len(self._votes),
                           float(np.mean(self._cover)) if self._cover else 0.0,
                           margin < self.min_margin, self._t0, t_end)
        self.events.append(ev)
        return ev

    def close(self) -> DecisionEvent | None:
        """Flush an open event at end-of-stream."""
        if self.state is GateState.PRESENT and self._votes:
            ev = self._emit(time.time())
            self.state = GateState.EMPTY
            return ev
        return None

    # ------------------------------------------------ batch entry point
    def process_video(self, video: str | Path, stride: int | None = None,
                      overlay_out: str | Path | None = None) -> DecisionEvent | None:
        stride = stride or self.stride
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vw = None
        if overlay_out:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = cv2.VideoWriter(str(overlay_out),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i, last = 0, None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = i / fps
            ev = self.feed(fr, timestamp=t)
            if ev:
                print(f"[event] {ev.decision} frac={ev.sand_frac} "
                      f"margin={ev.margin} n={ev.n_frames}")
                last = ev
            if vw:
                vw.write(fr)
            i += 1
        cap.release()
        if vw:
            vw.release()
        ev = self.close()
        return ev or last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="../runs/final_best/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--temporal-boost", type=float, default=0.0)
    ap.add_argument("--overlay", default=None)
    a = ap.parse_args()
    d = MaterialStateDecider(a.model, conf=a.conf, stride=a.stride,
                             temporal_boost=a.temporal_boost)
    ev = d.process_video(a.video, overlay_out=a.overlay)
    if ev:
        print("\n=== FINAL DECISION ===")
        print(f"  {ev.decision.upper()}  sand_frac={ev.sand_frac}  "
              f"margin={ev.margin}  frames={ev.n_frames}  "
              f"{'[UNCERTAIN -> human review]' if ev.uncertain else '[confident]'}")
    else:
        print("no event detected")


if __name__ == "__main__":
    main()
