import time

import cv2

from src.config import ROOT


def distance_color(d):
    """BGR box colour by distance: red <3 m, amber 3-8 m, green >8 m."""
    if d is None:
        return (200, 200, 200)
    if d < 3:
        return (0, 0, 255)
    if d < 8:
        return (0, 200, 255)
    return (0, 200, 0)


def draw_box(img, box, text, color):
    x1, y1, x2, y2 = map(int, box)
    scale = max(0.5, img.shape[1] / 1600)
    th = max(1, int(round(scale * 2)))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    (tw, t_h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    ytop = max(0, y1 - t_h - base - 4)
    cv2.rectangle(img, (x1, ytop), (x1 + tw + 4, ytop + t_h + base + 4), color, -1)
    cv2.putText(img, text, (x1 + 2, ytop + t_h + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), th, cv2.LINE_AA)


def display_name(class_name):
    return {
        "barrier": "Barrier",
        "cone": "Cone",
        "signBoard": "Sign Board",
        "trafficLight": "Traffic Light",
    }.get(class_name, class_name.replace("_", " ").title())


def draw_status(img, stop, fps):
    text = f"STOP (1)  |  FPS: {fps:.1f}" if stop else f"GO (0)  |  FPS: {fps:.1f}"
    color = (0, 0, 220) if stop else (0, 150, 0)
    scale = max(0.65, img.shape[1] / 1200)
    thickness = max(2, int(scale * 2))
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    cv2.rectangle(img, (0, 0), (width + 24, height + baseline + 18), color, -1)
    cv2.putText(img, text, (12, height + 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)


class FPSCounter:
    def __init__(self):
        self._last = time.perf_counter()
        self._fps = 0.0

    def tick(self):
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        return self._fps