"""Estimate the lit traffic-light colour from a detected crop."""
import cv2
import numpy as np

HSV_RANGES = {
    "red": ((0, 10), (165, 179)),
    "yellow": ((11, 35),),
    "green": ((40, 100),),
}
SATURATION_MIN = 70
VALUE_MIN = 170
MIN_COLOR_PIXEL_FRACTION = 0.003


def classify_light_color(crop_bgr, s_min=SATURATION_MIN, v_min=VALUE_MIN,
                         min_frac=MIN_COLOR_PIXEL_FRACTION):
    """Return (color, confidence), using saturated bright pixels in HSV."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0
    height, width = crop_bgr.shape[:2]
    if height < 6 or width < 3:
        return "unknown", 0.0
    margin_y, margin_x = int(height * 0.08), int(width * 0.08)
    crop = (
        crop_bgr[margin_y:height - margin_y, margin_x:width - margin_x]
        if height - 2 * margin_y > 2 and width - 2 * margin_x > 2
        else crop_bgr
    )
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    lit = (saturation >= s_min) & (value >= v_min)
    counts = {}
    for color, hue_ranges in HSV_RANGES.items():
        color_mask = np.zeros(hue.shape, dtype=bool)
        for lower, upper in hue_ranges:
            color_mask |= (hue >= lower) & (hue <= upper)
        counts[color] = int(np.count_nonzero(lit & color_mask))
    minimum_pixels = max(3, int(min_frac * lit.size))
    if max(counts.values(), default=0) < minimum_pixels:
        return "unknown", 0.0
    winner = max(counts, key=counts.get)
    confidence = counts[winner] / max(1, sum(counts.values()))
    return winner, float(confidence)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
