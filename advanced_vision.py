import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class Detection:
    contour: np.ndarray
    bbox: tuple
    center: tuple
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    extent: float
    solidity: float
    shape: str
    color: str
    color_confidence: float

class Config:
    # Segmentation
    min_area = 1200
    max_area_ratio = 0.65
    min_saturation = 45
    min_value = 35
    dark_value = 55
    # Morphology
    open_kernel = 5
    close_kernel = 9
    # Contour
    approx_epsilon = 0.025
    # Color
    color_min_confidence = 0.45

def preprocess(frame, max_width=960):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # Bilateral filtering keeps edges sharper than a large Gaussian blur.
    return cv2.GaussianBlur(frame, (5, 5), 0)

def make_hsv_mask(frame, cfg=Config):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Main mask: chromatic objects.
    chromatic = cv2.inRange(
        hsv,
        np.array([0, cfg.min_saturation, cfg.min_value], np.uint8),
        np.array([179, 255, 255], np.uint8),
    )

    # Secondary mask: very dark objects. This makes black/dark objects detectable.
    dark = cv2.inRange(v, 0, cfg.dark_value)

    mask = cv2.bitwise_or(chromatic, dark)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Fill small holes inside detected objects.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= cfg.min_area * 0.35:
            cv2.drawContours(clean, [c], -1, 255, -1)

    return hsv, clean

def _circular_mean_hue(hues):
    if len(hues) == 0:
        return None
    angles = hues.astype(np.float32) * (2.0 * np.pi / 180.0)
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    angle = np.arctan2(s, c)
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * 180.0 / (2.0 * np.pi))

def classify_color(hsv, contour, cfg=Config):
    # Erode the contour mask so border/background pixels contribute less.
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(mask, kernel, iterations=1)
    if cv2.countNonZero(inner) < 20:
        inner = mask

    pixels = hsv[inner > 0]
    if len(pixels) == 0:
        return "Unknown", 0.0

    H, S, V = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    sat = float(np.percentile(S, 60))
    val = float(np.median(V))

    # Achromatic colors first.
    if sat < 35:
        if val < 55:
            return "Black", min(1.0, (55 - val) / 25.0 + 0.5)
        if val > 205:
            return "White", min(1.0, (val - 205) / 35.0 + 0.5)
        return "Gray", 0.8

    hue = _circular_mean_hue(H[S >= 45])
    if hue is None:
        return "Unknown", 0.0

    ranges = [
        ("Red", 0, 10),
        ("Orange", 10, 22),
        ("Yellow", 22, 38),
        ("Green", 38, 85),
        ("Cyan", 85, 100),
        ("Blue", 100, 130),
        ("Purple", 130, 155),
        ("Pink", 155, 179),
    ]

    # Red wraps around 0/179.
    if hue <= 10 or hue >= 170:
        name = "Red"
        distance = min(hue, 180 - hue)
    else:
        name, distance = min(
            ((n, min(abs(hue-a), abs(hue-b))) for n, a, b in ranges if a <= hue <= b),
            key=lambda x: x[1],
            default=("Unknown", 90),
        )

    # Confidence combines hue separation and saturation.
    confidence = float(np.clip(1.0 - distance / 18.0, 0.0, 1.0))
    confidence *= float(np.clip((sat - 35.0) / 80.0, 0.35, 1.0))
    return name, confidence

def classify_shape(contour, cfg=Config):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return "Unknown", 0.0, 0.0, 0.0, 0.0

    x, y, w, h = cv2.boundingRect(contour)
    ar = w / float(max(h, 1))
    extent = area / float(max(w * h, 1))
    circularity = 4.0 * np.pi * area / float(perimeter * perimeter)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(max(hull_area, 1))

    eps = cfg.approx_epsilon * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    n = len(approx)

    # Use multiple geometric features instead of only vertex count.
    if n == 3:
        shape = "Triangle"
    elif n == 4:
        (_, _, _, _w) = (x, y, w, h)
        # A rectangle is accepted even when perspective/rotation changes its box ratio.
        rect_ratio = min(ar, 1.0 / ar) if ar > 0 else 0
        if rect_ratio > 0.82 and extent > 0.68 and solidity > 0.90:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif 5 <= n <= 6 and solidity > 0.90:
        shape = "Polygon"
    elif circularity > 0.78 and solidity > 0.92:
        shape = "Circle"
    elif circularity > 0.58 and solidity > 0.80:
        shape = "Ellipse"
    else:
        shape = "Irregular"

    return shape, circularity, ar, extent, solidity

def detect_objects(frame, cfg=Config):
    processed = preprocess(frame)
    hsv, mask = make_hsv_mask(processed, cfg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    results = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.min_area or area > image_area * cfg.max_area_ratio:
            continue

        perimeter = cv2.arcLength(contour, True)
        shape, circularity, ar, extent, solidity = classify_shape(contour, cfg)

        M = cv2.moments(contour)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        color, color_conf = classify_color(hsv, contour, cfg)
        x, y, w, h = cv2.boundingRect(contour)

        results.append(Detection(
            contour=contour,
            bbox=(x, y, w, h),
            center=(cx, cy),
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            aspect_ratio=ar,
            extent=extent,
            solidity=solidity,
            shape=shape,
            color=color,
            color_confidence=color_conf,
        ))

    results.sort(key=lambda d: d.area, reverse=True)
    return processed, hsv, mask, results

def draw_results(frame, detections, fps=None):
    out = frame.copy()
    for i, d in enumerate(detections, 1):
        cv2.drawContours(out, [d.contour], -1, (0, 220, 0), 2)
        x, y, w, h = d.bbox
        cx, cy = d.center

        cv2.rectangle(out, (x, y), (x+w, y+h), (255, 180, 0), 1)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        label = f"{i}: {d.color} {d.shape}"
        metrics = f"A:{int(d.area)} C:{d.circularity:.2f} S:{d.solidity:.2f}"
        conf = f"color:{d.color_confidence:.2f}"

        ty = max(22, y - 28)
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(out, metrics, (x, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(out, conf, (x, y+h+36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.1f} | Objects: {len(detections)}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return out
