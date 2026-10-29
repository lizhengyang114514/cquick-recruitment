import os

import cv2
import numpy as np

DEFAULT_COLOR = {
    "h_min": 5,
    "s_min": 110,
    "v_min": 90,
    "h_max": 22,
    "s_max": 255,
    "v_max": 255,
}


def detect(image, h_min, s_min, v_min, h_max, s_max, v_max, min_area):
    result = {
        "detected": False,
        "u": 0.0,
        "v": 0.0,
        "area": 0.0,
        "contour": None,
        "mask": None,
    }

    if image is None:
        return result

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([h_min, s_min, v_min], np.uint8),
                       np.array([h_max, s_max, v_max], np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a > min_area and a > best_area:
            best = c
            best_area = a

    result["mask"] = mask
    if best is None:
        return result

    m = cv2.moments(best)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(best)
        u, v = float(x + w / 2), float(y + h / 2)
    else:
        u, v = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

    result["detected"] = True
    result["u"] = u
    result["v"] = v
    result["area"] = float(best_area)
    result["contour"] = best
    return result


def annotate(image, result):
    out = image.copy()
    if not result["detected"]:
        cv2.putText(out, "NOT DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return out

    contour = result["contour"]
    x, y, w, h = cv2.boundingRect(contour)
    u, v = int(result["u"]), int(result["v"])
    cv2.drawContours(out, [contour], -1, (0, 255, 0), 3)
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(out, (u, v), 9, (0, 0, 255), -1)
    cv2.putText(out, "u=" + str(u) + " v=" + str(v), (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


def list_images(folder):
    names = []
    if not os.path.isdir(folder):
        return names
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            names.append(os.path.join(folder, name))
    return names
