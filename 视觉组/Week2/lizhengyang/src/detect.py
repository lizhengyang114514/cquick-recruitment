import cv2
import numpy as np
import os

# 从哪个目录运行都先切到项目根，不然 PyCharm 里读不到 test_images
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("outputs", exist_ok=True)

MIN_AREA = 2000
kernel = np.ones((7, 7), np.uint8)


# 第 1 张：橙色罐头，用橙色范围找
print("--- 01_orange_can.jpg ---")
img = cv2.imread("test_images/01_orange_can.jpg")
if img is None:
    print("读不到图片")
else:
    img = cv2.resize(img, (600, 800))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # S 下限 110 是为了别把灰桌子算进来
    mask = cv2.inRange(hsv, (5, 110, 90), (22, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("outputs/01_mask.png", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数", len(cnts))

    best = None
    max_area = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > MIN_AREA and a > max_area:
            best = c
            max_area = a

    if best is None:
        print("没找到目标")
    else:
        x, y, w, h = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M["m00"] == 0:
            u = x + w // 2
            v = y + h // 2
        else:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
        print("u =", u, " v =", v, " 面积 =", int(max_area))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(img, "u=%d v=%d" % (u, v), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite("outputs/01_result.png", img)


# 第 2 张：黄绿玩偶。这个玩偶其实是黄绿色，H 大概 25 左右，不是纯绿
print("--- 02_green_plush.jpg ---")
img = cv2.imread("test_images/02_green_plush.jpg")
if img is None:
    print("读不到图片")
else:
    img = cv2.resize(img, (600, 647))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 90, 50), (35, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("outputs/02_mask.png", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数", len(cnts))

    best = None
    max_area = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > MIN_AREA and a > max_area:
            best = c
            max_area = a

    if best is None:
        print("没找到目标")
    else:
        x, y, w, h = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M["m00"] == 0:
            u = x + w // 2
            v = y + h // 2
        else:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
        print("u =", u, " v =", v, " 面积 =", int(max_area))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(img, "u=%d v=%d" % (u, v), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite("outputs/02_result.png", img)


# 第 3 张：蓝色背包，故意用橙色范围去找，用来当误检的例子
print("--- 03_blue_bag.jpg 用橙色范围找 ---")
img = cv2.imread("test_images/03_blue_bag.jpg")
if img is None:
    print("读不到图片")
else:
    img = cv2.resize(img, (600, 800))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (5, 110, 90), (22, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("outputs/03_mask.png", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数", len(cnts))

    best = None
    max_area = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > MIN_AREA and a > max_area:
            best = c
            max_area = a

    if best is None:
        print("没找到目标")
    else:
        x, y, w, h = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M["m00"] == 0:
            u = x + w // 2
            v = y + h // 2
        else:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
        print("u =", u, " v =", v, " 面积 =", int(max_area), " <- 这个是误检")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(img, "u=%d v=%d" % (u, v), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite("outputs/03_result.png", img)


# 第 4 次：还是黄绿玩偶，但用橙色范围找，应该找不到
print("--- 02_green_plush.jpg 用橙色范围找 ---")
img = cv2.imread("test_images/02_green_plush.jpg")
if img is None:
    print("读不到图片")
else:
    img = cv2.resize(img, (600, 647))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (5, 110, 90), (22, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("outputs/04_mask.png", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数", len(cnts))

    best = None
    max_area = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > MIN_AREA and a > max_area:
            best = c
            max_area = a

    if best is None:
        print("没找到目标 detected=False")
        cv2.putText(img, "NOT DETECTED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        cv2.imwrite("outputs/04_result.png", img)
    else:
        x, y, w, h = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M["m00"] == 0:
            u = x + w // 2
            v = y + h // 2
        else:
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])
        print("u =", u, " v =", v, " 面积 =", int(max_area))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.imwrite("outputs/04_result.png", img)

print("done")
