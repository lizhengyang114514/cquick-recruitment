import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs("outputs", exist_ok=True)

# 参数对比都用同一张图：橙色罐头
img = cv2.imread("test_images/01_orange_can.jpg")
img = cv2.resize(img, (600, 800))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((7, 7), np.uint8)


# 实验 A：只把 S 下限从 110 改成 40
print("--- 实验A：改 S 下限 ---")
mask_a1 = cv2.inRange(hsv, (5, 110, 90), (22, 255, 255))
mask_a1 = cv2.morphologyEx(mask_a1, cv2.MORPH_OPEN, kernel)
mask_a1 = cv2.morphologyEx(mask_a1, cv2.MORPH_CLOSE, kernel)
cnts, _ = cv2.findContours(mask_a1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
a1 = 0
for c in cnts:
    if cv2.contourArea(c) > a1:
        a1 = cv2.contourArea(c)
print("S>=110 白占比", round(mask_a1.mean() / 255 * 100, 2), "% 最大面积", int(a1))

mask_a2 = cv2.inRange(hsv, (5, 40, 90), (22, 255, 255))
mask_a2 = cv2.morphologyEx(mask_a2, cv2.MORPH_OPEN, kernel)
mask_a2 = cv2.morphologyEx(mask_a2, cv2.MORPH_CLOSE, kernel)
cnts2, _ = cv2.findContours(mask_a2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
a2 = 0
c2 = None
for c in cnts2:
    if cv2.contourArea(c) > a2:
        a2 = cv2.contourArea(c)
        c2 = c
print("S>=40  白占比", round(mask_a2.mean() / 255 * 100, 2), "% 最大面积", int(a2))
if c2 is not None:
    M = cv2.moments(c2)
    if M["m00"] != 0:
        print("S>=40 中心变成", int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), "原来是 279 261")


# 实验 B：不加形态学 vs 加开闭运算
print("--- 实验B：加不加形态学 ---")
mask_b1 = cv2.inRange(hsv, (5, 110, 90), (22, 255, 255))
cnts3, _ = cv2.findContours(mask_b1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
b1 = 0
c3 = None
for c in cnts3:
    if cv2.contourArea(c) > b1:
        b1 = cv2.contourArea(c)
        c3 = c
if c3 is not None:
    x, y, w, h = cv2.boundingRect(c3)
    print("不加形态学 白占比", round(mask_b1.mean() / 255 * 100, 2), "% 最大面积", int(b1), " 框", x, y, w, h)

mask_b2 = cv2.inRange(hsv, (5, 110, 90), (22, 255, 255))
mask_b2 = cv2.morphologyEx(mask_b2, cv2.MORPH_OPEN, kernel)
mask_b2 = cv2.morphologyEx(mask_b2, cv2.MORPH_CLOSE, kernel)
cnts4, _ = cv2.findContours(mask_b2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
b2 = 0
c4 = None
for c in cnts4:
    if cv2.contourArea(c) > b2:
        b2 = cv2.contourArea(c)
        c4 = c
if c4 is not None:
    x, y, w, h = cv2.boundingRect(c4)
    print("加了开闭运算 白占比", round(mask_b2.mean() / 255 * 100, 2), "% 最大面积", int(b2), " 框", x, y, w, h)


# 实验 C：最小面积阈值 2000 改成 30000，看候选数量怎么变
print("--- 实验C：改最小面积阈值 ---")
n1 = 0
n2 = 0
for c in cnts4:
    a = cv2.contourArea(c)
    if a > 2000:
        n1 = n1 + 1
    if a > 30000:
        n2 = n2 + 1
print("阈值2000 候选", n1, "个   阈值30000 候选", n2, "个")


# 四张掩膜拼成一张对比图
a1_img = cv2.resize(cv2.cvtColor(mask_a1, cv2.COLOR_GRAY2BGR), (240, 320))
a2_img = cv2.resize(cv2.cvtColor(mask_a2, cv2.COLOR_GRAY2BGR), (240, 320))
b1_img = cv2.resize(cv2.cvtColor(mask_b1, cv2.COLOR_GRAY2BGR), (240, 320))
b2_img = cv2.resize(cv2.cvtColor(mask_b2, cv2.COLOR_GRAY2BGR), (240, 320))

cv2.putText(a1_img, "S110", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(a2_img, "S40", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(b1_img, "no morph", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.putText(b2_img, "open+close", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv2.imwrite("outputs/02_param_compare.png",
            np.vstack([np.hstack([a1_img, a2_img]), np.hstack([b1_img, b2_img])]))
print("对比图存到 outputs/02_param_compare.png")
