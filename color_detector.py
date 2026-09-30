import cv2
import numpy as np

def get_color(hsv, c): # Đã đổi 'image' thành 'hsv'
    # Tạo mask cho contour
    mask = np.zeros(hsv.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Lấy các pixel nằm trong vật thể trực tiếp từ biến hsv truyền vào
    pixels = hsv[mask == 255]

    if len(pixels) == 0:
        return "Unknown"

    # Lấy HSV trung vị
    median_hsv = np.median(pixels, axis=0)
    H = median_hsv[0]
    S = median_hsv[1]
    V = median_hsv[2]


    # ==========================================
    # NHẬN DẠNG MÀU
    # ==========================================

    # ĐỎ
    if (H <= 10 or H >= 170) and S > 80 and V > 50:
        return "Red"

    # CAM
    elif 10 < H <= 20 and S > 80:
        return "Orange"

    # VÀNG
    elif 20 < H <= 35 and S > 80:
        return "Yellow"

    # XANH LÁ
    elif 35 < H <= 85 and S > 60:
        return "Green"

    # XANH DƯƠNG
    elif 85 < H <= 130 and S > 60:
        return "Blue"

    # TÍM
    elif 130 < H <= 160 and S > 60:
        return "Purple"

    # HỒNG
    elif 160 < H < 170 and S > 60:
        return "Pink"

    # XÁM
    elif S < 50 and V < 200:
        return "Gray"

    # TRẮNG
    elif S < 50 and V >= 200:
        return "White"

    # ĐEN
    elif V < 50:
        return "Black"

    else:
        return "Unknown"