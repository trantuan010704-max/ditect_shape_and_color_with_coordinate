import cv2


def findCentroid(c):
    M = cv2.moments(c)

    # Kiểm tra để tránh lỗi chia cho 0 khi nhiễu tạo ra contour không hợp lệ
    if M["m00"] == 0:
        return 0, 0

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY