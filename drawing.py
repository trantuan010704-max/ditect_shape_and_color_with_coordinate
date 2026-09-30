import cv2

def draw_camera_axes(image, intrinsics):
    origin_x = int(round(intrinsics.ppx))
    origin_y = int(round(intrinsics.ppy))
    origin = (origin_x, origin_y)

    axis_length_x = 500
    axis_length_y = 300
    h, w = image.shape[:2]

    # Trục X
    x_end = (min(origin_x + axis_length_x, w - 10), origin_y)
    cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 4, tipLength=0.10)
    cv2.putText(image, "Xc", (min(x_end[0] + 10, w - 50), max(x_end[1] - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Trục Y
    y_end = (origin_x, min(origin_y + axis_length_y, h - 10))
    cv2.arrowedLine(image, origin, y_end, (255, 0, 0), 4, tipLength=0.10)
    cv2.putText(image, "Yc", (min(y_end[0] + 10, w - 45), min(y_end[1] + 30, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Gốc O và Ghi chú
    cv2.circle(image, origin, 6, (0, 255, 0), -1)
    cv2.putText(image, "O", (max(origin_x - 30, 5), max(origin_y - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "Camera: X -> right, Y -> down, Z -> forward", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return image, origin

def draw_object_projection(image, object_center, origin, X_mm, Y_mm, label_color=(0, 255, 255)):
    u, v = object_center
    ox, oy = origin

    x_projection = (u, oy)
    y_projection = (ox, v)

    cv2.line(image, (u, v), x_projection, label_color, 2, cv2.LINE_AA)
    cv2.line(image, (u, v), y_projection, label_color, 2, cv2.LINE_AA)
    cv2.circle(image, x_projection, 5, (0, 255, 255), -1)
    cv2.circle(image, y_projection, 5, (0, 255, 255), -1)

    # Label X
    x_text = f"Xc={X_mm:.0f} mm"
    x_text_x = max(5, min(int((u + ox) / 2 - 45), image.shape[1] - 150))
    x_text_y = max(20, int(oy - 10))
    (tw, th), _ = cv2.getTextSize(x_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(image, (x_text_x - 4, x_text_y - th - 4), (x_text_x + tw + 4, x_text_y + 4), (0, 0, 0), -1)
    cv2.putText(image, x_text, (x_text_x, x_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, label_color, 2, cv2.LINE_AA)

    # Label Y
    y_text = f"Yc={Y_mm:.0f} mm"
    y_text_x = max(5, min(int(ox + 10), image.shape[1] - 150))
    y_text_y = max(25, min(int((v + oy) / 2), image.shape[0] - 10))
    (tw, th), _ = cv2.getTextSize(y_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    cv2.rectangle(image, (y_text_x - 4, y_text_y - th - 4), (y_text_x + tw + 4, y_text_y + 4), (0, 0, 0), -1)
    cv2.putText(image, y_text, (y_text_x, y_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, label_color, 2, cv2.LINE_AA)

    return image