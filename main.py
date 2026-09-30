import cv2
import numpy as np
import pyrealsense2 as rs

# 1. IMPORT TỪ CÁC CHƯƠNG TRÌNH CON
from color_detector import get_color
from centre_of_shape import findCentroid
from shapedetector import detect
from spatial import get_median_depth, pixel_to_camera
from drawing import draw_camera_axes, draw_object_projection


def main():
    # Khởi tạo Camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()

    print("\n================ CAMERA INTRINSICS ================")
    print(f"fx = {color_intrinsics.fx}, fy = {color_intrinsics.fy}")
    print(f"cx = {color_intrinsics.ppx}, cy = {color_intrinsics.ppy}")
    print("====================================================\n")

    try:
        while True:
            # Lấy frame và align
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            aligned_depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # Tạo mask để tìm contours (Lọc màu và độ sáng mặt trên)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, np.array([0, 60, 30]), np.array([179, 255, 255]))
            bright_mask = cv2.inRange(hsv[:, :, 2], 120, 255)
            top_mask = cv2.bitwise_and(color_mask, bright_mask)

            kernel = np.ones((7, 7), np.uint8)
            top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            camera_origin = (int(round(color_intrinsics.ppx)), int(round(color_intrinsics.ppy)))
            object_id = 1

            for contour in contours:
                if cv2.contourArea(contour) < 2500:
                    continue

                # ====================================================
                # GỌI 3 HÀM TỪ 3 CHƯƠNG TRÌNH CŨ CỦA BẠN
                # ====================================================

                # 1. Tìm trọng tâm
                try:
                    cx, cy = findCentroid(contour)
                except ZeroDivisionError:
                    continue  # Bỏ qua nếu m00 = 0 để tránh lỗi văng chương trình

                # 2. Nhận dạng màu (Hàm của bạn dùng ảnh gốc image)
                color = get_color(image, contour)

                # 3. Nhận dạng hình dáng
                shape = detect(contour)

                result = f"{color} {shape}"

                # ====================================================
                # XỬ LÝ 3D DEPTH VÀ TỌA ĐỘ CAMERA
                # ====================================================
                depth_m = get_median_depth(depth_frame, cx, cy, radius=4)
                if depth_m > 0:
                    X, Y, Z = pixel_to_camera(cx, cy, depth_m, aligned_depth_intrinsics)
                    X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000
                else:
                    X_mm, Y_mm, Z_mm = 0, 0, 0

                # In ra terminal
                print(
                    f"{object_id}. {result} | Center=({cx},{cy}) | Depth={Z_mm:.1f} mm | Cam XYZ=({X_mm:.1f}, {Y_mm:.1f}, {Z_mm:.1f}) mm")

                # ====================================================
                # HIỂN THỊ ĐỒ HỌA
                # ====================================================
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
                cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(image, result, (max(cx - 80, 5), max(cy - 35, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                            (0, 0, 0), 2)
                cv2.putText(image, f"Pixel ({cx},{cy})", (max(cx - 70, 5), cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (0, 0, 0), 1)

                if depth_m > 0:
                    xyz_text = f"Cam XYZ ({X_mm:.0f},{Y_mm:.0f},{Z_mm:.0f})mm"
                    cv2.putText(image, xyz_text, (max(cx - 100, 5), cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),
                                1)
                    draw_object_projection(image, (cx, cy), camera_origin, X_mm, Y_mm)
                else:
                    cv2.putText(image, "Depth invalid", (max(cx - 60, 5), cy + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (0, 0, 255), 2)

                object_id += 1

            # Vẽ hệ trục Camera
            image, _ = draw_camera_axes(image, color_intrinsics)

            cv2.namedWindow("D435i Object Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("D435i Object Detection", image)
            cv2.namedWindow("Top Surface Mask", cv2.WINDOW_NORMAL)
            cv2.imshow("Top Surface Mask", top_mask)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()