import numpy as np

def get_median_depth(depth_frame, cx, cy, radius=4):
    values = []
    for y in range(max(0, cy - radius), min(depth_frame.get_height(), cy + radius + 1)):
        for x in range(max(0, cx - radius), min(depth_frame.get_width(), cx + radius + 1)):
            depth = depth_frame.get_distance(x, y)
            if depth > 0:
                values.append(depth)

    if len(values) == 0:
        return 0.0
    return float(np.median(values))

def pixel_to_camera(cx, cy, depth_m, intrinsics):
    fx = intrinsics.fx
    fy = intrinsics.fy
    camera_cx = intrinsics.ppx
    camera_cy = intrinsics.ppy

    # Công thức chiếu ngược: X = (u - cx)*Z/fx, Y = (v - cy)*Z/fy
    X = ((cx - camera_cx) * depth_m / fx)
    Y = ((cy - camera_cy) * depth_m / fy)
    Z = depth_m

    return X, Y, Z