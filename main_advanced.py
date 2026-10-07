import argparse
import time
import cv2
from advanced_vision import detect_objects, draw_results, Config

def parse_args():
    p = argparse.ArgumentParser(description="Advanced HSV shape/color detector")
    p.add_argument("--source", default="0", help="camera index, image path, or video path")
    p.add_argument("--show-mask", action="store_true", help="show segmentation mask")
    p.add_argument("--width", type=int, default=960, help="processing width")
    p.add_argument("--min-area", type=int, default=1200)
    p.add_argument("--min-sat", type=int, default=45)
    return p.parse_args()

def open_source(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)

def main():
    args = parse_args()
    Config.min_area = args.min_area
    Config.min_saturation = args.min_sat

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    # For live camera, request a moderate resolution; processing code can downscale further.
    if args.source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            processed, hsv, mask, detections = detect_objects(frame, Config)

            now = time.perf_counter()
            dt = max(now - prev, 1e-6)
            prev = now
            instant = 1.0 / dt
            fps = 0.9 * fps + 0.1 * instant if fps else instant

            view = draw_results(processed, detections, fps)
            cv2.imshow("Advanced Object Detection", view)

            if args.show_mask:
                cv2.imshow("HSV Segmentation Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                cv2.imshow("HSV Segmentation Mask", mask)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
